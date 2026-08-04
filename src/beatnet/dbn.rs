//! madmom `DBNDownBeatTrackingProcessor` port (BeatNet offline decoder).
//!
//! Faithful Rust translation of:
//! - `madmom.features.beats_hmm` (BarStateSpace, BarTransitionModel,
//!   RNNDownBeatTrackingObservationModel)
//! - `madmom.ml.hmm.HiddenMarkovModel.viterbi`
//! - `madmom.features.downbeats.DBNDownBeatTrackingProcessor`

use std::sync::OnceLock;

const MIN_BPM: f64 = 55.0;
const MAX_BPM: f64 = 215.0;
const NUM_TEMPI: usize = 60;
const TRANSITION_LAMBDA: f64 = 100.0;
const OBSERVATION_LAMBDA: i32 = 16;
const THRESHOLD: f32 = 0.05;
const FPS: f64 = 50.0;

/// Detected beat: time in seconds + bar position (1 = downbeat).
#[derive(Clone, Debug)]
pub struct BeatEvent {
	pub time: f32,
	pub beat_number: u32,
}

struct TransitionModel {
	/// CSR: states transitioning *to* s are in `states[pointers[s]..pointers[s+1]]`
	states: Vec<u32>,
	pointers: Vec<u32>,
	log_probabilities: Vec<f64>,
}

impl TransitionModel {
	fn num_states(&self) -> usize {
		self.pointers.len().saturating_sub(1)
	}

	fn make_sparse(dest: &[u32], prev: &[u32], probs: &[f64]) -> Self {
		assert_eq!(dest.len(), prev.len());
		assert_eq!(dest.len(), probs.len());
		let num_states = prev.iter().copied().max().unwrap_or(0) as usize + 1;

		// Build CSR by columns = destination state (madmom: csr_matrix((prob, (states, prev_states))))
		// scipy csr: data at (row=states/dest, col=prev_states)
		// indices = column indices of nonzeros = prev_states for each dest row
		// For each dest state s, tm_states[pointers[s]:pointers[s+1]] = previous states
		let mut triples: Vec<(u32, u32, f64)> = dest
			.iter()
			.zip(prev.iter())
			.zip(probs.iter())
			.map(|((&d, &p), &pr)| (d, p, pr))
			.collect();
		// Sort by dest then prev for CSR
		triples.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

		let mut states = Vec::with_capacity(triples.len());
		let mut probabilities = Vec::with_capacity(triples.len());
		let mut pointers = vec![0u32; num_states + 1];
		let mut cursor = 0u32;
		let mut current_dest = 0u32;

		for (d, p, pr) in triples {
			while current_dest < d {
				current_dest += 1;
				pointers[current_dest as usize] = cursor;
			}
			if current_dest == d {
				states.push(p);
				probabilities.push(pr);
				cursor += 1;
			}
		}
		while (current_dest as usize) < num_states {
			current_dest += 1;
			pointers[current_dest as usize] = cursor;
		}

		let log_probabilities: Vec<f64> = probabilities.iter().map(|&p| p.ln()).collect();
		Self {
			states,
			pointers,
			log_probabilities,
		}
	}
}

struct BarStateSpace {
	num_beats: usize,
	num_states: usize,
	state_positions: Vec<f64>,
	state_intervals: Vec<i32>,
	first_states: Vec<Vec<usize>>,
	last_states: Vec<Vec<usize>>,
}

impl BarStateSpace {
	fn new(num_beats: usize, min_interval: f64, max_interval: f64, num_intervals: Option<usize>) -> Self {
		let beat = BeatStateSpace::new(min_interval, max_interval, num_intervals);
		let mut state_positions = Vec::new();
		let mut state_intervals = Vec::new();
		let mut first_states = Vec::new();
		let mut last_states = Vec::new();
		let mut num_states = 0usize;

		for b in 0..num_beats {
			for &pos in &beat.state_positions {
				state_positions.push(pos + b as f64);
			}
			state_intervals.extend_from_slice(&beat.state_intervals);
			first_states.push(beat.first_states.iter().map(|s| s + num_states).collect());
			last_states.push(beat.last_states.iter().map(|s| s + num_states).collect());
			num_states += beat.num_states;
		}

		Self {
			num_beats,
			num_states,
			state_positions,
			state_intervals,
			first_states,
			last_states,
		}
	}
}

struct BeatStateSpace {
	num_states: usize,
	state_positions: Vec<f64>,
	state_intervals: Vec<i32>,
	first_states: Vec<usize>,
	last_states: Vec<usize>,
}

impl BeatStateSpace {
	fn new(min_interval: f64, max_interval: f64, num_intervals: Option<usize>) -> Self {
		let min_i = min_interval.round() as i32;
		let max_i = max_interval.round() as i32;
		let mut intervals: Vec<i32> = (min_i..=max_i).collect();

		if let Some(n) = num_intervals {
			if n < intervals.len() {
				let mut num_log = n;
				loop {
					let mut vals = Vec::with_capacity(num_log);
					if num_log == 1 {
						vals.push(((min_interval * max_interval).sqrt()).round() as i32);
					} else {
						for i in 0..num_log {
							let t = i as f64 / (num_log - 1) as f64;
							let log_v = min_interval.log2() + t * (max_interval.log2() - min_interval.log2());
							vals.push((2f64.powf(log_v)).round() as i32);
						}
					}
					vals.sort_unstable();
					vals.dedup();
					intervals = vals;
					if intervals.len() >= n {
						break;
					}
					num_log += 1;
				}
			}
		}

		let num_states = intervals.iter().map(|&i| i as usize).sum();
		let mut first_states = Vec::with_capacity(intervals.len());
		let mut last_states = Vec::with_capacity(intervals.len());
		let mut state_positions = vec![0.0; num_states];
		let mut state_intervals = vec![0i32; num_states];
		let mut idx = 0usize;
		let mut cum = 0i32;
		for (k, &interval) in intervals.iter().enumerate() {
			first_states.push(if k == 0 { 0 } else { cum as usize });
			cum += interval;
			last_states.push((cum - 1) as usize);
			let i = interval as usize;
			for j in 0..i {
				state_positions[idx + j] = j as f64 / interval as f64;
				state_intervals[idx + j] = interval;
			}
			idx += i;
		}

		Self {
			num_states,
			state_positions,
			state_intervals,
			first_states,
			last_states,
		}
	}
}

fn exponential_transition(from_int: &[i32], to_int: &[i32], lambda: f64) -> Vec<Vec<f64>> {
	let threshold = f64::EPSILON;
	let mut prob = vec![vec![0.0; to_int.len()]; from_int.len()];
	for (i, &fi) in from_int.iter().enumerate() {
		let mut row_sum = 0.0;
		for (j, &ti) in to_int.iter().enumerate() {
			let ratio = ti as f64 / fi as f64;
			let p = (-lambda * (ratio - 1.0).abs()).exp();
			if p > threshold {
				prob[i][j] = p;
				row_sum += p;
			}
		}
		if row_sum > 0.0 {
			for j in 0..to_int.len() {
				prob[i][j] /= row_sum;
			}
		}
	}
	prob
}

fn bar_transition_model(ss: &BarStateSpace, transition_lambda: f64) -> TransitionModel {
	let lambdas = vec![transition_lambda; ss.num_beats];

	let mut dest: Vec<u32> = Vec::new();
	let mut prev: Vec<u32> = Vec::new();
	let mut probs: Vec<f64> = Vec::new();

	// Same-tempo transitions within beats (all states except first states of each beat)
	let mut is_first = vec![false; ss.num_states];
	for beat_firsts in &ss.first_states {
		for &s in beat_firsts {
			is_first[s] = true;
		}
	}
	for s in 0..ss.num_states {
		if !is_first[s] {
			dest.push(s as u32);
			prev.push((s - 1) as u32);
			probs.push(1.0);
		}
	}

	// Tempo transitions at beat boundaries
	for beat in 0..ss.num_beats {
		let to_states = &ss.first_states[beat];
		let from_beat = if beat == 0 { ss.num_beats - 1 } else { beat - 1 };
		let from_states = &ss.last_states[from_beat];
		let from_int: Vec<i32> = from_states.iter().map(|&s| ss.state_intervals[s]).collect();
		let to_int: Vec<i32> = to_states.iter().map(|&s| ss.state_intervals[s]).collect();
		let prob = exponential_transition(&from_int, &to_int, lambdas[beat]);
		for (fi, &from_s) in from_states.iter().enumerate() {
			for (ti, &to_s) in to_states.iter().enumerate() {
				let p = prob[fi][ti];
				if p > 0.0 {
					dest.push(to_s as u32);
					prev.push(from_s as u32);
					probs.push(p);
				}
			}
		}
	}

	TransitionModel::make_sparse(&dest, &prev, &probs)
}

struct ObservationModel {
	/// Per-state pointer into density columns: 0=non-beat, 1=beat, 2=downbeat
	pointers: Vec<u32>,
	observation_lambda: f64,
}

impl ObservationModel {
	fn downbeat(ss: &BarStateSpace, observation_lambda: i32) -> Self {
		let border = 1.0 / observation_lambda as f64;
		let mut pointers = vec![0u32; ss.num_states];
		for (i, &pos) in ss.state_positions.iter().enumerate() {
			let frac = pos - pos.floor();
			// beat states: position within beat < border
			if frac < border {
				pointers[i] = 1;
			}
			// downbeat states: absolute position in first beat < border
			if pos < border {
				pointers[i] = 2;
			}
		}
		Self {
			pointers,
			observation_lambda: observation_lambda as f64,
		}
	}

	fn log_densities(&self, observations: &[[f32; 2]]) -> Vec<[f64; 3]> {
		let mut out = Vec::with_capacity(observations.len());
		let denom = (self.observation_lambda - 1.0).max(1.0);
		for &[beat, down] in observations {
			let beat = beat.clamp(1e-6, 1.0 - 1e-6) as f64;
			let down = down.clamp(1e-6, 1.0 - 1e-6) as f64;
			let rest = (1.0 - beat - down).clamp(1e-6, 1.0);
			out.push([
				(rest / denom).ln(),
				beat.ln(),
				down.ln(),
			]);
		}
		out
	}
}

struct Hmm {
	tm: TransitionModel,
	om: ObservationModel,
	state_space: BarStateSpace,
	initial_log: Vec<f64>,
}

impl Hmm {
	fn new(num_beats: usize) -> Self {
		let min_interval = 60.0 * FPS / MAX_BPM;
		let max_interval = 60.0 * FPS / MIN_BPM;
		let ss = BarStateSpace::new(num_beats, min_interval, max_interval, Some(NUM_TEMPI));
		let tm = bar_transition_model(&ss, TRANSITION_LAMBDA);
		let om = ObservationModel::downbeat(&ss, OBSERVATION_LAMBDA);
		let n = tm.num_states();
		let initial_log = vec![-(n as f64).ln(); n];
		Self {
			tm,
			om,
			state_space: ss,
			initial_log,
		}
	}

	fn viterbi(&self, observations: &[[f32; 2]]) -> (Vec<u32>, f64) {
		let num_states = self.tm.num_states();
		let num_obs = observations.len();
		if num_obs == 0 || num_states == 0 {
			return (Vec::new(), f64::NEG_INFINITY);
		}

		let densities = self.om.log_densities(observations);
		let mut prev = self.initial_log.clone();
		let mut curr = vec![0.0f64; num_states];
		let mut bt = vec![0u32; num_obs * num_states];

		for frame in 0..num_obs {
			for state in 0..num_states {
				let density = densities[frame][self.om.pointers[state] as usize];
				let start = self.tm.pointers[state] as usize;
				let end = self.tm.pointers[state + 1] as usize;
				let mut best = f64::NEG_INFINITY;
				let mut best_prev = 0u32;
				for ptr in start..end {
					let prev_state = self.tm.states[ptr] as usize;
					let tp = prev[prev_state] + self.tm.log_probabilities[ptr] + density;
					if tp > best {
						best = tp;
						best_prev = prev_state as u32;
					}
				}
				curr[state] = best;
				bt[frame * num_states + state] = best_prev;
			}
			prev.copy_from_slice(&curr);
		}

		let (mut state, log_prob) = curr
			.iter()
			.enumerate()
			.max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
			.map(|(i, &v)| (i as u32, v))
			.unwrap_or((0, f64::NEG_INFINITY));

		if !log_prob.is_finite() {
			return (Vec::new(), log_prob);
		}

		let mut path = vec![0u32; num_obs];
		for frame in (0..num_obs).rev() {
			path[frame] = state;
			state = bt[frame * num_states + state as usize];
		}
		(path, log_prob)
	}
}

fn threshold_activations(act: &[[f32; 2]], threshold: f32) -> (Vec<[f32; 2]>, usize) {
	let mut first = None;
	let mut last = 0usize;
	for (i, row) in act.iter().enumerate() {
		if row[0] >= threshold || row[1] >= threshold {
			if first.is_none() {
				first = Some(i);
			}
			last = i + 1;
		}
	}
	let first = first.unwrap_or(0);
	if last <= first {
		return (Vec::new(), 0);
	}
	(act[first..last].to_vec(), first)
}

fn extract_beats(
	hmm: &Hmm,
	path: &[u32],
	activations: &[[f32; 2]],
	first: usize,
	correct: bool,
) -> Vec<BeatEvent> {
	if path.is_empty() {
		return Vec::new();
	}

	let positions: Vec<f64> = path
		.iter()
		.map(|&s| hmm.state_space.state_positions[s as usize])
		.collect();
	let beat_numbers: Vec<u32> = positions.iter().map(|&p| p.floor() as u32 + 1).collect();

	let beat_frames: Vec<usize> = if correct {
		let beat_range: Vec<bool> = path
			.iter()
			.map(|&s| hmm.om.pointers[s as usize] >= 1)
			.collect();
		if !beat_range.iter().any(|&b| b) {
			return Vec::new();
		}

		let mut changes = Vec::new();
		if beat_range[0] {
			changes.push(0);
		}
		for i in 1..beat_range.len() {
			if beat_range[i] != beat_range[i - 1] {
				changes.push(i);
			}
		}
		if *beat_range.last().unwrap() {
			changes.push(beat_range.len());
		}

		let mut beats = Vec::new();
		for chunk in changes.chunks_exact(2) {
			let left = chunk[0];
			let right = chunk[1];
			let mut best_i = left;
			let mut best_v = f32::NEG_INFINITY;
			for i in left..right {
				let v = activations[i][0].max(activations[i][1]);
				if v > best_v {
					best_v = v;
					best_i = i;
				}
			}
			beats.push(best_i);
		}
		beats
	} else {
		let mut beats = Vec::new();
		for i in 1..beat_numbers.len() {
			if beat_numbers[i] != beat_numbers[i - 1] {
				beats.push(i);
			}
		}
		beats
	};

	beat_frames
		.into_iter()
		.map(|f| BeatEvent {
			time: (f + first) as f32 / FPS as f32,
			beat_number: beat_numbers.get(f).copied().unwrap_or(1),
		})
		.collect()
}

fn hmms() -> &'static [Hmm; 3] {
	static HMMS: OnceLock<[Hmm; 3]> = OnceLock::new();
	HMMS.get_or_init(|| [Hmm::new(2), Hmm::new(3), Hmm::new(4)])
}

/// Run BeatNet-compatible offline DBN on beat/downbeat activations `[T][2]`.
pub fn track_beats(beat: &[f32], downbeat: &[f32]) -> Vec<BeatEvent> {
	assert_eq!(beat.len(), downbeat.len());
	let act: Vec<[f32; 2]> = beat
		.iter()
		.zip(downbeat.iter())
		.map(|(&b, &d)| [b, d])
		.collect();

	let (segment, first) = threshold_activations(&act, THRESHOLD);
	if segment.is_empty() {
		return Vec::new();
	}

	let mut best_path = Vec::new();
	let mut best_log = f64::NEG_INFINITY;
	let mut best_hmm_idx = 0usize;

	for (i, hmm) in hmms().iter().enumerate() {
		let (path, log_p) = hmm.viterbi(&segment);
		if log_p > best_log {
			best_log = log_p;
			best_path = path;
			best_hmm_idx = i;
		}
	}

	if best_path.is_empty() {
		return Vec::new();
	}

	extract_beats(&hmms()[best_hmm_idx], &best_path, &segment, first, true)
}

/// BPM + first-beat offset from DBN beat events.
pub fn bpm_offset_from_beats(beats: &[BeatEvent]) -> Option<(f32, f32)> {
	if beats.len() < 4 {
		return None;
	}

	let mut iois: Vec<f32> = beats.windows(2).map(|w| w[1].time - w[0].time).collect();
	iois.retain(|&d| d > 60.0 / MAX_BPM as f32 && d < 60.0 / MIN_BPM as f32);
	if iois.len() < 3 {
		return None;
	}
	iois.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
	let median = iois[iois.len() / 2];
	let mut bpm = 60.0 / median;

	let nearest = bpm.round();
	if (bpm - nearest).abs() < 0.3 {
		bpm = nearest;
	}

	let offset = beats
		.iter()
		.find(|b| b.beat_number == 1)
		.map(|b| b.time)
		.unwrap_or(beats[0].time);

	Some((bpm, offset))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn dbn_recovers_synthetic_pulse_tempo() {
		let bpm = 128.0f32;
		let seconds = 20.0f32;
		let n = (seconds * FPS as f32) as usize;
		let period = FPS as f32 * 60.0 / bpm;
		let mut beat = vec![0.01f32; n];
		let mut down = vec![0.01f32; n];
		let mut t = period * 0.3;
		let mut k = 0usize;
		while (t as usize) + 1 < n {
			let i = t.round() as usize;
			beat[i] = 0.85;
			if k % 4 == 0 {
				down[i] = 0.9;
				beat[i] = 0.2;
			}
			k += 1;
			t += period;
		}

		let events = track_beats(&beat, &down);
		assert!(events.len() > 8, "too few beats: {}", events.len());
		let (got, _) = bpm_offset_from_beats(&events).expect("bpm");
		assert!((got - bpm).abs() < 3.0, "expected ~{bpm}, got {got}");
	}

	#[test]
	fn bar_state_space_sizes() {
		let ss = BarStateSpace::new(4, 60.0 * FPS / 215.0, 60.0 * FPS / 55.0, Some(60));
		assert!(ss.num_states > 100);
		assert_eq!(ss.first_states.len(), 4);
	}
}
