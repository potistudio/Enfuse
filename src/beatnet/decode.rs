//! Offline BeatNet decode: activations → beat times → BPM + offset.
//!
//! Mirrors BeatNet `mode='offline'` post-CRNN path in spirit: tempo and phase
//! come from the neural beat/downbeat activations (not spectral-flux).

use super::features::FPS;
use super::infer::Activations;

const MIN_BPM: f32 = 55.0;
const MAX_BPM: f32 = 215.0;
const SNAP_THRESHOLD: f32 = 0.3;

/// Decode BPM + first-beat offset from CRNN activations (beat & downbeat).
pub fn decode_bpm_offset(act: &Activations) -> Option<(f32, f32)> {
	if act.is_empty() {
		return None;
	}

	// BeatNet offline feeds DBN only beat+downbeat (not non-beat).
	let strength: Vec<f32> = act
		.beat
		.iter()
		.zip(act.downbeat.iter())
		.map(|(&b, &d)| b.max(d))
		.collect();

	let peaks = peak_pick_beats(&strength, &act.downbeat);
	if peaks.len() < 4 {
		return None;
	}

	let (mut bpm, period) = tempo_from_beat_times(&peaks, FPS)?;
	let beats = track_beats_dp(&strength, period);
	let beat_offset = if let Some(&t) = beats.first() {
		t as f32 / FPS
	} else {
		peaks[0] as f32 / FPS
	};

	// Prefer a downbeat near the first beat for grid phase.
	let beat_offset = refine_offset_with_downbeat(act, beat_offset, period);

	let nearest = bpm.round();
	if (bpm - nearest).abs() < SNAP_THRESHOLD {
		bpm = nearest;
	}

	Some((bpm, beat_offset))
}

fn tempo_from_beat_times(peaks: &[usize], fps: f32) -> Option<(f32, f32)> {
	let min_lag = (fps * 60.0 / MAX_BPM).floor() as usize;
	let max_lag = (fps * 60.0 / MIN_BPM).ceil() as usize;

	let mut iois: Vec<usize> = peaks.windows(2).map(|w| w[1] - w[0]).collect();
	iois.retain(|&d| d >= min_lag && d <= max_lag);
	if iois.len() < 3 {
		return None;
	}
	iois.sort_unstable();
	let med = iois[iois.len() / 2] as f32;

	// Histogram mode as a check against median (rejects mixed 3:2 IOIs).
	let mut best_lag = med;
	let mut best_count = 0usize;
	for &cand in &iois {
		let c = iois
			.iter()
			.filter(|&&d| d.abs_diff(cand) <= 1)
			.count();
		if c > best_count {
			best_count = c;
			best_lag = cand as f32;
		}
	}
	// Prefer mode when it is decisive; else median.
	let period = if best_count >= iois.len() / 3 {
		best_lag
	} else {
		med
	};

	let bpm = 60.0 * fps / period;
	Some((bpm, period))
}

/// Peak-pick beat activations; keep stronger peaks and prefer downbeat frames.
fn peak_pick_beats(strength: &[f32], downbeat: &[f32]) -> Vec<usize> {
	let n = strength.len();
	if n < 3 {
		return Vec::new();
	}

	// Adaptive threshold on upper quantile of activations (BeatNet peaks are sharp).
	let mut sorted: Vec<f32> = strength.to_vec();
	sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
	let q75 = sorted[(sorted.len() * 3) / 4];
	let q90 = sorted[(sorted.len() * 9) / 10];
	let max_v = sorted.last().copied().unwrap_or(0.0);
	let thresh = (q75 + (q90 - q75) * 0.5).max(max_v * 0.25).max(0.05);

	let min_dist = ((FPS * 60.0 / MAX_BPM).floor() as usize).max(1);
	let mut peaks = Vec::new();

	for i in 1..n - 1 {
		let v = strength[i];
		if v < thresh {
			continue;
		}
		if v < strength[i - 1] || v < strength[i + 1] {
			continue;
		}
		// Soft downbeat boost for tie-breaking when replacing nearby peaks
		let score = v + 0.15 * downbeat.get(i).copied().unwrap_or(0.0);
		if let Some(&last) = peaks.last() {
			if i - last < min_dist {
				let last_score = strength[last]
					+ 0.15 * downbeat.get(last).copied().unwrap_or(0.0);
				if score > last_score {
					*peaks.last_mut().unwrap() = i;
				}
				continue;
			}
		}
		peaks.push(i);
	}
	peaks
}

fn track_beats_dp(act: &[f32], period: f32) -> Vec<usize> {
	let n = act.len();
	if n == 0 || period < 1.0 {
		return Vec::new();
	}

	let period_i = period.round().max(1.0) as isize;
	let tol = ((period * 0.3).ceil() as isize).max(1);
	let min_step = (period_i - tol).max(1);
	let max_step = period_i + tol;

	let mut score = vec![0.0f32; n];
	let mut back = vec![usize::MAX; n];

	for t in 0..n {
		score[t] = act[t];
		let t_i = t as isize;
		for step in min_step..=max_step {
			let prev = t_i - step;
			if prev < 0 {
				continue;
			}
			let p = prev as usize;
			let tempo_pen = -((step as f32 - period).abs() / period) * 0.1;
			let cand = score[p] + act[t] + tempo_pen;
			if cand > score[t] {
				score[t] = cand;
				back[t] = p;
			}
		}
	}

	let start_search = n.saturating_sub(max_step as usize + 1);
	let mut best_end = start_search;
	let mut best_s = f32::NEG_INFINITY;
	for t in start_search..n {
		if score[t] > best_s {
			best_s = score[t];
			best_end = t;
		}
	}

	let mut beats = Vec::new();
	let mut cur = best_end;
	beats.push(cur);
	while back[cur] != usize::MAX {
		cur = back[cur];
		beats.push(cur);
	}
	beats.reverse();
	beats
}

fn refine_offset_with_downbeat(act: &Activations, offset_sec: f32, period: f32) -> f32 {
	if act.downbeat.is_empty() || period < 1.0 {
		return offset_sec;
	}
	let offset_frame = (offset_sec * FPS).round() as isize;
	let search = (period * 0.5).ceil() as isize;
	let n = act.downbeat.len() as isize;

	let mut best_f = offset_frame;
	let mut best_v = f32::NEG_INFINITY;
	for d in -search..=search {
		let f = offset_frame + d;
		if f < 0 || f >= n {
			continue;
		}
		let v = act.downbeat[f as usize] * 1.5 + act.beat[f as usize];
		if v > best_v {
			best_v = v;
			best_f = f;
		}
	}
	best_f as f32 / FPS
}

#[cfg(test)]
mod tests {
	use super::*;

	fn pulse_activations(bpm: f32, seconds: f32) -> Activations {
		let n = (seconds * FPS) as usize;
		let period = FPS * 60.0 / bpm;
		let mut beat = vec![0.02f32; n];
		let mut downbeat = vec![0.01f32; n];
		let mut t = period * 0.2;
		let mut count = 0usize;
		while (t as usize) < n {
			let i = t.round() as usize;
			if i < n {
				beat[i] = 0.9;
				if i + 1 < n {
					beat[i + 1] = 0.35;
				}
				if count % 4 == 0 {
					downbeat[i] = 0.85;
				}
			}
			count += 1;
			t += period;
		}
		Activations {
			beat,
			downbeat,
			non_beat: vec![0.1; n],
		}
	}

	#[test]
	fn decode_tempo_across_range() {
		for &bpm in &[90.0, 110.0, 128.0, 150.0, 160.0, 173.0, 200.0] {
			let act = pulse_activations(bpm, 16.0);
			let (got, _) = decode_bpm_offset(&act).expect("decode");
			assert!(
				(got - bpm).abs() < 8.0,
				"expected ~{bpm}, got {got}"
			);
		}
	}
}
