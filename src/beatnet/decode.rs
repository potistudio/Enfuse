//! Tempo-constrained DP beat tracking on BeatNet activations.

use super::features::FPS;
use super::infer::Activations;

const MIN_BPM: f32 = 55.0;
const MAX_BPM: f32 = 200.0;
const SNAP_THRESHOLD: f32 = 0.3;

/// Decode BPM + first-beat offset (seconds) from CRNN activations.
pub fn decode_bpm_offset(act: &Activations) -> Option<(f32, f32)> {
	if act.is_empty() {
		return None;
	}

	// Tempo from beat channel only — adding downbeats biases autocorr toward
	// half-tempo / bar harmonics (often landing near ~70 or ~140).
	let (mut bpm, period) = estimate_tempo(&act.beat, FPS)?;

	// Tracking can use a bit of downbeat emphasis for phase.
	let strength: Vec<f32> = act
		.beat
		.iter()
		.zip(act.downbeat.iter())
		.map(|(&b, &d)| b + 0.35 * d)
		.collect();

	let beats = track_beats_dp(&strength, period);
	let beat_offset = if let Some(&first) = beats.first() {
		first as f32 / FPS
	} else {
		let search = ((period * 2.0).ceil() as usize).min(strength.len());
		let idx = strength[..search]
			.iter()
			.enumerate()
			.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
			.map(|(i, _)| i)
			.unwrap_or(0);
		idx as f32 / FPS
	};

	let beat_offset = refine_offset_with_downbeat(act, beat_offset, period);

	let nearest = bpm.round();
	if (bpm - nearest).abs() < SNAP_THRESHOLD {
		bpm = nearest;
	}

	Some((bpm, beat_offset))
}

fn estimate_tempo(env: &[f32], fps: f32) -> Option<(f32, f32)> {
	let n = env.len();
	if n < 4 {
		return None;
	}

	let min_lag = (fps * 60.0 / MAX_BPM).floor() as usize;
	let max_lag = (fps * 60.0 / MIN_BPM).ceil() as usize;
	let max_lag = max_lag.min(n / 2).max(min_lag + 1);
	if min_lag == 0 || min_lag >= max_lag {
		return None;
	}

	let mean = env.iter().sum::<f32>() / n as f32;
	let centered: Vec<f32> = env.iter().map(|x| x - mean).collect();

	let mut corr = vec![0.0f32; max_lag + 1];
	for lag in min_lag..=max_lag {
		corr[lag] = autocorr_at(&centered, lag);
	}

	// Primary: median inter-onset interval. Autocorr harmonics (×2 / ÷2) and
	// fixed-grid F1 scores are unreliable under frame-period jitter and were
	// locking estimates onto ~70 / ~140 BPM.
	let best_lag_i = if let Some(ioi) = median_ioi_lag(env, min_lag, max_lag) {
		let mut lag = ioi;
		// If peak-picking skipped every other beat, IOI is ~2× true period.
		// Trust the half-lag only when its autocorr is clearly stronger.
		let half = ioi / 2;
		if half >= min_lag && corr[half] > corr[ioi] * 1.15 {
			lag = half;
		}
		// If IOI is a half-beat (too fast), prefer double when autocorr agrees.
		let dbl = ioi.saturating_mul(2);
		if dbl <= max_lag && corr[dbl] > corr[lag] * 1.25 && corr[dbl] > corr[ioi] {
			// Only when the IOI itself looks like a weak harmonic
			if corr[ioi] < corr[dbl] * 0.85 {
				lag = dbl;
			}
		}
		lag
	} else {
		// Fallback: strongest autocorr local maximum
		let mut best_c = f32::NEG_INFINITY;
		let mut best_l = min_lag;
		for lag in min_lag..=max_lag {
			let c = corr[lag];
			let left = if lag > min_lag {
				corr[lag - 1]
			} else {
				f32::NEG_INFINITY
			};
			let right = if lag < max_lag {
				corr[lag + 1]
			} else {
				f32::NEG_INFINITY
			};
			if c >= left && c >= right && c > best_c {
				best_c = c;
				best_l = lag;
			}
		}
		best_l
	};

	let mut best_lag = best_lag_i as f32;
	if best_lag_i > min_lag && best_lag_i < max_lag {
		let y0 = corr[best_lag_i - 1];
		let y1 = corr[best_lag_i];
		let y2 = corr[best_lag_i + 1];
		let denom = 2.0 * (2.0 * y1 - y2 - y0);
		if denom.abs() > 1e-6 {
			best_lag = best_lag_i as f32 + (y2 - y0) / denom;
		}
	}

	let bpm = 60.0 * fps / best_lag;
	Some((bpm, best_lag))
}

fn median_ioi_lag(env: &[f32], min_lag: usize, max_lag: usize) -> Option<usize> {
	let peaks = peak_pick(env);
	if peaks.len() < 4 {
		return None;
	}
	let mut iois: Vec<usize> = peaks.windows(2).map(|w| w[1] - w[0]).collect();
	iois.retain(|&d| d >= min_lag && d <= max_lag);
	if iois.len() < 3 {
		return None;
	}
	iois.sort_unstable();
	Some(iois[iois.len() / 2])
}

fn peak_pick(env: &[f32]) -> Vec<usize> {
	let n = env.len();
	if n < 3 {
		return Vec::new();
	}
	let max_v = env.iter().copied().fold(0.0f32, f32::max);
	let mean = env.iter().sum::<f32>() / n as f32;
	let thresh = (mean + (max_v - mean) * 0.35).max(mean * 1.5);

	let mut peaks = Vec::new();
	for i in 1..n - 1 {
		if env[i] >= thresh && env[i] >= env[i - 1] && env[i] >= env[i + 1] {
			if peaks.last().is_none_or(|p| i - p >= min_peak_distance()) {
				peaks.push(i);
			} else if env[i] > env[*peaks.last().unwrap()] {
				*peaks.last_mut().unwrap() = i;
			}
		}
	}
	peaks
}

fn min_peak_distance() -> usize {
	((FPS * 60.0 / MAX_BPM).floor() as usize).max(1)
}

fn autocorr_at(centered: &[f32], lag: usize) -> f32 {
	let n = centered.len();
	if lag >= n {
		return 0.0;
	}
	let mut corr = 0.0f32;
	let count = n - lag;
	for i in 0..count {
		corr += centered[i] * centered[i + lag];
	}
	corr / count as f32
}

/// Ellis-style DP: maximize activation along a tempo-constrained beat path.
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

	fn pulse_train(bpm: f32, seconds: f32, fps: f32) -> Vec<f32> {
		let n = (seconds * fps) as usize;
		let period = fps * 60.0 / bpm;
		let mut env = vec![0.02f32; n];
		let mut t = period * 0.25;
		while (t as usize) < n {
			let i = t.round() as usize;
			if i < n {
				env[i] = 1.0;
				if i + 1 < n {
					env[i + 1] = 0.4;
				}
			}
			t += period;
		}
		env
	}

	fn assert_tempo_near(bpm_true: f32, tol: f32) {
		let env = pulse_train(bpm_true, 16.0, FPS);
		let (bpm, _) = estimate_tempo(&env, FPS).expect("tempo");
		assert!(
			(bpm - bpm_true).abs() < tol,
			"expected ~{bpm_true} BPM, got {bpm}"
		);
	}

	#[test]
	fn estimate_tempo_across_range() {
		for &bpm in &[90.0, 100.0, 120.0, 128.0, 140.0, 160.0, 174.0] {
			assert_tempo_near(bpm, 4.0);
		}
	}

	#[test]
	fn estimate_tempo_not_pulled_to_130_from_160() {
		let mut env = pulse_train(160.0, 16.0, FPS);
		let distractor = pulse_train(130.0, 16.0, FPS);
		for (e, d) in env.iter_mut().zip(distractor.iter()) {
			*e += d * 0.35;
		}
		let (bpm, _) = estimate_tempo(&env, FPS).expect("tempo");
		assert!(
			(bpm - 160.0).abs() < 5.0,
			"expected ~160, got {bpm}"
		);
	}

	#[test]
	fn estimate_tempo_not_locked_to_70_or_140() {
		assert_tempo_near(110.0, 5.0);
		assert_tempo_near(95.0, 5.0);
	}
}
