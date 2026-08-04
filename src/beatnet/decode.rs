//! Tempo-constrained DP beat tracking on BeatNet activations.

use super::features::FPS;
use super::infer::Activations;

const MIN_BPM: f32 = 40.0;
const MAX_BPM: f32 = 220.0;
const SNAP_THRESHOLD: f32 = 0.3;

/// Decode BPM + first-beat offset (seconds) from CRNN activations.
pub fn decode_bpm_offset(act: &Activations) -> Option<(f32, f32)> {
	if act.is_empty() {
		return None;
	}

	// Combined strength: beats + downbeats (DBN-style dual stream, simplified)
	let strength: Vec<f32> = act
		.beat
		.iter()
		.zip(act.downbeat.iter())
		.map(|(&b, &d)| b + d)
		.collect();

	let (mut bpm, period) = estimate_tempo(&strength, FPS)?;
	let beats = track_beats_dp(&strength, period);
	let beat_offset = if let Some(&first) = beats.first() {
		first as f32 / FPS
	} else {
		// Fallback: strongest peak in first 2 beats
		let search = ((period * 2.0).ceil() as usize).min(strength.len());
		let idx = strength[..search]
			.iter()
			.enumerate()
			.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
			.map(|(i, _)| i)
			.unwrap_or(0);
		idx as f32 / FPS
	};

	// Prefer downbeat near the first tracked beat for offset refinement
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

	// Mean-center for autocorrelation
	let mean = env.iter().sum::<f32>() / n as f32;
	let centered: Vec<f32> = env.iter().map(|x| x - mean).collect();

	let mut best_corr = f32::NEG_INFINITY;
	let mut best_lag = min_lag as f32;

	for lag in min_lag..=max_lag {
		let mut corr = 0.0f32;
		let mut count = 0usize;
		for i in 0..n - lag {
			corr += centered[i] * centered[i + lag];
			count += 1;
		}
		if count == 0 {
			continue;
		}
		corr /= count as f32;

		let bpm_at = 60.0 * fps / lag as f32;
		// Mild preference around 120 BPM (same spirit as existing detector)
		let weight = (-((bpm_at.ln() - 120.0f32.ln()).powi(2)) / (2.0 * 0.45f32.powi(2))).exp();
		let weighted = corr * weight;
		if weighted > best_corr {
			best_corr = weighted;
			best_lag = lag as f32;
		}
	}

	if !best_corr.is_finite() {
		return None;
	}

	// Parabolic interpolation around peak lag
	let idx = best_lag.round() as usize;
	if idx > min_lag && idx < max_lag {
		let y0 = autocorr_at(&centered, idx - 1);
		let y1 = autocorr_at(&centered, idx);
		let y2 = autocorr_at(&centered, idx + 1);
		let denom = 2.0 * (2.0 * y1 - y2 - y0);
		if denom.abs() > 1e-6 {
			best_lag = idx as f32 + (y2 - y0) / denom;
		}
	}

	let bpm = 60.0 * fps / best_lag;
	Some((bpm, best_lag))
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
	// Search window ±30% of period
	let tol = ((period * 0.3).ceil() as isize).max(1);
	let min_step = (period_i - tol).max(1);
	let max_step = period_i + tol;

	let mut score = vec![0.0f32; n];
	let mut back = vec![usize::MAX; n];

	for t in 0..n {
		// Path starting at t
		score[t] = act[t];
		let t_i = t as isize;
		for step in min_step..=max_step {
			let prev = t_i - step;
			if prev < 0 {
				continue;
			}
			let p = prev as usize;
			// Tempo tightness: prefer steps near `period`
			let tempo_pen = -((step as f32 - period).abs() / period) * 0.1;
			let cand = score[p] + act[t] + tempo_pen;
			if cand > score[t] {
				score[t] = cand;
				back[t] = p;
			}
		}
	}

	// Backtrack from best score in the last period window
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
