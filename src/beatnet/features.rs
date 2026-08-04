//! madmom-compatible LOG_SPECT features for BeatNet (272-dim @ 50 fps).

use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::sync::OnceLock;

pub const SAMPLE_RATE: u32 = 22050;
pub const HOP_SIZE: usize = 441; // 20 ms @ 22050
pub const FRAME_SIZE: usize = 1411; // 64 ms @ 22050
pub const FPS: f32 = 50.0;
pub const NUM_BANDS: usize = 136;
pub const FEAT_DIM: usize = NUM_BANDS * 2; // log-spec + positive diff
const NUM_FFT_BINS: usize = FRAME_SIZE >> 1; // 705 — madmom excludes Nyquist
const FMIN: f32 = 30.0;
const FMAX: f32 = 17000.0;
const BANDS_PER_OCTAVE: i32 = 24;
const FREF: f32 = 440.0;

/// Interleaved → mono (mean of channels).
pub fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
	let ch = channels.max(1) as usize;
	if ch == 1 {
		return samples.to_vec();
	}
	let frames = samples.len() / ch;
	let mut mono = Vec::with_capacity(frames);
	for i in 0..frames {
		let mut sum = 0.0f32;
		for c in 0..ch {
			sum += samples[i * ch + c];
		}
		mono.push(sum / ch as f32);
	}
	mono
}

/// Linear-interpolation resampler (offline analysis quality is sufficient).
pub fn resample_to(samples: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
	if from_sr == 0 || samples.is_empty() {
		return Vec::new();
	}
	if from_sr == to_sr {
		return samples.to_vec();
	}
	let ratio = to_sr as f64 / from_sr as f64;
	let out_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
	let mut out = Vec::with_capacity(out_len);
	let max_idx = samples.len() - 1;
	for i in 0..out_len {
		let src = i as f64 / ratio;
		let i0 = src.floor() as usize;
		let frac = (src - i0 as f64) as f32;
		if i0 >= max_idx {
			out.push(samples[max_idx]);
		} else {
			out.push(samples[i0] * (1.0 - frac) + samples[i0 + 1] * frac);
		}
	}
	out
}

/// Extract LOG_SPECT + positive spectral difference features.
/// Returns row-major `[T * 272]` (each frame is 272 floats).
pub fn log_spect_features(mono_22k: &[f32]) -> Option<Vec<f32>> {
	let num_frames = ((mono_22k.len() as f64) / HOP_SIZE as f64).ceil() as usize;
	if num_frames == 0 {
		return None;
	}

	let filterbank = filterbank();
	let window = hann_window(FRAME_SIZE);

	let mut planner = FftPlanner::new();
	let fft = planner.plan_fft_forward(FRAME_SIZE);
	let mut buffer = vec![Complex::new(0.0, 0.0); FRAME_SIZE];
	let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];

	let mut log_spec = vec![0.0f32; num_frames * NUM_BANDS];
	let half = FRAME_SIZE / 2;

	for frame_idx in 0..num_frames {
		// madmom FramedSignal origin=0 (center): start = i*hop - frame/2
		let start = frame_idx as isize * HOP_SIZE as isize - half as isize;
		for n in 0..FRAME_SIZE {
			let src = start + n as isize;
			let sample = if src < 0 || src as usize >= mono_22k.len() {
				0.0
			} else {
				mono_22k[src as usize]
			};
			buffer[n] = Complex::new(sample * window[n], 0.0);
		}

		fft.process_with_scratch(&mut buffer, &mut scratch);

		// Magnitude of first NUM_FFT_BINS bins, filter, log(1 + mag)
		let mut bands = [0.0f32; NUM_BANDS];
		for b in 0..NUM_BANDS {
			let mut acc = 0.0f32;
			for k in 0..NUM_FFT_BINS {
				let w = filterbank[k * NUM_BANDS + b];
				if w != 0.0 {
					acc += buffer[k].norm() * w;
				}
			}
			bands[b] = (1.0 + acc).log10();
		}
		log_spec[frame_idx * NUM_BANDS..(frame_idx + 1) * NUM_BANDS].copy_from_slice(&bands);
	}

	// Positive first-order spectral difference (diff_frames=1), hstacked
	let mut feats = vec![0.0f32; num_frames * FEAT_DIM];
	for t in 0..num_frames {
		let base = t * FEAT_DIM;
		let ls = t * NUM_BANDS;
		feats[base..base + NUM_BANDS].copy_from_slice(&log_spec[ls..ls + NUM_BANDS]);
		if t == 0 {
			// first frame diffs are zero (madmom pads with +inf then clamps)
			for b in 0..NUM_BANDS {
				feats[base + NUM_BANDS + b] = 0.0;
			}
		} else {
			let prev = (t - 1) * NUM_BANDS;
			for b in 0..NUM_BANDS {
				let d = log_spec[ls + b] - log_spec[prev + b];
				feats[base + NUM_BANDS + b] = if d > 0.0 { d } else { 0.0 };
			}
		}
	}

	Some(feats)
}

fn hann_window(n: usize) -> Vec<f32> {
	(0..n)
		.map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n as f32 - 1.0)).cos()))
		.collect()
}

/// Cached filterbank as row-major `[NUM_FFT_BINS * NUM_BANDS]`.
fn filterbank() -> &'static [f32] {
	static FB: OnceLock<Vec<f32>> = OnceLock::new();
	FB.get_or_init(build_log_filterbank).as_slice()
}

fn build_log_filterbank() -> Vec<f32> {
	let bin_freqs = fft_bin_frequencies(NUM_FFT_BINS, SAMPLE_RATE);
	let center_freqs = log_frequencies(BANDS_PER_OCTAVE, FMIN, FMAX, FREF);
	let bins = frequencies_to_unique_bins(&center_freqs, &bin_freqs);

	// Triangular filters from consecutive triplets → len(bins) - 2 bands
	let mut filters: Vec<(usize, Vec<f32>)> = Vec::new();
	let mut i = 0;
	while i + 3 <= bins.len() {
		let start = bins[i];
		let center = bins[i + 1];
		let stop = bins[i + 2];
		if start <= center && center < stop {
			filters.push(triangular_filter(start, center, stop, true));
		}
		i += 1;
	}

	assert_eq!(
		filters.len(),
		NUM_BANDS,
		"expected {NUM_BANDS} log bands, got {}",
		filters.len()
	);

	let mut fb = vec![0.0f32; NUM_FFT_BINS * NUM_BANDS];
	for (band, (start, data)) in filters.iter().enumerate() {
		let mut s = *start as isize;
		let mut d = data.as_slice();
		if s < 0 {
			let skip = (-s) as usize;
			if skip >= d.len() {
				continue;
			}
			d = &d[skip..];
			s = 0;
		}
		let start = s as usize;
		let stop = (start + d.len()).min(NUM_FFT_BINS);
		let len = stop - start;
		for (k, &w) in d[..len].iter().enumerate() {
			let idx = (start + k) * NUM_BANDS + band;
			if w > fb[idx] {
				fb[idx] = w;
			}
		}
	}
	fb
}

fn fft_bin_frequencies(num_fft_bins: usize, sample_rate: u32) -> Vec<f32> {
	// madmom: np.fft.fftfreq(num_fft_bins * 2, 1/sr)[:num_fft_bins]
	let n = num_fft_bins * 2;
	(0..num_fft_bins)
		.map(|i| i as f32 * sample_rate as f32 / n as f32)
		.collect()
}

fn log_frequencies(bands_per_octave: i32, fmin: f32, fmax: f32, fref: f32) -> Vec<f32> {
	let left = ((fmin / fref).log2() * bands_per_octave as f32).floor() as i32;
	let right = ((fmax / fref).log2() * bands_per_octave as f32).ceil() as i32;
	let mut freqs: Vec<f32> = (left..right)
		.map(|i| fref * 2f32.powf(i as f32 / bands_per_octave as f32))
		.collect();
	freqs.retain(|&f| f >= fmin && f <= fmax);
	freqs
}

fn frequencies_to_unique_bins(frequencies: &[f32], bin_frequencies: &[f32]) -> Vec<usize> {
	let mut indices = Vec::with_capacity(frequencies.len());
	for &f in frequencies {
		let mut idx = bin_frequencies.partition_point(|&bf| bf < f);
		idx = idx.clamp(1, bin_frequencies.len() - 1);
		let left = bin_frequencies[idx - 1];
		let right = bin_frequencies[idx];
		if f - left < right - f {
			idx -= 1;
		}
		indices.push(idx);
	}
	indices.dedup();
	indices
}

fn triangular_filter(start: usize, center: usize, stop: usize, norm: bool) -> (usize, Vec<f32>) {
	let c = center - start;
	let s = stop - start;
	let mut data = vec![0.0f32; s];
	if c > 0 {
		for i in 0..c {
			data[i] = i as f32 / c as f32;
		}
	}
	let fall = s - c;
	if fall > 0 {
		for i in 0..fall {
			data[c + i] = 1.0 - i as f32 / fall as f32;
		}
	}
	if norm {
		let sum: f32 = data.iter().sum();
		if sum > 0.0 {
			for v in data.iter_mut() {
				*v /= sum;
			}
		}
	}
	(start, data)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn filterbank_has_expected_shape() {
		let fb = filterbank();
		assert_eq!(fb.len(), NUM_FFT_BINS * NUM_BANDS);
		// Each band is area-normalized ≈ 1
		for b in 0..NUM_BANDS {
			let mut sum = 0.0f32;
			for k in 0..NUM_FFT_BINS {
				sum += fb[k * NUM_BANDS + b];
			}
			assert!((sum - 1.0).abs() < 1e-3, "band {b} sum={sum}");
		}
	}

	#[test]
	fn feature_dim_is_272() {
		let sr = SAMPLE_RATE;
		let samples = vec![0.0f32; sr as usize]; // 1s silence
		let feats = log_spect_features(&samples).unwrap();
		assert_eq!(feats.len() % FEAT_DIM, 0);
		assert_eq!(feats.len() / FEAT_DIM, ((sr as usize) as f64 / HOP_SIZE as f64).ceil() as usize);
	}
}
