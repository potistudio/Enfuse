//! BeatNet offline inference: LOG_SPECT features → CRNN (ONNX) → beat decode.

mod decode;
mod features;
mod infer;

use crate::analysis::BpmResult;

/// Run BeatNet BPM + beat offset. Returns `None` on failure (caller may fall back).
pub fn detect_bpm(samples: &[f32], sample_rate: u32, channels: u16) -> Option<BpmResult> {
	if samples.is_empty() || sample_rate == 0 {
		return None;
	}

	let mono = features::to_mono(samples, channels);
	let mono_22k = features::resample_to(&mono, sample_rate, features::SAMPLE_RATE);
	if mono_22k.len() < features::FRAME_SIZE {
		return None;
	}

	let feats = features::log_spect_features(&mono_22k)?;
	if feats.is_empty() {
		return None;
	}

	let activations = infer::run_crnn(&feats)?;
	let (bpm, beat_offset) = decode::decode_bpm_offset(&activations)?;

	if bpm <= 0.0 || !bpm.is_finite() {
		return None;
	}

	Some(BpmResult { bpm, beat_offset })
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn detect_bpm_on_click_track() {
		let sr = 44100u32;
		let bpm_true = 160.0f32;
		let seconds = 10.0f32;
		let n = (sr as f32 * seconds) as usize;
		let samples_per_beat = (60.0 * sr as f32 / bpm_true) as usize;
		let mut samples = vec![0.0f32; n];
		let mut t = samples_per_beat / 4;
		while t + 200 < n {
			for i in 0..200 {
				let env = 1.0 - (i as f32 / 200.0);
				samples[t + i] =
					env * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin();
			}
			t += samples_per_beat;
		}

		let result = detect_bpm(&samples, sr, 1);
		assert!(result.is_some(), "BeatNet path should succeed on click track");
		let result = result.unwrap();
		assert!(
			(result.bpm - bpm_true).abs() < 8.0,
			"unexpected bpm {} (want ~{bpm_true})",
			result.bpm
		);
		assert!(result.beat_offset >= 0.0 && result.beat_offset < 2.0);
	}
}
