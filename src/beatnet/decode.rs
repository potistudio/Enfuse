//! BeatNet offline decode: CRNN activations → DBN → BPM + offset.

use super::dbn;
use super::infer::Activations;

/// Decode BPM + first-beat offset via madmom-compatible DBN (BeatNet offline).
pub fn decode_bpm_offset(act: &Activations) -> Option<(f32, f32)> {
	if act.is_empty() {
		return None;
	}
	let beats = dbn::track_beats(&act.beat, &act.downbeat);
	dbn::bpm_offset_from_beats(&beats)
}
