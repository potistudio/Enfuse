//! ONNX CRNN inference for BeatNet BDA (model_1).

use super::features::FEAT_DIM;
use ndarray::Array3;
use ort::session::Session;
use ort::value::TensorRef;
use std::sync::{Mutex, OnceLock};

static MODEL_BYTES: &[u8] = include_bytes!("../../assets/beatnet/model_1.onnx");

/// Softmax activations: beat, downbeat, non-beat per frame.
#[derive(Clone, Debug)]
pub struct Activations {
	pub beat: Vec<f32>,
	pub downbeat: Vec<f32>,
	#[allow(dead_code)]
	pub non_beat: Vec<f32>,
}

impl Activations {
	pub fn is_empty(&self) -> bool {
		self.beat.is_empty()
	}
}

fn session() -> Result<&'static Mutex<Session>, String> {
	static SESSION: OnceLock<Mutex<Session>> = OnceLock::new();
	if let Some(s) = SESSION.get() {
		return Ok(s);
	}
	let sess = Session::builder()
		.map_err(|e| format!("ort builder: {e}"))?
		.commit_from_memory(MODEL_BYTES)
		.map_err(|e| format!("ort load model: {e}"))?;
	let _ = SESSION.set(Mutex::new(sess));
	Ok(SESSION.get().expect("session set"))
}

/// Run CRNN on row-major `[T * 272]` features → per-frame activations.
pub fn run_crnn(features: &[f32]) -> Option<Activations> {
	if features.len() % FEAT_DIM != 0 {
		log::warn!("BeatNet features length {} not divisible by {FEAT_DIM}", features.len());
		return None;
	}
	let t = features.len() / FEAT_DIM;
	if t == 0 {
		return None;
	}

	run_crnn_inner(features, t).unwrap_or_else(|e| {
		log::warn!("BeatNet ONNX inference failed: {e}");
		None
	})
}

fn run_crnn_inner(features: &[f32], t: usize) -> Result<Option<Activations>, String> {
	let input = Array3::from_shape_vec((1, t, FEAT_DIM), features.to_vec())
		.map_err(|e| format!("shape features: {e}"))?;

	let sess = session()?;
	let mut sess = sess.lock().map_err(|_| "ort session mutex poisoned".to_string())?;

	let outputs = sess
		.run(ort::inputs![TensorRef::from_array_view(input.view()).map_err(|e| e.to_string())?])
		.map_err(|e| format!("ort run: {e}"))?;

	// Output: [1, 3, T] — classes beat / downbeat / non-beat
	let (_shape, data) = outputs[0]
		.try_extract_tensor::<f32>()
		.map_err(|e| format!("extract activations: {e}"))?;

	if data.len() != 3 * t {
		return Err(format!("unexpected activation length {} for T={t}", data.len()));
	}

	// Layout: [1, 3, T] contiguous → class-major: beat[0..T], downbeat[T..2T], nonbeat[2T..3T]
	let beat = data[..t].to_vec();
	let downbeat = data[t..2 * t].to_vec();
	let non_beat = data[2 * t..3 * t].to_vec();

	Ok(Some(Activations {
		beat,
		downbeat,
		non_beat,
	}))
}
