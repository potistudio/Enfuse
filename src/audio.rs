use crate::analysis;

use ringbuf::HeapRb;
use ringbuf::traits::{Producer, Split};
use rodio::*;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

type RingBuffer = HeapRb<f32>;
type RbProducer = <RingBuffer as Split>::Prod;
// Explicitly public type alias if needed or just use RbConsumer
#[allow(dead_code)]
pub type RbConsumer = <RingBuffer as Split>::Cons;

pub const DECK_COUNT: usize = 2;

pub struct AudioEngine {
	//...
	_stream: OutputStream,
	stream_handle: OutputStreamHandle,
	pub decks: Vec<Arc<Mutex<Deck>>>,
}

impl Default for AudioEngine {
	fn default() -> Self {
		Self::new()
	}
}

impl AudioEngine {
	pub fn new() -> Self {
		let (_stream, stream_handle) = OutputStream::try_default().unwrap();

		let decks = (0..DECK_COUNT)
			.map(|_| Arc::new(Mutex::new(Deck::new(stream_handle.clone()))))
			.collect();

		Self {
			_stream,
			stream_handle,
			decks,
		}
	}
}

#[derive(Clone, Debug)]
pub struct DeckData {
	pub samples: Vec<f32>,
	pub waveform: Vec<analysis::WaveformPoint>,
	pub channels: u16,
	pub sample_rate: u32,
	pub duration: Duration,
	pub bpm: f32,
	pub beat_offset: f32, // 秒単位での最初の拍の位置
	pub path: PathBuf,
}

pub fn decode_file_with_progress<F>(path: PathBuf, mut on_progress: F) -> Result<DeckData, String>
where
	F: FnMut(f32),
{
	let file = File::open(&path).map_err(|e| e.to_string())?;
	// Estimate size for progress (very rough for mp3)
	let _total_bytes = file.metadata().map(|m| m.len()).unwrap_or(0);

	let source = rodio::Decoder::new(BufReader::new(file)).map_err(|e| e.to_string())?;

	let channels = source.channels();
	let sample_rate = source.sample_rate();
	let total_duration = source.total_duration(); // Option<Duration>

	// We'll collect samples here
	let mut samples: Vec<f32> = Vec::with_capacity(sample_rate as usize * 60 * 3); // Pre-alloc 3 mins approx

	// Convert to iterator
	let mut sample_iter = source.convert_samples::<f32>();

	let mut count = 0;
	// Notify every 0.1s (sample_rate / 10)
	let notify_interval = (sample_rate as usize / 10).max(100);

	loop {
		// Read a chunk of samples
		let mut chunk_finished = true;
		for _ in 0..notify_interval {
			if let Some(sample) = sample_iter.next() {
				samples.push(sample);
				count += 1;
			} else {
				chunk_finished = false;
				break;
			}
		}

		// Progress Calculation
		let progress = if let Some(dur) = total_duration {
			let total_samples_est = dur.as_secs_f64() * sample_rate as f64 * channels as f64;
			(count as f64 / total_samples_est) as f32
		} else {
			// Fallback: This is hard without duration.
			// Just return small incremental progress to show "alive"
			// Or maybe 0.5?
			0.0 // Indeterminate
		};

		on_progress(progress.min(1.0));

		if !chunk_finished {
			break;
		}
	}

	// Calculate Duration exact
	let total_samples = samples.len() as u64;
	let total_frames = if channels > 0 {
		total_samples / channels as u64
	} else {
		0
	};
	let seconds = if sample_rate > 0 {
		total_frames as f64 / sample_rate as f64
	} else {
		0.0
	};
	let duration = Duration::from_secs_f64(seconds);

	// Parallelize Analysis (Waveform & BPM)
	let (waveform, bpm_result) = rayon::join(
		|| analysis::analyze_waveform(&samples, sample_rate),
		|| analysis::detect_bpm(&samples, sample_rate),
	);

	Ok(DeckData {
		samples,
		waveform,
		channels,
		sample_rate,
		duration,
		bpm: bpm_result.bpm,
		beat_offset: bpm_result.beat_offset,
		path,
	})
}

// Keep old one for compat or redirect
pub fn decode_file(path: PathBuf) -> Result<DeckData, String> {
	decode_file_with_progress(path, |_| {})
}

// ... imports ...
use std::f32::consts::PI;

pub struct Deck {
	handle: OutputStreamHandle,
	sink: Option<Sink>,

	// Audio Data
	pub samples: Arc<Vec<f32>>, // Arc for sharing
	pub waveform: Vec<analysis::WaveformPoint>,
	pub channels: u16,
	pub sample_rate: u32,
	pub duration: Duration,
	pub bpm: f32,
	pub beat_offset: f32, // 秒単位での最初の拍の位置

	pub file_path: Option<PathBuf>,
	pub is_playing: bool,
	pub volume: f32,
	pub user_speed: f32,                                  // The speed slider setting
	pub monitor_consumer: Arc<Mutex<Option<RbConsumer>>>, // For visualizer

	// Scratch Control
	pub control_cursor: Arc<AtomicU64>,
	pub control_speed: Arc<AtomicU32>,

	// EQ Control (0.0 - 1.0, 0.5 is flat)
	pub low: Arc<AtomicU32>,
	pub mid: Arc<AtomicU32>,
	pub high: Arc<AtomicU32>,

	// Metronome
	pub metronome_enabled: bool,
	last_beat_index: i32,
}

impl Deck {
	pub fn new(handle: OutputStreamHandle) -> Self {
		Self {
			handle,
			sink: None,
			samples: Arc::new(Vec::new()),
			waveform: Vec::new(),
			channels: 2,
			sample_rate: 44100,
			duration: Duration::from_secs(0),
			bpm: 0.0,
			beat_offset: 0.0,
			file_path: None,
			is_playing: false,
			volume: 1.0,
			user_speed: 1.0,
			monitor_consumer: Arc::new(Mutex::new(None)),
			control_cursor: Arc::new(AtomicU64::new(0)),
			control_speed: Arc::new(AtomicU32::new(f32_to_u32(1.0))),
			low: Arc::new(AtomicU32::new(f32_to_u32(0.5))),
			mid: Arc::new(AtomicU32::new(f32_to_u32(0.5))),
			high: Arc::new(AtomicU32::new(f32_to_u32(0.5))),
			metronome_enabled: false,
			last_beat_index: -1,
		}
	}

	pub fn load_data(&mut self, data: DeckData) {
		self.file_path = Some(data.path);
		self.samples = Arc::new(data.samples); // Convert to Arc
		self.waveform = data.waveform;
		self.channels = data.channels;
		self.sample_rate = data.sample_rate;
		self.duration = data.duration;
		self.bpm = data.bpm;
		self.beat_offset = data.beat_offset;

		// Reset Cursor
		self.control_cursor.store(0, Ordering::Relaxed);

		self.stop(); // Stop any previous playback
	}

	pub fn play(&mut self) {
		// メトロノームの拍カウントをリセット
		self.last_beat_index = -1;

		if self.sink.is_none() {
			if !self.samples.is_empty() {
				// Use ScratchSource
				let scratch_source = ScratchSource::new(
					self.samples.clone(),
					self.control_cursor.clone(),
					self.control_speed.clone(),
					self.channels,
					self.sample_rate,
				);

				// EQ Source
				let eq_source = EqSource::new(scratch_source, self.low.clone(), self.mid.clone(), self.high.clone());

				// Spy for Spectrum
				let rb = RingBuffer::new(4096);
				let (prod, cons) = rb.split();
				*self.monitor_consumer.lock().unwrap() = Some(cons);

				let source = SpySource::new(eq_source, prod);

				let sink = Sink::try_new(&self.handle).unwrap();
				sink.append(source);
				sink.set_volume(self.volume);
				// We do NOT use sink.set_speed, we use our internal speed

				self.sink = Some(sink);
			}
		} else if let Some(sink) = &self.sink {
			sink.play();
		}

		self.control_speed.store(f32_to_u32(self.user_speed), Ordering::Relaxed);
		self.is_playing = true;
	}

	pub fn pause(&mut self) {
		if let Some(sink) = &self.sink {
			sink.pause();
		}
		self.is_playing = false;
	}

	pub fn stop(&mut self) {
		if let Some(sink) = &self.sink {
			sink.stop();
		}
		self.sink = None;
		self.is_playing = false;
		self.control_cursor.store(0, Ordering::Relaxed);
	}

	pub fn set_volume(&mut self, volume: f32) {
		self.volume = volume;
		if let Some(sink) = &self.sink {
			sink.set_volume(volume);
		}
	}

	// Standard speed slider
	pub fn set_speed(&mut self, speed: f32) {
		self.user_speed = speed;
		// Only apply if we are "playing" (not scratching overrides)
		// For now, simple logic:
		self.control_speed.store(f32_to_u32(speed), Ordering::Relaxed);
	}

	// Direct speed control for scratching
	pub fn set_scratch_speed(&mut self, speed: f32) {
		self.control_speed.store(f32_to_u32(speed), Ordering::Relaxed);
	}

	pub fn seek_to(&mut self, pos: Duration) {
		let pos = if pos > self.duration { self.duration } else { pos };
		let sample_pos = (pos.as_secs_f64() * self.sample_rate as f64).round();
		self.control_cursor.store(f64_to_u64(sample_pos), Ordering::Relaxed);
	}

	pub fn get_position(&self) -> Duration {
		let pos_frame = u64_to_f64(self.control_cursor.load(Ordering::Relaxed));
		if self.sample_rate == 0 {
			return Duration::ZERO;
		}
		let secs = pos_frame / self.sample_rate as f64;
		Duration::from_secs_f64(secs)
	}

	pub fn set_low(&self, v: f32) {
		self.low.store(f32_to_u32(v), Ordering::Relaxed);
	}
	pub fn set_mid(&self, v: f32) {
		self.mid.store(f32_to_u32(v), Ordering::Relaxed);
	}
	pub fn set_high(&self, v: f32) {
		self.high.store(f32_to_u32(v), Ordering::Relaxed);
	}

	pub fn toggle_metronome(&mut self) {
		self.metronome_enabled = !self.metronome_enabled;
		self.last_beat_index = -1; // Reset
	}

	/// 現在位置から拍を検出し、新しい拍ならクリック音を再生
	pub fn tick_metronome(&mut self) {
		if !self.metronome_enabled || !self.is_playing || self.bpm <= 0.0 {
			return;
		}

		let pos_secs = self.get_position().as_secs_f32();
		let beat_duration = 60.0 / self.bpm;

		// 現在の拍番号を計算（beat_offsetを考慮）
		let beats_since_offset = (pos_secs - self.beat_offset + 0.033) / beat_duration;
		let current_beat = beats_since_offset.floor() as i32;

		// 新しい拍かつ有効な位置（beat_offset以降）なら音を鳴らす
		if current_beat > self.last_beat_index && beats_since_offset >= 0.0 {
			self.last_beat_index = current_beat;

			// クリック音を再生（高周波の短いビープ）
			// 4拍ごと（小節の頭）は高い音、それ以外は低い音
			let is_downbeat = (current_beat + 2).rem_euclid(4) == 0;
			let freq = if is_downbeat { 1200.0 } else { 880.0 };

			let click = rodio::source::SineWave::new(freq)
				.take_duration(Duration::from_millis(40))
				.amplify(0.5);

			if let Ok(sink) = Sink::try_new(&self.handle) {
				sink.append(click);
				sink.detach(); // 自動再生して解放
			}
		}
	}
}

// Atomic Float Helpers
fn f32_to_u32(val: f32) -> u32 {
	val.to_bits()
}
fn u32_to_f32(val: u32) -> f32 {
	f32::from_bits(val)
}
fn f64_to_u64(val: f64) -> u64 {
	val.to_bits()
}
fn u64_to_f64(val: u64) -> f64 {
	f64::from_bits(val)
}

// --- Biquad Implementation ---
#[derive(Clone, Copy, Debug)]
enum FilterType {
	LowShelf,
	HighShelf,
	Peaking,
}

#[derive(Clone, Debug)]
struct Biquad {
	b0: f32,
	b1: f32,
	b2: f32,
	a1: f32,
	a2: f32,
	x1: f32,
	x2: f32,
	y1: f32,
	y2: f32,
}

impl Biquad {
	fn new() -> Self {
		Self {
			b0: 1.0,
			b1: 0.0,
			b2: 0.0,
			a1: 0.0,
			a2: 0.0,
			x1: 0.0,
			x2: 0.0,
			y1: 0.0,
			y2: 0.0,
		}
	}

	fn calculate(&mut self, filter_type: FilterType, freq: f32, q: f32, db_gain: f32, sample_rate: u32) {
		let a = 10.0f32.powf(db_gain / 40.0);
		let w0 = 2.0 * PI * freq / sample_rate as f32;
		let cos_w0 = w0.cos();
		let sin_w0 = w0.sin();
		let alpha = sin_w0 / (2.0 * q);

		let (b0, b1, b2, a0, a1, a2) = match filter_type {
			FilterType::LowShelf => {
				let a_plus = a + 1.0;
				let a_minus = a - 1.0;
				let sqrt_a = a.sqrt();
				(
					a * (a_plus - a_minus * cos_w0 + 2.0 * sqrt_a * alpha),
					2.0 * a * (a_minus - a_plus * cos_w0),
					a * (a_plus - a_minus * cos_w0 - 2.0 * sqrt_a * alpha),
					a_plus + a_minus * cos_w0 + 2.0 * sqrt_a * alpha,
					-2.0 * (a_minus + a_plus * cos_w0),
					a_plus + a_minus * cos_w0 - 2.0 * sqrt_a * alpha,
				)
			}
			FilterType::HighShelf => {
				let a_plus = a + 1.0;
				let a_minus = a - 1.0;
				let sqrt_a = a.sqrt();
				(
					a * (a_plus + a_minus * cos_w0 + 2.0 * sqrt_a * alpha),
					-2.0 * a * (a_minus + a_plus * cos_w0),
					a * (a_plus + a_minus * cos_w0 - 2.0 * sqrt_a * alpha),
					a_plus - a_minus * cos_w0 + 2.0 * sqrt_a * alpha,
					2.0 * (a_minus - a_plus * cos_w0),
					a_plus - a_minus * cos_w0 - 2.0 * sqrt_a * alpha,
				)
			}
			FilterType::Peaking => {
				let a = 10.0f32.powf(db_gain / 40.0);
				(
					1.0 + alpha * a,
					-2.0 * cos_w0,
					1.0 - alpha * a,
					1.0 + alpha / a,
					-2.0 * cos_w0,
					1.0 - alpha / a,
				)
			}
		};

		self.b0 = b0 / a0;
		self.b1 = b1 / a0;
		self.b2 = b2 / a0;
		self.a1 = a1 / a0;
		self.a2 = a2 / a0;
	}

	fn process(&mut self, x: f32) -> f32 {
		let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
		self.x2 = self.x1;
		self.x1 = x;
		self.y2 = self.y1;
		self.y1 = y;
		y
	}
}

pub struct EqSource<I> {
	input: I,
	low_filter: Vec<Biquad>,
	mid_filter: Vec<Biquad>,
	high_filter: Vec<Biquad>,
	target_low: Arc<AtomicU32>,
	target_mid: Arc<AtomicU32>,
	target_high: Arc<AtomicU32>,
	current_low: f32,
	current_mid: f32,
	current_high: f32,
	sample_rate: u32,
	channels: usize,
	channel_cursor: usize,
	counter: usize,
}

impl<I> EqSource<I>
where
	I: Source<Item = f32>,
{
	pub fn new(input: I, low: Arc<AtomicU32>, mid: Arc<AtomicU32>, high: Arc<AtomicU32>) -> Self {
		let channels = input.channels() as usize;
		let sample_rate = input.sample_rate();
		let mut source = Self {
			input,
			low_filter: vec![Biquad::new(); channels],
			mid_filter: vec![Biquad::new(); channels],
			high_filter: vec![Biquad::new(); channels],
			target_low: low,
			target_mid: mid,
			target_high: high,
			current_low: 0.0,
			current_mid: 0.0,
			current_high: 0.0,
			sample_rate,
			channels,
			channel_cursor: 0,
			counter: 0,
		};
		source.update_coefficients(true);
		source
	}

	fn update_coefficients(&mut self, force: bool) {
		let t_low = u32_to_f32(self.target_low.load(Ordering::Relaxed));
		let t_mid = u32_to_f32(self.target_mid.load(Ordering::Relaxed));
		let t_high = u32_to_f32(self.target_high.load(Ordering::Relaxed));

		// 0.5 center -> 0dB
		// Range: 0.0..1.0
		// Map 0.0 -> -24dB, 0.5 -> 0dB, 1.0 -> 6dB
		let map_gain = |v: f32| {
			if v < 0.5 {
				(v - 0.5) * 48.0 // (0.0 - 0.5)*48 = -24
			} else {
				(v - 0.5) * 12.0 // (1.0 - 0.5)*12 = 6
			}
		};

		let low_db = map_gain(t_low);
		let mid_db = map_gain(t_mid);
		let high_db = map_gain(t_high);

		if force {
			self.current_low = low_db;
			self.current_mid = mid_db;
			self.current_high = high_db;
		} else {
			let smooth = 0.1;
			self.current_low += (low_db - self.current_low) * smooth;
			self.current_mid += (mid_db - self.current_mid) * smooth;
			self.current_high += (high_db - self.current_high) * smooth;
		}

		for bq in self.low_filter.iter_mut() {
			bq.calculate(FilterType::LowShelf, 200.0, 0.707, self.current_low, self.sample_rate);
		}
		for bq in self.mid_filter.iter_mut() {
			bq.calculate(FilterType::Peaking, 1000.0, 1.0, self.current_mid, self.sample_rate);
		}
		for bq in self.high_filter.iter_mut() {
			bq.calculate(
				FilterType::HighShelf,
				4000.0,
				0.707,
				self.current_high,
				self.sample_rate,
			);
		}
	}
}

impl<I> Iterator for EqSource<I>
where
	I: Source<Item = f32>,
{
	type Item = f32;

	fn next(&mut self) -> Option<Self::Item> {
		if self.counter == 0 {
			self.update_coefficients(false);
			self.counter = 64 * self.channels;
		}
		self.counter -= 1;

		let mut sample = self.input.next()?;

		let ch = self.channel_cursor;
		// Apply filters
		sample = self.low_filter[ch].process(sample);
		sample = self.mid_filter[ch].process(sample);
		sample = self.high_filter[ch].process(sample);

		self.channel_cursor = (self.channel_cursor + 1) % self.channels;

		Some(sample)
	}
}

impl<I> Source for EqSource<I>
where
	I: Source<Item = f32>,
{
	fn current_frame_len(&self) -> Option<usize> {
		self.input.current_frame_len()
	}
	fn channels(&self) -> u16 {
		self.input.channels()
	}
	fn sample_rate(&self) -> u32 {
		self.input.sample_rate()
	}
	fn total_duration(&self) -> Option<Duration> {
		self.input.total_duration()
	}
}

#[derive(Clone)]
pub struct ScratchSource {
	samples: Arc<Vec<f32>>,
	cursor: Arc<AtomicU64>, // f64 index
	speed: Arc<AtomicU32>,  // f32 speed
	channels: u16,
	sample_rate: u32,
	// Internal iteration state
	channel_cursor: usize,
}

impl ScratchSource {
	pub fn new(
		samples: Arc<Vec<f32>>,
		cursor: Arc<AtomicU64>,
		speed: Arc<AtomicU32>,
		channels: u16,
		sample_rate: u32,
	) -> Self {
		Self {
			samples,
			cursor,
			speed,
			channels,
			sample_rate,
			channel_cursor: 0,
		}
	}
}

impl Iterator for ScratchSource {
	type Item = f32;

	fn next(&mut self) -> Option<Self::Item> {
		let pos_bits = self.cursor.load(Ordering::Relaxed);
		let pos = u64_to_f64(pos_bits);
		let max_frame = (self.samples.len() / self.channels as usize) as f64 - 1.0;

		// Safety clamp
		if pos < 0.0 || pos > max_frame + 1.0 {
			return Some(0.0);
		}

		let index = (pos as usize) * (self.channels as usize) + self.channel_cursor;
		let sample = if index < self.samples.len() {
			self.samples[index]
		} else {
			0.0
		};

		self.channel_cursor += 1;
		if self.channel_cursor >= self.channels as usize {
			self.channel_cursor = 0;
			// Advance cursor
			let speed = u32_to_f32(self.speed.load(Ordering::Relaxed));
			let mut new_pos = pos + speed as f64;

			// Loop or clamp? Clamp.
			if new_pos < 0.0 {
				new_pos = 0.0;
			}
			if new_pos > max_frame {
				new_pos = max_frame;
			}

			self.cursor.store(f64_to_u64(new_pos), Ordering::Relaxed);
		}

		Some(sample)
	}
}

impl Source for ScratchSource {
	fn current_frame_len(&self) -> Option<usize> {
		None
	}
	fn channels(&self) -> u16 {
		self.channels
	}
	fn sample_rate(&self) -> u32 {
		self.sample_rate
	}
	fn total_duration(&self) -> Option<Duration> {
		None
	}
}

// Spy Source Wrapper for Analysis
pub struct SpySource<I, P>
where
	I: Source<Item = f32>,
	P: Producer<Item = f32>,
{
	input: I,
	producer: P,
}

impl<I, P> SpySource<I, P>
where
	I: Source<Item = f32>,
	P: Producer<Item = f32>,
{
	pub fn new(input: I, producer: P) -> Self {
		Self { input, producer }
	}
}

impl<I, P> Iterator for SpySource<I, P>
where
	I: Source<Item = f32>,
	P: Producer<Item = f32>,
{
	type Item = f32;

	fn next(&mut self) -> Option<Self::Item> {
		let sample = self.input.next()?;
		let _ = self.producer.try_push(sample); // push copies? push takes value.
		// If push fails (full), we just ignore.
		Some(sample)
	}
}

impl<I, P> Source for SpySource<I, P>
where
	I: Source<Item = f32>,
	P: Producer<Item = f32>,
{
	fn current_frame_len(&self) -> Option<usize> {
		self.input.current_frame_len()
	}

	fn channels(&self) -> u16 {
		self.input.channels()
	}

	fn sample_rate(&self) -> u32 {
		self.input.sample_rate()
	}

	fn total_duration(&self) -> Option<Duration> {
		self.input.total_duration()
	}
}
