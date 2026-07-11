use iced::futures::SinkExt;
use iced::widget::canvas::{Cache, Canvas};
use iced::widget::{button, column, container, progress_bar, row, slider, stack, text, vertical_slider};
use iced::{Color, Element, Length, Subscription, Task, Theme};
use ringbuf::traits::Consumer as ConsumerTrait;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::analysis;
use crate::audio::{self, AudioEngine, Deck, DeckData};
use crate::ui::disc::RotatingDisc;
use crate::ui::spectrum::Spectrum;
use crate::ui::waveform::{Waveform, WaveformOutput};
use crate::ui::wipe::WipeOverlay;

const WIPE_DURATION_SECS: f32 = 0.5;
const TICK_MS: u64 = 16;

pub(super) struct App {
	audio_engine: AudioEngine,
	loading_a: Option<PathBuf>,
	progress_a: f32,
	loading_b: Option<PathBuf>,
	progress_b: f32,

	// Mixer State
	volume_a: f32,
	volume_b: f32,
	crossfader: f32,

	// Spectrum
	spectrum_buf_a: VecDeque<f32>,
	spectrum_buf_b: VecDeque<f32>,
	spectrum_display: Vec<f32>,

	// Waveform Cache
	waveform_cache_a: Cache,
	waveform_cache_b: Cache,

	// Zoom
	zoom_a: f32,
	zoom_b: f32,

	// EQ
	high_a: f32,
	mid_a: f32,
	low_a: f32,
	high_b: f32,
	mid_b: f32,
	low_b: f32,

	// Disc Cache
	disc_cache_a: Cache,
	disc_cache_b: Cache,

	// Wipe Effect
	wipe_progress_a: f32,
	wipe_animating_a: bool,
	wipe_progress_b: f32,
	wipe_animating_b: bool,
}

impl Default for App {
	fn default() -> Self {
		Self {
			audio_engine: AudioEngine::new(),
			loading_a: None,
			progress_a: 0.0,
			loading_b: None,
			progress_b: 0.0,
			volume_a: 1.0,
			volume_b: 1.0,
			crossfader: 0.0,
			spectrum_buf_a: VecDeque::with_capacity(2048),
			spectrum_buf_b: VecDeque::with_capacity(2048),
			spectrum_display: Vec::new(),
			waveform_cache_a: Cache::default(),
			waveform_cache_b: Cache::default(),
			zoom_a: 1.0,
			zoom_b: 1.0,
			high_a: 0.5,
			mid_a: 0.5,
			low_a: 0.5,
			high_b: 0.5,
			mid_b: 0.5,
			low_b: 0.5,
			disc_cache_a: Cache::default(),
			disc_cache_b: Cache::default(),
			wipe_progress_a: 1.0,
			wipe_animating_a: false,
			wipe_progress_b: 1.0,
			wipe_animating_b: false,
		}
	}
}

#[derive(Debug, Clone)]
pub(super) enum Message {
	Tick(std::time::Instant),
	DeckAPlay,
	DeckAPause,
	DeckALoad,
	DeckAProgress(f32),
	DeckALoaded(Result<DeckData, String>),
	DeckASpeed(f32),
	DeckBPlay,
	DeckBPause,
	DeckBLoad,
	DeckBProgress(f32),
	DeckBLoaded(Result<DeckData, String>),
	DeckBSpeed(f32),
	DeckAZoom(f32),
	DeckBZoom(f32),

	// EQ
	DeckAHigh(f32),
	DeckAMid(f32),
	DeckALow(f32),
	DeckBHigh(f32),
	DeckBMid(f32),
	DeckBLow(f32),

	// Mixer
	VolumeAChanged(f32),
	VolumeBChanged(f32),
	CrossfaderChanged(f32),

	// Seek & Scratch
	DeckASeek(f32),
	DeckBSeek(f32),
	DeckAScratch(f32),
	DeckBScratch(f32),

	// Metronome
	DeckAMetronome,
	DeckBMetronome,

	// MIDI (FLX4)
	Midi(Vec<u8>),
}

impl App {
	pub(super) fn theme(&self) -> Theme {
		Theme::custom(
			"Enfuse Dark".to_string(),
			iced::theme::Palette {
				background: Color::from_rgb8(18, 18, 18),
				text: Color::WHITE,
				primary: Color::from_rgb8(100, 149, 237),
				success: Color::from_rgb8(50, 205, 50),
				danger: Color::from_rgb8(220, 20, 60),
			},
		)
	}

	fn apply_volumes(&self) {
		let mut deck_a = self.audio_engine.deck_a.lock().unwrap();
		let mut deck_b = self.audio_engine.deck_b.lock().unwrap();

		let vol_a = self.volume_a
			* if self.crossfader > 0.0 {
				1.0 - self.crossfader
			} else {
				1.0
			};
		let vol_b = self.volume_b
			* if self.crossfader < 0.0 {
				1.0 + self.crossfader
			} else {
				1.0
			};

		deck_a.set_volume(vol_a);
		deck_b.set_volume(vol_b);
	}

	pub(super) fn subscription(&self) -> Subscription<Message> {
		let sub_a = if let Some(path) = &self.loading_a {
			track_loader(0, path.clone())
		} else {
			Subscription::none()
		};

		let sub_b = if let Some(path) = &self.loading_b {
			track_loader(1, path.clone())
		} else {
			Subscription::none()
		};

		let is_playing = {
			let a = self.audio_engine.deck_a.lock().unwrap().is_playing;
			let b = self.audio_engine.deck_b.lock().unwrap().is_playing;
			a || b
		};

		let wipe_active = self.wipe_animating_a || self.wipe_animating_b;

		let time_sub = if is_playing || wipe_active {
			iced::time::every(std::time::Duration::from_millis(TICK_MS)).map(Message::Tick)
		} else {
			Subscription::none()
		};

		let midi_sub = crate::midi::listener(Message::Midi);

		Subscription::batch(vec![sub_a, sub_b, time_sub, midi_sub])
	}

	pub(super) fn update(&mut self, message: Message) -> Task<Message> {
		match message {
			Message::Tick(_) => {
				self.tick_wipe();
				self.tick_spectrum();
				self.audio_engine.deck_a.lock().unwrap().tick_metronome();
				self.audio_engine.deck_b.lock().unwrap().tick_metronome();
			}
			Message::VolumeAChanged(v) => {
				self.volume_a = v;
				self.apply_volumes();
			}
			Message::VolumeBChanged(v) => {
				self.volume_b = v;
				self.apply_volumes();
			}
			Message::CrossfaderChanged(v) => {
				self.crossfader = v;
				self.apply_volumes();
			}
			Message::DeckAPlay => {
				self.audio_engine.deck_a.lock().unwrap().play();
			}
			Message::DeckAPause => {
				self.audio_engine.deck_a.lock().unwrap().pause();
			}
			Message::DeckALoad => {
				if self.loading_a.is_none() {
					if let Some(path) = rfd::FileDialog::new().pick_file() {
						self.loading_a = Some(path);
						self.progress_a = 0.0;
					}
				}
			}
			Message::DeckAProgress(p) => {
				self.progress_a = p;
			}
			Message::DeckALoaded(result) => {
				self.loading_a = None;
				match result {
					Ok(data) => {
						let mut deck = self.audio_engine.deck_a.lock().unwrap();
						deck.load_data(data);
						deck.play();
					}
					Err(e) => {
						log::error!("Error loading track A: {}", e);
					}
				}
				self.apply_volumes();
				self.start_wipe_a();
			}
			Message::DeckASpeed(speed) => {
				self.audio_engine.deck_a.lock().unwrap().set_speed(speed);
			}
			Message::DeckBPlay => {
				self.audio_engine.deck_b.lock().unwrap().play();
			}
			Message::DeckBPause => {
				self.audio_engine.deck_b.lock().unwrap().pause();
			}
			Message::DeckBLoad => {
				if self.loading_b.is_none() {
					if let Some(path) = rfd::FileDialog::new().pick_file() {
						self.loading_b = Some(path);
						self.progress_b = 0.0;
					}
				}
			}
			Message::DeckBProgress(p) => {
				self.progress_b = p;
			}
			Message::DeckBLoaded(result) => {
				self.loading_b = None;
				match result {
					Ok(data) => {
						let mut deck = self.audio_engine.deck_b.lock().unwrap();
						deck.load_data(data);
						deck.play();
					}
					Err(e) => {
						log::error!("Error loading track B: {}", e);
					}
				}
				self.apply_volumes();
				self.start_wipe_b();
			}
			Message::DeckBSpeed(speed) => {
				self.audio_engine.deck_b.lock().unwrap().set_speed(speed);
			}
			Message::DeckASeek(p) => {
				let mut deck = self.audio_engine.deck_a.lock().unwrap();
				let dur = deck.duration.as_secs_f64() * p as f64;
				deck.seek_to(std::time::Duration::from_secs_f64(dur));
			}
			Message::DeckBSeek(p) => {
				let mut deck = self.audio_engine.deck_b.lock().unwrap();
				let dur = deck.duration.as_secs_f64() * p as f64;
				deck.seek_to(std::time::Duration::from_secs_f64(dur));
			}
			Message::DeckAScratch(v) => {
				self.audio_engine.deck_a.lock().unwrap().set_scratch_speed(v);
			}
			Message::DeckBScratch(v) => {
				self.audio_engine.deck_b.lock().unwrap().set_scratch_speed(v);
			}
			Message::DeckAZoom(z) => {
				self.zoom_a = z;
			}
			Message::DeckBZoom(z) => {
				self.zoom_b = z;
			}
			Message::DeckAHigh(v) => {
				self.high_a = v;
				self.audio_engine.deck_a.lock().unwrap().set_high(v);
			}
			Message::DeckAMid(v) => {
				self.mid_a = v;
				self.audio_engine.deck_a.lock().unwrap().set_mid(v);
			}
			Message::DeckALow(v) => {
				self.low_a = v;
				self.audio_engine.deck_a.lock().unwrap().set_low(v);
			}
			Message::DeckBHigh(v) => {
				self.high_b = v;
				self.audio_engine.deck_b.lock().unwrap().set_high(v);
			}
			Message::DeckBMid(v) => {
				self.mid_b = v;
				self.audio_engine.deck_b.lock().unwrap().set_mid(v);
			}
			Message::DeckBLow(v) => {
				self.low_b = v;
				self.audio_engine.deck_b.lock().unwrap().set_low(v);
			}
			Message::DeckAMetronome => {
				self.audio_engine.deck_a.lock().unwrap().toggle_metronome();
			}
			Message::DeckBMetronome => {
				self.audio_engine.deck_b.lock().unwrap().toggle_metronome();
			}
			Message::Midi(bytes) => {
				let hex = bytes
					.iter()
					.map(|b| format!("{b:02X}"))
					.collect::<Vec<_>>()
					.join(" ");
				log::info!("MIDI in: {hex}");
			}
		}
		Task::none()
	}

	fn start_wipe_a(&mut self) {
		self.wipe_progress_a = 0.0;
		self.wipe_animating_a = true;
	}

	fn start_wipe_b(&mut self) {
		self.wipe_progress_b = 0.0;
		self.wipe_animating_b = true;
	}

	fn tick_wipe(&mut self) {
		let dt = TICK_MS as f32 / 1000.0;
		if self.wipe_animating_a {
			self.wipe_progress_a += dt / WIPE_DURATION_SECS;
			if self.wipe_progress_a >= 1.0 {
				self.wipe_progress_a = 1.0;
				self.wipe_animating_a = false;
			}
		}
		if self.wipe_animating_b {
			self.wipe_progress_b += dt / WIPE_DURATION_SECS;
			if self.wipe_progress_b >= 1.0 {
				self.wipe_progress_b = 1.0;
				self.wipe_animating_b = false;
			}
		}
	}

	fn tick_spectrum(&mut self) {
		let fetch_deck = |deck_mutex: &Mutex<audio::Deck>, buf: &mut VecDeque<f32>| {
			let deck = deck_mutex.lock().unwrap();
			let mut cons_guard = deck.monitor_consumer.lock().unwrap();
			if let Some(consumer) = cons_guard.as_mut() {
				while let Some(s) = consumer.try_pop() {
					buf.push_back(s);
				}
			}
		};

		fetch_deck(&self.audio_engine.deck_a, &mut self.spectrum_buf_a);
		fetch_deck(&self.audio_engine.deck_b, &mut self.spectrum_buf_b);

		while self.spectrum_buf_a.len() > 2048 {
			self.spectrum_buf_a.pop_front();
		}
		while self.spectrum_buf_b.len() > 2048 {
			self.spectrum_buf_b.pop_front();
		}

		let window_size = 1024;
		let mut mix_buf = vec![0.0; window_size];

		let (vol_a_mult, vol_b_mult) = {
			let mut va = self.volume_a;
			let mut vb = self.volume_b;
			if self.crossfader > 0.0 {
				va *= 1.0 - self.crossfader;
			}
			if self.crossfader < 0.0 {
				vb *= 1.0 + self.crossfader;
			}
			(va, vb)
		};

		let len_a = self.spectrum_buf_a.len();
		let start_a = len_a.saturating_sub(window_size);
		for (i, sample) in self.spectrum_buf_a.iter().skip(start_a).take(window_size).enumerate() {
			if i < mix_buf.len() {
				mix_buf[i] += sample * vol_a_mult;
			}
		}

		let len_b = self.spectrum_buf_b.len();
		let start_b = len_b.saturating_sub(window_size);
		for (i, sample) in self.spectrum_buf_b.iter().skip(start_b).take(window_size).enumerate() {
			if i < mix_buf.len() {
				mix_buf[i] += sample * vol_b_mult;
			}
		}

		self.spectrum_display = analysis::compute_spectrum(&mix_buf);
	}

	pub(super) fn view(&self) -> Element<Message> {
		let deck_a = self.audio_engine.deck_a.lock().unwrap();
		let deck_b = self.audio_engine.deck_b.lock().unwrap();

		// --- Deck A Waveform ---
		let pos_a = if deck_a.duration.as_secs_f64() > 0.0 {
			deck_a.get_position().as_secs_f64() / deck_a.duration.as_secs_f64()
		} else {
			0.0
		};

		let waveform_a: Element<WaveformOutput> = Element::from(
			Canvas::new(Waveform::new(
				&deck_a.waveform,
				pos_a as f32,
				deck_a.bpm,
				deck_a.beat_offset,
				44100,
				self.zoom_a,
				&self.waveform_cache_a,
			))
			.width(Length::Fill)
			.height(Length::Fixed(100.0)),
		);
		let waveform_a: Element<Message> = waveform_a.map(|out| match out {
			WaveformOutput::Seek(p) => Message::DeckASeek(p),
			WaveformOutput::Scratch(v) => Message::DeckAScratch(v),
			WaveformOutput::Zoom(z) => Message::DeckAZoom(z),
			WaveformOutput::Released => Message::DeckAScratch(1.0),
		});

		// --- Deck B Waveform ---
		let pos_b = if deck_b.duration.as_secs_f64() > 0.0 {
			deck_b.get_position().as_secs_f64() / deck_b.duration.as_secs_f64()
		} else {
			0.0
		};

		let waveform_b: Element<WaveformOutput> = Element::from(
			Canvas::new(Waveform::new(
				&deck_b.waveform,
				pos_b as f32,
				deck_b.bpm,
				deck_b.beat_offset,
				44100,
				self.zoom_b,
				&self.waveform_cache_b,
			))
			.width(Length::Fill)
			.height(Length::Fixed(100.0)),
		);
		let waveform_b: Element<Message> = waveform_b.map(|out| match out {
			WaveformOutput::Seek(p) => Message::DeckBSeek(p),
			WaveformOutput::Scratch(v) => Message::DeckBScratch(v),
			WaveformOutput::Zoom(z) => Message::DeckBZoom(z),
			WaveformOutput::Released => Message::DeckBScratch(1.0),
		});

		// --- Deck Controls ---
		let deck_a_content = view_deck(
			"DECK A",
			&deck_a,
			self.loading_a.is_some(),
			self.progress_a,
			Message::DeckAPlay,
			Message::DeckAPause,
			Message::DeckALoad,
			Message::DeckASpeed,
			self.high_a,
			self.mid_a,
			self.low_a,
			Message::DeckAHigh,
			Message::DeckAMid,
			Message::DeckALow,
			Message::DeckAMetronome,
		);

		let deck_b_content = view_deck(
			"DECK B",
			&deck_b,
			self.loading_b.is_some(),
			self.progress_b,
			Message::DeckBPlay,
			Message::DeckBPause,
			Message::DeckBLoad,
			Message::DeckBSpeed,
			self.high_b,
			self.mid_b,
			self.low_b,
			Message::DeckBHigh,
			Message::DeckBMid,
			Message::DeckBLow,
			Message::DeckBMetronome,
		);

		// --- Wipe Effect Overlay ---
		let deck_a_final: Element<Message> = if self.wipe_animating_a {
			let wipe: Element<()> = Element::from(
				Canvas::new(WipeOverlay { progress: self.wipe_progress_a })
					.width(Length::Fill)
					.height(Length::Fill),
			);
			let wipe = wipe.map(|_| Message::Tick(std::time::Instant::now()));
			let deck_container = container(deck_a_content)
				.width(Length::Fill)
				.height(Length::Fill);
			stack![deck_container, wipe].into()
		} else {
			deck_a_content
		};

		let deck_b_final: Element<Message> = if self.wipe_animating_b {
			let wipe: Element<()> = Element::from(
				Canvas::new(WipeOverlay { progress: self.wipe_progress_b })
					.width(Length::Fill)
					.height(Length::Fill),
			);
			let wipe = wipe.map(|_| Message::Tick(std::time::Instant::now()));
			let deck_container = container(deck_b_content)
				.width(Length::Fill)
				.height(Length::Fill);
			stack![deck_container, wipe].into()
		} else {
			deck_b_content
		};

		// --- Mixer ---
		let fader_a = container(
			vertical_slider(0.0..=1.0, self.volume_a, Message::VolumeAChanged)
				.step(0.01)
				.width(26.0)
				.height(150)
				.style(fader_style),
		)
		.padding([12, 16])
		.style(fader_track_style);
		let fader_b = container(
			vertical_slider(0.0..=1.0, self.volume_b, Message::VolumeBChanged)
				.step(0.01)
				.width(26.0)
				.height(150)
				.style(fader_style),
		)
		.padding([12, 16])
		.style(fader_track_style);
		let crossfader = slider(-1.0..=1.0, self.crossfader, Message::CrossfaderChanged)
			.step(0.01)
			.style(slider_style);

		// --- Rotating Discs ---
		let rotation_a = pos_a as f32 * std::f32::consts::PI * 20.0;
		let rotation_b = pos_b as f32 * std::f32::consts::PI * 20.0;

		let disc_a: Element<Message> = Element::from(
			Canvas::new(RotatingDisc::new(rotation_a, deck_a.is_playing, &self.disc_cache_a))
				.width(Length::Fixed(100.0))
				.height(Length::Fixed(100.0)),
		)
		.map(|_| Message::Tick(std::time::Instant::now()));

		let disc_b: Element<Message> = Element::from(
			Canvas::new(RotatingDisc::new(rotation_b, deck_b.is_playing, &self.disc_cache_b))
				.width(Length::Fixed(100.0))
				.height(Length::Fixed(100.0)),
		)
		.map(|_| Message::Tick(std::time::Instant::now()));

		let vol_a_col = column![
			text("VOL A").size(10).color(Color::from_rgb8(160, 160, 160)),
			fader_a,
			text(format!("{:.0}%", self.volume_a * 100.0))
				.size(11)
				.color(Color::from_rgb8(140, 140, 140))
		]
		.spacing(8)
		.align_x(iced::Alignment::Center);

		let vol_b_col = column![
			text("VOL B").size(10).color(Color::from_rgb8(160, 160, 160)),
			fader_b,
			text(format!("{:.0}%", self.volume_b * 100.0))
				.size(11)
				.color(Color::from_rgb8(140, 140, 140))
		]
		.spacing(8)
		.align_x(iced::Alignment::Center);

		let mixer_view = column![
			row![
				row![disc_a, vol_a_col].spacing(8).align_y(iced::Alignment::Center),
				row![vol_b_col, disc_b].spacing(8).align_y(iced::Alignment::Center)
			]
			.spacing(30),
			text("CROSSFADER").size(10),
			crossfader
		]
		.spacing(16)
		.align_x(iced::Alignment::Center)
		.width(Length::Shrink);

		// --- Spectrum ---
		let spectrum: Element<Message> = Element::from(
			Canvas::new(Spectrum { data: &self.spectrum_display })
				.width(Length::Fill)
				.height(Length::Fixed(120.0)),
		)
		.map(|_: ()| Message::Tick(std::time::Instant::now()));

		// --- Layout ---
		let controls_row = row![deck_a_final, mixer_view, deck_b_final]
			.padding(20)
			.spacing(20);

		column![waveform_a, waveform_b, controls_row, spectrum]
			.spacing(10)
			.into()
	}
}

// --- Track Loader Subscription ---

fn track_loader(deck_id: u8, path: PathBuf) -> Subscription<Message> {
	Subscription::run_with_id(
		(deck_id, path.clone()),
		iced::stream::channel(100, move |mut output| async move {
			let (tx_internal, mut rx_internal) = tokio::sync::mpsc::unbounded_channel();

			tokio::task::spawn_blocking(move || {
				let res = audio::decode_file_with_progress(path, |p| {
					let _ = tx_internal.send(InternalState::Progress(p));
				});
				let _ = tx_internal.send(InternalState::Finished(res));
			});

			loop {
				if let Some(msg) = rx_internal.recv().await {
					match msg {
						InternalState::Progress(p) => {
							let _ = output
								.send(if deck_id == 0 {
									Message::DeckAProgress(p)
								} else {
									Message::DeckBProgress(p)
								})
								.await;
						}
						InternalState::Finished(res) => {
							let _ = output
								.send(if deck_id == 0 {
									Message::DeckALoaded(res)
								} else {
									Message::DeckBLoaded(res)
								})
								.await;
							break;
						}
					}
				} else {
					break;
				}
			}
		}),
	)
}

enum InternalState {
	Progress(f32),
	Finished(Result<DeckData, String>),
}

// --- Deck View ---

fn view_deck<'a>(
	title: &'a str,
	deck: &Deck,
	is_loading: bool,
	progress: f32,
	on_play: Message,
	on_pause: Message,
	on_load: Message,
	on_speed: fn(f32) -> Message,
	high: f32,
	mid: f32,
	low: f32,
	on_high: fn(f32) -> Message,
	on_mid: fn(f32) -> Message,
	on_low: fn(f32) -> Message,
	on_metronome: Message,
) -> Element<'a, Message> {
	let play_pause_btn = if deck.is_playing {
		button("PAUSE")
			.on_press(on_pause)
			.style(deck_button_style)
	} else {
		button("PLAY").on_press(on_play).style(deck_button_style)
	};

	let speed_slider = slider::<f32, Message, Theme>(0.5..=1.5, deck.user_speed, on_speed)
		.step(0.01)
		.style(slider_style);

	let load_content: Element<'a, Message> = if is_loading {
		Element::from(
			column![
				text("ANALYZING...").size(12),
				progress_bar::<Theme>(0.0..=1.0, progress).height(8)
			]
			.spacing(4)
			.align_x(iced::Alignment::Center),
		)
	} else {
		button("LOAD TRACK")
			.on_press(on_load)
			.style(deck_button_style)
			.into()
	};

	let eq_col = |label, val, msg| {
		column![
			text(label).size(10).color(Color::from_rgb8(160, 160, 160)),
			vertical_slider(0.0..=1.0, val, msg)
				.step(0.01)
				.height(80)
				.style(eq_slider_style)
		]
		.align_x(iced::Alignment::Center)
	};

	let eq_row = row![
		eq_col("HI", high, on_high),
		eq_col("MID", mid, on_mid),
		eq_col("LO", low, on_low)
	]
	.spacing(12);

	let status_text = if is_loading {
		"Loading..."
	} else if deck.is_playing {
		"Playing"
	} else {
		"Stopped"
	};

	let bpm_text = if deck.bpm > 0.0 {
		format!("{:.1} BPM", deck.bpm)
	} else {
		"--- BPM".to_string()
	};

	container(
		column![
			text(title).size(24).color(Color::from_rgb8(200, 200, 200)),
			row![
				text(status_text).size(12).color(Color::from_rgb8(140, 140, 140)),
				text("  ").size(12),
				text(bpm_text).size(14).color(Color::from_rgb8(100, 149, 237)),
			]
			.spacing(4),
			load_content,
			row![
				play_pause_btn,
				button(if deck.metronome_enabled { "METRO ON" } else { "METRO OFF" })
					.on_press(on_metronome)
					.style(deck_button_style)
			]
			.spacing(8),
			eq_row,
			text(format!("Speed: {:.2}x", deck.user_speed))
				.size(11)
				.color(Color::from_rgb8(140, 140, 140)),
			speed_slider
		]
		.spacing(8)
		.align_x(iced::Alignment::Center),
	)
	.width(Length::FillPortion(2))
	.into()
}

// --- Styles ---

fn deck_button_style(_theme: &Theme, status: button::Status) -> button::Style {
	let border_color = match status {
		button::Status::Active => Color::from_rgb8(100, 100, 100),
		button::Status::Hovered => Color::from_rgb8(180, 180, 180),
		button::Status::Pressed => Color::from_rgb8(60, 60, 60),
		button::Status::Disabled => Color::from_rgb8(50, 50, 50),
	};

	button::Style {
		background: Some(iced::Background::Color(Color::from_rgba(0.0, 0.0, 0.0, 0.4))),
		text_color: border_color,
		border: iced::Border {
			color: border_color,
			width: 1.0,
			radius: 4.0.into(),
		},
		shadow: iced::Shadow::default(),
	}
}

fn slider_style(_theme: &Theme, status: slider::Status) -> slider::Style {
	let handle_color = match status {
		slider::Status::Active => Color::from_rgb8(200, 200, 200),
		slider::Status::Hovered => Color::WHITE,
		slider::Status::Dragged => Color::from_rgb8(100, 149, 237),
	};

	slider::Style {
		rail: slider::Rail {
			backgrounds: (
				iced::Background::Color(Color::from_rgb8(60, 60, 60)),
				iced::Background::Color(Color::from_rgb8(30, 30, 30)),
			),
			border: iced::Border {
				radius: 2.0.into(),
				..iced::Border::default()
			},
			width: 4.0,
		},
		handle: slider::Handle {
			shape: slider::HandleShape::Rectangle {
				width: 8,
				border_radius: 2.0.into(),
			},
			background: iced::Background::Color(handle_color),
			border_color: Color::from_rgb8(80, 80, 80),
			border_width: 1.0,
		},
	}
}

fn fader_style(_theme: &Theme, status: slider::Status) -> slider::Style {
	let (handle_color, border_color) = match status {
		slider::Status::Active => (Color::from_rgb8(220, 220, 220), Color::from_rgb8(70, 70, 70)),
		slider::Status::Hovered => (Color::WHITE, Color::from_rgb8(100, 149, 237)),
		slider::Status::Dragged => (Color::from_rgb8(100, 149, 237), Color::from_rgb8(150, 190, 255)),
	};

	slider::Style {
		rail: slider::Rail {
			backgrounds: (
				iced::Background::Color(Color::from_rgb8(100, 149, 237)),
				iced::Background::Color(Color::from_rgb8(28, 28, 28)),
			),
			border: iced::Border {
				radius: 3.0.into(),
				width: 1.0,
				color: Color::from_rgb8(50, 50, 50),
			},
			width: 6.0,
		},
		handle: slider::Handle {
			shape: slider::HandleShape::Rectangle {
				width: 16,
				border_radius: 4.0.into(),
			},
			background: iced::Background::Color(handle_color),
			border_color,
			border_width: 1.5,
		},
	}
}

fn fader_track_style(_theme: &Theme) -> container::Style {
	container::Style {
		background: Some(iced::Background::Color(Color::from_rgb8(10, 10, 10))),
		border: iced::Border {
			color: Color::from_rgb8(45, 45, 45),
			width: 1.0,
			radius: 6.0.into(),
		},
		..container::Style::default()
	}
}

fn eq_slider_style(_theme: &Theme, status: slider::Status) -> slider::Style {
	let handle_color = match status {
		slider::Status::Active => Color::from_rgb8(200, 200, 200),
		slider::Status::Hovered => Color::WHITE,
		slider::Status::Dragged => Color::from_rgb8(211, 253, 80),
	};

	slider::Style {
		rail: slider::Rail {
			backgrounds: (
				iced::Background::Color(Color::from_rgb8(60, 60, 60)),
				iced::Background::Color(Color::from_rgb8(30, 30, 30)),
			),
			border: iced::Border {
				radius: 2.0.into(),
				..iced::Border::default()
			},
			width: 3.0,
		},
		handle: slider::Handle {
			shape: slider::HandleShape::Rectangle {
				width: 14,
				border_radius: 2.0.into(),
			},
			background: iced::Background::Color(handle_color),
			border_color: Color::from_rgb8(80, 80, 80),
			border_width: 1.0,
		},
	}
}
