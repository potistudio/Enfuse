use iced::futures::SinkExt;
use iced::widget::canvas::{Cache, Canvas};
use iced::widget::{button, column, container, progress_bar, row, slider, stack, text, vertical_slider};
use iced::{Color, Element, Length, Subscription, Task, Theme};
use ringbuf::traits::Consumer as ConsumerTrait;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::analysis;
use crate::audio::{self, AudioEngine, DECK_COUNT, Deck, DeckData};
use crate::ui::disc::RotatingDisc;
use crate::ui::knob::Knob;
use crate::ui::spectrum::Spectrum;
use crate::ui::waveform::{Waveform, WaveformOutput};
use crate::ui::wipe::WipeOverlay;

const WIPE_DURATION_SECS: f32 = 0.5;
const TICK_MS: u64 = 16;
const BPM_ANIM_DURATION_SECS: f32 = 0.35;

struct DeckState {
	volume: f32,
	loading: Option<PathBuf>,
	progress: f32,
	spectrum_buf: VecDeque<f32>,
	waveform_cache: Cache,
	zoom_level: f32,

	// EQ
	eq_high: f32,
	eq_mid: f32,
	eq_low: f32,

	disc_cache: Cache,
	wipe_progress: f32,
	wipe_animating: bool,

	// BPM Counter Animation
	bpm_display: f32,
	bpm_display_prev: f32,
	bpm_anim: f32,
}

impl Default for DeckState {
	fn default() -> Self {
		Self {
			volume: 1.0,
			loading: None,
			progress: 0.0,
			spectrum_buf: VecDeque::with_capacity(2048),
			waveform_cache: Cache::default(),
			zoom_level: 1.0,

			eq_high: 0.5,
			eq_mid: 0.5,
			eq_low: 0.5,

			disc_cache: Cache::default(),
			wipe_progress: 0.0,
			wipe_animating: false,

			bpm_display: 0.0,
			bpm_display_prev: 0.0,
			bpm_anim: 1.0,
		}
	}
}

pub(super) struct App {
	decks: Vec<DeckState>,
	audio_engine: AudioEngine,
	crossfader: f32,

	// Spectrum
	spectrum_display: Vec<f32>,
}

impl Default for App {
	fn default() -> Self {
		Self {
			audio_engine: AudioEngine::new(),
			crossfader: 0.0,
			spectrum_display: Vec::new(),
			decks: (0..DECK_COUNT).map(|_| DeckState::default()).collect(),
		}
	}
}

#[derive(Debug, Clone)]
pub(super) enum Message {
	Tick(std::time::Instant),
	DeckPlay(usize),
	DeckPause(usize),
	DeckLoad(usize),
	DeckProgress(usize, f32),
	DeckLoaded(usize, Result<DeckData, String>),
	DeckSpeed(usize, f32),
	DeckZoom(usize, f32),

	// EQ
	DeckHigh(usize, f32),
	DeckMid(usize, f32),
	DeckLow(usize, f32),

	// Mixer
	VolumeChanged(usize, f32),
	CrossfaderChanged(f32),

	// Seek & Scratch
	DeckSeek(usize, f32),
	DeckScratch(usize, f32),

	// Metronome
	DeckMetronome(usize),

	// Beat match: sync this deck's tempo + phase to the other deck
	DeckBeatMatch(usize),

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
				warning: Color::from_rgb8(255, 165, 0),
				danger: Color::from_rgb8(220, 20, 60),
			},
		)
	}

	// Deck 0 is the "A" side, Deck 1 is the "B" side of the crossfader.
	fn apply_volumes(&self) {
		let mut deck_a = self.audio_engine.decks[0].lock().unwrap();
		let mut deck_b = self.audio_engine.decks[1].lock().unwrap();

		let vol_a = self.decks[0].volume
			* if self.crossfader > 0.0 {
				1.0 - self.crossfader
			} else {
				1.0
			};
		let vol_b = self.decks[1].volume
			* if self.crossfader < 0.0 {
				1.0 + self.crossfader
			} else {
				1.0
			};

		deck_a.set_volume(vol_a);
		deck_b.set_volume(vol_b);
	}

	pub(super) fn subscription(&self) -> Subscription<Message> {
		let mut subs: Vec<Subscription<Message>> = self
			.decks
			.iter()
			.enumerate()
			.map(|(id, state)| {
				if let Some(path) = &state.loading {
					track_loader(id, path.clone())
				} else {
					Subscription::none()
				}
			})
			.collect();

		let is_playing = self
			.audio_engine
			.decks
			.iter()
			.any(|deck| deck.lock().unwrap().is_playing);
		let wipe_active = self.decks.iter().any(|state| state.wipe_animating);
		let bpm_anim_active = self.decks.iter().any(|state| state.bpm_anim < 1.0);

		let time_sub = if is_playing || wipe_active || bpm_anim_active {
			iced::time::every(std::time::Duration::from_millis(TICK_MS)).map(Message::Tick)
		} else {
			Subscription::none()
		};

		subs.push(time_sub);
		subs.push(crate::midi::listener(Message::Midi));

		Subscription::batch(subs)
	}

	pub(super) fn update(&mut self, message: Message) -> Task<Message> {
		match message {
			Message::Tick(_) => {
				self.tick_wipe();
				self.tick_spectrum();
				self.tick_bpm_anim();
				for deck in &self.audio_engine.decks {
					deck.lock().unwrap().tick_metronome();
				}
			}
			Message::VolumeChanged(id, v) => {
				self.decks[id].volume = v;
				self.apply_volumes();
			}
			Message::CrossfaderChanged(v) => {
				self.crossfader = v;
				self.apply_volumes();
			}
			Message::DeckPlay(id) => {
				self.audio_engine.decks[id].lock().unwrap().play();
			}
			Message::DeckPause(id) => {
				self.audio_engine.decks[id].lock().unwrap().pause();
			}
			Message::DeckLoad(id) => {
				if self.decks[id].loading.is_none() {
					if let Some(path) = rfd::FileDialog::new().pick_file() {
						self.decks[id].loading = Some(path);
						self.decks[id].progress = 0.0;
					}
				}
			}
			Message::DeckProgress(id, p) => {
				self.decks[id].progress = p;
			}
			Message::DeckLoaded(id, result) => {
				self.decks[id].loading = None;
				match result {
					Ok(data) => {
						let mut deck = self.audio_engine.decks[id].lock().unwrap();
						deck.load_data(data);
						deck.play();
					}
					Err(e) => {
						log::error!("Error loading track {}: {}", id, e);
					}
				}
				self.apply_volumes();
				self.start_wipe(id);
				let deck = self.audio_engine.decks[id].lock().unwrap();
				let effective_bpm = deck.bpm * deck.user_speed;
				drop(deck);
				self.set_bpm_display(id, effective_bpm);
			}
			Message::DeckSpeed(id, speed) => {
				let mut deck = self.audio_engine.decks[id].lock().unwrap();
				deck.set_speed(speed);
				let effective_bpm = deck.bpm * deck.user_speed;
				drop(deck);
				self.set_bpm_display(id, effective_bpm);
			}
			Message::DeckSeek(id, p) => {
				let mut deck = self.audio_engine.decks[id].lock().unwrap();
				let dur = deck.duration.as_secs_f64() * p as f64;
				deck.seek_to(std::time::Duration::from_secs_f64(dur));
			}
			Message::DeckScratch(id, v) => {
				self.audio_engine.decks[id].lock().unwrap().set_scratch_speed(v);
			}
			Message::DeckZoom(id, z) => {
				self.decks[id].zoom_level = z;
			}
			Message::DeckHigh(id, v) => {
				self.decks[id].eq_high = v;
				self.audio_engine.decks[id].lock().unwrap().set_high(v);
			}
			Message::DeckMid(id, v) => {
				self.decks[id].eq_mid = v;
				self.audio_engine.decks[id].lock().unwrap().set_mid(v);
			}
			Message::DeckLow(id, v) => {
				self.decks[id].eq_low = v;
				self.audio_engine.decks[id].lock().unwrap().set_low(v);
			}
			Message::DeckMetronome(id) => {
				self.audio_engine.decks[id].lock().unwrap().toggle_metronome();
			}
			Message::DeckBeatMatch(id) => {
				let other = 1 - id;
				let (m_bpm, m_speed, m_offset, m_pos, m_empty) = {
					let master = self.audio_engine.decks[other].lock().unwrap();
					(
						master.bpm,
						master.user_speed,
						master.beat_offset,
						master.get_position(),
						master.samples.is_empty(),
					)
				};
				if m_bpm <= 0.0 || m_empty {
					return Task::none();
				}

				let mut slave = self.audio_engine.decks[id].lock().unwrap();
				if !slave.beat_match_to(m_bpm, m_speed, m_offset, m_pos) {
					return Task::none();
				}
				let effective_bpm = slave.bpm * slave.user_speed;
				drop(slave);
				self.set_bpm_display(id, effective_bpm);
			}
			Message::Midi(bytes) => {
				let hex = bytes.iter().map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(" ");
				log::info!("MIDI in: {hex}");
			}
		}
		Task::none()
	}

	fn start_wipe(&mut self, id: usize) {
		self.decks[id].wipe_progress = 0.0;
		self.decks[id].wipe_animating = true;
	}

	fn set_bpm_display(&mut self, id: usize, value: f32) {
		let state = &mut self.decks[id];
		if (state.bpm_display - value).abs() > 0.05 {
			state.bpm_display_prev = state.bpm_display;
			state.bpm_display = value;
			state.bpm_anim = 0.0;
		}
	}

	fn tick_bpm_anim(&mut self) {
		let step = TICK_MS as f32 / 1000.0 / BPM_ANIM_DURATION_SECS;
		for state in &mut self.decks {
			if state.bpm_anim < 1.0 {
				state.bpm_anim = (state.bpm_anim + step).min(1.0);
			}
		}
	}

	fn tick_wipe(&mut self) {
		let dt = TICK_MS as f32 / 1000.0;
		for state in &mut self.decks {
			if state.wipe_animating {
				state.wipe_progress += dt / WIPE_DURATION_SECS;
				if state.wipe_progress >= 1.0 {
					state.wipe_progress = 1.0;
					state.wipe_animating = false;
				}
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

		for (id, deck) in self.audio_engine.decks.iter().enumerate() {
			fetch_deck(deck, &mut self.decks[id].spectrum_buf);
			while self.decks[id].spectrum_buf.len() > 2048 {
				self.decks[id].spectrum_buf.pop_front();
			}
		}

		let window_size = 1024;
		let mut mix_buf = vec![0.0; window_size];

		// Deck 0 ("A") is attenuated on the positive crossfader side, deck 1 ("B") on the negative side.
		for (id, state) in self.decks.iter().enumerate() {
			let mut vol_mult = state.volume;
			if id == 0 && self.crossfader > 0.0 {
				vol_mult *= 1.0 - self.crossfader;
			} else if id == 1 && self.crossfader < 0.0 {
				vol_mult *= 1.0 + self.crossfader;
			}

			let len = state.spectrum_buf.len();
			let start = len.saturating_sub(window_size);
			for (i, sample) in state.spectrum_buf.iter().skip(start).take(window_size).enumerate() {
				if i < mix_buf.len() {
					mix_buf[i] += sample * vol_mult;
				}
			}
		}

		self.spectrum_display = analysis::compute_spectrum(&mix_buf);
	}

	pub(super) fn view(&self) -> Element<'_, Message> {
		let deck_guards: Vec<_> = self.audio_engine.decks.iter().map(|d| d.lock().unwrap()).collect();

		let build_deck = |id: usize| -> (Element<'_, Message>, Element<'_, Message>, Element<'_, Message>) {
			let deck = &deck_guards[id];
			let state = &self.decks[id];

			let pos = if deck.duration.as_secs_f64() > 0.0 {
				deck.get_position().as_secs_f64() / deck.duration.as_secs_f64()
			} else {
				0.0
			};

			let waveform: Element<WaveformOutput> = Element::from(
				Canvas::new(Waveform::new(
					&deck.waveform,
					pos as f32,
					deck.bpm,
					deck.beat_offset,
					44100,
					state.zoom_level,
					&state.waveform_cache,
				))
				.width(Length::Fill)
				.height(Length::Fixed(100.0)),
			);
			let waveform: Element<Message> = waveform.map(move |out| match out {
				WaveformOutput::Seek(p) => Message::DeckSeek(id, p),
				WaveformOutput::Scratch(v) => Message::DeckScratch(id, v),
				WaveformOutput::Zoom(z) => Message::DeckZoom(id, z),
				WaveformOutput::Released => Message::DeckScratch(id, 1.0),
			});

			let deck_content = view_deck(id, deck, state);

			let deck_final: Element<Message> = if state.wipe_animating {
				let wipe: Element<()> = Element::from(
					Canvas::new(WipeOverlay {
						progress: state.wipe_progress,
					})
					.width(Length::Fill)
					.height(Length::Fill),
				);
				let wipe = wipe.map(|_| Message::Tick(std::time::Instant::now()));
				let deck_container = container(deck_content).width(Length::Fill).height(Length::Fill);
				stack![deck_container, wipe].into()
			} else {
				deck_content
			};

			let rotation = pos as f32 * std::f32::consts::PI * 20.0;
			let disc: Element<Message> = Element::from(
				Canvas::new(RotatingDisc::new(
					rotation,
					state.bpm_display_prev,
					state.bpm_display,
					state.bpm_anim,
					deck.is_playing,
					&state.disc_cache,
				))
				.width(Length::Fixed(160.0))
				.height(Length::Fixed(160.0)),
			)
			.map(|_| Message::Tick(std::time::Instant::now()));

			(waveform, deck_final, disc)
		};

		let (waveform_a, deck_a_final, disc_a) = build_deck(0);
		let (waveform_b, deck_b_final, disc_b) = build_deck(1);

		// --- Mixer ---
		let fader = |id: usize| {
			container(
				vertical_slider(0.0..=1.0, self.decks[id].volume, move |v| Message::VolumeChanged(id, v))
					.step(0.01)
					.width(26.0)
					.height(150)
					.style(fader_style),
			)
			.padding([12, 16])
			.style(fader_track_style)
		};

		let vol_col = |id: usize, label: &'static str| {
			column![
				text(label).size(10).color(Color::from_rgb8(160, 160, 160)),
				fader(id),
				text(format!("{:.0}%", self.decks[id].volume * 100.0))
					.size(11)
					.color(Color::from_rgb8(140, 140, 140))
			]
			.spacing(8)
			.align_x(iced::Alignment::Center)
		};

		let crossfader = slider(-1.0..=1.0, self.crossfader, Message::CrossfaderChanged)
			.step(0.01)
			.style(slider_style);

		let mixer_view = column![
			row![
				row![disc_a, vol_col(0, "VOL A")]
					.spacing(8)
					.align_y(iced::Alignment::Center),
				row![vol_col(1, "VOL B"), disc_b]
					.spacing(8)
					.align_y(iced::Alignment::Center)
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
			Canvas::new(Spectrum {
				data: &self.spectrum_display,
			})
			.width(Length::Fill)
			.height(Length::Fixed(120.0)),
		)
		.map(|_: ()| Message::Tick(std::time::Instant::now()));

		// --- Layout ---
		let controls_row = row![deck_a_final, mixer_view, deck_b_final].padding(20).spacing(20);

		column![waveform_a, waveform_b, controls_row, spectrum]
			.spacing(10)
			.into()
	}
}

// --- Track Loader Subscription ---

fn track_loader(deck_id: usize, path: PathBuf) -> Subscription<Message> {
	Subscription::run_with((deck_id, path), track_loader_stream)
}

fn track_loader_stream(
	(deck_id, path): &(usize, PathBuf),
) -> impl iced::futures::Stream<Item = Message> + use<> {
	let deck_id = *deck_id;
	let path = path.clone();

	iced::stream::channel(
		100,
		move |mut output: iced::futures::channel::mpsc::Sender<Message>| async move {
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
							let _ = output.send(Message::DeckProgress(deck_id, p)).await;
						}
						InternalState::Finished(res) => {
							let _ = output.send(Message::DeckLoaded(deck_id, res)).await;
							break;
						}
					}
				} else {
					break;
				}
			}
		},
	)
}

enum InternalState {
	Progress(f32),
	Finished(Result<DeckData, String>),
}

// --- Deck View ---

fn view_deck<'a>(id: usize, deck: &Deck, state: &'a DeckState) -> Element<'a, Message> {
	let title = if id == 0 { "DECK A" } else { "DECK B" };

	let play_pause_btn = if deck.is_playing {
		button("PAUSE")
			.on_press(Message::DeckPause(id))
			.style(deck_button_style)
	} else {
		button("PLAY").on_press(Message::DeckPlay(id)).style(deck_button_style)
	};

	let speed_slider = slider::<f32, Message, Theme>(0.5..=1.5, deck.user_speed, move |v| Message::DeckSpeed(id, v))
		.step(0.01)
		.style(slider_style);

	let load_content: Element<'a, Message> = if state.loading.is_some() {
		Element::from(
			column![
				text("ANALYZING...").size(12),
				progress_bar::<Theme>(0.0..=1.0, state.progress).girth(8)
			]
			.spacing(4)
			.align_x(iced::Alignment::Center),
		)
	} else {
		button("LOAD TRACK")
			.on_press(Message::DeckLoad(id))
			.style(deck_button_style)
			.into()
	};

	let eq_col = move |label: &'static str, val: f32, ctor: fn(usize, f32) -> Message| {
		let knob: Element<'a, f32> = Element::from(
			Canvas::new(Knob::new(val))
				.width(Length::Fixed(44.0))
				.height(Length::Fixed(44.0)),
		);
		column![
			text(label).size(10).color(Color::from_rgb8(160, 160, 160)),
			knob.map(move |v| ctor(id, v)),
		]
		.spacing(4)
		.align_x(iced::Alignment::Center)
	};

	let eq_row = row![
		eq_col("HI", state.eq_high, Message::DeckHigh),
		eq_col("MID", state.eq_mid, Message::DeckMid),
		eq_col("LO", state.eq_low, Message::DeckLow)
	]
	.spacing(16);

	let status_text = if state.loading.is_some() {
		"Loading..."
	} else if deck.is_playing {
		"Playing"
	} else {
		"Stopped"
	};

	let bpm_text = if deck.bpm > 0.0 {
		format!("{:.1} BPM", deck.bpm * deck.user_speed)
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
				button(if deck.metronome_enabled {
					"METRO ON"
				} else {
					"METRO OFF"
				})
				.on_press(Message::DeckMetronome(id))
				.style(deck_button_style),
				button("SYNC")
					.on_press(Message::DeckBeatMatch(id))
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
		snap: false,
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

