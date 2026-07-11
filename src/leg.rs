use iced::futures::SinkExt;
use iced::widget::canvas::{Cache, Canvas};
use iced::widget::{button, column, container, progress_bar, row, slider, text, vertical_slider};
use iced::{Color, Element, Length, Subscription, Task, Theme};
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer as ConsumerTrait, Split};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

type RingBuffer = HeapRb<f32>;
type RbConsumer = <RingBuffer as Split>::Cons;

mod analysis;
mod audio;
mod ui;
use audio::{AudioEngine, Deck, DeckData};
use ui::disc::RotatingDisc;
use ui::spectrum::Spectrum;
use ui::waveform::{Waveform, WaveformOutput};

pub fn main() -> iced::Result {
    env_logger::init();

    iced::application("Dix - Rust DJ App", DixApp::update, DixApp::view)
        .subscription(DixApp::subscription)
        .theme(DixApp::theme)
        .font(include_bytes!("../assets/fonts/Inter-Regular.ttf").as_slice())
        .default_font(iced::Font::with_name("Inter"))
        .run()
}

struct DixApp {
    audio_engine: AudioEngine,
    loading_a: Option<PathBuf>,
    progress_a: f32,
    loading_b: Option<PathBuf>,
    progress_b: f32,

    // Mixer State
    volume_a: f32,
    volume_b: f32,
    crossfader: f32, // -1.0 (A) to 1.0 (B)

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
}

impl Default for DixApp {
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
        }
    }
}

#[derive(Debug, Clone)]
enum Message {
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

    // Mixer Messages
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
}

impl DixApp {
    fn theme(&self) -> Theme {
        Theme::custom(
            "Dix Dark".to_string(),
            iced::theme::Palette {
                background: Color::from_rgb8(32, 32, 32),
                text: Color::WHITE,
                primary: Color::from_rgb8(100, 149, 237), // Cornflower blue
                success: Color::from_rgb8(50, 205, 50),   // Lime green
                danger: Color::from_rgb8(220, 20, 60),    // Crimson
            },
        )
    }

    fn apply_volumes(&self) {
        let mut deck_a = self.audio_engine.deck_a.lock().unwrap();
        let mut deck_b = self.audio_engine.deck_b.lock().unwrap();

        // Linear crossfader curve
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

    fn subscription(&self) -> Subscription<Message> {
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

        // Time subscription for playhead
        let is_playing = {
            let a = self.audio_engine.deck_a.lock().unwrap().is_playing;
            let b = self.audio_engine.deck_b.lock().unwrap().is_playing;
            a || b
        };

        let time_sub = if is_playing {
            iced::time::every(std::time::Duration::from_millis(16)).map(Message::Tick)
        } else {
            Subscription::none()
        };

        Subscription::batch(vec![sub_a, sub_b, time_sub])
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Tick(_) => {
                // Fetch samples
                let fetch_deck = |deck_mutex: &Mutex<audio::Deck>, buf: &mut VecDeque<f32>| {
                    let deck = deck_mutex.lock().unwrap();
                    let mut cons_guard = deck.monitor_consumer.lock().unwrap();
                    if let Some(consumer) = cons_guard.as_mut() {
                        // Pop all available
                        while let Some(s) = consumer.try_pop() {
                            buf.push_back(s);
                        }
                    }
                };

                fetch_deck(&self.audio_engine.deck_a, &mut self.spectrum_buf_a);
                fetch_deck(&self.audio_engine.deck_b, &mut self.spectrum_buf_b);

                // Limit buffers
                while self.spectrum_buf_a.len() > 2048 {
                    self.spectrum_buf_a.pop_front();
                }
                while self.spectrum_buf_b.len() > 2048 {
                    self.spectrum_buf_b.pop_front();
                }

                // Mix (1024 samples)
                let window_size = 1024;
                let mut mix_buf = vec![0.0; window_size];

                // Calculate effective volumes (A/B + Crossfader)
                // Simplified: linear volume + mix
                // We already have self.volume_a / b but crossfader applies to them or handled in deck?
                // Deck volume is `set_volume`. Fetch is pre-volume?
                // `SpySource` is after `PitchShiftSource`, before `Sink::set_volume`?
                // In `Deck::play`: `sink.set_volume`. `source` is appended.
                // So `SpySource` captures data BEFORE volume is applied by Sink.
                // So we MUST apply volume here.

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

                // Mix A
                let len_a = self.spectrum_buf_a.len();
                let start_a = len_a.saturating_sub(window_size);
                for (i, sample) in self
                    .spectrum_buf_a
                    .iter()
                    .skip(start_a)
                    .take(window_size)
                    .enumerate()
                {
                    if i < mix_buf.len() {
                        mix_buf[i] += sample * vol_a_mult;
                    }
                }

                // Mix B
                let len_b = self.spectrum_buf_b.len();
                let start_b = len_b.saturating_sub(window_size);
                for (i, sample) in self
                    .spectrum_buf_b
                    .iter()
                    .skip(start_b)
                    .take(window_size)
                    .enumerate()
                {
                    if i < mix_buf.len() {
                        mix_buf[i] += sample * vol_b_mult;
                    }
                }

                // Compute Spectrum
                self.spectrum_display = analysis::compute_spectrum(&mix_buf);

                // Metronome
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
                let mut deck = self.audio_engine.deck_a.lock().unwrap();
                deck.play();
            }
            Message::DeckAPause => {
                let mut deck = self.audio_engine.deck_a.lock().unwrap();
                deck.pause();
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
                        println!("Error loading track A: {}", e);
                    }
                }
                self.apply_volumes(); // Apply initial volume state
            }
            Message::DeckASpeed(speed) => {
                let mut deck = self.audio_engine.deck_a.lock().unwrap();
                deck.set_speed(speed);
            }
            Message::DeckBPlay => {
                let mut deck = self.audio_engine.deck_b.lock().unwrap();
                deck.play();
            }
            Message::DeckBPause => {
                let mut deck = self.audio_engine.deck_b.lock().unwrap();
                deck.pause();
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
                        println!("Error loading track B: {}", e);
                    }
                }
                self.apply_volumes();
            }
            Message::DeckBSpeed(speed) => {
                let mut deck = self.audio_engine.deck_b.lock().unwrap();
                deck.set_speed(speed);
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
                let mut deck = self.audio_engine.deck_a.lock().unwrap();
                deck.set_scratch_speed(v);
            }
            Message::DeckBScratch(v) => {
                let mut deck = self.audio_engine.deck_b.lock().unwrap();
                deck.set_scratch_speed(v);
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
        }
        Task::none()
    }

    fn view(&self) -> Element<Message> {
        let deck_a = self.audio_engine.deck_a.lock().unwrap();
        let deck_b = self.audio_engine.deck_b.lock().unwrap();

        // Waveforms
        let pos_a = if deck_a.duration.as_secs_f64() > 0.0 {
            deck_a.get_position().as_secs_f64() / deck_a.duration.as_secs_f64()
        } else {
            0.0
        };

        let waveform_a = Element::from(
            Canvas::new(Waveform::new(
                &deck_a.waveform,
                pos_a as f32,
                deck_a.bpm,
                deck_a.beat_offset,
                44100,
                self.zoom_a, // Pass zoom
                &self.waveform_cache_a,
            ))
            .width(Length::Fill)
            .height(Length::Fixed(100.0)),
        )
        .map(|out| match out {
            WaveformOutput::Seek(p) => Message::DeckASeek(p),
            WaveformOutput::Scratch(v) => Message::DeckAScratch(v),
            WaveformOutput::Zoom(z) => Message::DeckAZoom(z), // Handle zoom
            WaveformOutput::Released => Message::DeckAScratch(1.0),
        });

        let pos_b = if deck_b.duration.as_secs_f64() > 0.0 {
            deck_b.get_position().as_secs_f64() / deck_b.duration.as_secs_f64()
        } else {
            0.0
        };

        let waveform_b = Element::from(
            Canvas::new(Waveform::new(
                &deck_b.waveform,
                pos_b as f32,
                deck_b.bpm,
                deck_b.beat_offset,
                44100,
                self.zoom_b, // Pass zoom
                &self.waveform_cache_b,
            ))
            .width(Length::Fill)
            .height(Length::Fixed(100.0)),
        )
        .map(|out| match out {
            WaveformOutput::Seek(p) => Message::DeckBSeek(p),
            WaveformOutput::Scratch(v) => Message::DeckBScratch(v),
            WaveformOutput::Zoom(z) => Message::DeckBZoom(z), // Handle zoom
            WaveformOutput::Released => Message::DeckBScratch(1.0),
        });

        let deck_a_view = view_deck(
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

        let deck_b_view = view_deck(
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

        let fader_a = vertical_slider(0.0..=1.0, self.volume_a, Message::VolumeAChanged).step(0.01);

        let fader_b = vertical_slider(0.0..=1.0, self.volume_b, Message::VolumeBChanged).step(0.01);

        let crossfader = slider(-1.0..=1.0, self.crossfader, Message::CrossfaderChanged).step(0.01);

        let mixer_view = column![
            row![
                column![text("Vol A"), fader_a]
                    .spacing(10)
                    .align_x(iced::Alignment::Center),
                column![text("Vol B"), fader_b]
                    .spacing(10)
                    .align_x(iced::Alignment::Center)
            ]
            .spacing(40),
            text("Crossfader"),
            crossfader
        ]
        .spacing(20)
        .align_x(iced::Alignment::Center)
        .width(Length::FillPortion(1));

        let controls_row = row![deck_a_view, mixer_view, deck_b_view]
            .padding(20)
            .spacing(20);

        let spectrum = Element::from(
            Canvas::new(Spectrum {
                data: &self.spectrum_display,
            })
            .width(Length::Fill)
            .height(Length::Fixed(150.0)),
        );

        // Rotating Discs
        let rotation_a = pos_a as f32 * std::f32::consts::PI * 20.0; // 10 rotations per full track
        let rotation_b = pos_b as f32 * std::f32::consts::PI * 20.0;

        let disc_a: Element<Message> = Element::from(
            Canvas::new(RotatingDisc::new(
                rotation_a,
                deck_a.is_playing,
                &self.disc_cache_a,
            ))
            .width(Length::Fixed(100.0))
            .height(Length::Fixed(100.0)),
        )
        .map(|_| Message::Tick(std::time::Instant::now()));

        let disc_b: Element<Message> = Element::from(
            Canvas::new(RotatingDisc::new(
                rotation_b,
                deck_b.is_playing,
                &self.disc_cache_b,
            ))
            .width(Length::Fixed(100.0))
            .height(Length::Fixed(100.0)),
        )
        .map(|_| Message::Tick(std::time::Instant::now()));

        // Waveform rows with discs
        let waveform_row_a = row![disc_a, waveform_a]
            .spacing(10)
            .align_y(iced::Alignment::Center);
        let waveform_row_b = row![waveform_b, disc_b]
            .spacing(10)
            .align_y(iced::Alignment::Center);

        column![waveform_row_a, waveform_row_b, controls_row, spectrum]
            .spacing(10)
            .into()
    }
}

// Ensure unique ID for subscription using a struct or distinct key.
fn track_loader(deck_id: u8, path: PathBuf) -> Subscription<Message> {
    Subscription::run_with_id(
        (deck_id, path.clone()), // Use a tuple as a unique key for the subscription
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
                            // We don't break immediately, we let the stream hang?
                            // No, if we break, the stream ends.
                            break;
                        }
                    }
                } else {
                    // Channel closed, task finished
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

fn view_deck<'a>(
    title: &'a str,
    deck: &Deck,
    is_loading: bool,
    progress: f32, // 0.0 to 1.0
    on_play: Message,
    on_pause: Message,
    on_load: Message,
    on_speed: fn(f32) -> Message,
    // EQ
    high: f32,
    mid: f32,
    low: f32,
    on_high: fn(f32) -> Message,
    on_mid: fn(f32) -> Message,
    on_low: fn(f32) -> Message,
    // Metronome
    on_metronome: Message,
) -> Element<'a, Message> {
    let play_pause_btn = if deck.is_playing {
        button("PAUSE")
            .on_press(on_pause)
            .style(outline_button_style)
    } else {
        button("PLAY").on_press(on_play).style(outline_button_style)
    };

    let speed_slider =
        slider::<f32, Message, Theme>(0.5..=1.5, deck.user_speed, on_speed).step(0.01);

    let speed_controls = row![speed_slider]
        .spacing(10)
        .align_y(iced::Alignment::Center);

    let load_content: Element<'a, Message> = if is_loading {
        Element::from(
            column![
                text("ANALYZING..."),
                progress_bar::<Theme>(0.0..=1.0, progress).height(10)
            ]
            .spacing(5)
            .align_x(iced::Alignment::Center),
        )
    } else {
        button("LOAD TRACK")
            .on_press(on_load)
            .style(outline_button_style)
            .into()
    };

    // EQ Controls
    let eq_col = |label, val, msg| {
        column![
            text(label).size(12),
            vertical_slider(0.0..=1.0, val, msg).step(0.01).height(80)
        ]
        .align_x(iced::Alignment::Center)
    };

    let eq_row = row![
        eq_col("HIGH", high, on_high),
        eq_col("MID", mid, on_mid),
        eq_col("LOW", low, on_low)
    ]
    .spacing(15);

    container(
        column![
            text(title).size(30),
            text(format!(
                "Status: {}",
                if is_loading {
                    "Loading..."
                } else if deck.is_playing {
                    "Playing"
                } else {
                    "Stopped"
                }
            )),
            text(format!(
                "BPM: {}",
                if deck.bpm > 0.0 {
                    format!("{:.1}", deck.bpm)
                } else {
                    "---".to_string()
                }
            ))
            .size(20),
            load_content,
            row![
                play_pause_btn,
                button(if deck.metronome_enabled {
                    "🔔 ON"
                } else {
                    "🔔 OFF"
                })
                .on_press(on_metronome)
                .style(outline_button_style)
            ]
            .spacing(10),
            eq_row,
            text(format!("Speed: {:.2}x", deck.user_speed)),
            speed_controls
        ]
        .spacing(10)
        .align_x(iced::Alignment::Center),
    )
    .width(Length::FillPortion(2))
    .into()
}

/// 角丸ストロークボタンスタイル
fn outline_button_style(_theme: &Theme, status: button::Status) -> button::Style {
    let border_color = match status {
        button::Status::Active => Color::from_rgb8(180, 180, 180),
        button::Status::Hovered => Color::WHITE,
        button::Status::Pressed => Color::from_rgb8(100, 100, 100),
        button::Status::Disabled => Color::from_rgb8(80, 80, 80),
    };

    button::Style {
        background: None,
        text_color: border_color,
        border: iced::Border {
            color: border_color,
            width: 1.5,
            radius: 6.0.into(),
        },
        shadow: iced::Shadow::default(),
    }
}
