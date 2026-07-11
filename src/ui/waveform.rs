use crate::analysis;
use iced::mouse;
use iced::widget::canvas::{self, Cache, Geometry};
use iced::{Color, Point, Rectangle, Size, Theme};

const CENTER_LINE_COLOR: Color = Color::from_rgba(1.0, 1.0, 1.0, 1.0);
const BEAT_GRID_COLOR: Color = Color::from_rgba(1.0, 1.0, 1.0, 0.3); // 通常の拍（薄め）
const BAR_GRID_COLOR: Color = Color::from_rgba(1.0, 1.0, 1.0, 0.7); // 小節の最初の拍（強調）

#[derive(Debug, Clone, Copy)]
pub enum WaveformOutput {
    Seek(f32),
    Scratch(f32),
    Zoom(f32),
    Released,
}

#[derive(Debug, Clone, Copy)]
pub enum WaveformState {
    Idle,
    Dragging {
        last_x: f32,
        last_time: std::time::Instant,
    },
}

impl Default for WaveformState {
    fn default() -> Self {
        Self::Idle
    }
}

pub struct Waveform<'a> {
    data: Vec<analysis::WaveformPoint>,
    pos: f32,
    bpm: f32,
    beat_offset: f32, // 秒単位での最初の拍の位置
    sample_rate: u32,
    zoom: f32,
    cache: &'a Cache,
}

impl<'a> Waveform<'a> {
    pub fn new(
        data: &[analysis::WaveformPoint],
        pos: f32,
        bpm: f32,
        beat_offset: f32,
        sample_rate: u32,
        zoom: f32,
        cache: &'a Cache,
    ) -> Self {
        Self {
            data: data.to_vec(),
            pos,
            bpm,
            beat_offset,
            sample_rate,
            zoom,
            cache,
        }
    }
}

impl<'a> canvas::Program<WaveformOutput> for Waveform<'a> {
    type State = WaveformState;

    fn update(
        &self,
        state: &mut Self::State,
        event: canvas::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> (iced::event::Status, Option<WaveformOutput>) {
        let cursor_position = if let Some(p) = cursor.position_in(bounds) {
            p
        } else {
            return (iced::event::Status::Ignored, None);
        };

        match event {
            canvas::Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                match delta {
                    mouse::ScrollDelta::Lines { y, .. } | mouse::ScrollDelta::Pixels { y, .. } => {
                        let zoom_sensitivity = 0.02;
                        let new_zoom = (self.zoom + y * zoom_sensitivity).clamp(0.2, 5.0);

                        if (new_zoom - self.zoom).abs() > 0.001 {
                            return (
                                iced::event::Status::Captured,
                                Some(WaveformOutput::Zoom(new_zoom)),
                            );
                        }
                    }
                }
                (iced::event::Status::Ignored, None)
            }
            canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let count = self.data.len() as f32;

                *state = WaveformState::Dragging {
                    last_x: cursor_position.x,
                    last_time: std::time::Instant::now(),
                };

                if count > 0.0 {
                    // Stop/Hold the record on press
                    (
                        iced::event::Status::Captured,
                        Some(WaveformOutput::Scratch(0.0)),
                    )
                } else {
                    (iced::event::Status::Captured, None)
                }
            }
            canvas::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                *state = WaveformState::Idle;
                (
                    iced::event::Status::Captured,
                    Some(WaveformOutput::Released),
                )
            }
            canvas::Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                if let WaveformState::Dragging { last_x, last_time } = *state {
                    let now = std::time::Instant::now();
                    let dt = now.duration_since(last_time).as_secs_f32();

                    if dt > 0.001 {
                        let dx = cursor_position.x - last_x;

                        // Calculate sensitivity based on visual speed
                        let target_beat_width = 128.0 * self.zoom;
                        let bpm = if self.bpm > 0.0 { self.bpm } else { 120.0 }; // Fallback
                        let pixels_per_second = target_beat_width * (bpm / 60.0);

                        // sensitivity = 1.0 / pixels_per_second
                        // If we move pixels_per_second in 1 second, we want speed to be 1.0 (normal play speed).
                        // Note: scratching is "speed override", not "add to speed".
                        // Wait, previous implementation was returning speed ~0.0-2.0 range?
                        // No, usually scratch sends "velocity".
                        // Let's stick to 1:1 mapping.

                        let sensitivity = if pixels_per_second > 0.0 {
                            1.0 / pixels_per_second
                        } else {
                            0.005 // Fallback
                        };

                        let speed = -(dx / dt) * sensitivity;

                        *state = WaveformState::Dragging {
                            last_x: cursor_position.x,
                            last_time: now,
                        };

                        return (
                            iced::event::Status::Captured,
                            Some(WaveformOutput::Scratch(speed)),
                        );
                    }
                }
                (iced::event::Status::Ignored, None)
            }
            _ => (iced::event::Status::Ignored, None),
        }
    }

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &iced::Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        self.cache.clear();
        let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
            // Background
            frame.fill_rectangle(
                Point::ORIGIN,
                bounds.size(),
                Color::from_rgba(0.0, 0.0, 0.0, 0.2),
            );

            let width = bounds.width;
            let height = bounds.height;

            if !self.data.is_empty() {
                let count = self.data.len() as f32;
                let current_index = self.pos * count;
                let center_x = width / 2.0;

                // Fixed Resolution Constants
                const PIXELS_PER_BAR: f32 = 1.0;

                // Zoom affects how "fast" we step through the data
                let target_beat_width = 128.0 * self.zoom;
                let samples_per_point = 64.0; // Matches analysis stride

                let data_step = if self.bpm > 0.0 && self.sample_rate > 0 {
                    let samples_per_beat = (60.0 / self.bpm) * self.sample_rate as f32;
                    let points_per_beat = samples_per_beat / samples_per_point;
                    (points_per_beat / target_beat_width) * PIXELS_PER_BAR
                } else {
                    0.75 / self.zoom
                };

                // Calculate mapping from Data Index to Screen X
                // x = center_x + (index - current_index) * (PIXELS_PER_BAR / data_step)
                let pixels_per_point = if data_step > 0.0 {
                    PIXELS_PER_BAR / data_step
                } else {
                    0.0
                };

                // Determine visible range
                let visible_width = width / 2.0;
                let visible_points = if pixels_per_point > 0.0 {
                    (visible_width / pixels_per_point).ceil() as usize + 2
                } else {
                    0
                };

                let start_index =
                    (current_index as isize - visible_points as isize).max(0) as usize;
                let end_index = (current_index as usize + visible_points)
                    .min(self.data.len().saturating_sub(1));

                if start_index < self.data.len() && pixels_per_point > 0.0 {
                    let mut prev_x =
                        center_x + (start_index as f32 - current_index) * pixels_per_point;
                    let mut prev_y =
                        (height / 2.0) - (self.data[start_index].amplitude * (height / 2.0));

                    for i in (start_index + 1)..=end_index {
                        let p = &self.data[i];
                        let x = center_x + (i as f32 - current_index) * pixels_per_point;
                        let y = (height / 2.0) - (p.amplitude * (height / 2.0));

                        // 線で描画
                        let path = canvas::Path::new(|b| {
                            b.move_to(Point::new(prev_x, prev_y));
                            b.line_to(Point::new(x, y));
                        });

                        let stroke = canvas::Stroke {
                            style: canvas::Style::Solid(Color::from_rgb(
                                p.color[0], p.color[1], p.color[2],
                            )),
                            width: 1.5,
                            line_cap: canvas::LineCap::Round,
                            line_join: canvas::LineJoin::Round,
                            ..canvas::Stroke::default()
                        };

                        frame.stroke(&path, stroke);

                        prev_x = x;
                        prev_y = y;
                    }
                }

                // Center Line
                frame.fill_rectangle(
                    Point::new(center_x, 0.0),
                    Size::new(2.0, height),
                    CENTER_LINE_COLOR,
                );

                // Beat Grid
                if self.bpm > 0.0 && pixels_per_point > 0.0 {
                    let samples_per_beat = (60.0 / self.bpm) * self.sample_rate as f32;
                    let points_per_beat = samples_per_beat / samples_per_point;

                    // beat_offsetをポイントインデックスに変換
                    let beat_offset_in_points =
                        self.beat_offset * self.sample_rate as f32 / samples_per_point;

                    // Calculate visible index range
                    let min_index = current_index - visible_points as f32;
                    let max_index = current_index + visible_points as f32;

                    // オフセットを考慮したビート番号の計算
                    let start_beat =
                        ((min_index - beat_offset_in_points) / points_per_beat).ceil() as i32;
                    let end_beat =
                        ((max_index - beat_offset_in_points) / points_per_beat).floor() as i32;

                    for beat in start_beat..=end_beat {
                        // オフセットを加算して拍位置を計算
                        let beat_index = beat_offset_in_points + beat as f32 * points_per_beat;
                        let index_diff = beat_index - current_index;

                        let x = center_x + index_diff * pixels_per_point;

                        // 4拍ごと（小節の最初）を強調表示
                        let is_bar_start = beat.rem_euclid(4) == 0;
                        let (line_width, color) = if is_bar_start {
                            (2.0, BAR_GRID_COLOR) // 小節線: 太く明るく
                        } else {
                            (1.0, BEAT_GRID_COLOR) // 通常の拍: 細く暗く
                        };

                        frame.fill_rectangle(
                            Point::new(x - line_width / 2.0, 0.0),
                            Size::new(line_width, height),
                            color,
                        );
                    }
                }
            }
        });

        vec![geometry]
    }
}
