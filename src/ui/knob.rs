use iced::mouse;
use iced::widget::canvas::{self, Geometry, Path};
use iced::{Color, Point, Radians, Rectangle, Theme};
use std::f32::consts::PI;

const SWEEP: f32 = PI * 1.5; // 270°
const START_ANGLE: f32 = PI * 0.75; // bottom-left
const SENSITIVITY: f32 = 0.006;

pub struct Knob {
	value: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct KnobState {
	dragging: bool,
	last_y: f32,
}

impl Knob {
	pub fn new(value: f32) -> Self {
		Self {
			value: value.clamp(0.0, 1.0),
		}
	}

	fn angle(&self) -> f32 {
		START_ANGLE + self.value * SWEEP
	}
}

impl canvas::Program<f32> for Knob {
	type State = KnobState;

	fn update(
		&self,
		state: &mut Self::State,
		event: &canvas::Event,
		bounds: Rectangle,
		cursor: mouse::Cursor,
	) -> Option<canvas::Action<f32>> {
		match event {
			canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
				if !cursor.is_over(bounds) {
					return None;
				}
				state.dragging = true;
				if let Some(pos) = cursor.position() {
					state.last_y = pos.y;
				}
				Some(canvas::Action::capture())
			}
			canvas::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
				if !state.dragging {
					return None;
				}
				state.dragging = false;
				Some(canvas::Action::capture())
			}
			canvas::Event::Mouse(mouse::Event::CursorMoved { position }) => {
				if !state.dragging {
					return None;
				}
				let dy = state.last_y - position.y;
				state.last_y = position.y;
				let new_val = (self.value + dy * SENSITIVITY).clamp(0.0, 1.0);
				if (new_val - self.value).abs() < 0.001 {
					return Some(canvas::Action::capture());
				}
				Some(canvas::Action::publish(new_val).and_capture())
			}
			canvas::Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
				if !cursor.is_over(bounds) {
					return None;
				}
				let y = match delta {
					mouse::ScrollDelta::Lines { y, .. } | mouse::ScrollDelta::Pixels { y, .. } => *y,
				};
				let new_val = (self.value + y * 0.04).clamp(0.0, 1.0);
				if (new_val - self.value).abs() < 0.001 {
					return None;
				}
				Some(canvas::Action::publish(new_val).and_capture())
			}
			_ => None,
		}
	}

	fn draw(
		&self,
		state: &Self::State,
		renderer: &iced::Renderer,
		_theme: &Theme,
		bounds: Rectangle,
		cursor: mouse::Cursor,
	) -> Vec<Geometry> {
		let mut frame = canvas::Frame::new(renderer, bounds.size());
		let center = Point::new(bounds.width / 2.0, bounds.height / 2.0);
		let radius = bounds.width.min(bounds.height) / 2.0 - 2.0;

		let active = state.dragging || cursor.is_over(bounds);
		let accent = if state.dragging {
			Color::from_rgb8(211, 253, 80)
		} else if active {
			Color::WHITE
		} else {
			Color::from_rgb8(200, 200, 200)
		};

		// Body
		frame.fill(&Path::circle(center, radius), Color::from_rgb8(28, 28, 28));
		frame.stroke(
			&Path::circle(center, radius),
			canvas::Stroke {
				style: canvas::Style::Solid(Color::from_rgb8(70, 70, 70)),
				width: 1.5,
				..canvas::Stroke::default()
			},
		);

		// Arc track (inactive range)
		let track = Path::new(|b| {
			b.arc(canvas::path::Arc {
				center,
				radius: radius * 0.78,
				start_angle: Radians(START_ANGLE),
				end_angle: Radians(START_ANGLE + SWEEP),
			});
		});
		frame.stroke(
			&track,
			canvas::Stroke {
				style: canvas::Style::Solid(Color::from_rgb8(45, 45, 45)),
				width: 3.0,
				line_cap: canvas::LineCap::Round,
				..canvas::Stroke::default()
			},
		);

		// Value arc
		if self.value > 0.001 {
			let value_arc = Path::new(|b| {
				b.arc(canvas::path::Arc {
					center,
					radius: radius * 0.78,
					start_angle: Radians(START_ANGLE),
					end_angle: Radians(self.angle()),
				});
			});
			frame.stroke(
				&value_arc,
				canvas::Stroke {
					style: canvas::Style::Solid(accent),
					width: 3.0,
					line_cap: canvas::LineCap::Round,
					..canvas::Stroke::default()
				},
			);
		}

		// Indicator
		let angle = self.angle();
		let inner = radius * 0.25;
		let outer = radius * 0.65;
		let indicator = Path::new(|b| {
			b.move_to(Point::new(
				center.x + inner * angle.cos(),
				center.y + inner * angle.sin(),
			));
			b.line_to(Point::new(
				center.x + outer * angle.cos(),
				center.y + outer * angle.sin(),
			));
		});
		frame.stroke(
			&indicator,
			canvas::Stroke {
				style: canvas::Style::Solid(accent),
				width: 2.5,
				line_cap: canvas::LineCap::Round,
				..canvas::Stroke::default()
			},
		);

		// Center cap
		frame.fill(
			&Path::circle(center, radius * 0.12),
			Color::from_rgb8(50, 50, 50),
		);

		vec![frame.into_geometry()]
	}

	fn mouse_interaction(
		&self,
		state: &Self::State,
		bounds: Rectangle,
		cursor: mouse::Cursor,
	) -> mouse::Interaction {
		if state.dragging {
			mouse::Interaction::Grabbing
		} else if cursor.is_over(bounds) {
			mouse::Interaction::Grab
		} else {
			mouse::Interaction::default()
		}
	}
}
