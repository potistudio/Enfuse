use iced::widget::canvas::{self, Geometry};
use iced::{Color, Point, Rectangle, Size, Theme};

const ACCENT_COLOR: Color = Color::from_rgb(0.827, 0.988, 0.314);

pub struct WipeOverlay {
	pub progress: f32,
}

/// Simple integer hash -> normalized f32 in [0, 1)
fn hash_f01(mut x: u32) -> f32 {
	x ^= x >> 16;
	x = x.wrapping_mul(0x45d9f3b);
	x ^= x >> 16;
	x = x.wrapping_mul(0x45d9f3b);
	x ^= x >> 16;
	x as f32 / u32::MAX as f32
}

impl<Message> canvas::Program<Message> for WipeOverlay {
	type State = ();

	fn draw(
		&self,
		_state: &(),
		renderer: &iced::Renderer,
		_theme: &Theme,
		bounds: Rectangle,
		_cursor: iced::mouse::Cursor,
	) -> Vec<Geometry> {
		let mut frame = canvas::Frame::new(renderer, bounds.size());
		let width = bounds.width;
		let height = bounds.height;

		let spread = 0.5;
		let eased = 1.0 - (1.0 - self.progress).powi(3);
		let t = (-spread - 0.01) + eased * (1.0 + spread * 2.0 + 0.02);
		let wipe_x = t * width;
		let zone_radius = spread * width;
		let base_size = width / 10.0;

		// Solid overlay: right of transition zone (untouched)
		let solid_left = wipe_x + zone_radius;
		if solid_left < width {
			frame.fill_rectangle(
				Point::new(solid_left, 0.0),
				Size::new(width - solid_left, height),
				ACCENT_COLOR,
			);
		}

		// Transition zone: 6 bands, fine (left) -> coarse (right)
		let band_width = zone_radius * 2.0 / 4.0;
		let subs = [32u32, 16, 8, 4, 2, 1];

		for (band, &sub) in subs.iter().enumerate() {
			let band_left = (wipe_x - zone_radius) + band as f32 * band_width;
			let band_right = band_left + band_width;
			let block_size = base_size / sub as f32;

			let col_start = (band_left / block_size).floor() as u32;
			let col_end = (band_right / block_size).ceil() as u32;
			let row_end = (height / block_size).ceil() as u32;

			for row in 0u32..row_end {
				for col in col_start..col_end {
					let x = col as f32 * block_size;
					let y = row as f32 * block_size;
					if x < band_left || x >= band_right || x >= width || y >= height {
						continue;
					}

					let seed = (band as u32)
						.wrapping_mul(15485863)
						.wrapping_add(row.wrapping_mul(7919))
						.wrapping_add(col.wrapping_mul(104729));
					let rand = hash_f01(seed);
					let threshold = x / width + (rand - 0.5) * spread * 2.0;

					if t < threshold {
						frame.fill_rectangle(
							Point::new(x, y),
							Size::new(block_size + 1.0, block_size + 1.0),
							ACCENT_COLOR,
						);
					}
				}
			}
		}

		vec![frame.into_geometry()]
	}
}
