use iced::mouse;
use iced::widget::canvas::{self, Cache, Geometry, Path};
use iced::{Color, Point, Rectangle, Theme};

/// 回転ディスクの描画色
const DISC_COLOR: Color = Color::from_rgba(0.918, 0.918, 0.918, 1.0);
const GROOVE_COLOR: Color = Color::from_rgba(0.2, 0.2, 0.2, 1.0);
const LABEL_COLOR: Color = Color::from_rgba(0.3, 0.3, 0.3, 1.0);
const MARKER_COLOR: Color = Color::from_rgba(1.0, 0.3, 0.3, 1.0);

pub struct RotatingDisc<'a> {
	rotation: f32, // ラジアン
	bpm_prev: f32,
	bpm: f32,
	bpm_anim: f32, // 0.0(切替直後) -> 1.0(表示確定)
	is_playing: bool,
	cache: &'a Cache,
}

impl<'a> RotatingDisc<'a> {
	pub fn new(
		rotation: f32,
		bpm_prev: f32,
		bpm: f32,
		bpm_anim: f32,
		is_playing: bool,
		cache: &'a Cache,
	) -> Self {
		Self {
			rotation,
			bpm_prev,
			bpm,
			bpm_anim,
			is_playing,
			cache,
		}
	}
}

impl<'a> canvas::Program<()> for RotatingDisc<'a> {
	type State = ();

	fn draw(
		&self,
		_state: &Self::State,
		renderer: &iced::Renderer,
		_theme: &Theme,
		bounds: Rectangle,
		_cursor: mouse::Cursor,
	) -> Vec<Geometry> {
		// 再生中、またはBPM切替アニメーション中はキャッシュをクリア
		if self.is_playing || self.bpm_anim < 1.0 {
			self.cache.clear();
		}

		let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
			let size = bounds.width.min(bounds.height);
			let center = Point::new(bounds.width / 2.0, bounds.height / 2.0);
			let radius = size / 2.0 - 2.0;

			// ディスク本体（黒）
			let disc = Path::circle(center, radius);
			frame.stroke(
				&disc,
				canvas::Stroke {
					style: canvas::Style::Solid(DISC_COLOR),
					width: 2.0,
					..canvas::Stroke::default()
				},
			);

			// 中央ラベル（円）
			let label_radius = radius * 0.35;
			let label = Path::circle(center, label_radius);
			frame.fill(&label, LABEL_COLOR);

			// ラベル内の穴
			let hole_radius = radius * 0.05;
			let hole = Path::circle(center, hole_radius);
			frame.fill(&hole, Color::BLACK);

			// 回転マーカー（ラベル上）
			let marker_angles = [
				self.rotation * 10.0,
				self.rotation * 10.0 + std::f32::consts::PI / 1.5, // 30度ずらす
				self.rotation * 10.0 - std::f32::consts::PI / 1.5, // 30度ずらす
			];
			let marker_start = radius * 0.25;
			let marker_end = radius * 0.75;

			for &marker_angle in &marker_angles {
				let start_x = center.x + marker_start * marker_angle.cos();
				let start_y = center.y + marker_start * marker_angle.sin();
				let end_x = center.x + marker_end * marker_angle.cos();
				let end_y = center.y + marker_end * marker_angle.sin();

				let marker = Path::new(|b| {
					b.move_to(Point::new(start_x, start_y));
					b.line_to(Point::new(end_x, end_y));
				});

				frame.stroke(
					&marker,
					canvas::Stroke {
						style: canvas::Style::Solid(DISC_COLOR),
						width: 3.0,
						line_cap: canvas::LineCap::Round,
						..canvas::Stroke::default()
					},
				);
			}

			// BPM表示（マーカーより手前に描画して被らないようにする）
			if self.bpm > 0.0 || self.bpm_prev > 0.0 {
				let label_bg = Path::circle(center, label_radius);
				frame.fill(&label_bg, LABEL_COLOR);
				frame.fill(&hole, Color::BLACK);

				// カウンターのように上下にフェード/スライドしながら切り替わる
				let progress = self.bpm_anim.clamp(0.0, 1.0);
				let eased = 1.0 - (1.0 - progress) * (1.0 - progress); // ease-out
				let travel = label_radius * 0.9;

				if self.bpm_prev > 0.0 && progress < 1.0 {
					frame.fill_text(canvas::Text {
						content: format!("{:.0}", self.bpm_prev),
						position: Point::new(
							center.x,
							center.y - label_radius * 0.25 - eased * travel,
						),
						color: Color::from_rgba(1.0, 1.0, 1.0, 1.0 - eased),
						size: iced::Pixels(label_radius * 0.75),
						align_x: iced::widget::text::Alignment::Center,
						align_y: iced::alignment::Vertical::Center,
						..canvas::Text::default()
					});
				}

				if self.bpm > 0.0 {
					frame.fill_text(canvas::Text {
						content: format!("{:.0}", self.bpm),
						position: Point::new(
							center.x,
							center.y - label_radius * 0.25 + (1.0 - eased) * travel,
						),
						color: Color::from_rgba(1.0, 1.0, 1.0, eased),
						size: iced::Pixels(label_radius * 0.75),
						align_x: iced::widget::text::Alignment::Center,
						align_y: iced::alignment::Vertical::Center,
						..canvas::Text::default()
					});
					frame.fill_text(canvas::Text {
						content: "BPM".to_string(),
						position: Point::new(center.x, center.y + label_radius * 0.55),
						color: Color::from_rgba(1.0, 1.0, 1.0, 0.7 * eased),
						size: iced::Pixels(label_radius * 0.3),
						align_x: iced::widget::text::Alignment::Center,
						align_y: iced::alignment::Vertical::Center,
						..canvas::Text::default()
					});
				}
			}
		});

		vec![geometry]
	}
}
