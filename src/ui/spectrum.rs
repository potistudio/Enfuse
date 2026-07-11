use iced::mouse;
use iced::widget::canvas::{self, Geometry, LineCap, LineJoin, Path, Stroke, Style};
use iced::{Color, Point, Rectangle, Theme};

pub struct Spectrum<'a> {
    pub data: &'a [f32],
}

impl<'a, Message> canvas::Program<Message> for Spectrum<'a> {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &iced::Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());

        // Background
        frame.fill_rectangle(
            Point::ORIGIN,
            bounds.size(),
            Color::from_rgba(0.0, 0.0, 0.0, 0.9),
        );

        let width = bounds.width;
        let height = bounds.height;
        let data_len = self.data.len();

        if data_len > 0 {
            let num_points = 128; // Higher resolution

            // Freq mapped to Bins
            let nyquist = 22050.0f32;
            let min_freq = 20.0f32;
            let max_freq = 20000.0f32;
            let min_log = min_freq.ln();
            let max_log = max_freq.ln();
            let log_width = max_log - min_log;

            let norm_factor = 1.0 / (data_len as f32);

            let mut points = Vec::with_capacity(num_points);

            for i in 0..num_points {
                let t = i as f32 / (num_points - 1) as f32;

                // Frequency window
                let t_next = (i as f32 + 1.0) / num_points as f32;
                let f_start = (min_log + t * log_width).exp();
                let f_end = (min_log + t_next * log_width).exp();

                let start_idx = (f_start / nyquist * data_len as f32).floor() as usize;
                let end_idx = (f_end / nyquist * data_len as f32).ceil() as usize;

                let start_idx = start_idx.clamp(0, data_len - 1);
                let end_idx = end_idx.clamp(start_idx + 1, data_len);

                let mut sum = 0.0;
                for j in start_idx..end_idx {
                    if j < data_len {
                        sum += self.data[j];
                    }
                }
                let count = (end_idx - start_idx) as f32;
                let avg = if count > 0.0 { sum / count } else { 0.0 };

                // dB scaling
                let amp = avg * norm_factor;
                let db = 20.0 * amp.max(1e-6).log10();
                let normalized = ((db + 60.0) / 60.0).clamp(0.0, 1.0);

                let x = t * width;
                let y = height - (normalized * height);
                points.push(Point::new(x, y));
            }

            // Draw Line Segments for Gradient
            if !points.is_empty() {
                let mut last_pt = points[0];
                for (i, &pt) in points.iter().enumerate().skip(1) {
                    let t = i as f32 / num_points as f32;
                    let (r, g, b) = if t < 0.5 {
                        (1.0 - t * 2.0, t * 2.0, 0.0)
                    } else {
                        (0.0, 1.0 - (t - 0.5) * 2.0, (t - 0.5) * 2.0)
                    };

                    let segment = Path::new(|p| {
                        p.move_to(last_pt);
                        p.line_to(pt);
                    });

                    frame.stroke(
                        &segment,
                        Stroke {
                            style: Style::Solid(Color::from_rgb(r, g, b)),
                            width: 2.0,
                            line_cap: LineCap::Round,
                            line_join: LineJoin::Round,
                            ..Default::default()
                        },
                    );
                    // Note: We might want a stronger line or maybe 3.0 width

                    last_pt = pt;
                }
            }
        }

        vec![frame.into_geometry()]
    }
}
