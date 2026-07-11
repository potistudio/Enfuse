mod analysis;
mod app;
mod audio;
mod midi;
mod ui;

use app::App;

fn main() -> iced::Result {
	env_logger::init();

	iced::application(App::default, App::update, App::view)
		.title("Enfuse")
		.subscription(App::subscription)
		.theme(App::theme)
		.font(include_bytes!("../assets/fonts/Inter-Regular.ttf").as_slice())
		.default_font(iced::Font::with_name("Inter"))
		.run()
}
