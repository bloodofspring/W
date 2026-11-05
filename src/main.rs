mod app;
mod caret;
mod library;
mod segment;

use eframe::egui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("SendNudes")
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "SendNudes",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc)))),
    )
}
