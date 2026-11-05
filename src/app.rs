//! Two screens: photo grid and transcription split view.

use crate::caret::{
    advance_on_space, click_at, move_by_word, nudge_free, Caret, CaretMove,
};
use crate::library::Library;
use crate::segment::{normalize_reading_order, PageWords};
use eframe::egui::{self, Color32, FontFamily, FontId, Key, Pos2, Rect, RichText, Sense, Vec2};
use image::{GrayImage, RgbaImage};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const FINE_NUDGE: f32 = 8.0;
const AUTOSAVE_SECS: u64 = 3;
const ZOOM_STEP: f32 = 1.15;
const MIN_ZOOM: f32 = 0.5;
const MAX_ZOOM: f32 = 12.0;
const UI_SCALE_STEP: f32 = 1.12;
const UI_SCALE_MIN: f32 = 0.85;
const UI_SCALE_MAX: f32 = 2.5;
const UI_SCALE_DEFAULT: f32 = 1.35;
const RESEGMENT_DEBOUNCE: Duration = Duration::from_millis(250);
/// How many on-disk menu thumbs to upload to GPU per startup frame.
const STARTUP_THUMBS_PER_FRAME: usize = 16;
const STATUS_H: f32 = 36.0;
const MAP_ZOOM_MIN: f32 = 1.0;
const MAP_ZOOM_MAX: f32 = 8.0;
const MAP_ZOOM_STEP: f32 = 1.15;

const BG: Color32 = Color32::from_rgb(8, 10, 12);
const PANEL: Color32 = Color32::from_rgb(14, 17, 20);
const PANEL_2: Color32 = Color32::from_rgb(20, 24, 28);
const BORDER: Color32 = Color32::from_rgb(40, 48, 56);
const CARD_OUTLINE: Color32 = Color32::from_rgb(235, 240, 245);
const TEXT: Color32 = Color32::from_rgb(210, 220, 230);
const MUTED: Color32 = Color32::from_rgb(110, 125, 140);
const ACCENT: Color32 = Color32::from_rgb(61, 220, 255);
const ACCENT_2: Color32 = Color32::from_rgb(46, 230, 166);
const DANGER: Color32 = Color32::from_rgb(255, 80, 100);
const WARN: Color32 = Color32::from_rgb(255, 200, 80);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    List,
    Edit { card: usize },
}

/// Background-ish work that must not freeze the UI for many frames.
#[derive(Debug, Clone)]
enum Busy {
    Idle,
    /// First-launch bootstrap so the window can appear before any heavy IO.
    Startup {
        stage: StartupStage,
    },
    Import {
        queue: VecDeque<PathBuf>,
        total: usize,
        done: usize,
        ok: usize,
        err: usize,
    },
    Opening {
        card: usize,
        stem: String,
        phase: OpenPhase,
        /// Skip words-cache / preview path and always re-run segmentation.
        force_full: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StartupStage {
    /// Open Application Support library (index + settings).
    OpenLibrary,
    /// Decode one card thumbnail per frame.
    WarmThumbs { next: usize, total: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenPhase {
    /// Next frame: decode the photo.
    NeedImage,
    /// Next frame: segment (if no cache) or light preview textures.
    NeedTextures,
}

struct ThumbCache {
    textures: HashMap<String, egui::TextureHandle>,
}

/// Cached Sauvola + close output so merge/line_sep tweaks skip re-binarization.
struct SegmentBinaryCache {
    threshold: u8,
    close_radius_x: u32,
    rotation_deg: f32,
    bin0: GrayImage,
    bin: GrayImage,
    rotated: RgbaImage,
}

impl ThumbCache {
    fn new() -> Self {
        Self {
            textures: HashMap::new(),
        }
    }

    fn get_or_load(
        &mut self,
        ctx: &egui::Context,
        key: &str,
        rgba: &RgbaImage,
    ) -> egui::TextureHandle {
        if let Some(t) = self.textures.get(key) {
            return t.clone();
        }
        let size = [rgba.width() as usize, rgba.height() as usize];
        let pixels: Vec<egui::Color32> = rgba
            .pixels()
            .map(|p| Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]))
            .collect();
        let image = egui::ColorImage { size, pixels };
        let handle = ctx.load_texture(key.to_owned(), image, Default::default());
        self.textures.insert(key.to_owned(), handle.clone());
        handle
    }

    fn invalidate(&mut self, key: &str) {
        self.textures.remove(key);
    }
}

pub struct App {
    /// `None` only during the first startup frames before the library is opened.
    library: Option<Library>,
    screen: Screen,
    selected: usize,
    delete_armed: Option<usize>,
    pending_open: Option<usize>,
    text: String,
    caret: Caret,
    zoom: f32,
    map_zoom: f32,
    page: PageWords,
    edit_texture: Option<(String, egui::TextureHandle, u32, u32)>,
    /// Original scan (unrotated) for resegment.
    edit_rgba: Option<RgbaImage>,
    binary_texture: Option<egui::TextureHandle>,
    thumbs: ThumbCache,
    dirty: bool,
    last_edit: Instant,
    status: String,
    pending_resegment: Option<Instant>,
    /// True while a segment slider is being dragged (delay resegment until release).
    slider_held: bool,
    segment_bin_cache: Option<SegmentBinaryCache>,
    busy: Busy,
    /// Paths queued while another busy job (e.g. opening) is running.
    pending_ingest: Vec<PathBuf>,
    /// Card indices waiting for async thumbnail decode (never block a frame on many).
    pending_thumbs: VecDeque<usize>,
    /// Native DPI factor captured at launch (before our UI scale).
    native_ppp: f32,
    theme_applied: bool,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Keep construction cheap so the native window can appear immediately.
        let native_ppp = cc.egui_ctx.pixels_per_point();
        Self {
            library: None,
            screen: Screen::List,
            selected: 0,
            delete_armed: None,
            pending_open: None,
            text: String::new(),
            caret: Caret::Word(0),
            zoom: 2.0,
            map_zoom: 1.0,
            page: PageWords::default(),
            edit_texture: None,
            edit_rgba: None,
            binary_texture: None,
            thumbs: ThumbCache::new(),
            dirty: false,
            last_edit: Instant::now(),
            status: "starting…".into(),
            pending_resegment: None,
            slider_held: false,
            segment_bin_cache: None,
            busy: Busy::Startup {
                stage: StartupStage::OpenLibrary,
            },
            pending_ingest: Vec::new(),
            pending_thumbs: VecDeque::new(),
            native_ppp,
            theme_applied: false,
        }
    }

    fn is_busy(&self) -> bool {
        !matches!(self.busy, Busy::Idle)
    }

    fn is_starting(&self) -> bool {
        matches!(self.busy, Busy::Startup { .. })
    }

    fn library(&self) -> &Library {
        self.library
            .as_ref()
            .expect("library used before startup finished")
    }

    fn library_mut(&mut self) -> &mut Library {
        self.library
            .as_mut()
            .expect("library used before startup finished")
    }

    fn ui_scale(&self) -> f32 {
        self.library
            .as_ref()
            .map(|l| l.settings.ui_scale)
            .unwrap_or(UI_SCALE_DEFAULT)
    }

    fn effective_ppp(&self) -> f32 {
        self.native_ppp * self.ui_scale()
    }

    fn apply_theme(&mut self, ctx: &egui::Context) {
        if self.theme_applied {
            return;
        }
        let mut style = (*ctx.style()).clone();
        style.override_font_id = Some(FontId::new(16.0, FontFamily::Monospace));
        style.visuals.dark_mode = true;
        style.visuals.panel_fill = BG;
        style.visuals.window_fill = PANEL;
        style.visuals.extreme_bg_color = PANEL_2;
        style.visuals.faint_bg_color = PANEL_2;
        style.visuals.widgets.noninteractive.bg_fill = PANEL;
        style.visuals.widgets.inactive.bg_fill = PANEL_2;
        style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(28, 36, 44);
        style.visuals.widgets.active.bg_fill = Color32::from_rgb(32, 42, 52);
        style.visuals.selection.bg_fill = Color32::from_rgba_unmultiplied(61, 220, 255, 60);
        style.visuals.widgets.noninteractive.fg_stroke.color = TEXT;
        style.visuals.widgets.inactive.fg_stroke.color = TEXT;
        style.visuals.widgets.hovered.fg_stroke.color = ACCENT;
        style.visuals.widgets.active.fg_stroke.color = ACCENT;
        style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER);
        style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, BORDER);
        style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, ACCENT);
        style.spacing.item_spacing = Vec2::new(10.0, 8.0);
        style.spacing.slider_width = 180.0;
        style.spacing.interact_size = Vec2::new(40.0, 22.0);
        ctx.set_style(style);
        // Prevent bare Cmd+/- from scaling the whole UI (photo zoom uses those).
        ctx.options_mut(|o| {
            o.zoom_with_keyboard = false;
        });
        self.theme_applied = true;
    }

    fn apply_ui_scale(&self, ctx: &egui::Context) {
        let target = self.effective_ppp();
        if (ctx.pixels_per_point() - target).abs() > 0.001 {
            ctx.set_pixels_per_point(target);
        }
        // Drop accidental zoom deltas from trackpad/cmd so they don't fight us.
        ctx.input_mut(|i| {
            if i.modifiers.command || i.modifiers.mac_cmd {
                i.smooth_scroll_delta.y = 0.0;
            }
            let _ = i.zoom_delta();
            let _ = i.zoom_delta_2d();
        });
    }

    fn bump_ui_scale(&mut self, factor: f32) {
        let next = (self.ui_scale() * factor).clamp(UI_SCALE_MIN, UI_SCALE_MAX);
        let _ = self.library_mut().set_ui_scale(next);
        self.status = format!("ui scale {:.0}%", next * 100.0);
    }

    fn reset_ui_scale(&mut self) {
        let _ = self.library_mut().set_ui_scale(UI_SCALE_DEFAULT);
        self.status = format!("ui scale {:.0}%", UI_SCALE_DEFAULT * 100.0);
    }

    /// Cmd+Shift +/-/0 — UI chrome scale (works on both screens).
    fn handle_ui_scale_keys(&mut self, ctx: &egui::Context) {
        let cmd = ctx.input(|i| i.modifiers.mac_cmd || i.modifiers.command);
        let shift = ctx.input(|i| i.modifiers.shift);
        if !(cmd && shift) {
            return;
        }
        if ctx.input(|i| i.key_pressed(Key::Equals) || i.key_pressed(Key::Plus)) {
            self.bump_ui_scale(UI_SCALE_STEP);
        }
        if ctx.input(|i| i.key_pressed(Key::Minus)) {
            self.bump_ui_scale(1.0 / UI_SCALE_STEP);
        }
        if ctx.input(|i| i.key_pressed(Key::Num0)) {
            self.reset_ui_scale();
        }
    }

    fn card_count(&self) -> usize {
        self.library
            .as_ref()
            .map(|l| l.index.cards.len())
            .unwrap_or(0)
    }

    fn open_edit(&mut self, ctx: &egui::Context, card: usize) {
        self.flush_save();
        let Some(c) = self.library().index.cards.get(card).cloned() else {
            return;
        };
        self.text = self.library().load_text(&c.stem);
        self.zoom = c.zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        self.edit_texture = None;
        self.edit_rgba = None;
        self.binary_texture = None;
        self.segment_bin_cache = None;
        // Cache boxes, but always refresh reading-order / row assignment.
        // Rematch by geometry — normalize can reorder words so a saved index
        // would land on a different box (looks like a random teleport).
        let cached = self.library().load_words(&c.stem);
        let untangle = self.library().segment_params().untangle_overlap;
        let iw = c.img_w as f32;
        let ih = c.img_h as f32;
        let loaded = Caret::from(&c.caret).sanitize(&cached, iw, ih);
        let old_center = loaded.center(&cached);
        self.page = normalize_reading_order(cached, untangle);
        self.caret = loaded.rematch_after_page_change(old_center, &self.page, iw, ih);
        self.screen = Screen::Edit { card };
        self.dirty = false;
        self.delete_armed = None;
        self.pending_resegment = None;
        self.status = "opening…".into();
        self.busy = Busy::Opening {
            card,
            stem: c.stem.clone(),
            phase: OpenPhase::NeedImage,
            force_full: false,
        };
        ctx.request_repaint();
    }

    /// Cmd+R — drop binary + words caches and rebuild the map from the photo.
    fn hard_reload_map(&mut self, ctx: &egui::Context) {
        if self.is_busy() {
            return;
        }
        match self.screen {
            Screen::Edit { card } => {
                let Some(c) = self.library().index.cards.get(card).cloned() else {
                    return;
                };
                self.segment_bin_cache = None;
                // Keep caret as Free at the old ink so rematch after rebuild
                // doesn't snap to (0,0) from an emptied words list.
                let (cx, cy) = self.caret.center(&self.page);
                self.caret = Caret::Free { x: cx, y: cy };
                self.page = PageWords::default();
                self.pending_resegment = None;
                self.status = "hard reload…".into();
                let phase = if self.edit_rgba.is_some() {
                    OpenPhase::NeedTextures
                } else {
                    OpenPhase::NeedImage
                };
                self.busy = Busy::Opening {
                    card,
                    stem: c.stem,
                    phase,
                    force_full: true,
                };
                ctx.request_repaint();
            }
            Screen::List => {
                let Some(c) = self.library().index.cards.get(self.selected).cloned() else {
                    self.status = "nothing to reload".into();
                    return;
                };
                self.status = format!("hard reload {}…", c.display_name);
                let page = self.library().resegment_card(&c);
                self.status = format!(
                    "reloaded {} · {}w / {}r",
                    c.display_name,
                    page.words.len(),
                    page.rows.len()
                );
            }
        }
    }

    fn upload_edit_textures(
        &mut self,
        ctx: &egui::Context,
        stem: &str,
        rgba: &RgbaImage,
        binary: &GrayImage,
    ) {
        let key = format!("edit-{stem}");
        let size = [rgba.width() as usize, rgba.height() as usize];
        let pixels: Vec<Color32> = rgba
            .pixels()
            .map(|p| Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]))
            .collect();
        let handle = ctx.load_texture(
            key.clone(),
            egui::ColorImage { size, pixels },
            Default::default(),
        );
        self.edit_texture = Some((key, handle, rgba.width(), rgba.height()));

        // Tech-style binary: ink = cyan-ish light, paper = dark.
        let bsize = [binary.width() as usize, binary.height() as usize];
        let bpixels: Vec<Color32> = binary
            .pixels()
            .map(|p| {
                if p[0] == 0 {
                    Color32::from_rgb(180, 230, 255)
                } else {
                    Color32::from_rgb(18, 22, 26)
                }
            })
            .collect();
        self.binary_texture = Some(ctx.load_texture(
            format!("bin-{stem}"),
            egui::ColorImage {
                size: bsize,
                pixels: bpixels,
            },
            Default::default(),
        ));
    }

    /// Swap `page` after rebuild/reorder; rematch caret by geometry, not raw index.
    fn set_page_keeping_caret(&mut self, page: PageWords) {
        let old_center = self.caret.center(&self.page);
        self.page = page;
        let (iw, ih) = self.img_size();
        self.caret = self
            .caret
            .rematch_after_page_change(old_center, &self.page, iw, ih);
    }

    fn apply_segment_output(&mut self, ctx: &egui::Context, stem: &str) {
        let Some(src) = self.edit_rgba.clone() else {
            return;
        };
        let params = self.library().segment_params().clone().clamped();

        let reuse = self.segment_bin_cache.as_ref().is_some_and(|c| {
            !params.affects_binary(&crate::segment::SegmentParams {
                threshold: c.threshold,
                close_radius_x: c.close_radius_x,
                rotation_deg: c.rotation_deg,
                ..params.clone()
            })
        });

        let out = if reuse {
            let c = self.segment_bin_cache.as_ref().unwrap();
            crate::segment::segment_from_binary(&c.bin0, &c.bin, c.rotated.clone(), &params)
        } else {
            let (rotated, bin0, bin) = crate::segment::prepare_binary(&src, &params);
            self.segment_bin_cache = Some(SegmentBinaryCache {
                threshold: params.threshold,
                close_radius_x: params.close_radius_x,
                rotation_deg: params.rotation_deg,
                bin0: bin0.clone(),
                bin: bin.clone(),
                rotated: rotated.clone(),
            });
            crate::segment::segment_from_binary(&bin0, &bin, rotated, &params)
        };

        let _ = self.library().write_words_cache(stem, &out.page);
        self.set_page_keeping_caret(out.page);
        self.upload_edit_textures(ctx, stem, &out.rgba, &out.binary);
    }

    /// Use cached word boxes; only rebuild rotated/binary textures.
    fn apply_preview_textures(&mut self, ctx: &egui::Context, stem: &str) {
        let Some(src) = self.edit_rgba.clone() else {
            return;
        };
        let params = self.library().segment_params().clone().clamped();
        let (rotated, bin0, bin) = crate::segment::prepare_binary(&src, &params);
        self.segment_bin_cache = Some(SegmentBinaryCache {
            threshold: params.threshold,
            close_radius_x: params.close_radius_x,
            rotation_deg: params.rotation_deg,
            bin0,
            bin: bin.clone(),
            rotated: rotated.clone(),
        });
        self.upload_edit_textures(ctx, stem, &rotated, &bin);
    }

    fn back_to_list(&mut self) {
        self.flush_save();
        self.edit_texture = None;
        self.edit_rgba = None;
        self.binary_texture = None;
        self.segment_bin_cache = None;
        self.pending_resegment = None;
        self.slider_held = false;
        self.screen = Screen::List;
    }

    fn flush_save(&mut self) {
        if let Screen::Edit { card } = self.screen {
            if self.library.is_none() {
                return;
            }
            let zoom = self.zoom;
            let caret = self.caret;
            let text = self.text.clone();
            if let Some(c) = self.library().index.cards.get(card).cloned() {
                let _ = self.library().save_text(&c.stem, &text);
                let _ = self.library_mut().update_card_view(card, zoom, caret);
            }
            self.dirty = false;
            self.status = "сохранено".into();
        }
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
        self.last_edit = Instant::now();
        if self.status == "сохранено" {
            self.status.clear();
        }
    }

    fn maybe_autosave(&mut self) {
        if self.dirty && self.last_edit.elapsed() >= Duration::from_secs(AUTOSAVE_SECS) {
            self.flush_save();
        }
    }

    fn schedule_resegment(&mut self) {
        self.pending_resegment = Some(Instant::now());
    }

    fn maybe_resegment(&mut self, ctx: &egui::Context) {
        if self.is_busy() {
            return;
        }
        let Some(started) = self.pending_resegment else {
            return;
        };
        // Wait out slider drag so we don't resegment every pointer move.
        if self.slider_held {
            ctx.request_repaint_after(Duration::from_millis(50));
            return;
        }
        if started.elapsed() < RESEGMENT_DEBOUNCE {
            ctx.request_repaint_after(RESEGMENT_DEBOUNCE.saturating_sub(started.elapsed()));
            return;
        }
        self.pending_resegment = None;
        let Screen::Edit { card } = self.screen else {
            return;
        };
        let Some(c) = self.library().index.cards.get(card).cloned() else {
            return;
        };
        if self.edit_rgba.is_some() {
            self.apply_segment_output(ctx, &c.stem);
        } else {
            let page = self.library().resegment_card(&c);
            self.set_page_keeping_caret(page);
        }
        self.status = format!(
            "map {}w / {}r",
            self.page.words.len(),
            self.page.rows.len()
        );
    }

    fn add_photos_dialog(&mut self) {
        let files = rfd::FileDialog::new()
            .set_title("Add photos (multi-select)")
            .add_filter("Images", &["jpg", "jpeg", "png", "tif", "tiff", "webp"])
            .pick_files();
        if let Some(paths) = files {
            self.ingest_paths(paths);
        }
    }

    fn collect_image_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
        let mut images = Vec::new();
        for path in paths {
            if path.is_dir() {
                if let Ok(rd) = std::fs::read_dir(&path) {
                    let mut kids: Vec<PathBuf> = rd
                        .flatten()
                        .map(|e| e.path())
                        .filter(|p| Library::is_image_path(p))
                        .collect();
                    kids.sort();
                    images.extend(kids);
                }
            } else if Library::is_image_path(&path) {
                images.push(path);
            }
        }
        let mut seen = std::collections::HashSet::new();
        images.retain(|p| seen.insert(p.to_string_lossy().into_owned()));
        images
    }

    /// Import many images (and one-level folders), one file per frame so the UI stays alive.
    fn ingest_paths(&mut self, paths: Vec<PathBuf>) {
        let images = Self::collect_image_paths(paths);
        if images.is_empty() {
            self.status = "no images to add".into();
            return;
        }

        match std::mem::replace(&mut self.busy, Busy::Idle) {
            Busy::Import {
                mut queue,
                mut total,
                done,
                ok,
                err,
            } => {
                total += images.len();
                queue.extend(images);
                self.status = format!("adding {done}/{total}…");
                self.busy = Busy::Import {
                    queue,
                    total,
                    done,
                    ok,
                    err,
                };
            }
            Busy::Opening {
                card,
                stem,
                phase,
                force_full,
            } => {
                self.busy = Busy::Opening {
                    card,
                    stem,
                    phase,
                    force_full,
                };
                self.pending_ingest.extend(images);
                self.status = "queued after open…".into();
            }
            Busy::Idle => {
                let total = images.len();
                self.status = format!("adding 0/{total}…");
                self.busy = Busy::Import {
                    queue: images.into(),
                    total,
                    done: 0,
                    ok: 0,
                    err: 0,
                };
            }
            Busy::Startup { stage } => {
                // Finish boot first; queue photos for after.
                self.busy = Busy::Startup { stage };
                self.pending_ingest.extend(images);
                self.status = "queued after start…".into();
            }
        }
    }

    fn finish_import_status(ok: usize, err: usize) -> String {
        match (ok, err) {
            (1, 0) => "added 1 photo".into(),
            (n, 0) if n > 1 => format!("added {n} photos"),
            (n, e) if n > 0 => format!("added {n}, failed {e}"),
            (0, e) => format!("failed to add ({e} errors)"),
            _ => "ok".into(),
        }
    }

    fn enqueue_thumb(&mut self, idx: usize) {
        if self.pending_thumbs.contains(&idx) {
            return;
        }
        if let Some(lib) = self.library.as_ref() {
            if let Some(card) = lib.index.cards.get(idx) {
                let key = format!("thumb-{}", card.stem);
                if self.thumbs.textures.contains_key(&key) {
                    return;
                }
            }
        }
        self.pending_thumbs.push_back(idx);
    }

    fn pump_pending_thumb(&mut self, ctx: &egui::Context) -> bool {
        let Some(idx) = self.pending_thumbs.pop_front() else {
            return false;
        };
        let (thumb_key, rgba) = {
            let Some(lib) = self.library.as_ref() else {
                return false;
            };
            let Some(card) = lib.index.cards.get(idx).cloned() else {
                return !self.pending_thumbs.is_empty();
            };
            let thumb_key = format!("thumb-{}", card.stem);
            if self.thumbs.textures.contains_key(&thumb_key) {
                return !self.pending_thumbs.is_empty();
            }
            // Prefer disk menu JPEG; fall back to one-time full decode + write.
            (
                thumb_key,
                lib.load_menu_thumbnail(&card.stem)
                    .or_else(|| lib.ensure_menu_thumbnail(&card)),
            )
        };
        if let Some(rgba) = rgba {
            let _ = self.thumbs.get_or_load(ctx, &thumb_key, &rgba);
        }
        !self.pending_thumbs.is_empty()
    }

    /// Advance at most one heavy step per frame.
    fn pump_busy(&mut self, ctx: &egui::Context) {
        let job = std::mem::replace(&mut self.busy, Busy::Idle);
        match job {
            Busy::Idle => {
                if !self.pending_ingest.is_empty() {
                    let paths = std::mem::take(&mut self.pending_ingest);
                    self.ingest_paths(paths);
                } else if self.pump_pending_thumb(ctx) {
                    ctx.request_repaint();
                }
            }
            Busy::Startup { stage } => match stage {
                StartupStage::OpenLibrary => {
                    let library = Library::open_default().unwrap_or_else(|e| {
                        eprintln!("failed to open library: {e}");
                        let fallback = PathBuf::from(".").join("sendnudes-data");
                        Library::open(fallback).expect("fallback library")
                    });
                    let total = library.index.cards.len();
                    self.library = Some(library);
                    if total == 0 {
                        self.status.clear();
                        self.busy = Busy::Idle;
                    } else {
                        self.status = format!("loading previews 0/{total}…");
                        self.busy = Busy::Startup {
                            stage: StartupStage::WarmThumbs { next: 0, total },
                        };
                    }
                    ctx.request_repaint();
                }
                StartupStage::WarmThumbs { next, total } => {
                    if next >= total {
                        self.status = format!("ready · {total} photos");
                        self.busy = Busy::Idle;
                        ctx.request_repaint();
                    } else {
                        // Only open small menu JPEGs here — never full scans.
                        let end = (next + STARTUP_THUMBS_PER_FRAME).min(total);
                        let mut missing = Vec::new();
                        for idx in next..end {
                            let Some(card) = self.library().index.cards.get(idx).cloned() else {
                                continue;
                            };
                            let key = format!("thumb-{}", card.stem);
                            if self.thumbs.textures.contains_key(&key) {
                                continue;
                            }
                            if let Some(rgba) = self.library().load_menu_thumbnail(&card.stem) {
                                let _ = self.thumbs.get_or_load(ctx, &key, &rgba);
                            } else {
                                // Old cards without a menu thumb: build later in background.
                                missing.push(idx);
                            }
                        }
                        for idx in missing {
                            self.enqueue_thumb(idx);
                        }
                        if end >= total {
                            self.status = format!("ready · {total} photos");
                            self.busy = Busy::Idle;
                        } else {
                            self.status = format!("loading previews {end}/{total}…");
                            self.busy = Busy::Startup {
                                stage: StartupStage::WarmThumbs {
                                    next: end,
                                    total,
                                },
                            };
                        }
                        ctx.request_repaint();
                    }
                }
            },
            Busy::Import {
                mut queue,
                total,
                mut done,
                mut ok,
                mut err,
            } => {
                if let Some(path) = queue.pop_front() {
                    match self.library_mut().add_photo(&path) {
                        Ok(idx) => {
                            ok += 1;
                            self.selected = idx;
                            // Thumb JPEG was written during add — just upload to GPU.
                            self.enqueue_thumb(idx);
                        }
                        Err(_) => err += 1,
                    }
                    done += 1;
                    if queue.is_empty() {
                        self.status = Self::finish_import_status(ok, err);
                        self.busy = Busy::Idle;
                    } else {
                        self.status = format!("adding {done}/{total}…");
                        self.busy = Busy::Import {
                            queue,
                            total,
                            done,
                            ok,
                            err,
                        };
                    }
                    ctx.request_repaint();
                }
            }
            Busy::Opening {
                card,
                stem,
                phase,
                force_full,
            } => match phase {
                OpenPhase::NeedImage => {
                    if let Some(c) = self.library().index.cards.get(card).cloned() {
                        self.edit_rgba = self.library().load_rgba(&c);
                    }
                    self.status = if force_full {
                        "hard reload…".into()
                    } else {
                        "building map…".into()
                    };
                    self.busy = Busy::Opening {
                        card,
                        stem,
                        phase: OpenPhase::NeedTextures,
                        force_full,
                    };
                    ctx.request_repaint();
                }
                OpenPhase::NeedTextures => {
                    if self.edit_rgba.is_some() {
                        if force_full || self.page.words.is_empty() {
                            self.segment_bin_cache = None;
                            self.apply_segment_output(ctx, &stem);
                        } else {
                            let untangle = self.library().segment_params().untangle_overlap;
                            let old = std::mem::take(&mut self.page);
                            let old_center = self.caret.center(&old);
                            let page = normalize_reading_order(old, untangle);
                            let _ = self.library().write_words_cache(&stem, &page);
                            self.page = page;
                            let (iw, ih) = self.img_size();
                            self.caret = self.caret.rematch_after_page_change(
                                old_center,
                                &self.page,
                                iw,
                                ih,
                            );
                            self.apply_preview_textures(ctx, &stem);
                        }
                    } else {
                        let (iw, ih) = self.img_size();
                        self.caret = self.caret.sanitize(&self.page, iw, ih);
                    }
                    self.status = if force_full {
                        format!(
                            "reloaded {}w / {}r",
                            self.page.words.len(),
                            self.page.rows.len()
                        )
                    } else {
                        format!(
                            "map {}w / {}r",
                            self.page.words.len(),
                            self.page.rows.len()
                        )
                    };
                    self.busy = Busy::Idle;
                    if !self.pending_ingest.is_empty() {
                        let paths = std::mem::take(&mut self.pending_ingest);
                        self.ingest_paths(paths);
                    }
                    ctx.request_repaint();
                }
            },
        }
    }

    fn busy_progress(&self) -> Option<(String, f32)> {
        match &self.busy {
            Busy::Idle => None,
            Busy::Startup { stage } => match stage {
                StartupStage::OpenLibrary => Some(("starting…".into(), 0.08)),
                StartupStage::WarmThumbs { next, total } => {
                    let frac = if *total == 0 {
                        1.0
                    } else {
                        (0.12 + 0.88 * (*next as f32 / *total as f32)).clamp(0.0, 1.0)
                    };
                    Some((format!("loading previews {next}/{total}"), frac))
                }
            },
            Busy::Import { total, done, .. } => {
                let frac = if *total == 0 {
                    1.0
                } else {
                    (*done as f32 / *total as f32).clamp(0.0, 1.0)
                };
                Some((format!("adding {done}/{total}"), frac))
            }
            Busy::Opening { phase, .. } => {
                let (label, frac) = match phase {
                    OpenPhase::NeedImage => ("opening photo…", 0.3),
                    OpenPhase::NeedTextures => ("building map…", 0.7),
                };
                Some((label.into(), frac))
            }
        }
    }

    fn paint_startup_screen(&self, ui: &mut egui::Ui) {
        let (label, frac) = self.busy_progress().unwrap_or_else(|| ("starting…".into(), 0.0));
        ui.centered_and_justified(|ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() * 0.28);
                ui.label(
                    RichText::new("SendNudes")
                        .font(self.mono(36.0))
                        .color(TEXT),
                );
                ui.add_space(18.0);
                ui.label(
                    RichText::new(label)
                        .font(self.mono(15.0))
                        .color(MUTED),
                );
                ui.add_space(16.0);
                let bar_w = ui.available_width().min(360.0);
                let (rect, _) =
                    ui.allocate_exact_size(Vec2::new(bar_w, 10.0), Sense::hover());
                let painter = ui.painter();
                painter.rect_filled(rect, 4.0, PANEL_2);
                let fill_w = (bar_w * frac).clamp(0.0, bar_w);
                if fill_w > 0.5 {
                    painter.rect_filled(
                        Rect::from_min_size(rect.min, Vec2::new(fill_w, rect.height())),
                        4.0,
                        ACCENT,
                    );
                }
                ui.add_space(10.0);
                ui.label(
                    RichText::new(format!("{:.0}%", frac * 100.0))
                        .font(self.mono(12.0))
                        .color(MUTED),
                );
            });
        });
    }

    fn paint_busy_hud(&self, ctx: &egui::Context) {
        let Some((label, frac)) = self.busy_progress() else {
            return;
        };
        let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("busy_hud"));
        let painter = ctx.layer_painter(layer);
        let screen = ctx.screen_rect();

        let pad = 14.0;
        let bar_w = 220.0;
        let bar_h = 8.0;
        let panel_w = bar_w + 24.0;
        let panel_h = 52.0;
        let panel = Rect::from_min_size(
            Pos2::new(
                screen.max.x - panel_w - pad,
                screen.max.y - panel_h - pad - STATUS_H,
            ),
            Vec2::new(panel_w, panel_h),
        );
        painter.rect_filled(
            panel,
            6.0,
            Color32::from_rgba_unmultiplied(14, 17, 20, 230),
        );
        painter.rect_stroke(panel, 6.0, egui::Stroke::new(1.0, BORDER));

        let text_pos = Pos2::new(panel.min.x + 12.0, panel.min.y + 10.0);
        painter.text(
            text_pos,
            egui::Align2::LEFT_TOP,
            label,
            FontId::new(13.0, FontFamily::Monospace),
            TEXT,
        );

        let bar = Rect::from_min_size(
            Pos2::new(panel.min.x + 12.0, panel.max.y - 12.0 - bar_h),
            Vec2::new(bar_w, bar_h),
        );
        painter.rect_filled(bar, 3.0, PANEL_2);
        let fill_w = (bar_w * frac).clamp(0.0, bar_w);
        if fill_w > 0.5 {
            painter.rect_filled(
                Rect::from_min_size(bar.min, Vec2::new(fill_w, bar_h)),
                3.0,
                ACCENT,
            );
        }
    }

    /// Accept Finder / desktop drops of images or folders anywhere in the window.
    fn handle_file_drops(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if dropped.is_empty() {
            return;
        }
        self.delete_armed = None;
        // Cancel in-flight open so drops always land on the grid cleanly.
        if matches!(self.busy, Busy::Opening { .. }) {
            self.busy = Busy::Idle;
        }
        if !matches!(self.screen, Screen::List) {
            self.flush_save();
            self.screen = Screen::List;
            self.edit_texture = None;
            self.edit_rgba = None;
            self.binary_texture = None;
        }
        self.ingest_paths(dropped);
    }

    fn paint_drop_overlay(&self, ctx: &egui::Context) {
        let hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());
        if !hovering {
            return;
        }
        let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("drop_overlay"));
        let painter = ctx.layer_painter(layer);
        let rect = ctx.screen_rect();
        painter.rect_filled(
            rect,
            0.0,
            Color32::from_rgba_unmultiplied(8, 10, 12, 180),
        );
        painter.rect_stroke(
            rect.shrink(12.0),
            8.0,
            egui::Stroke::new(2.0, ACCENT),
        );
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "drop photos to add",
            self.mono(22.0),
            ACCENT,
        );
    }

    fn img_size(&self) -> (f32, f32) {
        if let Some((_, _, w, h)) = &self.edit_texture {
            (*w as f32, *h as f32)
        } else if let Screen::Edit { card } = self.screen {
            if let Some(c) = self.library().index.cards.get(card) {
                return (c.img_w as f32, c.img_h as f32);
            }
            (1.0, 1.0)
        } else {
            (1.0, 1.0)
        }
    }

    fn open_texts_folder(&mut self) {
        let dir = &self.library().texts_dir;
        let _ = std::fs::create_dir_all(dir);
        match std::process::Command::new("open").arg(dir).spawn() {
            Ok(_) => self.status = format!("opened {}", dir.display()),
            Err(e) => self.status = format!("open folder failed: {e}"),
        }
    }

    fn handle_list_keys(&mut self, ctx: &egui::Context) {
        let cmd = ctx.input(|i| i.modifiers.mac_cmd || i.modifiers.command);
        let shift = ctx.input(|i| i.modifiers.shift);
        if cmd && !shift && ctx.input(|i| i.key_pressed(Key::R)) {
            self.delete_armed = None;
            self.hard_reload_map(ctx);
            return;
        }

        let count = self.card_count();
        if ctx.input(|i| i.key_pressed(Key::N)) {
            self.delete_armed = None;
            self.add_photos_dialog();
            return;
        }
        if ctx.input(|i| i.key_pressed(Key::F)) {
            self.delete_armed = None;
            self.open_texts_folder();
            return;
        }
        if count == 0 {
            return;
        }
        self.selected = self.selected.min(count - 1);
        let cols = grid_cols(ctx.screen_rect().width());

        if ctx.input(|i| i.key_pressed(Key::ArrowRight)) {
            self.delete_armed = None;
            self.selected = (self.selected + 1).min(count - 1);
        }
        if ctx.input(|i| i.key_pressed(Key::ArrowLeft)) {
            self.delete_armed = None;
            self.selected = self.selected.saturating_sub(1);
        }
        if ctx.input(|i| i.key_pressed(Key::ArrowDown)) {
            self.delete_armed = None;
            self.selected = (self.selected + cols).min(count - 1);
        }
        if ctx.input(|i| i.key_pressed(Key::ArrowUp)) {
            self.delete_armed = None;
            self.selected = self.selected.saturating_sub(cols);
        }
        if ctx.input(|i| i.key_pressed(Key::Enter)) {
            self.delete_armed = None;
            self.pending_open = Some(self.selected);
        }

        if ctx.input(|i| i.key_pressed(Key::D)) {
            if self.delete_armed == Some(self.selected) {
                let sel = self.selected;
                let (photo_name, stem) = self
                    .library()
                    .index
                    .cards
                    .get(sel)
                    .map(|c| (c.photo_name.clone(), c.stem.clone()))
                    .unwrap_or_default();
                if let Some(name) = photo_name {
                    self.thumbs.invalidate(&name);
                }
                self.thumbs.invalidate(&format!("thumb-{stem}"));
                let _ = self.library_mut().delete_card(sel);
                self.delete_armed = None;
                let n = self.card_count();
                self.selected = if n == 0 {
                    0
                } else {
                    sel.min(n - 1)
                };
                self.status = "deleted".into();
            } else {
                self.delete_armed = Some(self.selected);
                self.status = "press D again to delete card".into();
            }
            return;
        }

        let clears = ctx.input(|i| {
            i.events.iter().any(|e| match e {
                egui::Event::Text(_) => true,
                egui::Event::Key {
                    pressed: true,
                    key,
                    ..
                } => {
                    *key != Key::D
                        && *key != Key::ArrowLeft
                        && *key != Key::ArrowRight
                        && *key != Key::ArrowUp
                        && *key != Key::ArrowDown
                        && *key != Key::N
                        && *key != Key::Enter
                }
                _ => false,
            })
        });
        if clears {
            self.delete_armed = None;
        }
    }

    fn handle_edit_keys(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.key_pressed(Key::Escape)) {
            self.back_to_list();
            return;
        }

        let (img_w, img_h) = self.img_size();
        let alt = ctx.input(|i| i.modifiers.alt);
        let cmd = ctx.input(|i| i.modifiers.mac_cmd || i.modifiers.command);
        let shift = ctx.input(|i| i.modifiers.shift);

        // Hard reload map: drop caches and re-segment.
        if cmd && !shift && ctx.input(|i| i.key_pressed(Key::R)) {
            self.hard_reload_map(ctx);
            return;
        }

        // Photo zoom: Cmd+/- without Shift (Shift is UI scale).
        if cmd && !shift && ctx.input(|i| i.key_pressed(Key::Equals) || i.key_pressed(Key::Plus))
        {
            self.zoom = (self.zoom * ZOOM_STEP).min(MAX_ZOOM);
            self.mark_dirty();
        }
        if cmd && !shift && ctx.input(|i| i.key_pressed(Key::Minus)) {
            self.zoom = (self.zoom / ZOOM_STEP).max(MIN_ZOOM);
            self.mark_dirty();
        }

        let left = ctx.input(|i| i.key_pressed(Key::ArrowLeft));
        let right = ctx.input(|i| i.key_pressed(Key::ArrowRight));
        let up = ctx.input(|i| i.key_pressed(Key::ArrowUp));
        let down = ctx.input(|i| i.key_pressed(Key::ArrowDown));
        let any_arrow = left || right || up || down;

        // Right-Shift+arrows: hop the photo caret by word / line.
        // Left-Shift+arrows stay with TextEdit (selection); plain arrows move the text caret.
        if right_shift_down() && !cmd && !alt && any_arrow {
            if left {
                self.caret = move_by_word(self.caret, &self.page, CaretMove::Left);
            }
            if right {
                self.caret = move_by_word(self.caret, &self.page, CaretMove::Right);
            }
            if up {
                self.caret = move_by_word(self.caret, &self.page, CaretMove::Up);
            }
            if down {
                self.caret = move_by_word(self.caret, &self.page, CaretMove::Down);
            }
            self.mark_dirty();
            consume_arrows(ctx);
            return;
        }

        // ⌘+arrows: fine pan on the photo.
        if cmd && !shift && !alt && any_arrow {
            let mut dx = 0.0;
            let mut dy = 0.0;
            if left {
                dx -= FINE_NUDGE;
            }
            if right {
                dx += FINE_NUDGE;
            }
            if up {
                dy -= FINE_NUDGE;
            }
            if down {
                dy += FINE_NUDGE;
            }
            self.caret = nudge_free(self.caret, &self.page, dx, dy, img_w, img_h);
            self.mark_dirty();
            consume_arrows(ctx);
        }

        let space = ctx.input(|i| {
            i.events.iter().any(|e| {
                matches!(
                    e,
                    egui::Event::Key {
                        key: Key::Space,
                        pressed: true,
                        ..
                    }
                )
            })
        });
        if space {
            self.caret = advance_on_space(self.caret, &self.page);
            self.mark_dirty();
        }
    }

    fn mono(&self, size: f32) -> FontId {
        FontId::new(size, FontFamily::Monospace)
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, PANEL);
        painter.hline(
            rect.x_range(),
            rect.min.y,
            egui::Stroke::new(1.0, BORDER),
        );

        let left = match self.screen {
            Screen::List => format!(
                "N add · F texts · drop · arrows · Enter · D delete · ⌘R reload · ⌘⇧± ui {:.0}%",
                self.ui_scale() * 100.0
            ),
            Screen::Edit { card } => {
                let name = self
                    .library
                    .as_ref()
                    .and_then(|l| l.index.cards.get(card))
                    .map(|c| c.display_name.as_str())
                    .unwrap_or("?");
                let addr = match self.caret {
                    Caret::Word(i) => self
                        .page
                        .line_col(i)
                        .map(|(l, c)| format!("L{l}.C{c} #{i}"))
                        .unwrap_or_else(|| format!("#{i}")),
                    Caret::Free { .. } => "free".into(),
                };
                format!(
                    "Esc · ⌘R · r⇧←→↑↓ words · arrows text · space · ⌘nudge · ⌘±={:.2} · map={:.1} · ui={:.0}% · {addr} · {name}",
                    self.zoom,
                    self.map_zoom,
                    self.ui_scale() * 100.0
                )
            }
        };
        let right = if self.dirty {
            "unsaved".to_string()
        } else if !self.status.is_empty() {
            self.status.clone()
        } else {
            "ok".to_string()
        };
        let right_color = if self.dirty { WARN } else { ACCENT_2 };

        painter.text(
            Pos2::new(rect.min.x + 12.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            left,
            self.mono(13.0),
            MUTED,
        );
        painter.text(
            Pos2::new(rect.max.x - 12.0, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            right,
            self.mono(13.0),
            right_color,
        );
    }

    fn ui_list(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let count = self.card_count();
        if count == 0 {
            ui.add_space(80.0);
            ui.vertical_centered(|ui| {
                ui.label(
                    RichText::new("no scans")
                        .font(self.mono(22.0))
                        .color(MUTED),
                );
                ui.label(
                    RichText::new("press N or drop several photos / a folder")
                        .font(self.mono(15.0))
                        .color(MUTED),
                );
            });
            return;
        }

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                // Fit the row exactly to content width (after scrollbar), then
                // split leftover equally so left/right padding match.
                const GAP: f32 = 12.0;
                let avail = ui.available_width();
                let cols = grid_cols(avail);
                let gaps = GAP * cols.saturating_sub(1) as f32;
                let card_w = ((avail - gaps) / cols as f32).floor().max(140.0);
                let card_h = card_w + 88.0;
                let row_w = card_w * cols as f32 + gaps;
                let side = ((avail - row_w) * 0.5).max(0.0);

                ui.horizontal(|ui| {
                    ui.add_space(side);
                    ui.vertical(|ui| {
                        ui.set_max_width(row_w);
                        egui::Grid::new("photo_grid")
                            .num_columns(cols)
                            .spacing([GAP, GAP])
                            .show(ui, |ui| {
                                for idx in 0..count {
                                    self.draw_card(ctx, ui, idx, card_w, card_h);
                                    if (idx + 1) % cols == 0 {
                                        ui.end_row();
                                    }
                                }
                            });
                    });
                    ui.add_space(side);
                });
            });
    }

    fn draw_card(
        &mut self,
        _ctx: &egui::Context,
        ui: &mut egui::Ui,
        idx: usize,
        card_w: f32,
        card_h: f32,
    ) {
        let card = self.library().index.cards[idx].clone();
        let selected = self.selected == idx;
        let armed = self.delete_armed == Some(idx);
        let border = if armed {
            DANGER
        } else if selected {
            ACCENT
        } else {
            CARD_OUTLINE
        };

        let (rect, response) =
            ui.allocate_exact_size(Vec2::new(card_w, card_h), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let fill = if response.hovered() || selected {
            PANEL_2
        } else {
            PANEL
        };
        // Fill full rect; stroke inset so painter_at clip doesn't eat the outline.
        painter.rect_filled(rect, 3.0, fill);
        painter.rect_stroke(
            rect.shrink(1.0),
            3.0,
            egui::Stroke::new(2.0, border),
        );

        let pad = 10.0;
        // Photo ~58% of card; meta (title + text) fills the rest and scales with UI.
        let thumb_h = ((card_h - pad * 3.0) * 0.58).max(72.0);
        let thumb_rect = Rect::from_min_size(
            rect.min + Vec2::new(pad, pad),
            Vec2::new(card_w - pad * 2.0, thumb_h),
        );
        painter.rect_filled(thumb_rect, 2.0, Color32::from_rgb(10, 12, 14));

        if card.photo_name.is_some() {
            let thumb_key = format!("thumb-{}", card.stem);
            let tex = self.thumbs.textures.get(&thumb_key).cloned();
            if let Some(tex) = tex {
                let size = tex.size_vec2();
                let fitted = fit_inside(thumb_rect.shrink(2.0), size);
                painter.image(
                    tex.id(),
                    fitted,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            } else {
                // Never decode many full-res photos in one frame — queue instead.
                self.enqueue_thumb(idx);
                painter.text(
                    thumb_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "…",
                    self.mono(14.0),
                    MUTED,
                );
            }
        } else {
            painter.text(
                thumb_rect.center(),
                egui::Align2::CENTER_CENTER,
                "no photo",
                self.mono(14.0),
                MUTED,
            );
        }

        let meta = Rect::from_min_max(
            Pos2::new(rect.min.x + pad, thumb_rect.max.y + 8.0),
            Pos2::new(rect.max.x - pad, rect.max.y - pad),
        );
        let title = format!("{:02}  {}", idx + 1, card.display_name);
        let raw = self.library().load_text(&card.stem);
        let preview = collapse_ws(&raw);
        let title_font = self.mono(13.0);
        let body_font = self.mono(11.0);
        let wrap_w = meta.width().max(1.0);

        let title_galley =
            ui.fonts(|f| f.layout(title, title_font, TEXT, wrap_w));
        let title_h = title_galley.size().y;
        painter
            .with_clip_rect(meta)
            .galley(meta.min, title_galley, TEXT);

        let mut body_top = meta.min.y + title_h + 4.0;
        if armed {
            let warn = ui.fonts(|f| {
                f.layout(
                    "D again: delete card".to_owned(),
                    self.mono(12.0),
                    DANGER,
                    wrap_w,
                )
            });
            let wh = warn.size().y;
            if body_top + wh <= meta.max.y {
                painter.with_clip_rect(meta).galley(
                    Pos2::new(meta.min.x, body_top),
                    warn,
                    DANGER,
                );
                body_top += wh + 2.0;
            }
        }

        let body_rect = Rect::from_min_max(Pos2::new(meta.min.x, body_top), meta.max);
        if body_rect.height() > 2.0 {
            if preview.is_empty() {
                let empty = ui.fonts(|f| {
                    f.layout("empty".to_owned(), body_font, DANGER, wrap_w)
                });
                painter
                    .with_clip_rect(body_rect)
                    .galley(body_rect.min, empty, DANGER);
            } else {
                // Only whole lines — never clip mid-glyph. Elide with …
                let line_h = ui.fonts(|f| f.row_height(&body_font)).max(1.0);
                let max_rows = (body_rect.height() / line_h).floor().max(0.0) as usize;
                if max_rows > 0 {
                    let mut job = egui::text::LayoutJob::single_section(
                        preview,
                        egui::text::TextFormat {
                            font_id: body_font,
                            color: MUTED,
                            ..Default::default()
                        },
                    );
                    job.wrap = egui::text::TextWrapping {
                        max_width: wrap_w,
                        max_rows,
                        break_anywhere: false,
                        overflow_character: Some('…'),
                    };
                    let body = ui.fonts(|f| f.layout_job(job));
                    painter.galley(body_rect.min, body, MUTED);
                }
            }
        }

        if response.clicked() {
            self.selected = idx;
            self.delete_armed = None;
        }
        if response.double_clicked() {
            self.pending_open = Some(idx);
        }
    }

    fn ui_edit(&mut self, ui: &mut egui::Ui) {
        let Screen::Edit { .. } = self.screen else {
            return;
        };

        let avail = ui.available_size();
        let left_w = (avail.x * 0.46).max(260.0);
        let right_w = (avail.x - left_w - 10.0).max(220.0);
        let body_h = avail.y.max(0.0);

        ui.allocate_ui_with_layout(
            Vec2::new(avail.x, body_h),
            egui::Layout::left_to_right(egui::Align::Min),
            |ui| {
                ui.allocate_ui_with_layout(
                    Vec2::new(left_w, body_h),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        let slider_h = 148.0;
                        let gap = 4.0;
                        let rest = (ui.available_height() - slider_h - gap * 2.0).max(120.0);
                        let photo_h = rest * 0.52;
                        let map_h = (rest - photo_h).max(60.0);

                        ui.allocate_ui_with_layout(
                            Vec2::new(left_w, photo_h),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                frame_panel(ui, |ui| self.draw_photo_viewport(ui));
                            },
                        );
                        ui.add_space(gap);
                        ui.allocate_ui_with_layout(
                            Vec2::new(left_w, map_h),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                frame_panel(ui, |ui| self.draw_word_map(ui));
                            },
                        );
                        ui.add_space(gap);
                        let rem = ui.available_height().max(0.0);
                        ui.allocate_ui_with_layout(
                            Vec2::new(left_w, rem),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                egui::ScrollArea::vertical()
                                    .id_salt("segment_params_scroll")
                                    .auto_shrink([false, false])
                                    .show(ui, |ui| self.draw_param_sliders(ui));
                            },
                        );
                    },
                );

                ui.add_space(8.0);

                ui.allocate_ui_with_layout(
                    Vec2::new(right_w, body_h),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        let panel = ui.available_size();
                        ui.set_min_size(panel);
                        ui.set_max_height(panel.y);
                        frame_panel(ui, |ui| {
                            let before = self.text.clone();
                            egui::ScrollArea::vertical()
                                .id_salt("transcript_scroll")
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    let edit = egui::TextEdit::multiline(&mut self.text)
                                        .font(FontId::new(17.0, FontFamily::Monospace))
                                        .desired_width(ui.available_width())
                                        .desired_rows(8)
                                        .frame(false)
                                        .text_color(TEXT);
                                    let response = ui.add(edit);
                                    if response.changed() || self.text != before {
                                        self.mark_dirty();
                                    }
                                });
                        });
                    },
                );
            },
        );
    }

    fn draw_param_sliders(&mut self, ui: &mut egui::Ui) {
        ui.spacing_mut().item_spacing.y = 6.0;
        ui.label(
            RichText::new("segment params (global)")
                .font(self.mono(12.0))
                .color(MUTED),
        );

        let mut params = self.library().settings.segment.clone();
        let mut changed = false;
        let mut held = false;

        let (c, h) = slider_u8(ui, self, "threshold", &mut params.threshold, 60.0..=220.0, "ink / shadow");
        changed |= c;
        held |= h;
        let (c, h) = slider_u32(ui, self, "merge_x  ", &mut params.merge_gap_x, 0.0..=24.0, "glue letters");
        changed |= c;
        held |= h;
        let (c, h) = slider_u32(ui, self, "merge_y  ", &mut params.merge_gap_y, 0.0..=12.0, "bridge rows");
        changed |= c;
        held |= h;
        let (c, h) = slider_f32(ui, self, "line_sep ", &mut params.line_sep, 0.50..=0.98, "split stuck lines");
        changed |= c;
        held |= h;
        let (c, h) = slider_u32(ui, self, "min_w    ", &mut params.min_word_w, 1.0..=30.0, "drop thin noise");
        changed |= c;
        held |= h;
        let (c, h) = slider_u32(ui, self, "min_h    ", &mut params.min_word_h, 1.0..=40.0, "drop short boxes");
        changed |= c;
        held |= h;
        let (c, h) = slider_u32(ui, self, "min_ink  ", &mut params.min_ink, 1.0..=200.0, "min ink in box");
        changed |= c;
        held |= h;
        let (c, h) = slider_u32(ui, self, "close_x  ", &mut params.close_radius_x, 0.0..=8.0, "binary close");
        changed |= c;
        held |= h;
        let (c, h) = slider_u32(ui, self, "band_gap ", &mut params.band_gap, 0.0..=8.0, "gap between lines");
        changed |= c;
        held |= h;
        let (c, h) = slider_f32(
            ui,
            self,
            "untangle ",
            &mut params.untangle_overlap,
            0.0..=0.8,
            "clip overlap (0=off)",
        );
        changed |= c;
        held |= h;

        ui.horizontal(|ui| {
            ui.label(RichText::new("rotate ° ").font(self.mono(12.0)).color(MUTED));
            let resp = ui.add(
                egui::Slider::new(&mut params.rotation_deg, -10.0..=10.0)
                    .show_value(true)
                    .step_by(0.5),
            );
            if resp.changed() {
                changed = true;
            }
            held |= resp.dragged() || resp.is_pointer_button_down_on();
            ui.label(
                RichText::new("manual tilt ±10°")
                    .font(self.mono(10.0))
                    .color(MUTED),
            );
            if ui
                .add(egui::Button::new(
                    RichText::new("reset").font(self.mono(12.0)),
                ))
                .clicked()
            {
                params = crate::segment::SegmentParams::default();
                changed = true;
                self.status = "params reset".into();
            }
        });

        self.slider_held = held;
        if changed {
            let _ = self.library_mut().set_segment_params(params);
            self.schedule_resegment();
        }
    }

    fn draw_photo_viewport(&mut self, ui: &mut egui::Ui) {
        let (img_w, img_h) = self.img_size();
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_rgb(6, 8, 10));

        let Some((_, tex, _, _)) = &self.edit_texture else {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "no photo — text still available",
                self.mono(15.0),
                MUTED,
            );
            return;
        };

        let (cx, cy) = self.caret.center(&self.page);
        let view_w = rect.width() / self.zoom;
        let view_h = rect.height() / self.zoom;
        let mut left = cx - view_w / 2.0;
        let mut top = cy - view_h / 2.0;
        left = left.clamp(0.0, (img_w - view_w).max(0.0));
        top = top.clamp(0.0, (img_h - view_h).max(0.0));
        if view_w >= img_w {
            left = (img_w - view_w) / 2.0;
        }
        if view_h >= img_h {
            top = (img_h - view_h) / 2.0;
        }

        let uv = Rect::from_min_max(
            Pos2::new(left / img_w, top / img_h),
            Pos2::new((left + view_w) / img_w, (top + view_h) / img_h),
        );
        painter.image(tex.id(), rect, uv, Color32::WHITE);

        if let Caret::Word(i) = self.caret {
            if let Some(w) = self.page.words.get(i) {
                let x0 = rect.min.x + (w.x as f32 - left) * self.zoom;
                let y0 = rect.min.y + (w.y as f32 - top) * self.zoom;
                let ww = w.w as f32 * self.zoom;
                let hh = w.h as f32 * self.zoom;
                let wr = Rect::from_min_size(Pos2::new(x0, y0), Vec2::new(ww, hh));
                painter.rect_stroke(wr, 0.0, egui::Stroke::new(1.5, ACCENT));
            }
        }

        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let px = left + (pos.x - rect.min.x) / self.zoom;
                let py = top + (pos.y - rect.min.y) / self.zoom;
                self.caret = click_at(&self.page, px, py);
                self.mark_dirty();
            }
        }
    }

    fn draw_word_map(&mut self, ui: &mut egui::Ui) {
        let (img_w, img_h) = self.img_size();
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click());

        if response.hovered() {
            let dy = ui.input(|i| i.smooth_scroll_delta.y);
            if dy.abs() > 0.5 {
                if dy > 0.0 {
                    self.map_zoom = (self.map_zoom * MAP_ZOOM_STEP).min(MAP_ZOOM_MAX);
                } else {
                    self.map_zoom = (self.map_zoom / MAP_ZOOM_STEP).max(MAP_ZOOM_MIN);
                }
            }
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_rgb(10, 12, 14));

        if img_w <= 1.0 || img_h <= 1.0 {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "map unavailable",
                self.mono(14.0),
                MUTED,
            );
            return;
        }

        let fitted = rect.shrink(4.0);
        painter.rect_filled(fitted, 0.0, Color32::from_rgb(18, 22, 26));

        // Independent map zoom: scale relative to fit-to-panel, pan on caret (or image center if free).
        let fit_scale = (fitted.width() / img_w).min(fitted.height() / img_h);
        let scale = fit_scale * self.map_zoom;
        let draw_w = img_w * scale;
        let draw_h = img_h * scale;

        let (focus_x, focus_y) = match self.caret {
            Caret::Word(_) => self.caret.center(&self.page),
            Caret::Free { .. } => (img_w * 0.5, img_h * 0.5),
        };

        let mut origin_x = fitted.center().x - focus_x * scale;
        let mut origin_y = fitted.center().y - focus_y * scale;
        if draw_w <= fitted.width() {
            origin_x = fitted.center().x - draw_w * 0.5;
        } else {
            origin_x = origin_x.clamp(fitted.max.x - draw_w, fitted.min.x);
        }
        if draw_h <= fitted.height() {
            origin_y = fitted.center().y - draw_h * 0.5;
        } else {
            origin_y = origin_y.clamp(fitted.max.y - draw_h, fitted.min.y);
        }

        let img_rect = Rect::from_min_size(Pos2::new(origin_x, origin_y), Vec2::new(draw_w, draw_h));
        let clip = painter.with_clip_rect(fitted);

        if let Some(bin_tex) = &self.binary_texture {
            clip.image(
                bin_tex.id(),
                img_rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }

        let current = self.caret.word_index_or_none();
        for (i, w) in self.page.words.iter().enumerate() {
            let wr = Rect::from_min_size(
                Pos2::new(
                    origin_x + w.x as f32 * scale,
                    origin_y + w.y as f32 * scale,
                ),
                Vec2::new(w.w as f32 * scale, w.h as f32 * scale),
            );
            let fill = if Some(i) == current {
                Color32::from_rgba_unmultiplied(255, 200, 60, 160)
            } else if i % 2 == 0 {
                Color32::from_rgba_unmultiplied(61, 220, 255, 70)
            } else {
                Color32::from_rgba_unmultiplied(46, 230, 166, 70)
            };
            clip.rect_filled(wr, 0.0, fill);
            clip.rect_stroke(
                wr,
                0.0,
                egui::Stroke::new(
                    1.0,
                    if Some(i) == current {
                        WARN
                    } else {
                        Color32::from_rgba_unmultiplied(255, 255, 255, 40)
                    },
                ),
            );
        }

        // Photo viewport outline in image coords (uses photo zoom, not map_zoom).
        let (cx, cy) = self.caret.center(&self.page);
        let photo_view_w = img_w / self.zoom;
        let photo_view_h = img_h / self.zoom;
        let mut left = cx - photo_view_w / 2.0;
        let mut top = cy - photo_view_h / 2.0;
        left = left.clamp(0.0, (img_w - photo_view_w).max(0.0));
        top = top.clamp(0.0, (img_h - photo_view_h).max(0.0));
        if photo_view_w >= img_w {
            left = (img_w - photo_view_w) / 2.0;
        }
        if photo_view_h >= img_h {
            top = (img_h - photo_view_h) / 2.0;
        }
        let vr = Rect::from_min_size(
            Pos2::new(origin_x + left * scale, origin_y + top * scale),
            Vec2::new(photo_view_w * scale, photo_view_h * scale),
        );
        clip.rect_stroke(vr, 0.0, egui::Stroke::new(1.0, ACCENT));

        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                if fitted.contains(pos) {
                    let px = (pos.x - origin_x) / scale;
                    let py = (pos.y - origin_y) / scale;
                    if px >= 0.0 && py >= 0.0 && px <= img_w && py <= img_h {
                        self.caret = click_at(&self.page, px, py);
                        self.mark_dirty();
                    }
                }
            }
        }

        let chain = match self.caret {
            Caret::Word(i) => {
                let next = move_by_word(Caret::Word(i), &self.page, CaretMove::Right);
                let next_i = match next {
                    Caret::Word(j) if j != i => format!("→#{j}"),
                    _ => "→|".into(),
                };
                self.page
                    .line_col(i)
                    .map(|(l, c)| format!("  L{l}.C{c} {next_i}"))
                    .unwrap_or_else(|| format!("  #{i} {next_i}"))
            }
            Caret::Free { .. } => String::new(),
        };
        painter.text(
            Pos2::new(rect.min.x + 6.0, rect.min.y + 4.0),
            egui::Align2::LEFT_TOP,
            format!(
                "map ×{:.1}  {}w / {}r{chain}",
                self.map_zoom,
                self.page.words.len(),
                self.page.rows.len()
            ),
            self.mono(12.0),
            MUTED,
        );
    }
}

impl Caret {
    fn word_index_or_none(self) -> Option<usize> {
        match self {
            Caret::Word(i) => Some(i),
            Caret::Free { .. } => None,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.apply_theme(ctx);
        self.handle_ui_scale_keys(ctx);
        self.apply_ui_scale(ctx);

        // Heavy import / open steps from the previous frame (HUD already shown).
        self.pump_busy(ctx);

        if !self.is_starting() {
            self.handle_file_drops(ctx);
            self.paint_drop_overlay(ctx);
        }

        egui::TopBottomPanel::bottom("status")
            .exact_height(STATUS_H)
            .frame(egui::Frame::none().fill(PANEL))
            .show_separator_line(false)
            .show(ctx, |ui| {
                self.status_bar(ui);
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(BG).inner_margin(12.0))
            .show(ctx, |ui| {
                if self.is_starting() {
                    self.paint_startup_screen(ui);
                    return;
                }
                match self.screen {
                    Screen::List => {
                        if !self.is_busy() {
                            self.handle_list_keys(ctx);
                        }
                        self.ui_list(ctx, ui);
                    }
                    Screen::Edit { .. } => {
                        if self.is_busy() {
                            // Esc cancels a stuck open and returns to the grid.
                            if ctx.input(|i| i.key_pressed(Key::Escape)) {
                                self.busy = Busy::Idle;
                                self.back_to_list();
                            }
                        } else {
                            self.handle_edit_keys(ctx);
                        }
                        if matches!(self.screen, Screen::Edit { .. }) {
                            self.ui_edit(ui);
                        }
                    }
                }
            });

        if let Some(idx) = self.pending_open.take() {
            if !self.is_busy() {
                self.open_edit(ctx, idx);
            } else {
                self.pending_open = Some(idx);
            }
        }

        self.maybe_resegment(ctx);
        self.maybe_autosave();
        // Corner HUD for import/open; startup uses the full-screen bar instead.
        if !self.is_starting() {
            self.paint_busy_hud(ctx);
        }

        if self.is_busy() || !self.pending_thumbs.is_empty() {
            ctx.request_repaint();
        } else if self.dirty || self.pending_resegment.is_some() {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.flush_save();
        if let Some(lib) = self.library.as_ref() {
            let _ = lib.save_settings();
        }
    }
}

fn frame_panel(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::none()
        .fill(PANEL)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .inner_margin(6.0)
        .show(ui, |ui| {
            add_contents(ui);
        });
}

fn grid_cols(width: f32) -> usize {
    ((width / 240.0).floor() as usize).clamp(2, 5)
}

fn fit_inside(outer: Rect, size: Vec2) -> Rect {
    let scale = (outer.width() / size.x).min(outer.height() / size.y);
    let w = size.x * scale;
    let h = size.y * scale;
    let min = Pos2::new(
        outer.min.x + (outer.width() - w) / 2.0,
        outer.min.y + (outer.height() - h) / 2.0,
    );
    Rect::from_min_size(min, Vec2::new(w, h))
}

fn collapse_ws(raw: &str) -> String {
    raw.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn slider_u8(
    ui: &mut egui::Ui,
    app: &App,
    name: &str,
    v: &mut u8,
    range: std::ops::RangeInclusive<f32>,
    tip: &str,
) -> (bool, bool) {
    let mut changed = false;
    let mut held = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new(name).font(app.mono(12.0)).color(MUTED));
        let mut t = *v as f32;
        let resp = ui.add(egui::Slider::new(&mut t, range).show_value(true));
        if resp.changed() {
            *v = t.round() as u8;
            changed = true;
        }
        held = resp.dragged() || resp.is_pointer_button_down_on();
        ui.label(RichText::new(tip).font(app.mono(10.0)).color(MUTED));
    });
    (changed, held)
}

fn slider_u32(
    ui: &mut egui::Ui,
    app: &App,
    name: &str,
    v: &mut u32,
    range: std::ops::RangeInclusive<f32>,
    tip: &str,
) -> (bool, bool) {
    let mut changed = false;
    let mut held = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new(name).font(app.mono(12.0)).color(MUTED));
        let mut t = *v as f32;
        let resp = ui.add(egui::Slider::new(&mut t, range).show_value(true));
        if resp.changed() {
            *v = t.round() as u32;
            changed = true;
        }
        held = resp.dragged() || resp.is_pointer_button_down_on();
        ui.label(RichText::new(tip).font(app.mono(10.0)).color(MUTED));
    });
    (changed, held)
}

fn slider_f32(
    ui: &mut egui::Ui,
    app: &App,
    name: &str,
    v: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    tip: &str,
) -> (bool, bool) {
    let mut changed = false;
    let mut held = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new(name).font(app.mono(12.0)).color(MUTED));
        let resp = ui.add(egui::Slider::new(v, range).show_value(true));
        if resp.changed() {
            changed = true;
        }
        held = resp.dragged() || resp.is_pointer_button_down_on();
        ui.label(RichText::new(tip).font(app.mono(10.0)).color(MUTED));
    });
    (changed, held)
}

fn consume_arrows(ctx: &egui::Context) {
    ctx.input_mut(|i| {
        i.events.retain(|e| {
            !matches!(
                e,
                egui::Event::Key {
                    key: Key::ArrowLeft
                        | Key::ArrowRight
                        | Key::ArrowUp
                        | Key::ArrowDown,
                    ..
                }
            )
        });
    });
}

/// egui only exposes a combined Shift; on macOS read the device-side bit.
#[cfg(target_os = "macos")]
fn right_shift_down() -> bool {
    // IOKit / HIToolbox: NX_DEVICERSHIFTKEYMASK
    const NX_DEVICERSHIFTKEYMASK: u64 = 0x0000_0004;
    const KCG_EVENT_SOURCE_STATE_COMBINED_SESSION: u32 = 0;
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGEventSourceFlagsState(state_id: u32) -> u64;
    }
    unsafe {
        CGEventSourceFlagsState(KCG_EVENT_SOURCE_STATE_COMBINED_SESSION) & NX_DEVICERSHIFTKEYMASK
            != 0
    }
}

#[cfg(not(target_os = "macos"))]
fn right_shift_down() -> bool {
    false
}
