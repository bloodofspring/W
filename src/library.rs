//! On-disk library: photo copies, text files, index, word-cache.

use crate::caret::Caret;
use crate::segment::{segment_full, PageWords, SegmentOutput, SegmentParams};
use directories::ProjectDirs;
use image::imageops::FilterType;
use image::{ImageBuffer, ImageFormat, Rgb, RgbImage, RgbaImage};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "tif", "tiff", "webp"];
/// Longest side of menu / grid preview JPEGs written under `cache/thumbs/`.
pub const MENU_THUMB_MAX: u32 = 256;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardState {
    /// Filename inside photos/ (empty if photo was deleted).
    pub photo_name: Option<String>,
    /// Stem used for texts/{stem}.txt and caches.
    pub stem: String,
    /// Original filename shown in UI.
    pub display_name: String,
    pub zoom: f32,
    /// Serialized caret.
    pub caret: StoredCaret,
    pub img_w: u32,
    pub img_h: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum StoredCaret {
    Word { index: usize },
    Free { x: f32, y: f32 },
}

impl From<Caret> for StoredCaret {
    fn from(c: Caret) -> Self {
        match c {
            Caret::Word(index) => StoredCaret::Word { index },
            Caret::Free { x, y } => StoredCaret::Free { x, y },
        }
    }
}

impl From<&StoredCaret> for Caret {
    fn from(c: &StoredCaret) -> Self {
        match *c {
            StoredCaret::Word { index } => Caret::Word(index),
            StoredCaret::Free { x, y } => Caret::Free { x, y },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexFile {
    pub cards: Vec<CardState>,
}

fn default_ui_scale() -> f32 {
    1.35
}

/// Global app settings. Old `settings.json` with only SegmentParams fields still loads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    #[serde(flatten)]
    pub segment: SegmentParams,
    /// Multiplier over native pixels_per_point (UI chrome scale).
    #[serde(default = "default_ui_scale")]
    pub ui_scale: f32,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            segment: SegmentParams::default(),
            ui_scale: default_ui_scale(),
        }
    }
}

impl AppSettings {
    pub fn clamped(mut self) -> Self {
        self.segment = self.segment.clamped();
        self.ui_scale = self.ui_scale.clamp(0.85, 2.5);
        self
    }
}

pub struct Library {
    pub photos_dir: PathBuf,
    pub texts_dir: PathBuf,
    pub cache_dir: PathBuf,
    /// Small JPEG previews for the grid (`{stem}.jpg`), separate from full scans.
    pub thumbs_dir: PathBuf,
    pub index_path: PathBuf,
    pub settings_path: PathBuf,
    pub index: IndexFile,
    pub settings: AppSettings,
}

impl Library {
    pub fn open_default() -> io::Result<Self> {
        let dirs = ProjectDirs::from("com", "SendNudes", "SendNudes").ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "cannot resolve Application Support")
        })?;
        Self::open(dirs.data_dir().to_path_buf())
    }

    pub fn open(root: PathBuf) -> io::Result<Self> {
        let photos_dir = root.join("photos");
        let texts_dir = root.join("texts");
        let cache_dir = root.join("cache");
        let thumbs_dir = cache_dir.join("thumbs");
        fs::create_dir_all(&photos_dir)?;
        fs::create_dir_all(&texts_dir)?;
        fs::create_dir_all(&cache_dir)?;
        fs::create_dir_all(&thumbs_dir)?;
        let index_path = root.join("index.json");
        let settings_path = root.join("settings.json");
        let index = if index_path.exists() {
            let raw = fs::read_to_string(&index_path)?;
            serde_json::from_str(&raw).unwrap_or_default()
        } else {
            IndexFile::default()
        };
        let settings = if settings_path.exists() {
            let raw = fs::read_to_string(&settings_path)?;
            serde_json::from_str::<AppSettings>(&raw)
                .unwrap_or_default()
                .clamped()
        } else {
            AppSettings::default()
        };
        Ok(Self {
            photos_dir,
            texts_dir,
            cache_dir,
            thumbs_dir,
            index_path,
            settings_path,
            index,
            settings,
        })
    }

    pub fn save_index(&self) -> io::Result<()> {
        let raw = serde_json::to_string_pretty(&self.index)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(&self.index_path, raw)
    }

    pub fn save_settings(&self) -> io::Result<()> {
        let raw = serde_json::to_string_pretty(&self.settings.clone().clamped())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(&self.settings_path, raw)
    }

    pub fn set_segment_params(&mut self, params: SegmentParams) -> io::Result<()> {
        self.settings.segment = params.clamped();
        self.save_settings()
    }

    pub fn set_ui_scale(&mut self, scale: f32) -> io::Result<()> {
        self.settings.ui_scale = scale.clamp(0.85, 2.5);
        self.save_settings()
    }

    pub fn segment_params(&self) -> &SegmentParams {
        &self.settings.segment
    }

    pub fn text_path(&self, stem: &str) -> PathBuf {
        self.texts_dir.join(format!("{stem}.txt"))
    }

    pub fn words_cache_path(&self, stem: &str) -> PathBuf {
        self.cache_dir.join(format!("{stem}.words.json"))
    }

    pub fn photo_path(&self, name: &str) -> PathBuf {
        self.photos_dir.join(name)
    }

    pub fn load_text(&self, stem: &str) -> String {
        fs::read_to_string(self.text_path(stem)).unwrap_or_default()
    }

    pub fn save_text(&self, stem: &str, text: &str) -> io::Result<()> {
        fs::write(self.text_path(stem), text)
    }

    pub fn unique_stem(photos_dir: &Path, base: &str) -> String {
        let mut stem = sanitize_stem(base);
        if stem.is_empty() {
            stem = format!("scan-{}", now_millis());
        }
        let mut candidate = stem.clone();
        let mut n = 2u32;
        while photos_dir.join(format!("{candidate}.jpg")).exists()
            || photos_dir.join(format!("{candidate}.png")).exists()
            || photos_dir.join(format!("{candidate}.tif")).exists()
            || photos_dir.join(format!("{candidate}.tiff")).exists()
            || photos_dir.join(format!("{candidate}.webp")).exists()
            || photos_dir.join(format!("{candidate}.jpeg")).exists()
        {
            candidate = format!("{stem}-{n}");
            n += 1;
        }
        candidate
    }

    /// Copy image into library, segment, create empty text if missing.
    pub fn add_photo(&mut self, source: &Path) -> io::Result<usize> {
        let ext = source
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();
        if !IMAGE_EXTS.contains(&ext.as_str()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported image type: {ext}"),
            ));
        }

        let display_name = source
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "photo".into());
        let base = source
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "photo".into());
        let stem = Self::unique_stem(&self.photos_dir, &base);
        let photo_name = format!("{stem}.{ext}");
        let dest = self.photo_path(&photo_name);
        fs::copy(source, &dest)?;

        let img = image::open(&dest)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
            .to_rgba8();
        // Menu preview first — startup must never need to reopen the full scan.
        self.write_menu_thumbnail(&stem, &img)?;
        let out = segment_full(&img, &self.settings.segment);
        self.write_words_cache(&stem, &out.page)?;
        // Store original scan dimensions; rotated size is applied in the editor.
        let (img_w, img_h) = img.dimensions();

        let text_path = self.text_path(&stem);
        if !text_path.exists() {
            fs::write(&text_path, "")?;
        }

        let card = CardState {
            photo_name: Some(photo_name),
            stem,
            display_name,
            zoom: 2.0,
            caret: StoredCaret::Word { index: 0 },
            img_w,
            img_h,
        };
        self.index.cards.push(card);
        self.save_index()?;
        Ok(self.index.cards.len() - 1)
    }

    /// Remove card entirely: photo, text, words cache, and index entry.
    pub fn delete_card(&mut self, card_idx: usize) -> io::Result<()> {
        if card_idx >= self.index.cards.len() {
            return Err(io::Error::new(io::ErrorKind::NotFound, "card missing"));
        }
        let card = self.index.cards.remove(card_idx);
        if let Some(name) = &card.photo_name {
            let _ = fs::remove_file(self.photos_dir.join(name));
        }
        let _ = fs::remove_file(self.text_path(&card.stem));
        let _ = fs::remove_file(self.words_cache_path(&card.stem));
        let _ = fs::remove_file(self.menu_thumb_path(&card.stem));
        // Legacy flat cache path from earlier builds.
        let _ = fs::remove_file(self.cache_dir.join(format!("{}.thumb.jpg", card.stem)));
        self.save_index()
    }

    pub fn write_words_cache(&self, stem: &str, page: &PageWords) -> io::Result<()> {
        let raw = serde_json::to_string_pretty(page)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(self.words_cache_path(stem), raw)
    }

    pub fn load_words(&self, stem: &str) -> PageWords {
        let path = self.words_cache_path(stem);
        if let Ok(raw) = fs::read_to_string(path) {
            if let Ok(page) = serde_json::from_str(&raw) {
                return page;
            }
        }
        PageWords::default()
    }

    /// Re-segment if cache missing but photo exists.
    pub fn ensure_words(&self, card: &CardState) -> PageWords {
        let cached = self.load_words(&card.stem);
        if !cached.words.is_empty() {
            return cached;
        }
        self.resegment_card(card)
    }

    /// Force re-segment with current global settings and refresh cache.
    pub fn resegment_card(&self, card: &CardState) -> PageWords {
        self.resegment_card_full(card)
            .map(|o| o.page)
            .unwrap_or_default()
    }

    pub fn resegment_card_full(&self, card: &CardState) -> Option<SegmentOutput> {
        let name = card.photo_name.as_ref()?;
        let path = self.photo_path(name);
        let rgba = image::open(&path).ok()?.to_rgba8();
        let out = segment_full(&rgba, &self.settings.segment);
        let _ = self.write_words_cache(&card.stem, &out.page);
        Some(out)
    }

    pub fn load_rgba(&self, card: &CardState) -> Option<RgbaImage> {
        let name = card.photo_name.as_ref()?;
        image::open(self.photo_path(name))
            .ok()
            .map(|i| i.to_rgba8())
    }

    pub fn menu_thumb_path(&self, stem: &str) -> PathBuf {
        self.thumbs_dir.join(format!("{stem}.jpg"))
    }

    fn resize_menu_thumb(img: &RgbaImage, max_side: u32) -> RgbaImage {
        let (w, h) = img.dimensions();
        let scale = max_side as f32 / w.max(h) as f32;
        if scale >= 1.0 {
            return img.clone();
        }
        let nw = ((w as f32) * scale).round().max(1.0) as u32;
        let nh = ((h as f32) * scale).round().max(1.0) as u32;
        image::imageops::resize(img, nw, nh, FilterType::Triangle)
    }

    /// Write a small JPEG used only by the photo grid (not the editor).
    pub fn write_menu_thumbnail(&self, stem: &str, rgba: &RgbaImage) -> io::Result<()> {
        fs::create_dir_all(&self.thumbs_dir)?;
        let thumb = Self::resize_menu_thumb(rgba, MENU_THUMB_MAX);
        // JPEG encoder needs RGB (no alpha).
        let rgb: RgbImage = ImageBuffer::from_fn(thumb.width(), thumb.height(), |x, y| {
            let p = thumb.get_pixel(x, y).0;
            Rgb([p[0], p[1], p[2]])
        });
        rgb.save_with_format(self.menu_thumb_path(stem), ImageFormat::Jpeg)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Fast path: read the menu JPEG only. Never opens the full scan.
    pub fn load_menu_thumbnail(&self, stem: &str) -> Option<RgbaImage> {
        let primary = self.menu_thumb_path(stem);
        if let Ok(cached) = image::open(&primary) {
            return Some(cached.to_rgba8());
        }
        // Migrate legacy flat cache into thumbs/.
        let legacy = self.cache_dir.join(format!("{stem}.thumb.jpg"));
        if let Ok(cached) = image::open(&legacy) {
            let rgba = cached.to_rgba8();
            let _ = self.write_menu_thumbnail(stem, &rgba);
            let _ = fs::remove_file(&legacy);
            return Some(rgba);
        }
        None
    }

    /// Build + save menu thumb from the full photo (one-time migration for old cards).
    pub fn ensure_menu_thumbnail(&self, card: &CardState) -> Option<RgbaImage> {
        if let Some(existing) = self.load_menu_thumbnail(&card.stem) {
            return Some(existing);
        }
        let img = self.load_rgba(card)?;
        let _ = self.write_menu_thumbnail(&card.stem, &img);
        self.load_menu_thumbnail(&card.stem)
    }

    pub fn update_card_view(
        &mut self,
        card_idx: usize,
        zoom: f32,
        caret: Caret,
    ) -> io::Result<()> {
        if let Some(card) = self.index.cards.get_mut(card_idx) {
            card.zoom = zoom;
            card.caret = caret.into();
            self.save_index()?;
        }
        Ok(())
    }

    pub fn is_image_path(path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
            .unwrap_or(false)
    }
}

fn sanitize_stem(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else if ch.is_whitespace() {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[allow(dead_code)]
fn guess_format(ext: &str) -> Option<ImageFormat> {
    match ext {
        "jpg" | "jpeg" => Some(ImageFormat::Jpeg),
        "png" => Some(ImageFormat::Png),
        "tif" | "tiff" => Some(ImageFormat::Tiff),
        "webp" => Some(ImageFormat::WebP),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn add_and_delete_card_removes_all() {
        let dir = tempfile::tempdir().unwrap();
        let mut lib = Library::open(dir.path().to_path_buf()).unwrap();

        let src = dir.path().join("in.png");
        let mut img = RgbImage::new(40, 30);
        for p in img.pixels_mut() {
            *p = Rgb([240, 240, 240]);
        }
        for y in 10..18 {
            for x in 5..25 {
                img.put_pixel(x, y, Rgb([10, 10, 10]));
            }
        }
        img.save(&src).unwrap();

        let idx = lib.add_photo(&src).unwrap();
        let stem = lib.index.cards[idx].stem.clone();
        let photo = lib.index.cards[idx].photo_name.clone().unwrap();
        lib.save_text(&stem, "hello").unwrap();
        assert!(lib.text_path(&stem).exists());
        assert!(lib.photo_path(&photo).exists());
        assert!(
            lib.menu_thumb_path(&stem).exists(),
            "add_photo must write a menu thumbnail"
        );
        assert!(lib.load_menu_thumbnail(&stem).is_some());

        lib.delete_card(idx).unwrap();
        assert!(lib.index.cards.is_empty());
        assert!(!lib.text_path(&stem).exists());
        assert!(!lib.photo_path(&photo).exists());
        assert!(!lib.words_cache_path(&stem).exists());
        assert!(!lib.menu_thumb_path(&stem).exists());
    }
}

#[cfg(test)]
mod seed {
    use super::*;

    #[test]
    fn seed_sample_manuscript_into_app_support() {
        if std::env::var("SENDNUDES_SEED").ok().as_deref() != Some("1") {
            return;
        }
        let sample = PathBuf::from("samples/manuscript.jpg");
        assert!(sample.exists(), "missing samples/manuscript.jpg");
        let mut lib = Library::open_default().expect("app support");
        // Avoid duplicates by stem.
        let already = lib.index.cards.iter().any(|c| c.display_name.contains("manuscript") || c.stem.starts_with("manuscript"));
        if already {
            eprintln!("sample already in library ({} cards)", lib.index.cards.len());
            return;
        }
        let idx = lib.add_photo(&sample).expect("add");
        eprintln!("seeded card #{idx}, stem={}", lib.index.cards[idx].stem);
        let words = lib.ensure_words(&lib.index.cards[idx]);
        eprintln!("words={} rows={}", words.words.len(), words.rows.len());
    }
}
