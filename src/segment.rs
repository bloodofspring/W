//! Find word boxes on a manuscript scan via local strip projections.

use image::{GrayImage, ImageBuffer, Luma, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};

/// Axis-aligned word rectangle in (possibly rotated) image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WordRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl WordRect {
    pub fn center(&self) -> (f32, f32) {
        (
            self.x as f32 + self.w as f32 / 2.0,
            self.y as f32 + self.h as f32 / 2.0,
        )
    }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x as f32
            && py >= self.y as f32
            && px < (self.x + self.w) as f32
            && py < (self.y + self.h) as f32
    }
}

/// All words on a page, grouped into reading-order rows.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PageWords {
    pub words: Vec<WordRect>,
    pub rows: Vec<(usize, usize)>,
}

impl PageWords {
    pub fn word_index_at(&self, px: f32, py: f32) -> Option<usize> {
        self.words.iter().position(|w| w.contains(px, py))
    }

    pub fn nearest_word(&self, px: f32, py: f32) -> Option<usize> {
        if self.words.is_empty() {
            return None;
        }
        self.words
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let (ax, ay) = a.center();
                let (bx, by) = b.center();
                let da = (ax - px).hypot(ay - py);
                let db = (bx - px).hypot(by - py);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    pub fn row_of(&self, word_idx: usize) -> Option<usize> {
        self.rows
            .iter()
            .position(|&(s, e)| word_idx >= s && word_idx < e)
    }

    /// Reading-order address: `(line_id, col)` within that line.
    pub fn line_col(&self, word_idx: usize) -> Option<(usize, usize)> {
        let row = self.row_of(word_idx)?;
        let (s, e) = self.rows[row];
        if word_idx >= e {
            return None;
        }
        Some((row, word_idx - s))
    }
}

/// Full segmentation result including binary preview and rotated raster.
pub struct SegmentOutput {
    pub page: PageWords,
    /// After rotate + binarize + close (0 = ink).
    pub binary: GrayImage,
    /// Color image after manual rotation (same size as binary).
    pub rgba: RgbaImage,
}

/// Tunable thresholds for handwriting scans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentParams {
    /// Bias for adaptive binarization (legacy global threshold as center = 155).
    pub threshold: u8,
    /// Floor / multiplier base for word-gap bridging (px).
    pub merge_gap_x: u32,
    #[serde(default)]
    pub merge_gap_y: u32,
    #[serde(default = "default_line_sep")]
    pub line_sep: f32,
    pub min_word_w: u32,
    #[serde(default = "default_min_word_h")]
    pub min_word_h: u32,
    #[serde(default = "default_min_ink")]
    pub min_ink: u32,
    #[serde(default = "default_close_radius_x")]
    pub close_radius_x: u32,
    /// Manual page rotation in degrees (−10…+10).
    #[serde(default)]
    pub rotation_deg: f32,
    /// Pixel gap left between adjacent line bands at midpoints.
    #[serde(default = "default_band_gap")]
    pub band_gap: u32,
    /// Clip cross-row boxes only if vertical overlap / min(h) exceeds this (0 = off).
    #[serde(default = "default_untangle_overlap")]
    pub untangle_overlap: f32,
}

fn default_line_sep() -> f32 {
    0.72
}
fn default_min_word_h() -> u32 {
    4
}
fn default_min_ink() -> u32 {
    12
}
fn default_close_radius_x() -> u32 {
    1
}
fn default_band_gap() -> u32 {
    1
}
fn default_untangle_overlap() -> f32 {
    0.25
}

impl Default for SegmentParams {
    fn default() -> Self {
        Self {
            threshold: 155,
            merge_gap_x: 5,
            merge_gap_y: 0,
            line_sep: 0.72,
            min_word_w: 5,
            min_word_h: 4,
            min_ink: 12,
            close_radius_x: 1,
            rotation_deg: 0.0,
            band_gap: 1,
            untangle_overlap: 0.25,
        }
    }
}

impl SegmentParams {
    pub fn clamped(mut self) -> Self {
        self.threshold = self.threshold.clamp(40, 240);
        self.merge_gap_x = self.merge_gap_x.clamp(0, 40);
        self.merge_gap_y = self.merge_gap_y.clamp(0, 12);
        self.line_sep = self.line_sep.clamp(0.50, 0.98);
        self.min_word_w = self.min_word_w.clamp(1, 40);
        self.min_word_h = self.min_word_h.clamp(1, 40);
        self.min_ink = self.min_ink.clamp(1, 500);
        self.close_radius_x = self.close_radius_x.clamp(0, 8);
        self.rotation_deg = self.rotation_deg.clamp(-10.0, 10.0);
        self.band_gap = self.band_gap.clamp(0, 8);
        self.untangle_overlap = self.untangle_overlap.clamp(0.0, 0.8);
        self
    }

    /// True when Sauvola / close / rotate must be recomputed (not just word geometry).
    pub fn affects_binary(&self, other: &Self) -> bool {
        self.threshold != other.threshold
            || self.close_radius_x != other.close_radius_x
            || (self.rotation_deg - other.rotation_deg).abs() > 0.01
    }
}

const STRIP_COUNT: usize = 8;

pub fn to_gray(rgba: &RgbaImage) -> GrayImage {
    ImageBuffer::from_fn(rgba.width(), rgba.height(), |x, y| {
        let p = rgba.get_pixel(x, y);
        let [r, g, b, _] = p.0;
        let yv = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;
        Luma([yv])
    })
}

pub fn binarize(gray: &GrayImage, threshold: u8) -> GrayImage {
    ImageBuffer::from_fn(gray.width(), gray.height(), |x, y| {
        let v = gray.get_pixel(x, y)[0];
        if v < threshold {
            Luma([0])
        } else {
            Luma([255])
        }
    })
}

/// Rotate RGBA around center. Output canvas is large enough to fit the rotated image;
/// empty pixels filled with paper white.
pub fn rotate_rgba(src: &RgbaImage, deg: f32) -> RgbaImage {
    if deg.abs() < 0.05 {
        return src.clone();
    }
    let (w, h) = src.dimensions();
    let rad = deg.to_radians();
    let (s, c) = (rad.sin(), rad.cos());
    let cx = (w as f32 - 1.0) / 2.0;
    let cy = (h as f32 - 1.0) / 2.0;

    // Bounding box of rotated corners.
    let corners = [(0.0f32, 0.0), (w as f32, 0.0), (w as f32, h as f32), (0.0, h as f32)];
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (x, y) in corners {
        let dx = x - cx;
        let dy = y - cy;
        xs.push(c * dx - s * dy);
        ys.push(s * dx + c * dy);
    }
    let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_y = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_y = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let out_w = (max_x - min_x).ceil().max(1.0) as u32;
    let out_h = (max_y - min_y).ceil().max(1.0) as u32;
    let ox = -min_x;
    let oy = -min_y;

    // Inverse map: for each out pixel, sample source.
    let inv_c = c;
    let inv_s = -s;
    ImageBuffer::from_fn(out_w, out_h, |x, y| {
        let dx = x as f32 - ox;
        let dy = y as f32 - oy;
        let sx = inv_c * dx - inv_s * dy + cx;
        let sy = inv_s * dx + inv_c * dy + cy;
        let ix = sx.round() as i64;
        let iy = sy.round() as i64;
        if ix >= 0 && iy >= 0 && (ix as u32) < w && (iy as u32) < h {
            *src.get_pixel(ix as u32, iy as u32)
        } else {
            Rgba([245, 245, 242, 255])
        }
    })
}

/// Sauvola adaptive threshold. `threshold` slider acts as bias around 155.
pub fn binarize_sauvola(gray: &GrayImage, threshold_bias: u8) -> GrayImage {
    let (w, h) = gray.dimensions();
    let radius = 20u32;
    let k = 0.34f32;
    let r = 128.0f32;
    let bias = (threshold_bias as i32 - 155) as f32 * 0.35;

    // Integral images for mean / variance.
    let mut sum = vec![0u64; ((w + 1) * (h + 1)) as usize];
    let mut sum_sq = vec![0u64; ((w + 1) * (h + 1)) as usize];
    let idx = |x: u32, y: u32| -> usize { (y * (w + 1) + x) as usize };
    for y in 1..=h {
        let mut row = 0u64;
        let mut row_sq = 0u64;
        for x in 1..=w {
            let v = gray.get_pixel(x - 1, y - 1)[0] as u64;
            row += v;
            row_sq += v * v;
            sum[idx(x, y)] = sum[idx(x, y - 1)] + row;
            sum_sq[idx(x, y)] = sum_sq[idx(x, y - 1)] + row_sq;
        }
    }

    let rect_sum = |s: &[u64], x0: u32, y0: u32, x1: u32, y1: u32| -> u64 {
        s[idx(x1, y1)] + s[idx(x0, y0)] - s[idx(x1, y0)] - s[idx(x0, y1)]
    };

    let mut out = ImageBuffer::new(w, h);
    let mut ink = 0u32;
    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(radius);
            let y0 = y.saturating_sub(radius);
            let x1 = (x + radius + 1).min(w);
            let y1 = (y + radius + 1).min(h);
            let n = ((x1 - x0) * (y1 - y0)) as f32;
            let s = rect_sum(&sum, x0, y0, x1, y1) as f32;
            let sq = rect_sum(&sum_sq, x0, y0, x1, y1) as f32;
            let mean = s / n;
            let var = (sq / n - mean * mean).max(0.0);
            let std = var.sqrt();
            let t = mean * (1.0 + k * (std / r - 1.0)) + bias;
            let v = gray.get_pixel(x, y)[0] as f32;
            if v < t {
                out.put_pixel(x, y, Luma([0]));
                ink += 1;
            } else {
                out.put_pixel(x, y, Luma([255]));
            }
        }
    }

    let total = (w * h).max(1);
    let ratio = ink as f32 / total as f32;
    // Fallback if adaptive is pathological.
    if ratio < 0.01 || ratio > 0.55 {
        return binarize(gray, threshold_bias);
    }
    out
}

fn close_horizontal(bin: &GrayImage, rx: u32) -> GrayImage {
    if rx == 0 {
        return bin.clone();
    }
    let (w, h) = bin.dimensions();
    let dilated = ImageBuffer::from_fn(w, h, |x, y| {
        let x0 = x.saturating_sub(rx);
        let x1 = (x + rx).min(w - 1);
        let mut ink = false;
        for xx in x0..=x1 {
            if bin.get_pixel(xx, y)[0] == 0 {
                ink = true;
                break;
            }
        }
        Luma([if ink { 0 } else { 255 }])
    });
    ImageBuffer::from_fn(w, h, |x, y| {
        let x0 = x.saturating_sub(rx);
        let x1 = (x + rx).min(w - 1);
        let mut all = true;
        for xx in x0..=x1 {
            if dilated.get_pixel(xx, y)[0] != 0 {
                all = false;
                break;
            }
        }
        Luma([if all { 0 } else { 255 }])
    })
}

fn ink_at(bin: &GrayImage, x: u32, y: u32) -> bool {
    bin.get_pixel(x, y)[0] == 0
}

fn strip_horizontal_projection(bin: &GrayImage, x0: u32, x1: u32) -> Vec<u32> {
    let h = bin.height();
    let mut proj = vec![0u32; h as usize];
    for y in 0..h {
        let mut c = 0u32;
        for x in x0..x1 {
            if ink_at(bin, x, y) {
                c += 1;
            }
        }
        proj[y as usize] = c;
    }
    proj
}

fn runs_from_projection(proj: &[u32], min_val: u32, max_gap: u32) -> Vec<(usize, usize)> {
    let mut raw: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < proj.len() {
        if proj[i] >= min_val {
            let start = i;
            i += 1;
            while i < proj.len() && proj[i] >= min_val {
                i += 1;
            }
            raw.push((start, i));
        } else {
            i += 1;
        }
    }
    if raw.is_empty() {
        return raw;
    }
    let mut merged = vec![raw[0]];
    for &(s, e) in &raw[1..] {
        let last = merged.last_mut().unwrap();
        let gap = s.saturating_sub(last.1);
        if gap <= max_gap as usize {
            last.1 = e;
        } else {
            merged.push((s, e));
        }
    }
    merged
}

fn smooth_u32(proj: &[u32], radius: usize) -> Vec<f32> {
    let n = proj.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let lo = i.saturating_sub(radius);
        let hi = (i + radius + 1).min(n);
        let slice = &proj[lo..hi];
        let sum: u32 = slice.iter().sum();
        out[i] = sum as f32 / slice.len() as f32;
    }
    out
}

fn line_peaks_in_projection(
    hproj: &[u32],
    row_min: u32,
    bridge_gap: u32,
    line_sep: f32,
    band_gap: u32,
) -> Vec<(usize, usize, usize)> {
    // Returns (peak_y, y0, y1) bands.
    if hproj.is_empty() {
        return Vec::new();
    }
    let sm = smooth_u32(hproj, 2);
    let min_peak = row_min as f32;
    let mut raw_peaks: Vec<usize> = Vec::new();
    for i in 2..sm.len().saturating_sub(2) {
        if sm[i] >= sm[i - 1]
            && sm[i] >= sm[i + 1]
            && sm[i] > min_peak
            && sm[i] >= sm[i - 2]
            && sm[i] >= sm[i + 2]
        {
            if let Some(last) = raw_peaks.last().copied() {
                if i - last <= 4 {
                    if sm[i] > sm[last] {
                        *raw_peaks.last_mut().unwrap() = i;
                    }
                    continue;
                }
            }
            raw_peaks.push(i);
        }
    }

    let mut peaks: Vec<usize> = Vec::new();
    for peak in raw_peaks {
        if let Some(&prev) = peaks.last() {
            let valley = sm[prev..=peak]
                .iter()
                .cloned()
                .fold(f32::MAX, f32::min);
            let sep = line_sep.clamp(0.50, 0.98);
            let barrier = sep * sm[prev].min(sm[peak]);
            if valley > barrier {
                if sm[peak] > sm[prev] {
                    *peaks.last_mut().unwrap() = peak;
                }
                continue;
            }
        }
        peaks.push(peak);
    }

    if peaks.is_empty() {
        return runs_from_projection(hproj, row_min, bridge_gap)
            .into_iter()
            .map(|(a, b)| ((a + b) / 2, a, b))
            .collect();
    }

    // Leave a small gap at midpoints so adjacent bands don't seal shut.
    let gap = band_gap as usize;
    let mut bands = Vec::new();
    for (pi, &peak) in peaks.iter().enumerate() {
        let raw_left = if pi == 0 {
            0
        } else {
            (peaks[pi - 1] + peak) / 2 + gap
        };
        let raw_right = if pi + 1 == peaks.len() {
            hproj.len()
        } else {
            ((peak + peaks[pi + 1]) / 2).saturating_sub(gap)
        };
        let left_bound = raw_left.min(peak);
        let right_bound = raw_right.max(peak + 1).min(hproj.len());
        let mut y0 = peak;
        let mut y1 = peak + 1;
        while y0 > left_bound && hproj[y0 - 1] >= row_min {
            y0 -= 1;
        }
        while y1 < right_bound && hproj[y1] >= row_min {
            y1 += 1;
        }
        y0 = y0.min(peak.saturating_sub(2)).max(left_bound);
        y1 = y1.max(peak + 3).min(right_bound);
        if y1 > y0 {
            bands.push((peak, y0, y1));
        }
    }
    bands
}

#[derive(Clone)]
struct StripPeak {
    peak: usize,
    y0: usize,
    y1: usize,
}

#[derive(Clone)]
struct LinePolyline {
    /// Per strip: optional (peak, y0, y1).
    nodes: Vec<Option<StripPeak>>,
}

fn merge_strip_peaks(
    strips: &[Vec<StripPeak>],
    max_dy: f32,
) -> Vec<LinePolyline> {
    if strips.is_empty() {
        return Vec::new();
    }
    let n = strips.len();
    let mut lines: Vec<LinePolyline> = strips[0]
        .iter()
        .map(|p| {
            let mut nodes = vec![None; n];
            nodes[0] = Some(p.clone());
            LinePolyline { nodes }
        })
        .collect();

    for si in 1..n {
        let mut used = vec![false; strips[si].len()];
        // Match existing lines to nearest peak in this strip.
        let mut assignments: Vec<(usize, usize, f32)> = Vec::new();
        for (li, line) in lines.iter().enumerate() {
            let Some(prev) = line.nodes[..si].iter().rev().find_map(|n| n.as_ref()) else {
                continue;
            };
            for (pi, peak) in strips[si].iter().enumerate() {
                let dy = (peak.peak as f32 - prev.peak as f32).abs();
                if dy <= max_dy {
                    assignments.push((li, pi, dy));
                }
            }
        }
        assignments.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        let mut line_taken = vec![false; lines.len()];
        for (li, pi, _) in assignments {
            if line_taken[li] || used[pi] {
                continue;
            }
            lines[li].nodes[si] = Some(strips[si][pi].clone());
            line_taken[li] = true;
            used[pi] = true;
        }
        for (pi, peak) in strips[si].iter().enumerate() {
            if used[pi] {
                continue;
            }
            let mut nodes = vec![None; n];
            nodes[si] = Some(peak.clone());
            lines.push(LinePolyline { nodes });
        }
    }

    // Sort by mean peak y.
    lines.sort_by(|a, b| {
        let ya = mean_peak(a);
        let yb = mean_peak(b);
        ya.partial_cmp(&yb).unwrap_or(std::cmp::Ordering::Equal)
    });
    lines
}

fn mean_peak(line: &LinePolyline) -> f32 {
    let mut s = 0.0;
    let mut n = 0.0;
    for node in line.nodes.iter().flatten() {
        s += node.peak as f32;
        n += 1.0;
    }
    if n == 0.0 {
        0.0
    } else {
        s / n
    }
}

fn interp_band_at_x(
    line: &LinePolyline,
    strip_bounds: &[(u32, u32)],
    x: u32,
) -> Option<(u32, u32)> {
    // Find strip containing x; if missing node, interpolate from neighbors.
    let si = strip_bounds
        .iter()
        .position(|&(a, b)| x >= a && x < b)?;
    if let Some(node) = &line.nodes[si] {
        return Some((node.y0 as u32, node.y1 as u32));
    }
    // Nearest left/right nodes.
    let left = (0..si).rev().find_map(|i| line.nodes[i].as_ref().map(|n| (i, n)));
    let right = ((si + 1)..line.nodes.len()).find_map(|i| line.nodes[i].as_ref().map(|n| (i, n)));
    match (left, right) {
        (Some((li, ln)), Some((ri, rn))) => {
            let lx = (strip_bounds[li].0 + strip_bounds[li].1) as f32 / 2.0;
            let rx = (strip_bounds[ri].0 + strip_bounds[ri].1) as f32 / 2.0;
            let t = if (rx - lx).abs() < 1.0 {
                0.5
            } else {
                ((x as f32) - lx) / (rx - lx)
            };
            let y0 = (ln.y0 as f32 * (1.0 - t) + rn.y0 as f32 * t).round() as u32;
            let y1 = (ln.y1 as f32 * (1.0 - t) + rn.y1 as f32 * t)
                .round()
                .max(y0 as f32 + 1.0) as u32;
            Some((y0, y1))
        }
        (Some((_, n)), None) | (None, Some((_, n))) => Some((n.y0 as u32, n.y1 as u32)),
        _ => None,
    }
}

fn vertical_projection_along_line(
    bin: &GrayImage,
    line: &LinePolyline,
    strip_bounds: &[(u32, u32)],
) -> Vec<u32> {
    let w = bin.width();
    let h = bin.height();
    let mut proj = vec![0u32; w as usize];
    for x in 0..w {
        if let Some((y0, y1)) = interp_band_at_x(line, strip_bounds, x) {
            let y0 = y0.min(h);
            let y1 = y1.min(h).max(y0);
            for y in y0..y1 {
                if ink_at(bin, x, y) {
                    proj[x as usize] += 1;
                }
            }
        }
    }
    proj
}

fn median_u32(values: &mut [u32]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

fn adaptive_merge_gap(proj: &[u32], col_min: u32, floor: u32) -> u32 {
    // Only letter-sized gaps — ignore larger word separators.
    let letter_max = floor.max(6).saturating_add(3);
    let mut gaps = Vec::new();
    let mut i = 0;
    while i < proj.len() {
        while i < proj.len() && proj[i] < col_min {
            i += 1;
        }
        if i >= proj.len() {
            break;
        }
        while i < proj.len() && proj[i] >= col_min {
            i += 1;
        }
        let gap_start = i;
        while i < proj.len() && proj[i] < col_min {
            i += 1;
        }
        if i < proj.len() {
            let gap = (i - gap_start) as u32;
            if gap > 0 && gap <= letter_max {
                gaps.push(gap);
            }
        }
    }
    if gaps.is_empty() {
        return floor;
    }
    gaps.sort_unstable();
    let q = gaps[gaps.len() / 4];
    (q.saturating_mul(14) / 10)
        .max(floor)
        .min(floor.saturating_mul(2).max(floor + 4))
}

fn tight_bounds(bin: &GrayImage, x0: u32, x1: u32, y0: u32, y1: u32) -> Option<WordRect> {
    let mut min_x = x1;
    let mut max_x = x0;
    let mut min_y = y1;
    let mut max_y = y0;
    let mut ink = 0u32;
    let h = bin.height();
    let w = bin.width();
    for y in y0..y1.min(h) {
        for x in x0..x1.min(w) {
            if ink_at(bin, x, y) {
                ink += 1;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
    }
    if ink == 0 || min_x > max_x {
        return None;
    }
    Some(WordRect {
        x: min_x,
        y: min_y,
        w: max_x - min_x + 1,
        h: max_y - min_y + 1,
    })
}

fn ink_count(bin: &GrayImage, r: &WordRect) -> u32 {
    let mut n = 0u32;
    for y in r.y..r.y.saturating_add(r.h).min(bin.height()) {
        for x in r.x..r.x.saturating_add(r.w).min(bin.width()) {
            if ink_at(bin, x, y) {
                n += 1;
            }
        }
    }
    n
}

fn words_from_line(
    bin0: &GrayImage,
    bin: &GrayImage,
    line: &LinePolyline,
    strip_bounds: &[(u32, u32)],
    params: &SegmentParams,
) -> Vec<WordRect> {
    let proj = vertical_projection_along_line(bin, line, strip_bounds);
    // Estimate line height from nodes.
    let mut heights = Vec::new();
    for node in line.nodes.iter().flatten() {
        heights.push((node.y1.saturating_sub(node.y0)) as u32);
    }
    let line_h = median_u32(&mut heights).max(params.min_word_h);
    let col_min = (line_h / 6).max(1);
    let gap = adaptive_merge_gap(&proj, col_min, params.merge_gap_x);
    let runs = runs_from_projection(&proj, col_min, gap);

    let mut words = Vec::new();
    for (wx0, wx1) in runs {
        let x0 = wx0 as u32;
        let x1 = wx1 as u32;
        if x1 <= x0 {
            continue;
        }
        // Union of bands along the run for tight bounds.
        let mut y0 = u32::MAX;
        let mut y1 = 0u32;
        for x in x0..x1 {
            if let Some((a, b)) = interp_band_at_x(line, strip_bounds, x) {
                y0 = y0.min(a);
                y1 = y1.max(b);
            }
        }
        if y0 == u32::MAX || y1 <= y0 {
            continue;
        }
        if let Some(rect) = tight_bounds(bin0, x0, x1, y0, y1) {
            if rect.w >= params.min_word_w
                && rect.h >= params.min_word_h
                && ink_count(bin0, &rect) >= params.min_ink
            {
                words.extend(split_word_by_valleys(bin0, bin, rect, params));
            }
        }
    }
    words
}

fn global_hproj(bin: &GrayImage) -> Vec<u32> {
    let w = bin.width();
    let mut hproj = vec![0u32; bin.height() as usize];
    for y in 0..bin.height() {
        let mut c = 0u32;
        for x in 0..w {
            if ink_at(bin, x, y) {
                c += 1;
            }
        }
        hproj[y as usize] = c;
    }
    hproj
}

/// Blend user line_sep with page-estimated separator depth from projection valleys.
fn estimate_adaptive_line_sep(hproj: &[u32], row_min: u32, user: f32) -> f32 {
    let sm = smooth_u32(hproj, 2);
    let min_peak = row_min as f32;
    let mut peaks: Vec<usize> = Vec::new();
    for i in 2..sm.len().saturating_sub(2) {
        if sm[i] >= sm[i - 1]
            && sm[i] >= sm[i + 1]
            && sm[i] > min_peak
            && sm[i] >= sm[i - 2]
            && sm[i] >= sm[i + 2]
        {
            if let Some(last) = peaks.last().copied() {
                if i - last <= 4 {
                    if sm[i] > sm[last] {
                        *peaks.last_mut().unwrap() = i;
                    }
                    continue;
                }
            }
            peaks.push(i);
        }
    }
    if peaks.len() < 2 {
        return user.clamp(0.50, 0.98);
    }
    let mut ratios: Vec<f32> = Vec::new();
    for w in peaks.windows(2) {
        let (a, b) = (w[0], w[1]);
        let valley = sm[a..=b].iter().cloned().fold(f32::MAX, f32::min);
        let base = sm[a].min(sm[b]).max(1.0);
        ratios.push((valley / base).clamp(0.0, 1.0));
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Barrier should sit between shallow within-line dips and deep inter-line valleys.
    let est = ratios[ratios.len() * 2 / 3].clamp(0.55, 0.95);
    (user * 0.35 + est * 0.65).clamp(0.50, 0.98)
}

fn union_rect(a: &WordRect, b: &WordRect) -> WordRect {
    let x0 = a.x.min(b.x);
    let y0 = a.y.min(b.y);
    let x1 = (a.x + a.w).max(b.x + b.w);
    let y1 = (a.y + a.h).max(b.y + b.h);
    WordRect {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    }
}

fn rect_area(w: &WordRect) -> u32 {
    w.w.saturating_mul(w.h)
}

fn horizontal_overlap_ratio(a: &WordRect, b: &WordRect) -> f32 {
    let x0 = a.x.max(b.x);
    let x1 = (a.x + a.w).min(b.x + b.w);
    if x1 <= x0 {
        return 0.0;
    }
    let inter = (x1 - x0) as f32;
    let min_w = a.w.min(b.w).max(1) as f32;
    inter / min_w
}

/// Attach ё/й-like floating marks to the nearest larger base glyph below (or barely above).
fn attach_floating_marks(mut comps: Vec<WordRect>, med_h: f32) -> Vec<WordRect> {
    if comps.len() < 2 {
        return comps;
    }
    let mark_h_max = (med_h * 0.45).max(3.0) as u32;
    let mark_area_max = (med_h * med_h * 0.40).max(16.0) as u32;
    let max_gap_above = (med_h * 0.55).max(4.0) as u32;
    let max_gap_below = (med_h * 0.25).max(2.0) as u32;

    let is_mark = |c: &WordRect| c.h <= mark_h_max && rect_area(c) <= mark_area_max;

    let mut guard = 0usize;
    loop {
        guard += 1;
        if guard > comps.len().saturating_mul(4).max(8) {
            break;
        }
        let n = comps.len();
        let mut pair: Option<(usize, usize)> = None;
        'search: for i in 0..n {
            if !is_mark(&comps[i]) {
                continue;
            }
            let mark = comps[i];
            let mut best: Option<(usize, u32)> = None;
            for j in 0..n {
                if i == j || is_mark(&comps[j]) {
                    continue;
                }
                let base = comps[j];
                if rect_area(&base) < rect_area(&mark).saturating_mul(2) {
                    continue;
                }
                if horizontal_overlap_ratio(&mark, &base) < 0.30 {
                    continue;
                }
                // Mark stacked above base (ё / й).
                if mark.y.saturating_add(mark.h) <= base.y {
                    let gap = base.y.saturating_sub(mark.y.saturating_add(mark.h));
                    if gap <= max_gap_above {
                        if best.map(|(_, g)| gap < g).unwrap_or(true) {
                            best = Some((j, gap));
                        }
                    }
                }
                // Tiny mark just under a taller body (rare).
                if base.y.saturating_add(base.h) <= mark.y {
                    let gap = mark.y.saturating_sub(base.y.saturating_add(base.h));
                    if gap <= max_gap_below {
                        if best.map(|(_, g)| gap < g).unwrap_or(true) {
                            best = Some((j, gap));
                        }
                    }
                }
            }
            if let Some((j, _)) = best {
                pair = Some((i, j));
                break 'search;
            }
        }
        let Some((i, j)) = pair else {
            break;
        };
        let merged = union_rect(&comps[i], &comps[j]);
        let (keep, drop) = if i < j { (i, j) } else { (j, i) };
        comps[keep] = merged;
        comps.remove(drop);
    }
    comps
}

/// Letter-sized X gap from neighbouring boxes in a row (same policy as adaptive_merge_gap).
fn letter_gap_from_row_boxes(row: &[WordRect], floor: u32) -> u32 {
    let letter_max = floor.max(6).saturating_add(3);
    let mut sorted: Vec<&WordRect> = row.iter().collect();
    sorted.sort_by_key(|w| w.x);
    let mut gaps = Vec::new();
    for w in sorted.windows(2) {
        let gap = w[1].x.saturating_sub(w[0].x.saturating_add(w[0].w));
        if gap > 0 && gap <= letter_max {
            gaps.push(gap);
        }
    }
    if gaps.is_empty() {
        return floor;
    }
    gaps.sort_unstable();
    let q = gaps[gaps.len() / 4];
    (q.saturating_mul(14) / 10)
        .max(floor)
        .min(floor.saturating_mul(2).max(floor + 4))
}

/// 4-connected ink components (letter/stroke blobs).
fn connected_components(bin: &GrayImage, min_ink: u32, min_w: u32, min_h: u32) -> Vec<WordRect> {
    let (w, h) = bin.dimensions();
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let mut visited = vec![false; (w as usize) * (h as usize)];
    let idx = |x: u32, y: u32| (y as usize) * (w as usize) + (x as usize);
    let mut out = Vec::new();

    for y0 in 0..h {
        for x0 in 0..w {
            let i0 = idx(x0, y0);
            if visited[i0] || !ink_at(bin, x0, y0) {
                continue;
            }
            let mut stack = vec![(x0, y0)];
            visited[i0] = true;
            let mut min_x = x0;
            let mut max_x = x0;
            let mut min_y = y0;
            let mut max_y = y0;
            let mut ink = 0u32;
            while let Some((x, y)) = stack.pop() {
                ink += 1;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                for (nx, ny) in [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ] {
                    if nx >= w || ny >= h {
                        continue;
                    }
                    let ni = idx(nx, ny);
                    if visited[ni] || !ink_at(bin, nx, ny) {
                        continue;
                    }
                    visited[ni] = true;
                    stack.push((nx, ny));
                }
            }
            let bw = max_x - min_x + 1;
            let bh = max_y - min_y + 1;
            if ink >= min_ink && bw >= min_w && bh >= min_h {
                out.push(WordRect {
                    x: min_x,
                    y: min_y,
                    w: bw,
                    h: bh,
                });
            }
        }
    }
    out
}

fn merge_row_horizontal(mut row: Vec<WordRect>, gap: u32) -> Vec<WordRect> {
    if row.is_empty() {
        return row;
    }
    row.sort_by_key(|w| w.x);
    let mut out: Vec<WordRect> = Vec::new();
    for w in row {
        if let Some(prev) = out.last_mut() {
            let gx = w.x.saturating_sub(prev.x.saturating_add(prev.w));
            if gx <= gap && vertical_overlap_ratio(prev, &w) >= 0.25 {
                *prev = union_rect(prev, &w);
                continue;
            }
        }
        out.push(w);
    }
    out
}

/// Split a glued multi-line box on deep horizontal ink valleys (inter-line gaps).
fn split_tall_box_by_lines(
    bin: &GrayImage,
    r: WordRect,
    med_line_h: u32,
    line_sep: f32,
    params: &SegmentParams,
) -> Vec<WordRect> {
    let tall_thresh = ((med_line_h as f32) * 1.8)
        .max(params.min_word_h as f32 * 3.0)
        .round() as u32;
    if r.h <= tall_thresh.max(1) {
        return vec![r];
    }

    let mut proj = vec![0u32; r.h as usize];
    for dy in 0..r.h {
        let y = r.y + dy;
        for dx in 0..r.w {
            if ink_at(bin, r.x + dx, y) {
                proj[dy as usize] += 1;
            }
        }
    }
    let row_min = (r.w / 20).max(2);
    let mut bands = line_peaks_in_projection(&proj, row_min, 0, line_sep, params.band_gap);
    if bands.len() < 2 {
        // Softer barrier for stubborn paragraph blobs.
        let soft = (line_sep * 0.82).clamp(0.50, 0.98);
        bands = line_peaks_in_projection(&proj, row_min, 0, soft, params.band_gap);
    }
    if bands.len() < 2 {
        return vec![r];
    }

    let mut parts = Vec::new();
    for &(_peak, y0, y1) in &bands {
        let gy0 = r.y + y0 as u32;
        let gy1 = r.y + y1 as u32;
        if gy1 <= gy0 {
            continue;
        }
        if let Some(t) = tight_bounds(bin, r.x, r.x + r.w, gy0, gy1) {
            if t.h >= params.min_word_h && ink_count(bin, &t) >= params.min_ink.max(3) {
                parts.push(t);
            }
        }
    }
    if parts.len() >= 2 {
        parts
    } else {
        vec![r]
    }
}

/// Split a glued multi-word box on deep vertical ink valleys (gaps wider than letter merge).
/// `proj_bin` may be the closed binary (clearer gutters); tight bounds use `bound_bin` (usually raw).
fn split_word_by_valleys(
    bound_bin: &GrayImage,
    proj_bin: &GrayImage,
    r: WordRect,
    params: &SegmentParams,
) -> Vec<WordRect> {
    let min_part = params.min_word_w.max(4);
    // Wide enough for multiple words, or leftover after a tall line split.
    let wide_enough = r.w >= min_part.saturating_mul(2).max(r.h.saturating_mul(2));
    if !wide_enough {
        return vec![r];
    }
    let mut proj = vec![0u32; r.w as usize];
    for dx in 0..r.w {
        let x = r.x + dx;
        for dy in 0..r.h {
            if ink_at(proj_bin, x, r.y + dy) {
                proj[dx as usize] += 1;
            }
        }
    }
    let maxp = proj.iter().copied().max().unwrap_or(0);
    if maxp == 0 {
        return vec![r];
    }
    let col_min = (maxp / 10).max(1);
    // Same adaptive letter gap as projection merge paths.
    let letter_gap = adaptive_merge_gap(&proj, col_min, params.merge_gap_x).max(2);
    let runs = runs_from_projection(&proj, col_min, letter_gap);
    if runs.len() < 2 {
        return vec![r];
    }

    let mut parts = Vec::new();
    for &(a, b) in &runs {
        let x0 = r.x + a as u32;
        let x1 = r.x + b as u32;
        if let Some(t) = tight_bounds(bound_bin, x0, x1, r.y, r.y + r.h) {
            if t.w >= params.min_word_w
                && t.h >= params.min_word_h
                && ink_count(bound_bin, &t) >= params.min_ink
            {
                parts.push(t);
            }
        }
    }
    if parts.len() >= 2 {
        parts
    } else {
        vec![r]
    }
}

/// Line-split then X-valley-split a merged box; retry soft Y-split if still paragraph-sized.
fn explode_merged_box(
    bin0: &GrayImage,
    bin: &GrayImage,
    w: WordRect,
    med_h: u32,
    adaptive_sep: f32,
    params: &SegmentParams,
) -> Vec<WordRect> {
    let mut line_parts = split_tall_box_by_lines(bin0, w, med_h, adaptive_sep, params);
    if line_parts.len() == 1 {
        let only = line_parts[0];
        let paragraphish = only.h as f32 > med_h as f32 * 2.4 && only.w as f32 > med_h as f32 * 4.0;
        if paragraphish {
            let soft = (adaptive_sep * 0.75).clamp(0.50, 0.98);
            line_parts = split_tall_box_by_lines(bin0, only, med_h, soft, params);
        }
    }

    let mut words = Vec::new();
    for lp in line_parts {
        // Prefer closed binary for gutters; fall back to raw if needed.
        let mut parts = split_word_by_valleys(bin0, bin, lp, params);
        if parts.len() == 1 {
            parts = split_word_by_valleys(bin0, bin0, lp, params);
        }
        for p in parts {
            if p.w >= params.min_word_w
                && p.h >= params.min_word_h
                && ink_count(bin0, &p) >= params.min_ink
            {
                words.push(p);
            }
        }
    }
    if words.is_empty() {
        vec![w]
    } else {
        words
    }
}

fn page_from_components(
    bin0: &GrayImage,
    bin: &GrayImage,
    comps: Vec<WordRect>,
    adaptive_sep: f32,
    params: &SegmentParams,
) -> PageWords {
    if comps.is_empty() {
        return PageWords::default();
    }
    let mut heights: Vec<u32> = comps.iter().map(|c| c.h).collect();
    let med_h_u = median_u32(&mut heights).max(params.min_word_h);
    let med_h = med_h_u as f32;
    let comps = attach_floating_marks(comps, med_h);

    // Recompute height after diacritic attach.
    let mut heights2: Vec<u32> = comps.iter().map(|c| c.h).collect();
    let med_h_u = median_u32(&mut heights2).max(params.min_word_h);
    let med_h = med_h_u as f32;

    // Higher sep → tighter Y join → more rows.
    let join = (med_h * (1.20 - (adaptive_sep - 0.50) * 0.85)).max(med_h * 0.45);

    let mut sorted = comps;
    sorted.sort_by(|a, b| {
        let (ax, ay) = a.center();
        let (bx, by) = b.center();
        ay.partial_cmp(&by)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut row_vecs: Vec<Vec<WordRect>> = Vec::new();
    for c in sorted {
        let cy = c.center().1;
        if let Some(row) = row_vecs.last_mut() {
            let ry: f32 = row.iter().map(|w| w.center().1).sum::<f32>() / row.len() as f32;
            if (cy - ry).abs() <= join {
                row.push(c);
                continue;
            }
        }
        row_vecs.push(vec![c]);
    }

    let mut page = PageWords::default();
    for row in row_vecs {
        let gap = letter_gap_from_row_boxes(&row, params.merge_gap_x);
        let merged = merge_row_horizontal(row, gap);
        let mut words = Vec::new();
        for w in merged {
            words.extend(explode_merged_box(
                bin0,
                bin,
                w,
                med_h_u,
                adaptive_sep,
                params,
            ));
        }
        if words.is_empty() {
            continue;
        }
        words.sort_by_key(|w| w.x);
        let start = page.words.len();
        page.words.extend(words);
        page.rows.push((start, page.words.len()));
    }
    page
}

fn segment_binary(bin0: &GrayImage, bin: &GrayImage, params: &SegmentParams) -> PageWords {
    let (w, h) = bin.dimensions();
    if w == 0 || h == 0 {
        return PageWords::default();
    }

    let hproj = global_hproj(bin);
    let row_min = ((w as f32) * 0.006).ceil().max(2.0) as u32;
    let adaptive_sep = estimate_adaptive_line_sep(&hproj, row_min, params.line_sep);

    // CC on closed binary so letter strokes within a word tend to connect.
    let letter_ink = (params.min_ink / 3).max(3);
    let comps = connected_components(bin, letter_ink, 1, 2);
    if comps.len() >= 2 {
        let page = page_from_components(bin0, bin, comps, adaptive_sep, params);
        if !page.words.is_empty() {
            return normalize_reading_order(page, params.untangle_overlap);
        }
    }

    // Fallback: strip polylines (wide pages) or global bands.
    if w < 360 {
        return segment_projection_bands(bin0, bin, params, &hproj, adaptive_sep);
    }

    let strip_n = STRIP_COUNT;
    let mut strip_bounds = Vec::with_capacity(strip_n);
    let mut strip_peaks: Vec<Vec<StripPeak>> = Vec::with_capacity(strip_n);
    for i in 0..strip_n {
        let x0 = (w as u64 * i as u64 / strip_n as u64) as u32;
        let x1 = (w as u64 * (i as u64 + 1) / strip_n as u64) as u32;
        strip_bounds.push((x0, x1));
        let sw = (x1 - x0).max(1);
        let shproj = strip_horizontal_projection(bin, x0, x1);
        let s_row_min = ((sw as f32) * 0.04).ceil().max(2.0) as u32;
        let bands = line_peaks_in_projection(
            &shproj,
            s_row_min,
            params.merge_gap_y,
            adaptive_sep,
            params.band_gap,
        );
        strip_peaks.push(
            bands
                .into_iter()
                .map(|(peak, y0, y1)| StripPeak { peak, y0, y1 })
                .collect(),
        );
    }

    let mut heights = Vec::new();
    for peaks in &strip_peaks {
        for p in peaks {
            heights.push((p.y1.saturating_sub(p.y0)) as u32);
        }
    }
    let med_h = median_u32(&mut heights).max(8);
    let max_dy = med_h as f32 * 1.2;
    let lines = merge_strip_peaks(&strip_peaks, max_dy);

    let mut page = PageWords::default();
    for line in &lines {
        if line.nodes.iter().flatten().count() == 0 {
            continue;
        }
        let words = words_from_line(bin0, bin, line, &strip_bounds, params);
        if words.is_empty() {
            continue;
        }
        let start = page.words.len();
        page.words.extend(words);
        page.rows.push((start, page.words.len()));
    }

    if page.words.is_empty() {
        return segment_projection_bands(bin0, bin, params, &hproj, adaptive_sep);
    }
    let _ = h;
    normalize_reading_order(page, params.untangle_overlap)
}

fn segment_projection_bands(
    bin0: &GrayImage,
    bin: &GrayImage,
    params: &SegmentParams,
    hproj: &[u32],
    adaptive_sep: f32,
) -> PageWords {
    let w = bin.width();
    let row_min = ((w as f32) * 0.006).ceil().max(3.0) as u32;
    let bands = line_peaks_in_projection(
        hproj,
        row_min,
        params.merge_gap_y,
        adaptive_sep,
        params.band_gap,
    );
    let mut page = PageWords::default();
    for (_peak, y0, y1) in bands {
        let y0 = y0 as u32;
        let y1 = y1 as u32;
        let line_h = y1.saturating_sub(y0);
        if line_h < params.min_word_h {
            continue;
        }
        let mut proj = vec![0u32; w as usize];
        for y in y0..y1 {
            for x in 0..w {
                if ink_at(bin, x, y) {
                    proj[x as usize] += 1;
                }
            }
        }
        let col_min = (line_h / 6).max(1);
        let gap = adaptive_merge_gap(&proj, col_min, params.merge_gap_x);
        let start = page.words.len();
        for (a, b) in runs_from_projection(&proj, col_min, gap) {
            if let Some(rect) = tight_bounds(bin0, a as u32, b as u32, y0, y1) {
                if rect.w >= params.min_word_w
                    && rect.h >= params.min_word_h
                    && ink_count(bin0, &rect) >= params.min_ink
                {
                    for part in split_word_by_valleys(bin0, bin, rect, params) {
                        page.words.push(part);
                    }
                }
            }
        }
        if page.words.len() > start {
            // Re-sort this row by X after valley splits.
            let row = &mut page.words[start..];
            row.sort_by_key(|w| w.x);
            page.rows.push((start, page.words.len()));
        }
    }
    normalize_reading_order(page, params.untangle_overlap)
}

fn word_area(w: &WordRect) -> u32 {
    w.w.saturating_mul(w.h)
}

fn rect_iou(a: &WordRect, b: &WordRect) -> f32 {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.w).min(b.x + b.w);
    let y1 = (a.y + a.h).min(b.y + b.h);
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let inter = (x1 - x0) * (y1 - y0);
    let union = word_area(a) + word_area(b) - inter;
    if union == 0 {
        0.0
    } else {
        inter as f32 / union as f32
    }
}

fn rect_coverage(a: &WordRect, b: &WordRect) -> f32 {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.w).min(b.x + b.w);
    let y1 = (a.y + a.h).min(b.y + b.h);
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let inter = (x1 - x0) * (y1 - y0);
    let denom = word_area(a).min(word_area(b)).max(1);
    inter as f32 / denom as f32
}

/// Drop overlapping duplicates (same ink claimed twice), keep larger box.
fn dedupe_words(words: Vec<WordRect>) -> Vec<WordRect> {
    let mut keep = vec![true; words.len()];
    for i in 0..words.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..words.len() {
            if !keep[j] {
                continue;
            }
            let (cix, ciy) = words[i].center();
            let (cjx, cjy) = words[j].center();
            let near_center = (cix - cjx).hypot(ciy - cjy) < 10.0
                && vertical_overlap_ratio(&words[i], &words[j]) > 0.4;
            let overlap = rect_iou(&words[i], &words[j]) > 0.30
                || rect_coverage(&words[i], &words[j]) > 0.45
                || near_center;
            if !overlap {
                continue;
            }
            let ai = word_area(&words[i]);
            let aj = word_area(&words[j]);
            // Prefer earlier in list when equal (stable reading order).
            if aj > ai {
                keep[i] = false;
                break;
            } else {
                keep[j] = false;
            }
        }
    }
    words
        .into_iter()
        .zip(keep)
        .filter_map(|(w, k)| k.then_some(w))
        .collect()
}

/// True when `w` sits horizontally between at least one word to the left and one to the right.
fn x_interleaved(w: &WordRect, row: &[WordRect]) -> bool {
    let cx = w.center().0;
    let mut left = false;
    let mut right = false;
    for o in row {
        let ocx = o.center().0;
        if ocx < cx - 2.0 {
            left = true;
        }
        if ocx > cx + 2.0 {
            right = true;
        }
    }
    left && right
}

fn row_mean_y(row: &[WordRect]) -> f32 {
    if row.is_empty() {
        return 0.0;
    }
    row.iter().map(|w| w.center().1).sum::<f32>() / row.len() as f32
}

/// Staggered handwriting: a word whose Y fell into the wrong cluster but sits
/// between left/right neighbours of another nearby row belongs on that row.
/// Without this, → skips the word (it's already "passed" earlier in the TB chain).
fn reseat_x_interleaved_words(rows: &mut Vec<Vec<WordRect>>) {
    if rows.len() < 2 {
        return;
    }
    let mut heights: Vec<u32> = rows.iter().flatten().map(|w| w.h).collect();
    let med_h = median_u32(&mut heights).max(1) as f32;

    for _ in 0..12 {
        let n = rows.len();
        let means: Vec<f32> = rows.iter().map(|r| row_mean_y(r)).collect();

        let mut op: Option<(usize, usize, usize)> = None; // from, wi, to
        'find: for ri in 0..n {
            for wi in 0..rows[ri].len() {
                let w = rows[ri][wi];
                let cy = w.center().1;

                // Candidate multi-word rows that bracket this word in X.
                let mut best: Option<(usize, f32)> = None; // (row, |dy|)
                for rj in 0..n {
                    if ri == rj || rows[rj].len() < 2 {
                        continue;
                    }
                    if (means[ri] - means[rj]).abs() > med_h * 1.75
                        && (cy - means[rj]).abs() > med_h * 1.75
                    {
                        continue;
                    }
                    if !x_interleaved(&w, &rows[rj]) {
                        continue;
                    }
                    let dy = (cy - means[rj]).abs();
                    if best.map(|(_, d)| dy < d).unwrap_or(true) {
                        best = Some((rj, dy));
                    }
                }
                let Some((rj, dest_dy)) = best else {
                    continue;
                };

                let singleton = rows[ri].len() == 1;
                let home_interleaved = x_interleaved(&w, &rows[ri]);
                let home_dy = (cy - means[ri]).abs();
                // Singleton mid-rows always look "perfect" by mean Y — force reseat
                // when a real line brackets the word in X. If both rows bracket it,
                // keep the closer mean Y (staggered word between two lines).
                let should_move = if singleton {
                    true
                } else if home_interleaved {
                    dest_dy + 1.0 < home_dy
                } else {
                    dest_dy <= home_dy + med_h * 0.45
                };
                if should_move && rj != ri {
                    op = Some((ri, wi, rj));
                    break 'find;
                }
            }
        }

        let Some((ri, wi, rj)) = op else {
            break;
        };
        let w = rows[ri].remove(wi);
        rows[rj].push(w);
        if rows[ri].is_empty() {
            rows.remove(ri);
        }
        for row in rows.iter_mut() {
            row.sort_by_key(|w| w.x);
        }
    }
}

/// Strict reading order: rows by mean Y (top→bottom), words by X (left→right).
/// Untangle first, then dedupe so one ink blob is never listed twice.
pub fn normalize_reading_order(page: PageWords, untangle_overlap: f32) -> PageWords {
    if page.words.is_empty() {
        return page;
    }

    let mut row_vecs: Vec<Vec<WordRect>> = if page.rows.is_empty() {
        vec![page.words]
    } else {
        page.rows
            .iter()
            .map(|&(s, e)| page.words[s..e].to_vec())
            .filter(|r| !r.is_empty())
            .collect()
    };

    row_vecs.sort_by(|a, b| {
        let ya: f32 = a.iter().map(|w| w.center().1).sum::<f32>() / a.len().max(1) as f32;
        let yb: f32 = b.iter().map(|w| w.center().1).sum::<f32>() / b.len().max(1) as f32;
        ya.partial_cmp(&yb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Fix mid-line words that Y-clustering assigned to the wrong row
    // *before* untangle — clipping can pull a staggered word into the wrong band.
    reseat_x_interleaved_words(&mut row_vecs);

    // Separate stuck lines before dropping overlaps.
    untangle_adjacent_row_vecs(&mut row_vecs, untangle_overlap);

    // Again after untangle in case clipping emptied a row / shifted means.
    reseat_x_interleaved_words(&mut row_vecs);

    for row in &mut row_vecs {
        row.sort_by_key(|w| w.x);
        *row = dedupe_words(std::mem::take(row));
        row.sort_by_key(|w| w.x);
    }
    row_vecs.retain(|r| !r.is_empty());

    // Cross-row: drop the smaller of any still-overlapping pair.
    let mut flat_tagged: Vec<(usize, WordRect)> = Vec::new();
    for (ri, row) in row_vecs.iter().enumerate() {
        for w in row {
            flat_tagged.push((ri, *w));
        }
    }
    let mut keep = vec![true; flat_tagged.len()];
    for i in 0..flat_tagged.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..flat_tagged.len() {
            if !keep[j] || flat_tagged[i].0 == flat_tagged[j].0 {
                continue;
            }
            let (a, b) = (&flat_tagged[i].1, &flat_tagged[j].1);
            let overlap = rect_iou(a, b) > 0.30 || rect_coverage(a, b) > 0.45;
            if !overlap {
                continue;
            }
            if word_area(b) > word_area(a) {
                keep[i] = false;
                break;
            } else {
                keep[j] = false;
            }
        }
    }

    let mut by_row: std::collections::BTreeMap<usize, Vec<WordRect>> =
        std::collections::BTreeMap::new();
    for (k, (ri, w)) in keep.into_iter().zip(flat_tagged) {
        if k {
            by_row.entry(ri).or_default().push(w);
        }
    }

    let mut rows: Vec<Vec<WordRect>> = by_row.into_values().collect();
    rows.sort_by(|a, b| {
        let ya: f32 = a.iter().map(|w| w.center().1).sum::<f32>() / a.len().max(1) as f32;
        let yb: f32 = b.iter().map(|w| w.center().1).sum::<f32>() / b.len().max(1) as f32;
        ya.partial_cmp(&yb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out = PageWords::default();
    for mut row in rows {
        row.retain(|w| w.w > 0 && w.h > 0);
        if row.is_empty() {
            continue;
        }
        row.sort_by_key(|w| w.x);
        let start = out.words.len();
        out.words.extend(row);
        out.rows.push((start, out.words.len()));
    }
    // `out.words` is the sole reading-order chain: TB by rows, LR within row.
    debug_assert!(out.rows.windows(2).all(|w| w[0].1 == w[1].0));
    out
}

fn vertical_overlap_ratio(a: &WordRect, b: &WordRect) -> f32 {
    let a0 = a.y;
    let a1 = a.y.saturating_add(a.h);
    let b0 = b.y;
    let b1 = b.y.saturating_add(b.h);
    let top = a0.max(b0);
    let bot = a1.min(b1);
    if bot <= top {
        return 0.0;
    }
    let overlap = (bot - top) as f32;
    let min_h = a.h.min(b.h).max(1) as f32;
    overlap / min_h
}

/// Clip words that cross the valley between neighbouring rows, only on strong overlap.
fn untangle_adjacent_row_vecs(rows: &mut [Vec<WordRect>], threshold: f32) {
    if threshold <= 0.0 {
        return;
    }
    for i in 0..rows.len().saturating_sub(1) {
        if rows[i].is_empty() || rows[i + 1].is_empty() {
            continue;
        }
        let ya: f32 =
            rows[i].iter().map(|w| w.center().1).sum::<f32>() / rows[i].len() as f32;
        let yb: f32 =
            rows[i + 1].iter().map(|w| w.center().1).sum::<f32>() / rows[i + 1].len() as f32;
        let valley = ((ya + yb) * 0.5).round().max(0.0) as u32;

        let upper0 = rows[i].clone();
        let lower0 = rows[i + 1].clone();
        for w in &mut rows[i] {
            let max_ov = lower0
                .iter()
                .map(|v| vertical_overlap_ratio(w, v))
                .fold(0.0_f32, f32::max);
            if max_ov <= threshold {
                continue;
            }
            let bottom = w.y.saturating_add(w.h);
            if bottom > valley {
                if w.y >= valley {
                    w.h = 0;
                } else {
                    w.h = valley - w.y;
                }
            }
        }
        for w in &mut rows[i + 1] {
            let max_ov = upper0
                .iter()
                .map(|v| vertical_overlap_ratio(v, w))
                .fold(0.0_f32, f32::max);
            if max_ov <= threshold {
                continue;
            }
            let bottom = w.y.saturating_add(w.h);
            if w.y < valley {
                if bottom <= valley {
                    w.h = 0;
                } else {
                    w.h = bottom - valley;
                    w.y = valley;
                }
            }
        }
    }
    for row in rows.iter_mut() {
        row.retain(|w| w.w > 0 && w.h > 0);
    }
}

/// Rotate + Sauvola + horizontal close. Shared by preview and full segment.
pub fn prepare_binary(
    rgba: &RgbaImage,
    params: &SegmentParams,
) -> (RgbaImage, GrayImage, GrayImage) {
    let params = params.clone().clamped();
    let rotated = rotate_rgba(rgba, params.rotation_deg);
    let gray = to_gray(&rotated);
    let bin0 = binarize_sauvola(&gray, params.threshold);
    let bin = close_horizontal(&bin0, params.close_radius_x);
    (rotated, bin0, bin)
}

/// Rotate + binarize for editor textures without recomputing word boxes.
#[allow(dead_code)]
pub fn preview_rasters(rgba: &RgbaImage, params: &SegmentParams) -> (RgbaImage, GrayImage) {
    let (rotated, _bin0, bin) = prepare_binary(rgba, params);
    (rotated, bin)
}

/// Re-run geometry on already prepared binary (skip Sauvola/close when only merge params change).
pub fn segment_from_binary(
    bin0: &GrayImage,
    bin: &GrayImage,
    rotated: RgbaImage,
    params: &SegmentParams,
) -> SegmentOutput {
    let params = params.clone().clamped();
    let page = segment_binary(bin0, bin, &params);
    SegmentOutput {
        page,
        binary: bin.clone(),
        rgba: rotated,
    }
}

/// Full pipeline with binary preview and rotated raster.
pub fn segment_full(rgba: &RgbaImage, params: &SegmentParams) -> SegmentOutput {
    let params = params.clone().clamped();
    let (rotated, bin0, bin) = prepare_binary(rgba, &params);
    segment_from_binary(&bin0, &bin, rotated, &params)
}

#[allow(dead_code)]
pub fn segment_rgba(rgba: &RgbaImage, params: &SegmentParams) -> PageWords {
    segment_full(rgba, params).page
}

#[cfg(test)]
pub fn segment_gray(gray: &GrayImage, params: &SegmentParams) -> PageWords {
    // Wrap gray as fake RGBA for the shared pipeline without rotation side-effects on color.
    let rgba = ImageBuffer::from_fn(gray.width(), gray.height(), |x, y| {
        let v = gray.get_pixel(x, y)[0];
        Rgba([v, v, v, 255])
    });
    let mut p = params.clone();
    // Synthetic pages: use global threshold path via bias; skip sauvola pathology by close match.
    // Force classic binarize for pure black/white synthetics by using sauvola which falls back.
    p.rotation_deg = 0.0;
    segment_full(&rgba, &p).page
}

#[cfg(test)]
pub fn synthetic_page(width: u32, height: u32, rects: &[(u32, u32, u32, u32)]) -> GrayImage {
    let mut img = ImageBuffer::from_pixel(width, height, Luma([255u8]));
    for &(x, y, w, h) in rects {
        for yy in y..y.saturating_add(h).min(height) {
            for xx in x..x.saturating_add(w).min(width) {
                img.put_pixel(xx, yy, Luma([0]));
            }
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> SegmentParams {
        SegmentParams {
            threshold: 128,
            merge_gap_x: 4,
            merge_gap_y: 0,
            line_sep: 0.72,
            min_word_w: 8,
            min_word_h: 4,
            min_ink: 10,
            close_radius_x: 0,
            rotation_deg: 0.0,
            band_gap: 1,
            untangle_overlap: 0.25,
        }
    }

    #[test]
    fn untangle_clips_vertical_overlap_between_rows() {
        let page = PageWords {
            words: vec![
                WordRect {
                    x: 10,
                    y: 10,
                    w: 40,
                    h: 30, // extends into next row
                },
                WordRect {
                    x: 10,
                    y: 25,
                    w: 40,
                    h: 30, // starts in previous row
                },
            ],
            rows: vec![(0, 1), (1, 2)],
        };
        let out = normalize_reading_order(page, 0.25);
        assert_eq!(out.rows.len(), 2);
        let (s0, e0) = out.rows[0];
        let (s1, e1) = out.rows[1];
        for a in &out.words[s0..e0] {
            for b in &out.words[s1..e1] {
                let a_bottom = a.y + a.h;
                assert!(
                    a_bottom <= b.y,
                    "rows still overlap: upper ends at {}, lower starts at {}",
                    a_bottom,
                    b.y
                );
            }
        }
    }

    #[test]
    fn untangle_skips_light_touch() {
        // ~8% vertical overlap — below default 0.25 threshold.
        let page = PageWords {
            words: vec![
                WordRect {
                    x: 10,
                    y: 10,
                    w: 40,
                    h: 20,
                },
                WordRect {
                    x: 10,
                    y: 28,
                    w: 40,
                    h: 20,
                },
            ],
            rows: vec![(0, 1), (1, 2)],
        };
        let before_h0 = page.words[0].h;
        let before_y1 = page.words[1].y;
        let out = normalize_reading_order(page, 0.25);
        assert_eq!(out.words[0].h, before_h0);
        assert_eq!(out.words[1].y, before_y1);
    }

    #[test]
    fn reseat_mid_stagger_word_into_lower_row() {
        // Top row + bottom row; middle word Y is between them but X sits between
        // the two bottom words (screenshot: «рождается» skipped by →).
        let page = PageWords {
            words: vec![
                WordRect {
                    x: 10,
                    y: 10,
                    w: 40,
                    h: 16,
                },
                WordRect {
                    x: 80,
                    y: 12,
                    w: 50,
                    h: 16,
                },
                WordRect {
                    x: 160,
                    y: 10,
                    w: 55,
                    h: 16,
                },
                WordRect {
                    x: 70,
                    y: 28,
                    w: 70,
                    h: 18,
                }, // mid alone (bad Y cluster)
                WordRect {
                    x: 10,
                    y: 40,
                    w: 45,
                    h: 16,
                },
                WordRect {
                    x: 160,
                    y: 42,
                    w: 40,
                    h: 16,
                },
            ],
            rows: vec![(0, 3), (3, 4), (4, 6)],
        };
        let out = normalize_reading_order(page, 0.25);
        assert_eq!(out.rows.len(), 2, "mid word should join a neighbour row: {:?}", out.rows);
        let (s, e) = out.rows[1];
        let bottom: Vec<_> = out.words[s..e].iter().map(|w| w.x).collect();
        assert_eq!(bottom.len(), 3, "bottom should have 3 words, got {:?}", bottom);
        assert!(bottom[0] < bottom[1] && bottom[1] < bottom[2]);
        assert_eq!(out.words[s + 1].w, 70);
    }

    #[test]
    fn reseat_mid_word_wrongly_glued_into_upper_row() {
        // Same geometry, but mid word was merged into the top cluster.
        let page = PageWords {
            words: vec![
                WordRect {
                    x: 10,
                    y: 10,
                    w: 40,
                    h: 16,
                },
                WordRect {
                    x: 80,
                    y: 12,
                    w: 50,
                    h: 16,
                },
                WordRect {
                    x: 70,
                    y: 28,
                    w: 70,
                    h: 18,
                }, // mid, wrongly in top row
                WordRect {
                    x: 160,
                    y: 10,
                    w: 55,
                    h: 16,
                },
                WordRect {
                    x: 10,
                    y: 40,
                    w: 45,
                    h: 16,
                },
                WordRect {
                    x: 160,
                    y: 42,
                    w: 40,
                    h: 16,
                },
            ],
            rows: vec![(0, 4), (4, 6)],
        };
        let out = normalize_reading_order(page, 0.25);
        assert_eq!(out.rows.len(), 2, "expected 2 rows, got {:?}", out.rows);
        let (s, e) = out.rows[1];
        assert_eq!(e - s, 3, "bottom should absorb mid word");
        assert_eq!(out.words[s + 1].w, 70);
    }

    #[test]
    fn normalize_dedupes_iou_and_sorts_rows() {
        let page = PageWords {
            words: vec![
                WordRect {
                    x: 10,
                    y: 40,
                    w: 30,
                    h: 12,
                },
                WordRect {
                    x: 50,
                    y: 42,
                    w: 28,
                    h: 12,
                },
                WordRect {
                    x: 10,
                    y: 10,
                    w: 30,
                    h: 12,
                },
                WordRect {
                    x: 12,
                    y: 11,
                    w: 26,
                    h: 10,
                }, // near-duplicate of previous
                WordRect {
                    x: 50,
                    y: 10,
                    w: 28,
                    h: 12,
                },
            ],
            rows: vec![(0, 2), (2, 5)], // deliberately unsorted rows
        };
        let out = normalize_reading_order(page, 0.25);
        assert!(out.words.len() <= 4, "duplicate should be dropped, got {}", out.words.len());
        assert!(out.rows.len() >= 2);
        // Reading order: mean Y ascending; within row X ascending.
        for row_i in 0..out.rows.len().saturating_sub(1) {
            let (s0, e0) = out.rows[row_i];
            let (s1, e1) = out.rows[row_i + 1];
            let y0: f32 = out.words[s0..e0]
                .iter()
                .map(|w| w.center().1)
                .sum::<f32>()
                / (e0 - s0).max(1) as f32;
            let y1: f32 = out.words[s1..e1]
                .iter()
                .map(|w| w.center().1)
                .sum::<f32>()
                / (e1 - s1).max(1) as f32;
            assert!(y0 <= y1 + 0.5, "row {row_i} Y {y0} > next {y1}");
        }
        for &(s, e) in &out.rows {
            for i in s..e.saturating_sub(1) {
                assert!(
                    out.words[i].x <= out.words[i + 1].x,
                    "words not sorted by X in row"
                );
            }
        }
        // Flat list: no large upward Y jump between consecutive words in a row.
        for &(s, e) in &out.rows {
            for i in s..e.saturating_sub(1) {
                let y0 = out.words[i].center().1;
                let y1 = out.words[i + 1].center().1;
                assert!(
                    y1 + 10.0 >= y0,
                    "neighbor Y jumped up: {y0} -> {y1}"
                );
            }
        }
    }

    #[test]
    fn reading_order_is_tb_lr_without_duplicate_boxes() {
        let rects = [
            (10, 10, 40, 12),
            (60, 10, 35, 12),
            (110, 10, 50, 12),
            (10, 40, 45, 12),
            (70, 40, 55, 12),
        ];
        let gray = synthetic_page(200, 80, &rects);
        let page = segment_gray(&gray, &test_params());
        assert!(page.words.len() >= 4);
        // Top→bottom rows.
        for i in 0..page.rows.len().saturating_sub(1) {
            let (s0, e0) = page.rows[i];
            let (s1, e1) = page.rows[i + 1];
            let y0: f32 = page.words[s0..e0].iter().map(|w| w.center().1).sum::<f32>()
                / (e0 - s0) as f32;
            let y1: f32 = page.words[s1..e1].iter().map(|w| w.center().1).sum::<f32>()
                / (e1 - s1) as f32;
            assert!(y0 <= y1 + 1.0);
        }
        // Left→right within row; no IoU duplicates anywhere.
        for &(s, e) in &page.rows {
            for i in s..e.saturating_sub(1) {
                assert!(page.words[i].x <= page.words[i + 1].x);
            }
        }
        for i in 0..page.words.len() {
            for j in (i + 1)..page.words.len() {
                assert!(
                    rect_iou(&page.words[i], &page.words[j]) < 0.35,
                    "duplicate-like boxes {i} and {j}"
                );
            }
        }
    }

    #[test]
    fn valley_split_separates_wide_glued_words() {
        // Two wide ink blocks with a clear empty gutter — one wide bbox should split.
        let mut img = ImageBuffer::from_pixel(120u32, 40u32, Luma([255u8]));
        for y in 10..28 {
            for x in 5..45 {
                img.put_pixel(x, y, Luma([0]));
            }
            for x in 70..110 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        let wide = WordRect {
            x: 5,
            y: 10,
            w: 105,
            h: 18,
        };
        let mut params = test_params();
        params.min_word_w = 8;
        params.min_ink = 5;
        let parts = split_word_by_valleys(&img, &img, wide, &params);
        assert!(
            parts.len() >= 2,
            "expected valley split, got {} parts",
            parts.len()
        );
        assert!(parts[0].x < parts[1].x);
    }

    #[test]
    fn attach_diacritic_dot_to_body() {
        // Tall body + small floating mark above (ё / й style).
        let body = WordRect {
            x: 20,
            y: 20,
            w: 18,
            h: 22,
        };
        let dot = WordRect {
            x: 26,
            y: 10,
            w: 5,
            h: 4,
        };
        let out = attach_floating_marks(vec![body, dot], 22.0);
        assert_eq!(out.len(), 1, "diacritic should attach, got {:?}", out);
        assert!(out[0].y <= 10);
        assert!(out[0].h >= 26);
    }

    #[test]
    fn tall_box_splits_into_lines() {
        // Three ink bands stacked with clear gutters inside one tall bbox.
        let mut img = ImageBuffer::from_pixel(80u32, 90u32, Luma([255u8]));
        for y in 5..18 {
            for x in 5..70 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 35..48 {
            for x in 5..70 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 65..78 {
            for x in 5..70 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        let tall = WordRect {
            x: 5,
            y: 5,
            w: 65,
            h: 73,
        };
        let mut params = test_params();
        params.min_word_h = 4;
        params.min_ink = 5;
        let parts = split_tall_box_by_lines(&img, tall, 14, 0.72, &params);
        assert!(
            parts.len() >= 3,
            "expected ≥3 line parts, got {} {:?}",
            parts.len(),
            parts
        );
    }

    #[test]
    fn wide_gutter_keeps_two_words_with_moderate_merge() {
        // Two words with a large gutter — must not glue even with merge_x=8.
        let rects = [(10, 20, 30, 12), (70, 20, 30, 12)];
        let gray = synthetic_page(120, 60, &rects);
        let mut params = test_params();
        params.merge_gap_x = 8;
        params.min_word_w = 5;
        params.min_ink = 5;
        let page = segment_gray(&gray, &params);
        assert!(
            page.words.len() >= 2,
            "wide gutter should stay 2 words, got {}",
            page.words.len()
        );
    }

    #[test]
    fn finds_two_rows_of_words() {
        let rects = [
            (10, 10, 40, 12),
            (60, 10, 35, 12),
            (110, 10, 50, 12),
            (10, 40, 45, 12),
            (70, 40, 55, 12),
        ];
        let gray = synthetic_page(200, 80, &rects);
        let page = segment_gray(&gray, &test_params());
        assert_eq!(page.rows.len(), 2, "expected 2 rows, got {:?}", page.rows);
        assert_eq!(page.rows[0].1 - page.rows[0].0, 3);
        assert_eq!(page.rows[1].1 - page.rows[1].0, 2);
        assert_eq!(page.words.len(), 5);
        assert!(page.words[0].x < page.words[1].x);
        assert!(page.words[1].x < page.words[2].x);
    }

    #[test]
    fn uneven_line_gaps_still_two_rows() {
        // Large gap between rows — variable spacing.
        let rects = [
            (10, 8, 40, 10),
            (60, 8, 35, 10),
            (110, 8, 40, 10),
            (10, 55, 45, 10),
            (70, 55, 50, 10),
        ];
        let gray = synthetic_page(200, 90, &rects);
        let page = segment_gray(&gray, &test_params());
        assert_eq!(page.rows.len(), 2, "expected 2 rows, got {:?}", page.rows);
        assert_eq!(page.words.len(), 5);
    }

    #[test]
    fn merges_close_strokes_into_one_word() {
        let rects = [(20, 20, 15, 10), (38, 20, 15, 10)];
        let gray = synthetic_page(100, 60, &rects);
        let mut params = test_params();
        params.merge_gap_x = 8;
        params.min_word_w = 5;
        params.min_ink = 5;
        let page = segment_gray(&gray, &params);
        assert_eq!(page.words.len(), 1);
        assert!(page.words[0].w >= 30);
    }

    #[test]
    fn rotation_changes_output_size_or_coords() {
        let path = std::path::Path::new("samples/manuscript.jpg");
        if !path.exists() {
            return;
        }
        let rgba = image::open(path).unwrap().to_rgba8();
        let a = segment_full(&rgba, &SegmentParams::default());
        let mut p = SegmentParams::default();
        p.rotation_deg = 3.0;
        let b = segment_full(&rgba, &p);
        assert_ne!(
            (a.rgba.width(), a.rgba.height()),
            (b.rgba.width(), b.rgba.height()),
            "rotation should expand canvas"
        );
        assert!(!b.binary.is_empty());
        assert!(b.page.words.len() > 50);
    }

    #[test]
    fn segments_real_manuscript() {
        let path = std::path::Path::new("samples/manuscript.jpg");
        if !path.exists() {
            eprintln!("skip: samples/manuscript.jpg missing");
            return;
        }
        let rgba = image::open(path).unwrap().to_rgba8();
        let out = segment_full(&rgba, &SegmentParams::default());
        let page = &out.page;
        eprintln!(
            "manuscript: {}x{}, words={}, rows={}, avg_words/row={:.1}",
            out.rgba.width(),
            out.rgba.height(),
            page.words.len(),
            page.rows.len(),
            page.words.len() as f32 / page.rows.len().max(1) as f32
        );
        let mut overlay = out.rgba.clone();
        for (i, w) in page.words.iter().enumerate() {
            let color = if i % 2 == 0 {
                Rgba([255, 40, 40, 255])
            } else {
                Rgba([40, 180, 255, 255])
            };
            for x in w.x..w.x.saturating_add(w.w).min(overlay.width()) {
                overlay.put_pixel(x, w.y, color);
                let yb = w.y.saturating_add(w.h.saturating_sub(1)).min(overlay.height() - 1);
                overlay.put_pixel(x, yb, color);
            }
            for y in w.y..w.y.saturating_add(w.h).min(overlay.height()) {
                overlay.put_pixel(w.x, y, color);
                let xr = w.x.saturating_add(w.w.saturating_sub(1)).min(overlay.width() - 1);
                overlay.put_pixel(xr, y, color);
            }
        }
        overlay
            .save("samples/manuscript_words.png")
            .unwrap();
        assert!(page.words.len() > 120, "got {}", page.words.len());
        assert!(page.rows.len() > 20, "got {}", page.rows.len());
    }

    #[test]
    fn line_sep_and_merge_gap_change_counts() {
        let path = std::path::Path::new("samples/manuscript.jpg");
        if !path.exists() {
            return;
        }
        let rgba = image::open(path).unwrap().to_rgba8();
        let mut tight = SegmentParams::default();
        tight.line_sep = 0.85;
        tight.merge_gap_x = 2;
        let mut loose = SegmentParams::default();
        loose.line_sep = 0.55;
        loose.merge_gap_x = 12;
        let a = segment_rgba(&rgba, &tight);
        let b = segment_rgba(&rgba, &loose);
        eprintln!(
            "tight words/rows={}/{} loose={}/{}",
            a.words.len(),
            a.rows.len(),
            b.words.len(),
            b.rows.len()
        );
        assert!(
            a.rows.len() != b.rows.len() || a.words.len() != b.words.len(),
            "params should change segmentation"
        );
    }
}
