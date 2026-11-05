//! Virtual photo caret: snaps to segmented words (Right-Shift+arrows) or free-pans with ⌘+arrows.
//!
//! ←/→ follow visual neighbours (same line by Y affinity, else wrap to next/prev line).
//! ↑/↓ use neighbouring `rows`, matching preferred X (not slot index).

use crate::segment::{PageWords, WordRect};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Caret {
    /// Index into `PageWords::words`.
    Word(usize),
    /// Free pan center in image coordinates.
    Free { x: f32, y: f32 },
}

impl Default for Caret {
    fn default() -> Self {
        Self::Word(0)
    }
}

impl Caret {
    pub fn center(&self, page: &PageWords) -> (f32, f32) {
        match *self {
            Caret::Word(i) => page
                .words
                .get(i)
                .map(|w| w.center())
                .unwrap_or((0.0, 0.0)),
            Caret::Free { x, y } => (x, y),
        }
    }

    /// Clamp free point to image bounds; keep word index in range.
    pub fn sanitize(self, page: &PageWords, img_w: f32, img_h: f32) -> Self {
        match self {
            Caret::Word(_i) if page.words.is_empty() => Caret::Free {
                x: img_w / 2.0,
                y: img_h / 2.0,
            },
            Caret::Word(i) => Caret::Word(i.min(page.words.len().saturating_sub(1))),
            Caret::Free { x, y } => Caret::Free {
                x: x.clamp(0.0, img_w.max(0.0)),
                y: y.clamp(0.0, img_h.max(0.0)),
            },
        }
    }

    /// After `page.words` is rebuilt/reordered, keep the caret on the same ink
    /// by nearest center — raw indices become meaningless across rebuilds.
    pub fn rematch_after_page_change(
        self,
        old_center: (f32, f32),
        page: &PageWords,
        img_w: f32,
        img_h: f32,
    ) -> Self {
        if page.words.is_empty() {
            return Caret::Free {
                x: old_center.0.clamp(0.0, img_w.max(0.0)),
                y: old_center.1.clamp(0.0, img_h.max(0.0)),
            };
        }
        match page.nearest_word(old_center.0, old_center.1) {
            Some(i) => Caret::Word(i),
            None => self.sanitize(page, img_w, img_h),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CaretMove {
    Left,
    Right,
    Up,
    Down,
}

fn chain_index(caret: Caret, page: &PageWords) -> usize {
    match caret {
        Caret::Word(i) => i.min(page.words.len() - 1),
        Caret::Free { x, y } => page.nearest_word(x, y).unwrap_or(0),
    }
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

/// Same handwriting line: solid vertical overlap, or centers within ±tol.
fn same_line(a: &WordRect, b: &WordRect) -> bool {
    if vertical_overlap_ratio(a, b) >= 0.22 {
        return true;
    }
    let dy = (a.center().1 - b.center().1).abs();
    let tol = (a.h.min(b.h) as f32 * 0.55).max(5.0);
    dy <= tol
}

/// Y extent of the current visual line (self + same_line neighbours).
fn line_y_band(page: &PageWords, idx: usize) -> (f32, f32, f32) {
    let w = &page.words[idx];
    let mut y0 = w.y as f32;
    let mut y1 = (w.y + w.h) as f32;
    let mut h_sum = w.h as f32;
    let mut n = 1.0_f32;
    for (j, o) in page.words.iter().enumerate() {
        if j == idx || !same_line(w, o) {
            continue;
        }
        y0 = y0.min(o.y as f32);
        y1 = y1.max((o.y + o.h) as f32);
        h_sum += o.h as f32;
        n += 1.0;
    }
    (y0, y1, (h_sum / n).max(1.0))
}

fn clearly_to_the_right(cur: &WordRect, other: &WordRect) -> bool {
    // Must start at / past our right edge (tiny overlap OK), not merely have a
    // center a few px to the right while sitting mostly behind us.
    let cur_right = cur.x + cur.w;
    other.x + 2 >= cur_right || other.x >= cur.x + cur.w / 2 && other.center().0 > cur.center().0 + 4.0
}

fn clearly_to_the_left(cur: &WordRect, other: &WordRect) -> bool {
    let other_right = other.x + other.w;
    cur.x + 2 >= other_right
        || other.x + other.w / 2 <= cur.x && other.center().0 + 4.0 < cur.center().0
}

/// Next word to the right on the same visual line; else first word of the line below.
fn move_right_visual(page: &PageWords, idx: usize) -> Caret {
    let w = &page.words[idx];
    let mut best: Option<(usize, f32)> = None;
    for (j, o) in page.words.iter().enumerate() {
        if j == idx || !same_line(w, o) || !clearly_to_the_right(w, o) {
            continue;
        }
        let dx = (o.x as f32 - (w.x + w.w) as f32).max(0.0) + 0.1;
        if best.map(|(_, s)| dx < s).unwrap_or(true) {
            best = Some((j, dx));
        }
    }
    if let Some((j, _)) = best {
        return Caret::Word(j);
    }

    // No neighbour on this Y-band → wrap to the next line (leftmost below the band).
    wrap_to_line_below(page, idx)
}

fn wrap_to_line_below(page: &PageWords, idx: usize) -> Caret {
    let w = &page.words[idx];
    let (_y0, band_y1, med_h) = line_y_band(page, idx);
    // Must clear the whole current line band — not just sit a few px lower on it.
    let threshold = band_y1 + (med_h * 0.20).max(3.0);
    let mut below: Vec<usize> = page
        .words
        .iter()
        .enumerate()
        .filter(|(j, o)| *j != idx && !same_line(w, o) && o.center().1 >= threshold)
        .map(|(j, _)| j)
        .collect();
    if below.is_empty() {
        return Caret::Word(idx);
    }
    let min_y = below
        .iter()
        .map(|&j| page.words[j].center().1)
        .fold(f32::INFINITY, f32::min);
    let band = (med_h * 0.85).max(10.0);
    below.retain(|&j| page.words[j].center().1 <= min_y + band);
    below.sort_by_key(|&j| page.words[j].x);
    Caret::Word(below[0])
}

fn wrap_to_line_above(page: &PageWords, idx: usize) -> Caret {
    let w = &page.words[idx];
    let (band_y0, _y1, med_h) = line_y_band(page, idx);
    let threshold = band_y0 - (med_h * 0.20).max(3.0);
    let mut above: Vec<usize> = page
        .words
        .iter()
        .enumerate()
        .filter(|(j, o)| *j != idx && !same_line(w, o) && o.center().1 <= threshold)
        .map(|(j, _)| j)
        .collect();
    if above.is_empty() {
        return Caret::Word(idx);
    }
    let max_y = above
        .iter()
        .map(|&j| page.words[j].center().1)
        .fold(f32::NEG_INFINITY, f32::max);
    let band = (med_h * 0.85).max(10.0);
    above.retain(|&j| page.words[j].center().1 >= max_y - band);
    above.sort_by_key(|&j| page.words[j].x);
    Caret::Word(*above.last().unwrap())
}

/// Previous word to the left on the same visual line; else last word of the line above.
fn move_left_visual(page: &PageWords, idx: usize) -> Caret {
    let w = &page.words[idx];
    let mut best: Option<(usize, f32)> = None;
    for (j, o) in page.words.iter().enumerate() {
        if j == idx || !same_line(w, o) || !clearly_to_the_left(w, o) {
            continue;
        }
        let dx = ((w.x as f32) - (o.x + o.w) as f32).max(0.0) + 0.1;
        if best.map(|(_, s)| dx < s).unwrap_or(true) {
            best = Some((j, dx));
        }
    }
    if let Some((j, _)) = best {
        return Caret::Word(j);
    }
    wrap_to_line_above(page, idx)
}

/// Jump along visual line neighbours or between rows. From Free, snaps nearest first.
pub fn move_by_word(caret: Caret, page: &PageWords, dir: CaretMove) -> Caret {
    if page.words.is_empty() {
        return caret;
    }
    let idx = chain_index(caret, page);
    match dir {
        CaretMove::Right => move_right_visual(page, idx),
        CaretMove::Left => move_left_visual(page, idx),
        CaretMove::Down => move_vertical(page, idx, 1),
        CaretMove::Up => move_vertical(page, idx, -1),
    }
}

/// ↑/↓: neighbouring `rows` (line_id ± 1), picking nearest center-X — not slot index.
fn move_vertical(page: &PageWords, idx: usize, row_delta: i32) -> Caret {
    let Some((row_i, _)) = page.line_col(idx) else {
        return move_vertical_visual(page, idx, row_delta);
    };
    let target_row = row_i as i32 + row_delta;
    if target_row < 0 || target_row as usize >= page.rows.len() {
        return Caret::Word(idx);
    }
    let (s, e) = page.rows[target_row as usize];
    if s >= e {
        return Caret::Word(idx);
    }
    let cx = page.words[idx].center().0;
    let cy = page.words[idx].center().1;
    let best = nearest_x_in_range(page, s, e, cx);
    // If `rows` skip a visual gap (bad cache), don't jump halfway down the page.
    let med_h = page.words[idx].h.max(1) as f32;
    let oy = page.words[best].center().1;
    if (oy - cy).abs() > med_h * 6.0 {
        return move_vertical_visual(page, idx, row_delta);
    }
    Caret::Word(best)
}

fn nearest_x_in_range(page: &PageWords, s: usize, e: usize, cx: f32) -> usize {
    (s..e)
        .min_by(|&a, &b| {
            let da = (page.words[a].center().0 - cx).abs();
            let db = (page.words[b].center().0 - cx).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(s)
}

/// Geometric ↑/↓: next visual line band, then nearest X on that band.
fn move_vertical_visual(page: &PageWords, idx: usize, row_delta: i32) -> Caret {
    let w = &page.words[idx];
    let cx = w.center().0;
    let (band_y0, band_y1, med_h) = line_y_band(page, idx);
    let mut cands: Vec<usize> = if row_delta > 0 {
        let threshold = band_y1 + (med_h * 0.20).max(3.0);
        page.words
            .iter()
            .enumerate()
            .filter(|(j, o)| *j != idx && !same_line(w, o) && o.center().1 >= threshold)
            .map(|(j, _)| j)
            .collect()
    } else {
        let threshold = band_y0 - (med_h * 0.20).max(3.0);
        page.words
            .iter()
            .enumerate()
            .filter(|(j, o)| *j != idx && !same_line(w, o) && o.center().1 <= threshold)
            .map(|(j, _)| j)
            .collect()
    };
    if cands.is_empty() {
        return Caret::Word(idx);
    }
    let band = (med_h * 0.85).max(10.0);
    if row_delta > 0 {
        let min_y = cands
            .iter()
            .map(|&j| page.words[j].center().1)
            .fold(f32::INFINITY, f32::min);
        cands.retain(|&j| page.words[j].center().1 <= min_y + band);
    } else {
        let max_y = cands
            .iter()
            .map(|&j| page.words[j].center().1)
            .fold(f32::NEG_INFINITY, f32::max);
        cands.retain(|&j| page.words[j].center().1 >= max_y - band);
    }
    let best = cands
        .into_iter()
        .min_by(|&a, &b| {
            let da = (page.words[a].center().0 - cx).abs();
            let db = (page.words[b].center().0 - cx).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    Caret::Word(best)
}

/// Space: next word on the visual line (end of line → first of next).
pub fn advance_on_space(caret: Caret, page: &PageWords) -> Caret {
    move_by_word(caret, page, CaretMove::Right)
}

/// Fine pan without snapping to words.
pub fn nudge_free(caret: Caret, page: &PageWords, dx: f32, dy: f32, img_w: f32, img_h: f32) -> Caret {
    let (x, y) = caret.center(page);
    Caret::Free {
        x: (x + dx).clamp(0.0, img_w.max(0.0)),
        y: (y + dy).clamp(0.0, img_h.max(0.0)),
    }
}

/// Click: snap to word under cursor, or free point if miss.
pub fn click_at(page: &PageWords, px: f32, py: f32) -> Caret {
    if let Some(i) = page.word_index_at(px, py) {
        Caret::Word(i)
    } else {
        Caret::Free { x: px, y: py }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::WordRect;

    fn sample_page() -> PageWords {
        // Row0: w0 w1 w2 | Row1: w3 w4  — already TB→LR chain.
        PageWords {
            words: vec![
                WordRect {
                    x: 10,
                    y: 10,
                    w: 20,
                    h: 10,
                },
                WordRect {
                    x: 40,
                    y: 10,
                    w: 20,
                    h: 10,
                },
                WordRect {
                    x: 70,
                    y: 10,
                    w: 20,
                    h: 10,
                },
                WordRect {
                    x: 10,
                    y: 40,
                    w: 20,
                    h: 10,
                },
                WordRect {
                    x: 40,
                    y: 40,
                    w: 20,
                    h: 10,
                },
            ],
            rows: vec![(0, 3), (3, 5)],
        }
    }

    #[test]
    fn space_wraps_to_next_row() {
        let page = sample_page();
        assert_eq!(advance_on_space(Caret::Word(2), &page), Caret::Word(3));
    }

    #[test]
    fn space_stays_on_last() {
        let page = sample_page();
        assert_eq!(advance_on_space(Caret::Word(4), &page), Caret::Word(4));
    }

    #[test]
    fn arrows_follow_visual_line() {
        let page = sample_page();
        assert_eq!(
            move_by_word(Caret::Word(0), &page, CaretMove::Right),
            Caret::Word(1)
        );
        assert_eq!(
            move_by_word(Caret::Word(1), &page, CaretMove::Left),
            Caret::Word(0)
        );
    }

    #[test]
    fn right_does_not_skip_mid_stagger_word() {
        // Bottom-left → mid (higher Y) → bottom-right — must not jump over mid.
        let page = PageWords {
            words: vec![
                WordRect {
                    x: 10,
                    y: 10,
                    w: 40,
                    h: 16,
                },
                WordRect {
                    x: 160,
                    y: 10,
                    w: 40,
                    h: 16,
                },
                WordRect {
                    x: 10,
                    y: 40,
                    w: 45,
                    h: 16,
                }, // start
                WordRect {
                    x: 70,
                    y: 28,
                    w: 70,
                    h: 18,
                }, // mid stagger
                WordRect {
                    x: 160,
                    y: 42,
                    w: 40,
                    h: 16,
                },
            ],
            // Bad rows: mid stuck alone between lines (old bug).
            rows: vec![(0, 2), (3, 4), (2, 3), (4, 5)],
        };
        let mid = move_by_word(Caret::Word(2), &page, CaretMove::Right);
        assert_eq!(mid, Caret::Word(3), "→ should land on mid stagger word");
        let right = move_by_word(mid, &page, CaretMove::Right);
        assert_eq!(right, Caret::Word(4));
    }

    #[test]
    fn down_preserves_column() {
        let page = sample_page();
        assert_eq!(
            move_by_word(Caret::Word(1), &page, CaretMove::Down),
            Caret::Word(4)
        );
        assert_eq!(page.line_col(1), Some((0, 1)));
        assert_eq!(page.line_col(4), Some((1, 1)));
    }

    #[test]
    fn down_matches_x_not_column_index() {
        // Top: 2 words, caret on the right (col 1, x≈400).
        // Bottom: 5 words across the page. Column-index ↓ would land on
        // bottom[1] (x≈50) — a teleport left across the image.
        let page = PageWords {
            words: vec![
                WordRect {
                    x: 10,
                    y: 10,
                    w: 30,
                    h: 14,
                },
                WordRect {
                    x: 400,
                    y: 10,
                    w: 30,
                    h: 14,
                }, // caret
                WordRect {
                    x: 10,
                    y: 50,
                    w: 30,
                    h: 14,
                },
                WordRect {
                    x: 50,
                    y: 50,
                    w: 30,
                    h: 14,
                },
                WordRect {
                    x: 120,
                    y: 50,
                    w: 30,
                    h: 14,
                },
                WordRect {
                    x: 250,
                    y: 50,
                    w: 30,
                    h: 14,
                },
                WordRect {
                    x: 400,
                    y: 50,
                    w: 30,
                    h: 14,
                }, // expected
            ],
            rows: vec![(0, 2), (2, 7)],
        };
        assert_eq!(
            move_by_word(Caret::Word(1), &page, CaretMove::Down),
            Caret::Word(6),
            "↓ must keep X column, not row slot index"
        );
        assert_eq!(
            move_by_word(Caret::Word(6), &page, CaretMove::Up),
            Caret::Word(1),
            "↑ must keep X column"
        );
    }

    #[test]
    fn option_nudge_makes_free() {
        let page = sample_page();
        let c = nudge_free(Caret::Word(0), &page, 5.0, 0.0, 200.0, 100.0);
        match c {
            Caret::Free { x, y } => {
                assert!((x - 25.0).abs() < 0.01);
                assert!((y - 15.0).abs() < 0.01);
            }
            _ => panic!("expected Free"),
        }
    }

    #[test]
    fn click_word_and_miss() {
        let page = sample_page();
        assert_eq!(click_at(&page, 15.0, 12.0), Caret::Word(0));
        assert_eq!(click_at(&page, 5.0, 5.0), Caret::Free { x: 5.0, y: 5.0 });
    }

    #[test]
    fn free_then_arrow_snaps_nearest_then_chain() {
        let page = sample_page();
        let c = move_by_word(Caret::Free { x: 42.0, y: 12.0 }, &page, CaretMove::Right);
        // nearest is word 1, right → word 2 along the line
        assert_eq!(c, Caret::Word(2));
    }

    #[test]
    fn wrap_uses_next_line_not_geometry_under() {
        let page = PageWords {
            words: vec![
                WordRect {
                    x: 10,
                    y: 10,
                    w: 20,
                    h: 10,
                },
                WordRect {
                    x: 70,
                    y: 10,
                    w: 20,
                    h: 10,
                },
                WordRect {
                    x: 10,
                    y: 40,
                    w: 20,
                    h: 10,
                },
                WordRect {
                    x: 70,
                    y: 36,
                    w: 20,
                    h: 10,
                },
            ],
            rows: vec![(0, 2), (2, 4)],
        };
        assert_eq!(advance_on_space(Caret::Word(1), &page), Caret::Word(2));
    }

    #[test]
    fn right_at_line_end_wraps_even_if_far_word_exists_elsewhere() {
        // End of line 0; a word far right but on a much lower Y must not steal →.
        let page = PageWords {
            words: vec![
                WordRect {
                    x: 10,
                    y: 10,
                    w: 30,
                    h: 14,
                },
                WordRect {
                    x: 50,
                    y: 10,
                    w: 30,
                    h: 14,
                }, // line end
                WordRect {
                    x: 10,
                    y: 45,
                    w: 30,
                    h: 14,
                }, // next line start
                WordRect {
                    x: 200,
                    y: 80,
                    w: 30,
                    h: 14,
                }, // unrelated lower-right
            ],
            rows: vec![(0, 2), (2, 3), (3, 4)],
        };
        assert_eq!(
            move_by_word(Caret::Word(1), &page, CaretMove::Right),
            Caret::Word(2)
        );
    }

    #[test]
    fn rematch_follows_geometry_after_reorder() {
        let old = PageWords {
            words: vec![
                WordRect {
                    x: 10,
                    y: 10,
                    w: 20,
                    h: 10,
                },
                WordRect {
                    x: 100,
                    y: 10,
                    w: 20,
                    h: 10,
                },
            ],
            rows: vec![(0, 2)],
        };
        let new = PageWords {
            words: vec![
                WordRect {
                    x: 100,
                    y: 10,
                    w: 20,
                    h: 10,
                }, // was index 1
                WordRect {
                    x: 10,
                    y: 10,
                    w: 20,
                    h: 10,
                }, // was index 0
            ],
            rows: vec![(0, 2)],
        };
        let caret = Caret::Word(1);
        let center = caret.center(&old);
        assert_eq!(
            caret.rematch_after_page_change(center, &new, 200.0, 100.0),
            Caret::Word(0),
            "saved index 1 must rematch to new index 0 after reorder"
        );
    }

    #[test]
    fn right_at_line_end_does_not_jump_back_on_stagger() {
        // A — B(end). B is slightly higher; A sits a bit lower. Old bug: wrap
        // treated A as "next line" and jumped back halfway.
        let page = PageWords {
            words: vec![
                WordRect {
                    x: 10,
                    y: 14,
                    w: 40,
                    h: 16,
                }, // A
                WordRect {
                    x: 70,
                    y: 10,
                    w: 40,
                    h: 16,
                }, // B (caret)
                WordRect {
                    x: 10,
                    y: 48,
                    w: 40,
                    h: 16,
                }, // next line
            ],
            rows: vec![(0, 2), (2, 3)],
        };
        assert_eq!(
            move_by_word(Caret::Word(1), &page, CaretMove::Right),
            Caret::Word(2),
            "→ must wrap down, not jump back to A"
        );
    }

}
