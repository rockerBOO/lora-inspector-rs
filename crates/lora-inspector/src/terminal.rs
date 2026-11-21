// `terminal` is the first of four modules that assemble the `inspect` report;
// its items are consumed by the render tier in a later task, so nothing here is
// referenced from `main.rs` yet.
#![allow(dead_code)]

use std::io::IsTerminal;

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Detected terminal capabilities used by every rendering tier.
///
/// `width` is the usable column count (always at least [`MIN_WIDTH`]); `color`
/// says whether ANSI styles may be emitted; `unicode` selects the Unicode
/// glyph set (`false` is the `--ascii` mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Terminal {
    pub width: usize,
    pub color: bool,
    pub unicode: bool,
}

/// The `--color` flag. `Auto` enables colour only on a clean TTY; `Always`
/// emits colour unconditionally; `Never` is always off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ColorChoice {
    Auto,
    Always,
    Never,
}

/// The styles the report can apply. `Ok` is the default (no decoration).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    Good,
    Ok,
    Weak,
    Poor,
    Collapsed,
    Dim,
    Bold,
}

const MIN_WIDTH: usize = 20;
const FALLBACK_WIDTH: usize = 80;

impl Terminal {
    /// Resolve the terminal settings for this invocation.
    ///
    /// Width is resolved from `--width`, then `COLUMNS`, then the terminal
    /// size reported for stdout (stderr/stdin fall back as the crate does),
    /// then [`FALLBACK_WIDTH`], and is clamped to [`MIN_WIDTH`]..=`usize::MAX`.
    pub fn detect(flag_width: Option<usize>, flag_color: ColorChoice, ascii: bool) -> Terminal {
        Terminal {
            width: resolve_width(flag_width),
            color: resolve_color(flag_color),
            unicode: !ascii,
        }
    }

    /// Display width of `s` in columns, ignoring any ANSI escape sequences.
    ///
    /// This is always the width the terminal will actually occupy, so CJK and
    /// other wide characters count twice; it is never `s.chars().count()`.
    pub fn text_width(&self, s: &str) -> usize {
        strip_ansi(s).width()
    }

    /// Truncate `s` to at most `max` columns, replacing the middle with an
    /// ellipsis (`…`, or `~` in ASCII mode). The result never exceeds `max`
    /// columns and never splits a grapheme, but an all-wide-character input can
    /// under-fill the target; callers compensate with [`Terminal::pad_right`].
    pub fn truncate_middle(&self, s: &str, max: usize) -> String {
        if self.text_width(s) <= max {
            return s.to_string();
        }

        let ellipsis = if self.unicode { "…" } else { "~" };
        let ellipsis_width = ellipsis.width();
        if max < ellipsis_width {
            return take_prefix(s, max);
        }

        let available = max - ellipsis_width;
        let prefix_width = available / 2;
        let suffix_width = available - prefix_width;

        let mut out = String::with_capacity(max);
        out.push_str(&take_prefix(s, prefix_width));
        out.push_str(ellipsis);
        out.push_str(&take_suffix(s, suffix_width));
        out
    }

    /// Pad `s` on the right with spaces until it is exactly `width` columns
    /// wide (using display width). Returns `s` unchanged when it already fills
    /// `width` columns.
    pub fn pad_right(&self, s: &str, width: usize) -> String {
        let used = self.text_width(s);
        if used >= width {
            return s.to_string();
        }

        let mut out = String::with_capacity(s.len() + (width - used));
        out.push_str(s);
        for _ in used..width {
            out.push(' ');
        }
        out
    }

    /// Render a fraction as a horizontal bar exactly `width` columns wide.
    ///
    /// The fraction is clamped to `0.0..=1.0` (`NaN` becomes `0`); the filled
    /// portion is `floor(width × frac)` full blocks and the remainder spaces.
    pub fn bar(&self, frac: f64, width: usize) -> String {
        let frac = if frac.is_nan() {
            0.0
        } else {
            frac.clamp(0.0, 1.0)
        };
        let filled = ((width as f64) * frac).floor() as usize;
        let glyph = if self.unicode { '█' } else { '#' };

        let mut out = String::with_capacity(width * glyph.len_utf8() + (width - filled));
        for _ in 0..filled {
            out.push(glyph);
        }
        for _ in filled..width {
            out.push(' ');
        }
        out
    }

    /// Render `values` as a sparkline exactly `width` columns wide.
    ///
    /// The sequence is resampled with nearest-neighbour indexing (column `j`
    /// reads value `round(j × (n − 1) / (width − 1))`) and each value is
    /// mapped onto the eight-level ramp `▁▂▃▄▅▆▇█`. An empty slice, all-equal
    /// values, or values with no finite member produce `width` spaces.
    /// Non-finite values are drawn at the lowest level.
    pub fn sparkline(&self, values: &[f64], width: usize) -> String {
        let Some((min, max)) = finite_bounds(values) else {
            return " ".repeat(width);
        };

        let glyphs: [char; 8] = if self.unicode {
            ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        } else {
            ['.', ':', '-', '=', '+', '*', '#', '@']
        };

        let mut out = String::with_capacity(width * 4);
        for column in 0..width {
            let index = nearest_index(column, values.len(), width);
            let value = values[index];
            let frac = if value.is_finite() {
                ((value - min) / (max - min)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let level = ((frac * 7.0).round() as usize).min(7);
            out.push(glyphs[level]);
        }
        out
    }

    /// Apply `style` to `s`. When colour is off (or the style is [`Style::Ok`])
    /// this is a no-op and returns `s` unchanged.
    pub fn paint(&self, style: Style, s: &str) -> String {
        if !self.color {
            return s.to_string();
        }

        let code = match style {
            Style::Good => "32",
            Style::Ok => return s.to_string(),
            Style::Weak => "33",
            Style::Poor => "31",
            Style::Collapsed => "1;31",
            Style::Dim => "2",
            Style::Bold => "1",
        };
        format!("\x1b[{code}m{s}\x1b[0m")
    }
}

/// `--width` → `COLUMNS` → terminal size on stdout → fallback, clamped up to
/// the usable minimum.
fn resolve_width(flag_width: Option<usize>) -> usize {
    let width = if let Some(width) = flag_width {
        width
    } else if let Some(width) = std::env::var("COLUMNS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
    {
        width
    } else if let Some((terminal_size::Width(width), _)) = terminal_size::terminal_size() {
        width as usize
    } else {
        FALLBACK_WIDTH
    };

    width.max(MIN_WIDTH)
}

/// `Never` is always off; `Always` is unconditionally on (an explicit user
/// request outranks `NO_COLOR` and `TERM=dumb`); `Auto` needs a TTY and a clean
/// environment.
fn resolve_color(choice: ColorChoice) -> bool {
    resolve_color_with(
        choice,
        std::io::stdout().is_terminal(),
        no_color_set(std::env::var_os("NO_COLOR")),
        std::env::var("TERM")
            .map(|term| term == "dumb")
            .unwrap_or(false),
    )
}

/// `NO_COLOR` disables colour only when it is present and non-empty, per
/// no-color.org (an empty value is a common accidental export and must not
/// turn colour off).
fn no_color_set(value: Option<std::ffi::OsString>) -> bool {
    value.is_some_and(|v| !v.is_empty())
}

/// Pure decision table for `--color`, separated from the environment reads so
/// every branch is testable without mutating process-global environment.
fn resolve_color_with(
    choice: ColorChoice,
    is_tty: bool,
    no_color_set: bool,
    term_is_dumb: bool,
) -> bool {
    match choice {
        ColorChoice::Never => false,
        ColorChoice::Always => true,
        ColorChoice::Auto => is_tty && !no_color_set && !term_is_dumb,
    }
}

/// The finite min/max of `values`, or `None` when there is no finite value or
/// the range is empty (so a sparkline would have nothing to vary).
fn finite_bounds(values: &[f64]) -> Option<(f64, f64)> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &value in values {
        if value.is_finite() {
            if value < min {
                min = value;
            }
            if value > max {
                max = value;
            }
        }
    }
    (max > min).then_some((min, max))
}

/// Nearest-neighbour source index for output column `column` when `len` input
/// values are resampled to `width` columns. Maps the first and last columns to
/// the first and last values.
fn nearest_index(column: usize, len: usize, width: usize) -> usize {
    if width <= 1 {
        return 0;
    }
    let position = column as f64 * (len as f64 - 1.0) / (width as f64 - 1.0);
    (position.round() as usize).clamp(0, len - 1)
}

/// Strip CSI escape sequences (`ESC [ … final`) so `text_width` measures only
/// the visible text of a painted string.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            chars.next(); // consume '['
            while let Some(&d) = chars.peek() {
                chars.next();
                if ('\u{40}'..='\u{7e}').contains(&d) {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Grapheme spans as `(start, end, width)`, grouping a base character with any
/// following zero-width marks. This keeps truncation from cutting between a
/// base character and its combining marks; plain CJK/ASCII map one span per
/// character.
fn grapheme_spans(s: &str) -> Vec<(usize, usize, usize)> {
    let mut spans: Vec<(usize, usize, usize)> = Vec::new();
    let mut current: Option<(usize, usize, usize)> = None;

    for (index, c) in s.char_indices() {
        let width = c.width().unwrap_or(0);
        match current.as_mut() {
            Some(span) => {
                if width > 0 {
                    spans.push(std::mem::replace(
                        span,
                        (index, index + c.len_utf8(), width),
                    ));
                } else {
                    span.1 = index + c.len_utf8();
                }
            }
            None => current = Some((index, index + c.len_utf8(), width)),
        }
    }

    if let Some(span) = current {
        spans.push(span);
    }
    spans
}

/// Longest grapheme-safe prefix of `s` whose display width is at most `max`.
fn take_prefix(s: &str, max: usize) -> String {
    let mut width = 0usize;
    let mut end = 0usize;
    for (_, stop, w) in grapheme_spans(s) {
        if w > 0 && width + w > max {
            break;
        }
        width += w;
        end = stop;
    }
    s[..end].to_string()
}

/// Longest grapheme-safe suffix of `s` whose display width is at most `max`.
fn take_suffix(s: &str, max: usize) -> String {
    let mut width = 0usize;
    let mut start = s.len();
    for (grapheme_start, _, w) in grapheme_spans(s).into_iter().rev() {
        if w > 0 && width + w > max {
            break;
        }
        width += w;
        start = grapheme_start;
    }
    s[start..].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_names_are_measured_by_display_width() {
        let t = Terminal {
            width: 40,
            color: false,
            unicode: true,
        };
        assert_eq!(t.text_width("Klein 一edit-consistency"), 24);
        assert_eq!(t.text_width("复古风格"), 8);
    }

    #[test]
    fn truncation_never_splits_a_grapheme() {
        let t = Terminal {
            width: 40,
            color: false,
            unicode: true,
        };
        let out = t.truncate_middle("retro-comic-illustration-复古风格", 12);
        assert_eq!(t.text_width(&out), 12);
        assert!(out.contains('…'));
    }

    #[test]
    fn width_resolution_clamps_to_a_usable_minimum() {
        assert_eq!(
            Terminal::detect(Some(0), ColorChoice::Never, false).width,
            20
        );
        assert_eq!(
            Terminal::detect(Some(1), ColorChoice::Never, false).width,
            20
        );
        assert_eq!(
            Terminal::detect(Some(200), ColorChoice::Never, false).width,
            200
        );
    }

    #[test]
    fn bars_never_exceed_their_width_even_for_bad_input() {
        let t = Terminal {
            width: 80,
            color: false,
            unicode: true,
        };
        assert_eq!(t.text_width(&t.bar(f64::NAN, 10)), 10);
        assert_eq!(t.text_width(&t.bar(-1.0, 10)), 10);
        assert_eq!(t.text_width(&t.bar(5.0, 10)), 10);
    }

    #[test]
    fn sparkline_resamples_to_exact_width() {
        let t = Terminal {
            width: 80,
            color: false,
            unicode: true,
        };
        let values: Vec<f64> = (0..26).map(|i| i as f64).collect();
        assert_eq!(t.text_width(&t.sparkline(&values, 12)), 12);
        assert_eq!(t.text_width(&t.sparkline(&[], 12)), 12);
    }

    #[test]
    fn color_is_off_when_not_auto_or_not_tty() {
        let t = Terminal::detect(None, ColorChoice::Never, false);
        assert_eq!(t.paint(Style::Good, "x"), "x");
    }

    #[test]
    fn color_never_is_always_off() {
        assert!(!resolve_color_with(ColorChoice::Never, true, false, false));
        assert!(!resolve_color_with(ColorChoice::Never, false, true, true));
    }

    #[test]
    fn color_always_overrides_environment() {
        // An explicit `Always` wins over a non-TTY, `NO_COLOR`, and `TERM=dumb`.
        assert!(resolve_color_with(ColorChoice::Always, false, true, true));
        assert!(Terminal::detect(None, ColorChoice::Always, false).color);
    }

    #[test]
    fn color_auto_requires_tty_and_clean_environment() {
        assert!(resolve_color_with(ColorChoice::Auto, true, false, false));
        assert!(!resolve_color_with(ColorChoice::Auto, false, false, false)); // non-TTY
        assert!(!resolve_color_with(ColorChoice::Auto, true, true, false)); // NO_COLOR
        assert!(!resolve_color_with(ColorChoice::Auto, true, false, true)); // TERM=dumb
    }

    #[test]
    fn no_color_is_honored_only_when_non_empty() {
        assert!(!no_color_set(None));
        assert!(!no_color_set(Some(std::ffi::OsString::new())));
        assert!(no_color_set(Some(std::ffi::OsString::from("1"))));
    }

    #[test]
    fn ascii_sparkline_uses_eight_level_ramp() {
        let t = Terminal {
            width: 40,
            color: false,
            unicode: false,
        };
        assert_eq!(
            t.sparkline(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 8),
            ".:-=+*#@"
        );
    }

    #[test]
    fn bars_and_sparklines_fill_zero_and_one_column_widths() {
        let t = Terminal {
            width: 80,
            color: false,
            unicode: true,
        };
        for x in [0.0, 0.5, 1.0, f64::NAN] {
            assert_eq!(t.text_width(&t.bar(x, 0)), 0);
            assert_eq!(t.text_width(&t.bar(x, 1)), 1);
        }
        assert_eq!(t.text_width(&t.sparkline(&[1.0, 2.0], 1)), 1);
        assert_eq!(t.text_width(&t.sparkline(&[], 0)), 0);
    }

    #[test]
    fn pad_right_fills_to_display_width() {
        let t = Terminal {
            width: 40,
            color: false,
            unicode: true,
        };
        assert_eq!(t.pad_right("复古", 6), "复古  ");
        assert_eq!(t.pad_right("ab", 1), "ab");
    }

    #[test]
    fn ascii_mode_emits_only_ascii() {
        let t = Terminal {
            width: 40,
            color: false,
            unicode: false,
        };
        assert!(t.bar(0.5, 10).is_ascii());
        assert!(t.sparkline(&[1.0, 2.0, 3.0], 10).is_ascii());

        // `--ascii` swaps the report's glyphs (including the ellipsis), not the
        // user's filename, so a CJK name keeps its CJK characters.
        let truncated = t.truncate_middle("retro-comic-illustration-复古风格", 12);
        assert_eq!(t.text_width(&truncated), 12);
        assert!(truncated.contains('~'));
        assert!(!truncated.contains('…'));
        assert!(t
            .truncate_middle("retro-comic-illustration-long-name", 12)
            .is_ascii());
        assert_eq!(t.text_width(&t.bar(0.5, 10)), 10);
    }

    #[test]
    fn text_width_ignores_ansi_escapes() {
        let t = Terminal {
            width: 40,
            color: false,
            unicode: true,
        };
        assert_eq!(t.text_width("\x1b[32m复古\x1b[0m"), 4);
    }
}
