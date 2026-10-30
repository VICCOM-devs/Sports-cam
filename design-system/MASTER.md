# Sports Cam – design system (MASTER)

Look: **broadcast dark**. Near-black surfaces, one ice-blue accent, team colours taken from the
jerseys drive the overlays so every match looks like its own broadcast. Code reads these tokens
from `web/css/tokens.css`; never hard-code a colour or size in a screen.

## Colour

| Token | Value | Use |
|---|---|---|
| `--bg` | `#0A0C10` | page background |
| `--surface-1` | `#12151B` | cards |
| `--surface-2` | `#1A1E26` | raised cards, inputs |
| `--surface-3` | `#232833` | pressed / hover |
| `--stroke` | `#2C323E` | hairlines (dark UI shows depth with lighter surface + hairline, not shadows) |
| `--text` | `#F3F5F8` | primary text |
| `--text-dim` | `#A6ADBA` | secondary |
| `--text-faint` | `#6B7380` | hints, timestamps |
| `--accent` | `#3DDCFF` | ice blue – primary action, live indicator |
| `--accent-soft` | `rgba(61,220,255,.14)` | accent fills |
| `--accent-deep` | `#0FA7C8` | pressed accent |
| `--success` | `#3DDC97` | |
| `--warn` | `#FFB02E` | battery, heat |
| `--danger` | `#FF4D5E` | REC, errors |
| `--home` / `--away` | from the match | team colours, set per match at runtime |

Contrast: text on bg ≥ 12:1, dim ≥ 6:1, faint is decorative only.

## Type

System UI stack (`Inter`, `-apple-system`, `Roboto`) so phones render instantly offline.

| Role | Size / weight / tracking |
|---|---|
| display | 40 / 800 / -0.03em (score, speed) |
| title | 22 / 700 / -0.02em |
| body | 16 / 400 |
| label | 12 / 700 / +0.08em, uppercase |
| mono | tabular numbers for clocks (`font-variant-numeric: tabular-nums`) |

## Space, shape, motion

- 4-pt grid: 4, 8, 12, 16, 24, 32, 48. Touch targets ≥ 56 px (gloves, cold hands), ≥ 8 px apart.
- Radii: 10 chips, 16 cards, 24 sheets. Strokes 1 px.
- Motion: fast 120 ms, base 200 ms, slow 320 ms; easing `cubic-bezier(.2,.8,.2,1)`. Angle switch = 200 ms crossfade. Respect `prefers-reduced-motion`.
- Feedback within 100 ms: pressed state + light haptic (`navigator.vibrate(8)` where allowed).

## States (every screen)

loading = skeleton blocks, never a spinner over everything · empty = what it is + one action ·
error = plain words + retry · offline = chip, keep showing last data · success = brief confirmation ·
long content = long Swedish team names, 130 % font.

## Copy

Swedish first, English second, all strings in `web/i18n/*.json`. Buttons are verbs. Errors say
what happened and what to do.
