"""Draw filterax's icon and logo: ``icon.svg`` and ``logo-{light,dark}.svg``.

The icon is the sequential picture: ensemble members spread during each
forecast and pinch together at each observation. Standard library plus
loguru; the tile and wordmark follow the other repos of the stack (radtrax,
gaussx, kernellib).
Run from the repository root::

    uv run --no-project --with loguru python docs/assets/make_assets.py
"""

import itertools
import math
from pathlib import Path

from loguru import logger


HERE = Path(__file__).resolve().parent

NAME = "filterax"
WORDMARK = ("filter", "ax")
LOGO_WIDTH = 500
# Fuchsia: a sibling of vardax's rose, since both do data assimilation.
GRADIENT = ("#E879F9", "#A21CAF")
THEMES = {
    "light": {"ink": "#1F2937", "accent": "#C026D3"},
    "dark": {"ink": "#E5E7EB", "accent": "#E879F9"},
}
SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
WHITE = "#FFFFFF"


def svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{NAME}">\n'
        f"<title>{NAME}</title>\n{body}\n</svg>\n"
    )


def tile(gradient_id: str) -> str:
    start, stop = GRADIENT
    return (
        f'<defs><linearGradient id="{gradient_id}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{start}"/>'
        f'<stop offset="1" stop-color="{stop}"/></linearGradient></defs>'
        '<rect x="4" y="4" width="120" height="120" rx="26" '
        f'fill="url(#{gradient_id})"/>'
    )


def polyline(
    points: list[tuple[float, float]], *, width: float, opacity: float = 1.0
) -> str:
    commands = [
        f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(points)
    ]
    return (
        f'<path d="{" ".join(commands)}" fill="none" stroke="{WHITE}" '
        f'stroke-width="{width}" stroke-opacity="{opacity}" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def dot(x: float, y: float, radius: float) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{WHITE}"/>'


def glyph() -> str:
    """Five members spread during each forecast and pinch at each observation."""
    observation_x = [22, 64, 106]

    def ensemble_mean(x: float) -> float:
        return 64 - 12 * math.sin((x - 22) / 84 * 2 * math.pi)

    body = ""
    for start, stop in itertools.pairwise(observation_x):
        for member, offset in enumerate([-12, -6, 0, 6, 12]):
            points = []
            for step in range(31):
                fraction = step / 30
                x = start + (stop - start) * fraction
                points.append((x, ensemble_mean(x) + offset * (0.1 + 0.9 * fraction)))
            opacity = 0.4 + 0.15 * (2 - abs(member - 2))
            body += polyline(points, width=2.4, opacity=opacity)
    for x in observation_x:
        body += (
            f'<circle cx="{x}" cy="{ensemble_mean(x):.1f}" r="6" fill="none" '
            f'stroke="{WHITE}" stroke-width="2.4"/>'
        )
    return body


def icon() -> str:
    return svg(128, 128, tile(f"{NAME}-icon") + glyph())


def logo(theme: str) -> str:
    colours = THEMES[theme]
    head, tail = WORDMARK
    mark = f'<g transform="translate(6 6)">{tile(f"{NAME}-{theme}")}{glyph()}</g>'
    wordmark = (
        f'<text x="154" y="92" font-family="{SANS}" font-size="72" font-weight="700" '
        f'letter-spacing="-1.5" fill="{colours["ink"]}">{head}'
        f'<tspan fill="{colours["accent"]}">{tail}</tspan></text>'
    )
    return svg(LOGO_WIDTH, 140, mark + wordmark)


def main() -> None:
    outputs = {"icon.svg": icon()}
    for theme in THEMES:
        outputs[f"logo-{theme}.svg"] = logo(theme)
    for filename, content in outputs.items():
        (HERE / filename).write_text(content)
        logger.info("wrote {}", HERE / filename)


if __name__ == "__main__":
    main()
