"""Regenerate the PWA icons from the graph mark (requires Pillow)."""

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 2048
SCALE = SIZE / 24
DESTINATION = Path(__file__).resolve().parent.parent / "public" / "icons"


def point(x, y):
    return round(x * SCALE), round(y * SCALE)


def curve(start, control1, control2, end):
    points = []
    for index in range(101):
        t = index / 100
        u = 1 - t
        x = (
            u**3 * start[0]
            + 3 * u**2 * t * control1[0]
            + 3 * u * t**2 * control2[0]
            + t**3 * end[0]
        )
        y = (
            u**3 * start[1]
            + 3 * u**2 * t * control1[1]
            + 3 * u * t**2 * control2[1]
            + t**3 * end[1]
        )
        points.append(point(x, y))
    return points


image = Image.new("RGB", (SIZE, SIZE), "#0066cc")
draw = ImageDraw.Draw(image)
stroke = round(1.75 * SCALE)
for center in [(6.5, 7), (6.5, 17), (17.5, 12)]:
    x, y = point(*center)
    radius = round(2 * SCALE)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius), outline="white", width=stroke
    )

for points in [
    curve((8.5, 7.5), (12.5, 7.5), (11.5, 12), (15.5, 12)),
    curve((8.5, 16.5), (12.5, 16.5), (11.5, 12), (15.5, 12)),
]:
    draw.line(points, fill="white", width=stroke, joint="curve")

DESTINATION.mkdir(parents=True, exist_ok=True)
for size in (180, 192, 512):
    image.resize((size, size), Image.Resampling.LANCZOS).save(
        DESTINATION / f"icon-{size}.png", optimize=True
    )
