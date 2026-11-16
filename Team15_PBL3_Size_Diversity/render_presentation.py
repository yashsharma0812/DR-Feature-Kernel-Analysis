"""Render the generated PPTX to PNG review images and a companion PDF.

This lightweight renderer covers the shape types used by this project's deck:
filled rectangles, text boxes, pictures, and tables. It is used for deterministic
slide-by-slide QA on systems where Office/Keynote automation is unavailable.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN


ROOT = Path(__file__).resolve().parent
PPTX = ROOT / "presentation" / "Team15_PBL3_Final.pptx"
RENDER_DIR = ROOT / "presentation" / "rendered_slides"
PDF = ROOT / "presentation" / "Team15_PBL3_Final.pdf"
CONTACT = ROOT / "presentation" / "Team15_PBL3_Contact_Sheet.png"
WIDTH, HEIGHT = 1600, 900
REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")


def rgb(value, fallback=(255, 255, 255)):
    try:
        color = value.rgb
        return tuple(color) if color is not None else fallback
    except (AttributeError, TypeError, ValueError):
        return fallback


def font(size_pt, bold=False):
    size_px = max(8, round(float(size_pt or 12) * WIDTH / 13.333 / 72))
    return ImageFont.truetype(str(BOLD if bold else REGULAR), size_px)


def emu_box(shape, slide_w, slide_h):
    return (
        round(shape.left / slide_w * WIDTH),
        round(shape.top / slide_h * HEIGHT),
        round((shape.left + shape.width) / slide_w * WIDTH),
        round((shape.top + shape.height) / slide_h * HEIGHT),
    )


def wrap(draw, value, fnt, max_width):
    words = value.split()
    if not words:
        return [""]
    lines, current = [], words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if draw.textlength(candidate, font=fnt) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_text_frame(canvas, box, frame, default_color=(37, 48, 62), table_mode=False):
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    pad_x = 8 if not table_mode else 4
    pad_y = 5 if not table_mode else 3
    available = max(5, x1 - x0 - pad_x * 2)
    rendered = []
    for paragraph in frame.paragraphs:
        value = paragraph.text
        runs = paragraph.runs
        first = runs[0] if runs else None
        size = first.font.size.pt if first and first.font.size else (10 if table_mode else 12)
        bold = bool(first.font.bold) if first else False
        fnt = font(size, bold)
        color = rgb(first.font.color, default_color) if first else default_color
        for hard_line in value.split("\n") or [""]:
            for line in wrap(draw, hard_line, fnt, available):
                rendered.append((line, fnt, color, paragraph.alignment))
        rendered.append((None, fnt, color, paragraph.alignment))
    if rendered and rendered[-1][0] is None:
        rendered.pop()
    heights = []
    for value, fnt, _, _ in rendered:
        if value is None:
            heights.append(max(2, fnt.size // 4))
        else:
            heights.append(round(fnt.size * 1.18))
    total_h = sum(heights)
    anchor = getattr(frame, "vertical_anchor", None)
    if anchor == MSO_ANCHOR.MIDDLE:
        y = y0 + max(pad_y, (y1 - y0 - total_h) / 2)
    elif anchor == MSO_ANCHOR.BOTTOM:
        y = y1 - total_h - pad_y
    else:
        y = y0 + pad_y
    for (value, fnt, color, alignment), line_h in zip(rendered, heights):
        if value is None:
            y += line_h
            continue
        width = draw.textlength(value, font=fnt)
        if alignment == PP_ALIGN.CENTER:
            x = x0 + (x1 - x0 - width) / 2
        elif alignment == PP_ALIGN.RIGHT:
            x = x1 - width - pad_x
        else:
            x = x0 + pad_x
        draw.text((x, y), value, font=fnt, fill=color)
        y += line_h


def draw_auto_shape(canvas, shape, box):
    draw = ImageDraw.Draw(canvas)
    fill = rgb(shape.fill.fore_color, (255, 255, 255)) if shape.fill.type else None
    line = rgb(shape.line.color, fill or (218, 228, 237))
    width = max(1, round((shape.line.width or 0) / 12700))
    radius = 14 if "Rounded" in shape.name or getattr(shape, "auto_shape_type", None) == 5 else 0
    if fill is not None:
        if radius:
            draw.rounded_rectangle(box, radius=radius, fill=fill, outline=line, width=width)
        else:
            draw.rectangle(box, fill=fill, outline=line, width=width)
    elif line:
        draw.rectangle(box, outline=line, width=width)
    if getattr(shape, "has_text_frame", False):
        draw_text_frame(canvas, box, shape.text_frame)


def draw_table(canvas, shape, box):
    table = shape.table
    x0, y0, x1, y1 = box
    widths = [column.width for column in table.columns]
    heights = [row.height for row in table.rows]
    total_w, total_h = sum(widths), sum(heights)
    xs = [x0]
    ys = [y0]
    for w in widths:
        xs.append(xs[-1] + (x1 - x0) * w / total_w)
    for h in heights:
        ys.append(ys[-1] + (y1 - y0) * h / total_h)
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(table.rows):
        for c, cell in enumerate(row.cells):
            cell_box = (round(xs[c]), round(ys[r]), round(xs[c + 1]), round(ys[r + 1]))
            fill = rgb(cell.fill.fore_color, (255, 255, 255)) if cell.fill.type else (255, 255, 255)
            draw.rectangle(cell_box, fill=fill, outline=(218, 228, 237), width=1)
            draw_text_frame(canvas, cell_box, cell.text_frame, table_mode=True)


def render_slide(slide, slide_w, slide_h):
    try:
        background = rgb(slide.background.fill.fore_color, (255, 255, 255))
    except (AttributeError, TypeError, ValueError):
        background = (255, 255, 255)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), background)
    for shape in slide.shapes:
        box = emu_box(shape, slide_w, slide_h)
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            with Image.open(BytesIO(shape.image.blob)) as im:
                image = im.convert("RGB")
                image.thumbnail((max(1, box[2] - box[0]), max(1, box[3] - box[1])), Image.Resampling.LANCZOS)
                canvas.paste(image, (box[0] + (box[2] - box[0] - image.width) // 2,
                                     box[1] + (box[3] - box[1] - image.height) // 2))
        elif shape.shape_type == MSO_SHAPE_TYPE.TABLE:
            draw_table(canvas, shape, box)
        elif shape.shape_type in (MSO_SHAPE_TYPE.AUTO_SHAPE, MSO_SHAPE_TYPE.TEXT_BOX, MSO_SHAPE_TYPE.PLACEHOLDER):
            if shape.shape_type == MSO_SHAPE_TYPE.TEXT_BOX:
                draw_text_frame(canvas, box, shape.text_frame)
            else:
                draw_auto_shape(canvas, shape, box)
    return canvas


def build_contact_sheet(images):
    thumb_w, thumb_h = 400, 225
    sheet = Image.new("RGB", (thumb_w * 3, thumb_h * 5), (230, 235, 240))
    for index, image in enumerate(images):
        thumb = image.copy()
        thumb.thumbnail((thumb_w - 8, thumb_h - 8), Image.Resampling.LANCZOS)
        x = (index % 3) * thumb_w + 4
        y = (index // 3) * thumb_h + 4
        sheet.paste(thumb, (x, y))
    sheet.save(CONTACT, dpi=(150, 150))


def main():
    if RENDER_DIR.exists():
        shutil.rmtree(RENDER_DIR)
    RENDER_DIR.mkdir(parents=True)
    prs = Presentation(PPTX)
    images = []
    for index, slide in enumerate(prs.slides, 1):
        image = render_slide(slide, prs.slide_width, prs.slide_height)
        image.save(RENDER_DIR / f"slide_{index:02d}.png", dpi=(150, 150))
        images.append(image)
    images[0].save(PDF, "PDF", resolution=150, save_all=True, append_images=images[1:])
    build_contact_sheet(images)
    print(f"Rendered {len(images)} slides to {RENDER_DIR}")
    print(f"Saved {PDF}")
    print(f"Saved {CONTACT}")


if __name__ == "__main__":
    main()
