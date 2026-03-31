"""
Chandra OCR inference using local llama.cpp server (GGUF version).
Reuses utilities from the official chandra repo.

Usage:
    python ocr_infer.py <image_or_pdf_path> [--layout] [--markdown] [--pages 1-3,5]
"""

import argparse
import base64
import hashlib
import io
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None

try:
    from markdownify import MarkdownConverter, re_whitespace
except ImportError:  # pragma: no cover - optional dependency
    MarkdownConverter = None
    re_whitespace = None

try:
    import six
except ImportError:  # pragma: no cover - optional dependency
    six = None

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

API_URL       = "http://localhost:8000/v1/chat/completions"
MODEL_NAME    = "chandra-ocr-2.Q8_0.gguf"
MAX_TOKENS    = 12384
MIN_IMAGE_DIM = 1536
IMAGE_DPI     = 192
MIN_PDF_DIM   = 1024
MAX_RETRIES   = 6
BBOX_SCALE    = 1000

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    raw: str
    token_count: int
    error: bool = False


@dataclass
class LayoutBlock:
    bbox: List[int]
    label: str
    content: str


# ---------------------------------------------------------------------------
# Prompts  (from prompts.py)
# ---------------------------------------------------------------------------

ALLOWED_TAGS = [
    "math","br","i","b","u","del","sup","sub","table","tr","td","p","th",
    "div","pre","h1","h2","h3","h4","h5","ul","ol","li","input","a","span",
    "img","hr","tbody","small","caption","strong","thead","big","code","chem",
]
ALLOWED_ATTRIBUTES = [
    "class","colspan","rowspan","display","checked","type","border","value",
    "style","href","alt","align","data-bbox","data-label",
]

PROMPT_ENDING = f"""
Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Use display for block math.
* Tables: Use colspan and rowspan to match table structure.
* Formatting: Maintain consistent formatting with the image.
* Images: Include a description in the alt attribute. Convert charts to data, diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: Join lines into paragraphs using <p>...</p>. Use <br> only when necessary.
* Chemistry: Use <chem>...</chem> tags for chemical formulas.
* Lists: Preserve indents and proper list markers.
* Use the simplest HTML structure that accurately represents the content.
""".strip()

OCR_PROMPT = f"OCR this image to HTML.\n\n{PROMPT_ENDING}"

OCR_LAYOUT_PROMPT = f"""
OCR this image to HTML, arranged as layout blocks. Each layout block should be a div
with the data-bbox attribute (x0 y0 x1 y1, normalized 0-1000) and data-label attribute.

Labels: Caption, Footnote, Equation-Block, List-Group, Page-Header, Page-Footer,
Image, Section-Header, Table, Text, Complex-Block, Code-Block, Form,
Table-Of-Contents, Figure, Chemical-Block, Diagram, Bibliography, Blank-Page

{PROMPT_ENDING}
""".strip()

PROMPT_MAPPING = {
    "ocr": OCR_PROMPT,
    "ocr_layout": OCR_LAYOUT_PROMPT,
}


# ---------------------------------------------------------------------------
# Post-processing helpers (from output.py)
# ---------------------------------------------------------------------------

@lru_cache
def _hash_html(html: str) -> str:
    return hashlib.md5(html.encode("utf-8")).hexdigest()


def get_image_name(html: str, div_idx: int) -> str:
    return f"{_hash_html(html)}_{div_idx}_img.webp"


def extract_images(
    html: str,
    chunks: List[Dict[str, object]],
    image: Image.Image,
) -> Dict[str, Image.Image]:
    if BeautifulSoup is None:
        return {}
    images: Dict[str, Image.Image] = {}
    div_idx = 0
    for chunk in chunks:
        div_idx += 1
        if chunk.get("label") not in {"Image", "Figure"}:
            continue
        img = BeautifulSoup(chunk.get("content", ""), "html.parser").find("img")
        if not img:
            continue
        bbox = chunk.get("bbox", [0, 0, image.width, image.height])
        try:
            block_image = image.crop(bbox)
        except Exception:
            continue
        img_name = get_image_name(html, div_idx)
        images[img_name] = block_image
    return images


def parse_html(
    html: str,
    include_headers_footers: bool = False,
    include_images: bool = True,
) -> str:
    if BeautifulSoup is None:
        return html
    soup = BeautifulSoup(html, "html.parser")
    top_level_divs = soup.find_all("div", recursive=False)
    out_html = ""
    image_idx = 0
    div_idx = 0
    for div in top_level_divs:
        div_idx += 1
        label = div.get("data-label")

        if label == "Blank-Page":
            continue

        if label in {"Page-Header", "Page-Footer"} and not include_headers_footers:
            continue

        if label in {"Image", "Figure"} and not include_images:
            continue

        if label in {"Image", "Figure"}:
            img = div.find("img")
            img_src = get_image_name(html, div_idx)
            if img:
                img["src"] = img_src
                image_idx += 1
            else:
                img = BeautifulSoup(f"<img src='{img_src}'/>", "html.parser")
                div.append(img)

        if label not in {"Image", "Figure"}:
            for img_tag in div.find_all("img"):
                if not img_tag.get("src"):
                    img_tag.decompose()

        if label == "Text" and not re.search(
            "<.+>", str(div.decode_contents()).strip()
        ):
            text_content = str(div.decode_contents()).strip()
            text_content = f"<p>{text_content}</p>"
            div.clear()
            div.append(BeautifulSoup(text_content, "html.parser"))

        content = str(div.decode_contents())
        out_html += content
    return out_html


if MarkdownConverter and re_whitespace and six:

    class Markdownify(MarkdownConverter):
        def __init__(
            self,
            inline_math_delimiters,
            block_math_delimiters,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.inline_math_delimiters = inline_math_delimiters
            self.block_math_delimiters = block_math_delimiters

        def convert_math(self, el, text, parent_tags):
            block = el.has_attr("display") and el["display"] == "block"
            if block:
                return (
                    "\n"
                    + self.block_math_delimiters[0]
                    + text.strip()
                    + self.block_math_delimiters[1]
                    + "\n"
                )
            return (
                " "
                + self.inline_math_delimiters[0]
                + text.strip()
                + self.inline_math_delimiters[1]
                + " "
            )

        def convert_table(self, el, text, parent_tags):
            return "\n\n" + str(el) + "\n\n"

        def convert_a(self, el, text, parent_tags):
            text = self.escape(text)
            text = re.sub(r"([\[\]()])", r"\\\1", text)
            return super().convert_a(el, text, parent_tags)

        def escape(self, text, parent_tags=None):
            text = super().escape(text, parent_tags)
            if self.options.get("escape_dollars", False):
                text = text.replace("$", r"\$")
            return text

        def process_text(self, el, parent_tags=None):
            text = six.text_type(el) or ""
            if not el.find_parent("pre"):
                text = re_whitespace.sub(" ", text)
            if not el.find_parent(["pre", "code", "kbd", "samp", "math"]):
                text = self.escape(text)
            if el.parent.name == "li" and (
                not el.next_sibling or el.next_sibling.name in ["ul", "ol"]
            ):
                text = text.rstrip()
            return text

else:

    Markdownify = None  # type: ignore


def parse_markdown(
    html: str,
    include_headers_footers: bool = False,
    include_images: bool = True,
) -> str:
    cleaned_html = parse_html(
        html,
        include_headers_footers=include_headers_footers,
        include_images=include_images,
    )
    if Markdownify is None:
        return re.sub(r"<[^>]+>", "", cleaned_html).strip()

    md_cls = Markdownify(
        heading_style="ATX",
        bullets="-",
        escape_misc=False,
        escape_underscores=True,
        escape_asterisks=True,
        escape_dollars=True,
        sub_symbol="<sub>",
        sup_symbol="<sup>",
        inline_math_delimiters=("$", "$"),
        block_math_delimiters=("$$", "$$"),
    )
    try:
        markdown = md_cls.convert(cleaned_html)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Error converting HTML to Markdown: {exc}")
        markdown = ""
    return markdown.strip()


def parse_layout(
    html: str,
    image: Image.Image,
    bbox_scale: int = BBOX_SCALE,
) -> List[LayoutBlock]:
    if BeautifulSoup is None:
        return []
    soup = BeautifulSoup(html, "html.parser")
    top_level_divs = soup.find_all("div", recursive=False)
    width, height = image.size
    width_scaler = width / bbox_scale if bbox_scale else 1
    height_scaler = height / bbox_scale if bbox_scale else 1
    layout_blocks: List[LayoutBlock] = []
    for div in top_level_divs:
        label = div.get("data-label")
        if label == "Blank-Page":
            continue

        bbox_attr = div.get("data-bbox")
        try:
            bbox_values = [int(x) for x in bbox_attr.split(" ")]
            assert len(bbox_values) == 4
        except Exception:
            bbox_values = [0, 0, bbox_scale, bbox_scale]

        normalized_bbox = [
            max(0, int(bbox_values[0] * width_scaler)),
            max(0, int(bbox_values[1] * height_scaler)),
            min(int(bbox_values[2] * width_scaler), width),
            min(int(bbox_values[3] * height_scaler), height),
        ]

        if BeautifulSoup is None:
            content = str(div.decode_contents())
        else:
            content_soup = BeautifulSoup(str(div.decode_contents()), "html.parser")
            for tag in content_soup.find_all(attrs={"data-bbox": True}):
                del tag["data-bbox"]
            content = str(content_soup)

        layout_blocks.append(
            LayoutBlock(
                bbox=normalized_bbox,
                label=label or "block",
                content=content,
            )
        )
    return layout_blocks


def parse_chunks(
    html: str,
    image: Image.Image,
    bbox_scale: int = BBOX_SCALE,
) -> List[Dict[str, object]]:
    return [asdict(block) for block in parse_layout(html, image, bbox_scale=bbox_scale)]

# ---------------------------------------------------------------------------
# Utils  (from util.py)
# ---------------------------------------------------------------------------

def scale_to_fit(
    img: Image.Image,
    max_size: Tuple[int, int] = (3072, 2048),
    min_size: Tuple[int, int] = (1792, 28),
    grid_size: int = 28,
) -> Image.Image:
    width, height = img.size
    if width <= 0 or height <= 0:
        return img

    original_ar = width / height
    current_pixels = width * height
    max_pixels = max_size[0] * max_size[1]
    min_pixels = min_size[0] * min_size[1]

    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5

    w_blocks = max(1, round((width * scale) / grid_size))
    h_blocks = max(1, round((height * scale) / grid_size))

    while (w_blocks * h_blocks * grid_size * grid_size) > max_pixels:
        if w_blocks == 1 and h_blocks == 1:
            break
        if w_blocks == 1:
            h_blocks -= 1
            continue
        if h_blocks == 1:
            w_blocks -= 1
            continue
        ar_w_loss = abs(((w_blocks - 1) / h_blocks) - original_ar)
        ar_h_loss = abs((w_blocks / (h_blocks - 1)) - original_ar)
        if ar_w_loss < ar_h_loss:
            w_blocks -= 1
        else:
            h_blocks -= 1

    new_width  = w_blocks * grid_size
    new_height = h_blocks * grid_size
    if (new_width, new_height) == (width, height):
        return img
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def detect_repeat_token(
    predicted_tokens: str,
    base_max_repeats: int = 4,
    window_size: int = 500,
    cut_from_end: int = 0,
    scaling_factor: float = 3.0,
) -> bool:
    if cut_from_end > 0:
        predicted_tokens = predicted_tokens[:-cut_from_end]
    for seq_len in range(1, window_size // 2 + 1):
        candidate_seq = predicted_tokens[-seq_len:]
        max_repeats = int(base_max_repeats * (1 + scaling_factor / seq_len))
        repeat_count = 0
        pos = len(predicted_tokens) - seq_len
        if pos < 0:
            continue
        while pos >= 0:
            if predicted_tokens[pos: pos + seq_len] == candidate_seq:
                repeat_count += 1
                pos -= seq_len
            else:
                break
        if repeat_count > max_repeats:
            return True
    return False


# ---------------------------------------------------------------------------
# Input loading  (from input.py)
# ---------------------------------------------------------------------------

def load_image(filepath: str) -> Image.Image:
    image = Image.open(filepath).convert("RGB")
    if image.width < MIN_IMAGE_DIM or image.height < MIN_IMAGE_DIM:
        scale = MIN_IMAGE_DIM / min(image.width, image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def load_pdf_images(filepath: str, page_range: List[int] = None) -> List[Image.Image]:
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_c
    except ImportError:
        print("pypdfium2 not installed. Run: pip install pypdfium2")
        sys.exit(1)

    doc = pdfium.PdfDocument(filepath)
    doc.init_forms()
    images = []
    for page_idx in range(len(doc)):
        if page_range and page_idx not in page_range:
            continue
        page_obj = doc[page_idx]
        min_page_dim = min(page_obj.get_width(), page_obj.get_height())
        scale_dpi = max((MIN_PDF_DIM / min_page_dim) * 72, IMAGE_DPI)
        pdfium_c.FPDFPage_Flatten(page_obj, pdfium_c.FLAT_NORMALDISPLAY)
        page_obj = doc[page_idx]
        pil_image = page_obj.render(scale=scale_dpi / 72).to_pil().convert("RGB")
        images.append(pil_image)
    doc.close()
    return images


def parse_range_str(range_str: str) -> List[int]:
    page_lst = []
    for part in range_str.split(","):
        if "-" in part:
            start, end = part.split("-")
            page_lst += list(range(int(start), int(end) + 1))
        else:
            page_lst.append(int(part))
    return sorted(set(page_lst))


def load_file(filepath: str, page_range_str: str = None) -> List[Image.Image]:
    page_range = parse_range_str(page_range_str) if page_range_str else None
    if filepath.lower().endswith(".pdf"):
        return load_pdf_images(filepath, page_range)
    return [load_image(filepath)]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _call_api(image: Image.Image, prompt: str, temperature: float = 0.0, top_p: float = 0.1) -> GenerationResult:
    image_b64 = image_to_base64(scale_to_fit(image))
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "top_p": top_p,
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        raw_content = message.get("content", "")

        if isinstance(raw_content, list):
            text_parts: List[str] = []
            for segment in raw_content:
                if not isinstance(segment, dict):
                    continue
                if segment.get("type") in {"text", "output_text"}:
                    text_parts.append(segment.get("text", ""))
            raw_content = "".join(text_parts)

        if (not raw_content or not str(raw_content).strip()) and message.get("reasoning_content"):
            raw_content = message.get("reasoning_content", "")

        raw    = str(raw_content)
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return GenerationResult(raw=raw, token_count=tokens, error=False)
    except Exception as e:
        print(f"  API error: {e}")
        return GenerationResult(raw="", token_count=0, error=True)


def generate(image: Image.Image, prompt_type: str = "ocr") -> GenerationResult:
    prompt = PROMPT_MAPPING[prompt_type]
    result = _call_api(image, prompt)

    for attempt in range(1, MAX_RETRIES + 1):
        has_repeat = detect_repeat_token(result.raw) or (
            len(result.raw) > 50 and detect_repeat_token(result.raw, cut_from_end=50)
        )
        if not result.error and not has_repeat:
            break
        reason = "repeat tokens" if has_repeat else "API error"
        print(f"  Retrying ({attempt}/{MAX_RETRIES}) — {reason}...")
        time.sleep(2 * attempt)
        retry_temp = min(0.2 * attempt, 0.8)
        result = _call_api(image, prompt, temperature=retry_temp, top_p=0.95)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chandra OCR — GGUF via llama.cpp")
    parser.add_argument("file",              help="Image or PDF file path")
    parser.add_argument("--layout",          action="store_true", help="Layout-aware OCR (bounding boxes)")
    parser.add_argument("--markdown",        action="store_true", help="Convert HTML output to Markdown")
    parser.add_argument("--pages",           default=None,        help="PDF page range, e.g. 0-2,4")
    parser.add_argument("--output", "-o",    default=None,        help="Save output to this file")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Write the raw model response without any cleaning or conversions",
    )
    parser.set_defaults(include_images=True, include_headers=False)
    parser.add_argument(
        "--include-images",
        dest="include_images",
        action="store_true",
        help="Include image blocks in the cleaned HTML/Markdown output (default)",
    )
    parser.add_argument(
        "--exclude-images",
        dest="include_images",
        action="store_false",
        help="Strip image/figure blocks from the cleaned HTML/Markdown output",
    )
    parser.add_argument(
        "--include-headers",
        dest="include_headers",
        action="store_true",
        help="Keep page headers/footers in the cleaned output",
    )
    parser.add_argument(
        "--exclude-headers",
        dest="include_headers",
        action="store_false",
        help="Remove page headers/footers (default)",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory to save cropped image blocks (matches reference post-processing)",
    )
    parser.add_argument(
        "--bbox-scale",
        type=int,
        default=BBOX_SCALE,
        help="BBox normalization scale used by the model (default: 1000)",
    )
    args = parser.parse_args()

    prompt_type = "ocr_layout" if args.layout else "ocr"

    images_dir = Path(args.images_dir).expanduser() if args.images_dir else None
    if images_dir:
        images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.file}")
    images = load_file(args.file, args.pages)
    print(f"Pages to process: {len(images)}")

    all_output = []

    for i, image in enumerate(images):
        print(f"\n[Page {i+1}/{len(images)}] Running {prompt_type}...")
        result = generate(image, prompt_type=prompt_type)

        if result.error:
            print(f"  Failed.")
            all_output.append(f"<!-- ERROR on page {i+1} -->")
            continue

        print(f"  Done — tokens used: {result.token_count}")

        cleaned_html: Optional[str] = None
        if not args.raw:
            cleaned_html = parse_html(
                result.raw,
                include_headers_footers=args.include_headers,
                include_images=args.include_images,
            )

        chunks: List[Dict[str, object]] = []
        if args.layout or images_dir is not None:
            chunks = parse_chunks(result.raw, image, bbox_scale=args.bbox_scale)

        if args.layout:
            text = json.dumps(chunks, ensure_ascii=False, indent=2)
        elif args.raw:
            text = result.raw
        elif args.markdown:
            text = parse_markdown(
                result.raw,
                include_headers_footers=args.include_headers,
                include_images=args.include_images,
            )
        else:
            text = cleaned_html if cleaned_html is not None else result.raw

        if images_dir:
            page_dir = images_dir / f"page_{i+1:03d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            extracted = extract_images(result.raw, chunks, image)
            for name, pil_image in extracted.items():
                pil_image.save(page_dir / name)
            if extracted:
                print(f"  Saved {len(extracted)} cropped image(s) -> {page_dir}")

        all_output.append(text)

    final = "\n\n---\n\n".join(all_output)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(final)
        print(f"\nSaved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print(final)


if __name__ == "__main__":
    main()