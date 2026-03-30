"""
Chandra OCR inference using local llama.cpp server (GGUF version).
Reuses utilities from the official chandra repo.

Usage:
    python ocr_infer.py <image_or_pdf_path> [--layout] [--markdown] [--pages 1-3,5]
"""

import argparse
import base64
import io
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import requests
from PIL import Image

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

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    raw: str
    token_count: int
    error: bool = False


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
        raw    = data["choices"][0]["message"]["content"]
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
# Output parsing  (from output.py)
# ---------------------------------------------------------------------------

def parse_html_blocks(html: str) -> str:
    """Strip layout wrapper divs, skip headers/footers/blank pages."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        top_divs = soup.find_all("div", recursive=False)
        if not top_divs:
            return html
        out = ""
        for div in top_divs:
            label = div.get("data-label", "")
            if label in ["Blank-Page", "Page-Header", "Page-Footer"]:
                continue
            out += str(div.decode_contents())
        return out
    except ImportError:
        return html


def to_markdown(html: str) -> str:
    try:
        from markdownify import markdownify as md
        return md(html, heading_style="ATX").strip()
    except ImportError:
        return re.sub(r"<[^>]+>", "", html).strip()


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
    args = parser.parse_args()

    prompt_type = "ocr_layout" if args.layout else "ocr"

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

        if args.markdown:
            text = to_markdown(result.raw)
        elif args.layout:
            text = parse_html_blocks(result.raw)
        else:
            text = result.raw

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
