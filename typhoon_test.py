import base64
import io
import json
import re
import urllib.request
import urllib.error
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required: pip install pillow")

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_DIR   = "ids"
OUTPUT_FILE = "ocr_results.json"
MODEL       = "scb10x/typhoon-ocr1.5-3b:latest"
OLLAMA_URL  = "http://localhost:11434/api/chat"
EXTENSIONS  = {".jpg", ".jpeg", ".png"}

# Max long edge for image, can increase if you have enough GPU vRAM as it will increase tokens (will possibly improve accuracy)
MAX_LONG_EDGE = 1024

# Step 1: Ask the model to dump all visible text
EXTRACT_PROMPT = """\
You are an OCR engine. Read this Thai National ID card image and output every line of text you can see, exactly as printed.
Output raw text only. Preserve both Thai and English lines. No formatting, no JSON, no explanation."""

# Step 2: Ask the model to structure the raw text into a fixed JSON schema
STRUCTURE_PROMPT = """\
You are a data extraction engine. Given raw OCR text from a Thai National ID card, extract the fields into a JSON object.

Field mapping rules:
- "เลขประจำตัวประชาชน" or "Identification Number" → "id_number" (digits only, remove spaces)
- "ชื่อตัวและชื่อสกุล" (Thai name line) → "name_th"
- "Name" + "Last name" (English lines) → combine into "name_en", skip the literal word "Last name" if it appears as a label, format as "Title. Firstname Lastname" with a space after the title/prefix
- "เกิดวันที่" or "Date of Birth" → "date_of_birth" (English date preferred, format DD/MM/YYYY)
- "ศาสนา" → "religion"
- "ที่อยู่" → "address_th"
- "วันออกบัตร" or "Date of Issue" → "date_of_issue" (English date preferred, format DD/MM/YYYY)
- "วันบัตรหมดอายุ" or "Date of Expiry" → "date_of_expiry" (English date preferred, format DD/MM/YYYY)

Rules:
- Return ONLY a valid JSON object, no markdown, no explanation
- If a field is missing or unreadable use null
- For dates prefer the English version if both Thai and English are present
- Remove any MRZ/barcode lines (long digit strings at the bottom)

RAW OCR TEXT:
{raw_text}"""

OPTIONS_EXTRACT = {
    "temperature": 0,
    "num_predict": 800,
    "repeat_penalty": 1.2,
}

OPTIONS_STRUCTURE = {
    "temperature": 0,
    "num_predict": 512,
    "repeat_penalty": 1.1,
}

# ── Thai/English month mappings (copied from PaddleOCR pipeline) ────────────────────

_TH_MONTHS = {
    "มกราคม": "01", "ม.ค.": "01", "ม.ค": "01",
    "กุมภาพันธ์": "02", "ก.พ.": "02", "ก.พ": "02",
    "มีนาคม": "03", "มี.ค.": "03", "มี.ค": "03",
    "เมษายน": "04", "เม.ย.": "04", "เม.ย": "04",
    "พฤษภาคม": "05", "พ.ค.": "05", "พ.ค": "05",
    "มิถุนายน": "06", "มิ.ย.": "06", "มิ.ย": "06",
    "กรกฎาคม": "07", "ก.ค.": "07", "ก.ค": "07",
    "สิงหาคม": "08", "ส.ค.": "08", "ส.ค": "08",
    "กันยายน": "09", "ก.ย.": "09", "ก.ย": "09",
    "ตุลาคม": "10", "ต.ค.": "10", "ต.ค": "10",
    "พฤศจิกายน": "11", "พ.ย.": "11", "พ.ย": "11",
    "ธันวาคม": "12", "ธ.ค.": "12", "ธ.ค": "12",
}

_EN_MONTHS = {
    "january": "01", "jan": "01", "jan.": "01",
    "february": "02", "feb": "02", "feb.": "02",
    "march": "03", "mar": "03", "mar.": "03",
    "april": "04", "apr": "04", "apr.": "04",
    "may": "05",
    "june": "06", "jun": "06", "jun.": "06",
    "july": "07", "jul": "07", "jul.": "07",
    "august": "08", "aug": "08", "aug.": "08",
    "september": "09", "sep": "09", "sep.": "09",
    "october": "10", "oct": "10", "oct.": "10",
    "november": "11", "nov": "11", "nov.": "11",
    "december": "12", "dec": "12", "dec.": "12",
}

# Common OCR misreads for Thai abbreviated months (copied from PaddleOCR pipeline)
_OCR_MONTH_FIXES = {
    "พ.8.": "พ.ย.", "พ.8": "พ.ย",
    "ม.1.": "ม.ค.", "ม.1": "ม.ค",
    "ก.1.": "ก.ค.", "ก.1": "ก.ค",
    "ก.8.": "ก.ย.", "ก.8": "ก.ย",
    "ส.1.": "ส.ค.", "ส.1": "ส.ค",
    "ต.1.": "ต.ค.", "ต.1": "ต.ค",
    "ธ.1.": "ธ.ค.", "ธ.1": "ธ.ค",
    "มิ.8.": "มิ.ย.", "มิ.8": "มิ.ย",
    "มี.1.": "มี.ค.", "มี.1": "มี.ค",
    "เม.8.": "เม.ย.", "เม.8": "เม.ย",
}

_TH_NAME_PREFIXES = (
    "นาย", "นาง", "นางสาว",
    "น.ส.", "น.ส", "นส.", "นส",
    "ด.ช.", "ด.ช", "ด.ญ.", "ด.ญ",
)
_TH_NAME_PREFIXES_NORMALIZED = {
    "นาย": "นาย", "นาง": "นาง", "นางสาว": "นางสาว",
    "น.ส.": "น.ส.", "น.ส": "น.ส.", "นส.": "น.ส.", "นส": "น.ส.",
    "ด.ช.": "ด.ช.", "ด.ช": "ด.ช.", "ด.ญ.": "ด.ญ.", "ด.ญ": "ด.ญ.",
}

# ── Post-processing helpers ───────────────────────────────────────────────────

def fix_name_en(name: str) -> str:
    """Remove literal 'Last name' label and ensure space after title prefix."""
    if not name:
        return name
    # Remove literal OCR label artifacts
    name = re.sub(r"\bLast\s*name\b", "", name, flags=re.IGNORECASE).strip()
    # Ensure space after common English title prefixes (Mr., Mrs., Miss, Ms., Dr.)
    name = re.sub(r"\b(Mr\.|Mrs\.|Miss|Ms\.|Dr\.)(?=\S)", r"\1 ", name)
    # Collapse multiple spaces
    name = re.sub(r" {2,}", " ", name).strip()
    return name

def fix_name_th(name: str) -> str:
    """Ensure a space after common Thai name prefixes/honorifics."""
    if not name:
        return name
    for prefix in sorted(_TH_NAME_PREFIXES, key=len, reverse=True):
        if name.startswith(prefix):
            normalized = _TH_NAME_PREFIXES_NORMALIZED[prefix]
            rest = name[len(prefix):]
            if rest and not rest.startswith(" "):
                name = normalized + " " + rest
            break
    name = re.sub(r" {2,}", " ", name).strip()
    return name

def fix_ocr_month(text: str) -> str:
    """Fix common OCR misreads in Thai month abbreviations before parsing."""
    for wrong, right in _OCR_MONTH_FIXES.items():
        if wrong in text:
            text = text.replace(wrong, right)
    return text

def parse_thai_date(text: str) -> str:
    """Parse a Thai or English date string → DD/MM/YYYY. Returns '' if unparseable."""
    text = fix_ocr_month(text.strip())

    # Try Thai months (longest match first to avoid partial matches)
    for th_month, mm in sorted(_TH_MONTHS.items(), key=lambda x: -len(x[0])):
        if th_month in text:
            parts = re.findall(r"\d+", text)
            if len(parts) >= 2:
                day = parts[0].zfill(2)
                year_be = int(parts[-1])
                # Convert Buddhist year to Christian year
                year_ce = year_be - 543 if year_be > 2400 else year_be
                return f"{day}/{mm}/{year_ce}"

    # Try English months
    text_lower = text.lower()
    for en_month, mm in sorted(_EN_MONTHS.items(), key=lambda x: -len(x[0])):
        if en_month in text_lower:
            parts = re.findall(r"\d+", text)
            if len(parts) >= 2:
                day = parts[0].zfill(2)
                year = int(parts[-1])
                if year < 100:
                    year += 1900 if year > 50 else 2000
                return f"{day}/{mm}/{year}"

    return ""


def extract_id_number(texts: list[str]) -> str:
    """
    Extract the 13-digit Thai national ID from OCR text lines.
    """
    combined = " ".join(texts)
    clean = re.sub(r"[^0-9\s]", " ", combined)

    # Pattern matching Thai ID card grouping: X XXXX XXXXX XX X
    match = re.search(r"\b(\d\s?\d{4}\s?\d{5}\s?\d{2}\s?\d)\b", clean)
    if match:
        return re.sub(r"\s", "", match.group(1))

    # Fallback: any 13-digit sequence in the text
    digits_only = re.sub(r"\s+", "", clean)
    match = re.search(r"(\d{13})", digits_only)
    if match:
        return match.group(1)

    # Last resort: any single line that has 13 digits after removing non-digit characters
    for text in texts:
        d = re.sub(r"[^0-9]", "", text)
        if len(d) == 13:
            return d

    return ""


def fix_dates_in_result(result: dict, raw_lines: list[str]) -> dict:
    """
    Re-parse date fields from raw OCR lines if the LLM returned garbled dates.
    """
    date_fields = {
        "date_of_birth": ["เกิดวันที่", "วันเกิด", "birth", "Date of Birth"],
        "date_of_issue": ["วันออกบัตร", "Date of Issue"],
        "date_of_expiry": ["วันบัตรหมดอายุ", "วันหมดอายุ", "Date of Expiry"],
    }

    for field, keywords in date_fields.items():
        val = result.get(field)
        # If already a clean DD/MM/YYYY, do nothing
        if val and re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(val)):
            continue

        # Search nearby lines for the keyword and try to parse a date
        found = ""
        for i, line in enumerate(raw_lines):
            if any(kw.lower() in line.lower() for kw in keywords):
                # Try the keyword line itself and the next two lines
                for candidate in raw_lines[i:i+3]:
                    found = parse_thai_date(candidate)
                    if found:
                        break
            if found:
                break

        if found:
            result[field] = found

    return result


def fix_id_number_in_result(result: dict, raw_lines: list[str]) -> dict:
    """Re-extract ID number from raw lines if the LLM missed or garbled it."""
    val = result.get("id_number", "")
    if val and re.match(r"^\d{13}$", str(val).replace(" ", "")):
        result["id_number"] = str(val).replace(" ", "")
        return result

    extracted = extract_id_number(raw_lines)
    if extracted:
        result["id_number"] = extracted

    return result


def post_process(result: dict, raw_text: str) -> dict:
    """
    Apply all post-processing fixes to the LLM-structured result:
    1. Re-extract 13-digit ID
    2. Re-parse any garbled dates using Thai/English month maps
    3. Strip whitespace from all string values
    """
    raw_lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    result = fix_id_number_in_result(result, raw_lines)
    result = fix_dates_in_result(result, raw_lines)

    # ── NEW: fix name_en ──
    if result.get("name_en"):
        result["name_en"] = fix_name_en(result["name_en"])

    # ── NEW: fix name_th ──
    if result.get("name_th"):
        result["name_th"] = fix_name_th(result["name_th"])

    for k, v in result.items():
        if isinstance(v, str):
            result[k] = v.strip()

    return result


# ── Ollama helpers ────────────────────────────────────────────────────────────

# Resize to the max long edge and encode to base64
def resize_and_encode(path: str, max_long_edge: int = MAX_LONG_EDGE) -> tuple[str, tuple]:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_long_edge:
        scale = max_long_edge / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), img.size

# Call Ollama API with messages and options, return the model's response
def call_ollama(messages: list, options: dict) -> str:
    payload = json.dumps({
        "model": MODEL,
        "stream": False,
        "options": options,
        "messages": messages,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content", "").strip()


# Step 1: Pass the image to the model and return the raw extracted text
def extract_raw_text(image_path: str) -> tuple[str, tuple]:
    b64, size = resize_and_encode(image_path)
    raw = call_ollama(
        messages=[{"role": "user", "content": EXTRACT_PROMPT, "images": [b64]}],
        options=OPTIONS_EXTRACT,
    )
    return raw, size


# Step 2: Send raw text (no image) to the model and parse into a JSON dict
def _try_structure_to_json(raw_text: str) -> tuple[dict | None, str]:
    """Single attempt: ask the model to structure raw_text, return (parsed_dict_or_None, raw_response)."""
    prompt = STRUCTURE_PROMPT.format(raw_text=raw_text)
    response = call_ollama(
        messages=[{"role": "user", "content": prompt}],
        options=OPTIONS_STRUCTURE,
    )

    # Strip markdown fences if any
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", response, flags=re.DOTALL).strip()

    # Unwrap quotes if exists, then try to parse as JSON
    if clean.startswith('"') and clean.endswith('"'):
        try:
            clean = json.loads(clean)
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(clean), response
    except json.JSONDecodeError:
        pass

    # Fallback: extract the substring between the first '{' and the last '}'
    # in case the model added stray text/labels before or after the JSON object
    start, end = clean.find("{"), clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(clean[start:end + 1]), response
        except json.JSONDecodeError:
            pass

    return None, response


def build_fallback_result(raw_text: str) -> dict:
    """
    Build a result dict using only deterministic regex extraction from raw OCR
    text, for use when the structuring LLM call fails entirely (e.g. it ignores
    the JSON instruction and returns unrelated text).
    """
    raw_lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    result = {
        "id_number": None,
        "name_th": None,
        "name_en": None,
        "date_of_birth": None,
        "religion": None,
        "address_th": None,
        "date_of_issue": None,
        "date_of_expiry": None,
    }

    # ID number
    result["id_number"] = extract_id_number(raw_lines) or None

    # Thai name line (the honorific often follows a label like "ชื่อตัวและชื่อสกุล")
    for line in raw_lines:
        for prefix in sorted(_TH_NAME_PREFIXES, key=len, reverse=True):
            idx = line.find(prefix)
            if idx != -1:
                result["name_th"] = line[idx:].strip()
                break
        if result["name_th"]:
            break

    # English name (look for a "Name" line and an optional following "Last name" line)
    for i, line in enumerate(raw_lines):
        if re.match(r"^Name\b", line, re.IGNORECASE):
            parts = [line]
            if i + 1 < len(raw_lines) and "last name" in raw_lines[i + 1].lower():
                parts.append(raw_lines[i + 1])
            combined = re.sub(r"^Name\b", "", " ".join(parts), flags=re.IGNORECASE).strip()
            result["name_en"] = combined or None
            break

    # Religion
    for line in raw_lines:
        if "ศาสนา" in line:
            val = line.replace("ศาสนา", "").strip()
            if val:
                result["religion"] = val
            break

    # Address
    for line in raw_lines:
        if "ที่อยู่" in line or re.search(r"หมู่ที่|ต\.|อ\.|จ\.", line):
            result["address_th"] = line
            break

    # Dates, reusing the same keyword-based search as fix_dates_in_result
    result = fix_dates_in_result(result, raw_lines)

    if result.get("name_th"):
        result["name_th"] = fix_name_th(result["name_th"])
    if result.get("name_en"):
        result["name_en"] = fix_name_en(result["name_en"])

    result["_fallback_extraction"] = True
    return result


# Step 2: Send raw text (no image) to the model and parse into a JSON dict.
# Retries once on failure, then falls back to deterministic regex extraction.
def structure_to_json(raw_text: str) -> dict:
    last_response = ""
    for attempt in range(2):
        parsed, last_response = _try_structure_to_json(raw_text)
        if parsed is not None:
            return parsed

    # Both LLM attempts failed to produce valid JSON — fall back to regex extraction
    fallback = build_fallback_result(raw_text)
    fallback["_structured_response"] = last_response
    return fallback


# Run both steps, then apply post-processing fixes
def ocr_image(image_path: str) -> tuple[dict, str]:
    try:
        raw_text, (w, h) = extract_raw_text(image_path)
        print(f"({w}x{h}) raw✓", end=" ", flush=True)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        return {"_error": f"Step 1 failed: {body}"}, ""

    try:
        result = structure_to_json(raw_text)
        print("json✓", end=" ", flush=True)
    except Exception as e:
        return {"_error": f"Step 2 failed: {e}", "_raw": raw_text}, raw_text

    # Step 3: deterministic post-processing on top of LLM output
    if "_parse_error" not in result and "_error" not in result:
        result = post_process(result, raw_text)
        print("fix✓", end=" ", flush=True)

    return result, raw_text


def main():

    # Ensure dirs
    image_dir = Path(IMAGE_DIR)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory '{IMAGE_DIR}' not found.")

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXTENSIONS)

    if not images:
        print(f"No images found in '{IMAGE_DIR}'.")
        return

    print(f"Found {len(images)} image(s). Running OCR with {MODEL}...\n")

    all_results = {}

    # Process each image
    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name} ... ", end="", flush=True)

        fields, _ = ocr_image(str(img_path))
        all_results[img_path.name] = fields

        preview = json.dumps(fields, ensure_ascii=False)[:100]
        print(f"→  {preview}{'...' if len(str(fields)) > 100 else ''}")

    # Write results to output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(all_results, out, ensure_ascii=False, indent=2)

    print(f"\nDone. Results saved to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()