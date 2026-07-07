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
IMAGE_DIR   = "test"
OUTPUT_FILE = "test-pre.json"
MODEL       = "scb10x/typhoon-ocr1.5-3b:latest"
OLLAMA_URL  = "http://localhost:11434/api/chat"
EXTENSIONS  = {".jpg", ".jpeg", ".png"}

# Max long edge for image, can increase if you have enough GPU vRAM as it will increase tokens (will possibly improve accuracy)
MAX_LONG_EDGE = 1024

# Ask the model to dump all visible text
EXTRACT_PROMPT = """\
You are an OCR engine. Read this Thai National ID card image and output every line of text you can see, exactly as printed.
Output raw text only. Preserve both Thai and English lines. No formatting, no JSON, no explanation."""

OPTIONS_EXTRACT = {
    "temperature": 0,
    "num_predict": 800,
    "repeat_penalty": 1.2, # Prevent the model from repeating the same line over and over
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
    "ด.ช.", "ด.ช", "ด.ญ.", "ด.ญ","พระ","จ.ส.อ.",
    "พ.ต.ท.","ร.ต.ท.","ว่าที่ ร.ต.","ว่าที่ ร.ต.หญิง","ส.อ.","หม่อมหลวง"
)
_TH_NAME_PREFIXES_NORMALIZED = {
    "นาย": "นาย", "นาง": "นาง", "นางสาว": "นางสาว",
    "น.ส.": "น.ส.", "น.ส": "น.ส.", "นส.": "น.ส.", "นส": "น.ส.",
    "ด.ช.": "ด.ช.", "ด.ช": "ด.ช.", "ด.ญ.": "ด.ญ.", "ด.ญ": "ด.ญ.",
    "พระ":"พระ","พ.ต.ท.":"พ.ต.ท.","ร.ต.ท.":"ร.ต.ท.",
    "ว่าที่ ร.ต.":"ว่าที่ ร.ต.","ว่าที่ ร.ต.หญิง":"ว่าที่ ร.ต.หญิง","ว่าที่ร.ต.":"ว่าที่ ร.ต.","ว่าที่ร.ต.หญิง":"ว่าที่ ร.ต.หญิง",
    "ส.อ.":"ส.อ.","หม่อมหลวง":"หม่อมหลวง","จ.ส.อ.":"จ.ส.อ.","จสอ":"จ.ส.อ."
}

_EN_NAME_PREFIXES = {
    "Mr.": "Mr.",
    "Mrs.": "Mrs.",
    "Miss": "Miss",
    "Ms.": "Ms.",
    "Dr.": "Dr.",
    "Sm 1": "Sm 1",
    "Acting SubLT":"Acting SubLT",
}

_EN_NAME_PREFIXES_NORMALIZED = {
    "Mr.": "Mr.",
    "Mrs.": "Mrs.",
    "Miss": "Miss",
    "Ms.": "Ms.",
    "Dr.": "Dr.",
    "Sm 1": "Sm 1",
    "Smt 1": "Smt 1",
    "Acting Sub.LT": "Acting Sub Lt.",   # ปรับรูปแบบ normalized ตามที่ต้องการจริง
    "Acting Sub Lt.": "Acting Sub Lt.",
}

# เรียงจากยาว -> สั้น ครั้งเดียว ใช้ร่วมกันทั้งสองฟังก์ชัน
_SORTED_PREFIXES = sorted(_EN_NAME_PREFIXES_NORMALIZED, key=len, reverse=True)

# สร้าง regex pattern จากรายการเดียวกัน (escape ทุกตัวกันอักขระพิเศษ เช่น '.')
_PREFIX_PATTERN = "|".join(re.escape(p) for p in _SORTED_PREFIXES)
_PREFIX_RE = re.compile(rf"^({_PREFIX_PATTERN})(?=[.\s]|$)", flags=re.IGNORECASE)


# ── Post-processing helpers ───────────────────────────────────────────────────

def fix_name_en(name: str) -> str:
    if not name:
        return name

    name = name.strip()

    m = _PREFIX_RE.match(name)
    if m:
        matched = m.group(1)
        normalized = _EN_NAME_PREFIXES_NORMALIZED[
            next(p for p in _SORTED_PREFIXES if p.lower() == matched.lower())
        ]
        rest = name[m.end():].lstrip(". ")
        name = f"{normalized} {rest}".strip() if rest else normalized

    name = re.sub(r"\bLast\s*name\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name

def fix_name_th(name: str) -> str:
    """Ensure a space after common Thai name prefixes/honorifics.
    e.g. "นายชีโน่" -> "นาย ชีโน่"""

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


def split_th_name(name: str) -> dict:
    """
    Split a Thai full name (optionally including a title/prefix) into its
    prefix, first name, and last name components.
    e.g. "เคิร์ต โคเบน" -> {"first_name_th": "เคิร์ต", "last_name_th": "โคเบน"}    
    """
    parts_result = {"prefix_th": None, "first_name_th": None, "last_name_th": None}
    if not name:
        return parts_result

    name = fix_name_th(name)
    rest = name
    for prefix in sorted(_TH_NAME_PREFIXES, key=len, reverse=True):
        normalized = _TH_NAME_PREFIXES_NORMALIZED[prefix]
        if name.startswith(normalized):
            parts_result["prefix_th"] = normalized
            rest = name[len(normalized):].strip()
            break

    tokens = rest.split(None, 1)
    if tokens:
        parts_result["first_name_th"] = tokens[0].strip() or None
    if len(tokens) > 1:
        parts_result["last_name_th"] = tokens[1].strip() or None

    return parts_result


def split_en_name(name: str) -> dict:
    parts_result = {
        "prefix_en": None,
        "first_name_en": None,
        "last_name_en": None,
    }

    if not name:
        return parts_result

    name = fix_name_en(name)
    rest = name

    # ใช้ prefix pattern ชุดเดียวกับ fix_name_en แทน hardcode ซ้ำ
    normalized_prefixes = sorted(
        set(_EN_NAME_PREFIXES_NORMALIZED.values()), key=len, reverse=True
    )
    prefix_pattern = "|".join(re.escape(p) for p in normalized_prefixes)
    match = re.match(rf"^({prefix_pattern})\s+(.*)$", name, flags=re.IGNORECASE)

    if match:
        parts_result["prefix_en"] = match.group(1)
        rest = match.group(2).strip()

    tokens = rest.split()

    if tokens:
        parts_result["first_name_en"] = tokens[0].strip() or None

    if len(tokens) > 1:
        parts_result["last_name_en"] = " ".join(tokens[1:]).strip() or None

    return parts_result


def fix_ocr_month(text: str) -> str:
    """Fix common OCR misreads in Thai month abbreviations before parsing.
    e.g. "พ.8." -> "พ.ย.", "ม.1" -> "ม.ค" """

    for wrong, right in _OCR_MONTH_FIXES.items():
        if wrong in text:
            text = text.replace(wrong, right)
    return text


# Anchor regex patterns for Thai and English dates, using the month mappings above.
_TH_MONTH_ALT = "|".join(re.escape(m) for m in sorted(_TH_MONTHS, key=len, reverse=True))
_TH_DATE_PATTERN = re.compile(r"(\d{1,2})\s*(" + _TH_MONTH_ALT + r")\s*(\d{2,4})")

_EN_MONTH_ALT = "|".join(re.escape(m) for m in sorted(_EN_MONTHS, key=len, reverse=True))
_EN_DATE_PATTERN = re.compile(r"(\d{1,2})\s*(" + _EN_MONTH_ALT + r")\.?,?\s*(\d{2,4})", re.IGNORECASE)


def parse_thai_date(text: str) -> str:
    """Parse a Thai or English date string → DD/MM/YYYY. Returns '' if unparseable.

    e.g. "23 ก.ย. 2540" → "23/09/1997", "23 Sep. 1997" → "23/09/1997"
    """
    text = fix_ocr_month(text.strip())

    # Try Thai months first
    m = _TH_DATE_PATTERN.search(text)
    if m:
        day = m.group(1).zfill(2)
        mm = _TH_MONTHS[m.group(2)]
        year_be = int(m.group(3))
        if year_be < 100:
            year_be += 2500
        # Convert Buddhist year to Christian year
        year_ce = year_be - 543 if year_be > 2400 else year_be
        return f"{day}/{mm}/{year_ce}"

    # Try English months
    m = _EN_DATE_PATTERN.search(text)
    if m:
        day = m.group(1).zfill(2)
        mm = _EN_MONTHS[m.group(2).lower()]
        year = int(m.group(3))
        if year < 100:
            year += 1900 if year > 50 else 2000
        return f"{day}/{mm}/{year}"

    return ""


def window_from(text: str, start_idx: int, stop_keywords: tuple = (), max_len: int = 80) -> str | None:
    """
    Return the window of text starting at `start_idx` and extending up to `max_len` characters,
    this is for extracting text after a keyword in the below functions.
    """
    segment = text[start_idx:start_idx + max_len]
    cut = len(segment)
    for sk in stop_keywords:
        pos = segment.find(sk)
        if pos != -1:
            cut = min(cut, pos)
    segment = segment[:cut]
    # OCR output frequently breaks a single field across multiple lines
    # (e.g. "Name Miss Rungaroon\nLast name Khemthong"), so collapse any
    # newlines/whitespace runs inside the window into single spaces before
    # returning it -- otherwise downstream regexes anchored with ^...$ (like
    # the prefix match in split_en_name) silently fail to match past the
    # embedded newline, and everything gets shoved into the wrong field.
    segment = re.sub(r"\s+", " ", segment).strip()
    return segment or None


def extract_after_keyword(text: str, keyword: str, stop_keywords: tuple = (), max_len: int = 80) -> str | None:
    """Find `keyword` in `text` and return a bounded window of text right after it."""
    idx = text.find(keyword)
    if idx == -1:
        return None
    return window_from(text, idx + len(keyword), stop_keywords, max_len)


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


def find_date_near_keyword(text: str, keywords: list[str], window: int = 40) -> str:
    """
    Search `text` for any of the `keywords`, and if found, look for a Thai or English date
    around that keyword within `window` characters. Returns the first parsed date found, or '' if none.
    """
    lower = text.lower()
    for kw in keywords:
        kw_lower = kw.lower()
        search_start = 0
        while True:
            idx = lower.find(kw_lower, search_start)
            if idx == -1:
                break
            segment = text[idx: idx + len(kw) + window]
            found = parse_thai_date(segment)
            if found:
                return found
            search_start = idx + len(kw)
    return ""


def find_date_raw_and_parsed(text: str, keywords: list[str], pattern: re.Pattern, window: int = 40) -> tuple[str, str]:
    """
    Search `text` for any of the `keywords`, and if found, look for a Thai or English date
    around that keyword within `window` characters. Returns a tuple of (raw_date_string, parsed_date_string) or ("", "") if none found.
    """
    lower = text.lower()
    for kw in keywords:
        kw_lower = kw.lower()
        search_start = 0
        while True:
            idx = lower.find(kw_lower, search_start)
            if idx == -1:
                break
            segment = text[idx: idx + len(kw) + window]
            m = pattern.search(fix_ocr_month(segment))
            if m:
                raw = m.group(0).strip()
                parsed = parse_thai_date(raw)
                if parsed:
                    return raw, parsed
            search_start = idx + len(kw)
    return "", ""


def fix_dates_in_result(result: dict, raw_text: str) -> dict:
    """
    Re-parse issue/expiry date fields from the raw OCR text if the LLM
    returned garbled dates.
    """
    date_fields = {
        "date_of_issue": ["วันออกบัตร", "Date of Issue"],
        "date_of_expiry": ["วันบัตรหมดอายุ", "วันหมดอายุ", "Date of Expiry"],
    }

    for field, keywords in date_fields.items():
        val = result.get(field)
        # If already a clean DD/MM/YYYY, do nothing
        if val and re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(val)):
            continue

        found = find_date_near_keyword(raw_text, keywords)
        if found:
            result[field] = found

    return result


_ADDRESS_PART_PATTERNS = {
    "tambon": r"(?:ต\.|ตำบล|แขวง)\s*(\S+)",
    "amphoe": r"(?:อ\.|อำเภอ|เขต)\s*(\S+)",
    "province": r"(?:จ\.|จังหวัด)\s*(\S+)",
}


def extract_address_parts(address: str) -> dict:
    """
    Split tambon/amphoe/province back out of the full address string.
    Markers on the card are abbreviated with no space before the value
    (e.g. "ต.หนองแก อ.หัวหิน จ.ประจวบคีรีขันธ์"), so each part is captured
    as the next non-whitespace token after its marker (ต./ตำบล/แขวง for
    tambon, อ./อำเภอ/เขต for amphoe, จ./จังหวัด for province). Bangkok
    addresses often skip the จ. marker and just print "กรุงเทพมหานคร"
    directly, so that's handled as a fallback for province.
    """
    parts_result = {"tambon": None, "amphoe": None, "province": None}
    if not address:
        return parts_result

    for key, pattern in _ADDRESS_PART_PATTERNS.items():
        m = re.search(pattern, address)
        if m:
            parts_result[key] = m.group(1).strip()

    if not parts_result["province"] and "กรุงเทพมหานคร" in address:
        parts_result["province"] = "กรุงเทพมหานคร"

    return parts_result


def extract_birthdate_info(raw_text: str) -> dict:
    """
    Extract date of birth in both Thai and English form (as printed) plus a
    unified DD/MM/YYYY value, and flag whether the two printed dates agree for sanity checking.
    """
    th_raw, th_parsed = find_date_raw_and_parsed(
        raw_text, ["เกิดวันที่", "วันเกิด"], _TH_DATE_PATTERN
    )
    en_raw, en_parsed = find_date_raw_and_parsed(
        raw_text, ["Date of Birth", "Birth"], _EN_DATE_PATTERN
    )

    match = None
    if th_parsed and en_parsed:
        match = th_parsed == en_parsed

    return {
        "date_of_birth_th": th_raw or None,
        "date_of_birth_en": en_raw or None,
        "date_of_birth": th_parsed or en_parsed or None,
        "date_of_birth_match": match,
    }


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


# Step 2: Extract structured fields from the raw OCR text using deterministic rules
def extract_fields_rule_based(raw_text: str) -> dict:
    """
    Build the structured result dict using only deterministic regex/rule-based
    extraction from the raw OCR text. No LLM call involved here, since the
    structuring LLM call was found to hallucinate/garble fields, especially
    dates and names.
    """
    raw_lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    result = {
        "id_number": None,
        "prefix_th": None,
        "first_name_th": None,
        "last_name_th": None,
        "prefix_en": None,
        "first_name_en": None,
        "last_name_en": None,
        "date_of_birth_th": None,
        "date_of_birth_en": None,
        "date_of_birth": None,
        "date_of_birth_match": None,
        "religion": None,
        "address": None,
        "tambon": None,
        "amphoe": None,
        "province": None,
        "date_of_issue": None,
        "date_of_expiry": None,
    }

    # ID number 
    result["id_number"] = extract_id_number(raw_lines) or None

    # Thai name: anchor on the printed label "ชื่อตัวและชื่อสกุล" and read only the text up to the next label
    name_th_raw = extract_after_keyword(
        raw_text, "ชื่อตัวและชื่อสกุล", stop_keywords=("Name",), max_len=60
    )
    if not name_th_raw:
        # Fallback: locate the first Thai honorific directly and window from there
        for prefix in sorted(_TH_NAME_PREFIXES, key=len, reverse=True):
            idx = raw_text.find(prefix)
            if idx != -1:
                name_th_raw = window_from(raw_text, idx, stop_keywords=("Name",), max_len=60)
                break
    result.update(split_th_name(name_th_raw))

    # English name: anchor on the printed label "Name" and read only the text up to the next label
    name_en_raw = extract_after_keyword(
        raw_text, "Name", stop_keywords=("เกิดวันที่", "Date of Birth"), max_len=60
    )
    result.update(split_en_name(name_en_raw))

    # Religion: anchor on the printed label "ศาสนา" and read only the text up to the next label
    result["religion"] = extract_after_keyword(
        raw_text, "ศาสนา", stop_keywords=("ที่อยู่",), max_len=30
    )

    # Address: anchor on the printed label "ที่อยู่" and read only the text up to the next label
    result["address"] = extract_after_keyword(
        raw_text, "ที่อยู่", stop_keywords=("วันออกบัตร", "Date of Issue"), max_len=200
    )

    # Split tambon/amphoe/province back out of the full address string above
    result.update(extract_address_parts(result["address"]))

    # Date of birth: anchor on the printed label "เกิดวันที่" and read only the text up to the next label
    result.update(extract_birthdate_info(raw_text))

    # Issue / expiry dates (Thai only on the card)
    result = fix_dates_in_result(result, raw_text)

    # Strip whitespace from all string values
    for k, v in result.items():
        if isinstance(v, str):
            result[k] = v.strip()

    return result


# Run the OCR step, then extract structured fields with pure rule-based logic
def ocr_image(image_path: str) -> tuple[dict, str]:
    try:
        raw_text, (w, h) = extract_raw_text(image_path)
        print(f"({w}x{h}) raw✓", end=" ", flush=True)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        return {"_error": f"OCR step failed: {body}", "raw_ocr_text": ""}, ""

    try:
        result = extract_fields_rule_based(raw_text)
        print("fields✓", end=" ", flush=True)
    except Exception as e:
        return {"_error": f"Field extraction failed: {e}", "raw_ocr_text": raw_text}, raw_text

    # Always include the raw OCR text alongside the structured fields
    result["raw_ocr_text"] = raw_text

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

        preview_full = json.dumps(fields, ensure_ascii=False)
        preview = preview_full[:100]
        print(f"→  {preview}{'...' if len(preview_full) > 100 else ''}")

    # Write results to output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(all_results, out, ensure_ascii=False, indent=2)

    print(f"\nDone. Results saved to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()