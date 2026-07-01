# Thai National ID OCR Pipeline

# Setup

## 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

(Everything else the script uses — `base64`, `io`, `json`, `re`, `urllib`, `pathlib` — is in the Python standard library.)

## 2. Install Ollama

**macOS**
```bash
brew install ollama
```
or download the app from https://ollama.com/download

**Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**
Download the installer from https://ollama.com/download

## 3. Start the Ollama server

Ollama usually installs itself as a background service and starts automatically. If it's not already running:

```bash
ollama serve
```

Leave this running in its own terminal (skip this step if `ollama` was installed as a system service and is already listening).

## 4. Pull the vision model

```bash
ollama pull scb10x/typhoon-ocr1.5-3b:latest
```

This is a ~3B-parameter vision-language model, so expect a multi-GB download. Confirm it's available:

```bash
ollama list
```

Optional sanity check that the model can actually see images:

```bash
ollama run scb10x/typhoon-ocr1.5-3b:latest
```
(type a prompt, or exit with `/bye` — this just confirms the model loads)

## 5. Add your images

Put the ID card photos into the `ids/` folder next to the script (create it if it doesn't exist):

```bash
mkdir -p ids
# copy .jpg / .jpeg / .png files into ids/
```

## 6. Run the pipeline

```bash
python typhoon_test.py
```

Results are written to `ocr_results.json` in the same directory.

## Notes / troubleshooting

- The script calls the Ollama HTTP API at `http://localhost:11434/api/chat`. If you changed Ollama's host/port, update `OLLAMA_URL` in the script.
- If you have limited VRAM, the default `MAX_LONG_EDGE = 1024` keeps images small; lower it further (e.g. `768`) if you hit out-of-memory errors, or raise it if you have headroom and want potentially better OCR accuracy on small text.
- First request to a freshly pulled model can be slow while Ollama loads it into memory — this is normal, later requests are faster.
- If `ocr_image()` reports an `_error` for every file, check that `ollama serve` is running and that `ollama list` shows the model.
  
## Overview

This script extracts structured data (name, ID number, dates, address, etc.) from photos of Thai National ID cards using a local vision-language model served by Ollama. It works in two stages: raw OCR, then deterministic, regex-based field extraction. There is no second LLM call to "structure" the data — an earlier version tried that and found the model would hallucinate or garble fields, especially dates and names, so extraction is now pure rule-based on top of the raw OCR text.

## Pipeline

### Step 1 — Raw OCR (image → text)

Each image is resized (long edge capped at 1024px) and sent to the vision model with a simple instruction: read every visible line of text, Thai and English, and output it raw — no formatting, no JSON. This keeps the model's job in this step narrow and reliable: just transcription, not interpretation.

### Step 2 — Field extraction (text → JSON, rule-based)

The raw OCR text is parsed directly with deterministic regex and keyword-anchored extraction — no model call involved:

- **Keyword anchoring** — Since the OCR engine doesn't reliably emit newlines between fields (the whole card can come back as one unbroken block of text), each field is located by finding its printed label (e.g. `ที่อยู่`, `ศาสนา`, `Name`) and reading a bounded window of text right after it, cut off at whichever comes first: the next label or a max length. When a field does get split across two OCR lines (e.g. `"Name Miss Rungaroon"` / `"Last name Khemthong"`), internal whitespace/newlines are collapsed to a single space so it's still read correctly.
- **ID number** — extracted with a pattern matching the card's digit grouping (`X XXXX XXXXX XX X`), falling back to any 13-digit run in the text.
- **Dates** — Thai dates use the Buddhist calendar and abbreviated month names that OCR often garbles (e.g. `พ.8.` instead of `พ.ย.`). Lookup tables of Thai and English month names/abbreviations, plus a table of common OCR misreads, are used to parse `date_of_issue` and `date_of_expiry` from the text right next to each date's own label — this keeps the parser from matching whichever date happens to appear first if several are present. Buddhist years are converted to the Christian calendar.
- **Date of birth (with cross-check)** — Thai ID cards print the date of birth twice: once in Thai (Buddhist calendar) and once in English (Christian calendar). Both are extracted and parsed independently, giving `date_of_birth_th`, `date_of_birth_en` (raw, as printed), a unified `date_of_birth`, and `date_of_birth_match` — whether the two printed dates agree, as a sanity check on OCR quality.
- **Names** — the Thai name is split into `prefix_th` / `first_name_th` / `last_name_th` using a list of honorifics (`นาย`, `นาง`, `นางสาว`, `น.ส.`, `ด.ช.`, `ด.ญ.`, etc.), inserting a space after the prefix if OCR ran it together with the name. The English name is split into `prefix_en` / `first_name_en` / `last_name_en` the same way (`Mr.`, `Mrs.`, `Miss`, `Ms.`, `Dr.`), with the literal `"Last name"` label stripped out if OCR echoed it.
- **Address** — the full registered address is captured as printed (`address`), and `tambon`, `amphoe`, and `province` are additionally split back out of it using the `ต./ตำบล/แขวง`, `อ./อำเภอ/เขต`, and `จ./จังหวัด` markers respectively (Bangkok addresses, which use `แขวง`/`เขต` and often print `กรุงเทพมหานคร` with no `จ.` marker, are handled as well).
- **Whitespace** — all string fields are trimmed.

### Error handling

- If Step 1 (the OCR call to Ollama) fails, e.g. a network or API error, the record gets an `_error` field describing the failure and an empty `raw_ocr_text`.
- If Step 2 (field extraction) throws an unexpected exception, the record gets an `_error` field, but `raw_ocr_text` is preserved so the raw transcription can still be inspected.

## Output Fields

| Field | Description |
|---|---|
| `id_number` | 13-digit Thai national ID number, digits only (no spaces). |
| `prefix_th` | Thai honorific prefix (e.g. `นาย`, `นาง`, `นางสาว`, `น.ส.`). |
| `first_name_th` | First name in Thai. |
| `last_name_th` | Last name in Thai. |
| `prefix_en` | English title (e.g. `Mr.`, `Mrs.`, `Miss`, `Ms.`, `Dr.`). |
| `first_name_en` | First name in English. |
| `last_name_en` | Last name in English. |
| `date_of_birth_th` | Date of birth as printed in Thai (Buddhist calendar), e.g. `23 ก.ย. 2540`. |
| `date_of_birth_en` | Date of birth as printed in English (Christian calendar), e.g. `23 Sep. 1997`. |
| `date_of_birth` | Unified date of birth, `DD/MM/YYYY`, Christian calendar. |
| `date_of_birth_match` | `true`/`false` if both the Thai and English printed dates were found and compared; `null` if either couldn't be read. |
| `religion` | Religion as printed on the card (e.g. `พุทธ`). |
| `address` | Full registered address in Thai, as printed. |
| `tambon` | Sub-district, split out of `address`. |
| `amphoe` | District, split out of `address`. |
| `province` | Province, split out of `address`. |
| `date_of_issue` | Card issue date, `DD/MM/YYYY`, Christian calendar. |
| `date_of_expiry` | Card expiry date, `DD/MM/YYYY`, Christian calendar. |
| `raw_ocr_text` | The full, unprocessed OCR transcription for this image — always included, useful for debugging any field above. |

Any field that's missing, unreadable, or not present on the card is `null`.

A field appears only if something went wrong:

| Field | Description |
|---|---|
| `_error` | Present if Step 1 or Step 2 failed outright (e.g. network/API error, or an exception during extraction); other fields on the record may be missing or unreliable in this case. |

All results across the processed images are collected into a single `ocr_results.json`, keyed by filename.
