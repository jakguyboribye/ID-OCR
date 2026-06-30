# Thai National ID OCR Pipeline

## Overview

This script extracts structured data (name, ID number, dates, address, etc.) from photos of Thai National ID cards using a local vision-language model served by Ollama. It works in three stages: raw OCR, LLM structuring, and deterministic post-processing — with a regex-based fallback if the LLM stumbles.

## Pipeline

### Step 1 — Raw OCR (image → text)

Each image is resized (long edge capped at 1024px) and sent to the vision model with a simple instruction: read every visible line of text, Thai and English, and output it raw — no formatting, no JSON. This keeps the model's job in this step narrow and reliable: just transcription, not interpretation.

### Step 2 — Structuring (text → JSON)

The raw OCR text is sent back to the model (no image this time) with a second prompt that defines the target schema and field-mapping rules — e.g. `เลขประจำตัวประชาชน` → `id_number`, the Thai name line → `name_th`, English `Name` + `Last name` lines combined into `name_en`, and so on. The model is asked to return only a JSON object.

This step is retried once if the response can't be parsed as JSON (the model occasionally ignores the instruction and returns something else). If both attempts fail, the script falls back to **Step 3b** below instead of giving up on the record.

### Step 3 — Post-processing (deterministic fixes)

LLMs are good at structuring but unreliable at exact digit/date transcription, so a set of deterministic, regex-based passes run on top of the LLM's JSON to correct the fields that matter most for accuracy:

- **ID number** — re-extracted from the raw OCR text with a pattern matching the card's digit grouping (`X XXXX XXXXX XX X`), falling back to any 13-digit run, used to override the LLM's value if it's missing or malformed.
- **Dates** — Thai dates use the Buddhist calendar and abbreviated month names that OCR often garbles (e.g. `พ.8.` instead of `พ.ย.`). A lookup table of Thai and English month names/abbreviations (plus a table of common OCR misreads) is used to re-parse `date_of_birth`, `date_of_issue`, and `date_of_expiry` straight from the raw text whenever the LLM's value doesn't already look like a clean `DD/MM/YYYY` string, with the Buddhist year converted to the Christian year.
- **Names** — `name_en` has the literal label "Last name" stripped out if the model echoed it, and a space is enforced after English titles (`Mr.`, `Mrs.`, etc). `name_th` gets the same treatment for Thai honorific prefixes (`นาย`, `นาง`, `นางสาว`, `น.ส.`, `ด.ช.`, `ด.ญ.`), inserting a space after the prefix only if one is missing.
- **Whitespace** — all string fields are trimmed.

### Step 3b — Fallback extraction (used only if Step 2 fails twice)

If the structuring model fails to return valid JSON on both attempts, the script doesn't discard the record — it builds one directly from the raw OCR text using the same regex logic as the post-processing step: ID number, dates, religion, address, and the Thai/English name lines are pulled out independently. The result is flagged with `_fallback_extraction: true` and includes `_structured_response` (what the model actually returned) so the failure can be inspected.

## Output Fields

| Field | Description |
|---|---|
| `id_number` | 13-digit Thai national ID number, digits only (no spaces). |
| `name_th` | Full name in Thai, including honorific prefix (e.g. `นาย`, `นาง`, `นางสาว`, `น.ส.`). |
| `name_en` | Full name in English, formatted as `Title. Firstname Lastname`. |
| `date_of_birth` | Date of birth, `DD/MM/YYYY`, Christian calendar. |
| `religion` | Religion as printed on the card (e.g. `พุทธ`). |
| `address_th` | Registered address in Thai. |
| `date_of_issue` | Card issue date, `DD/MM/YYYY`, Christian calendar. |
| `date_of_expiry` | Card expiry date, `DD/MM/YYYY`, Christian calendar. |

Any field that's missing, unreadable, or not present on the card is `null`.

A few extra fields appear only when something went wrong:

| Field | Description |
|---|---|
| `_fallback_extraction` | `true` if the LLM's structuring step failed and the record was built via regex extraction instead (see Step 3b). |
| `_structured_response` | The model's raw, unparseable response — only present when `_fallback_extraction` is `true`, useful for debugging. |
| `_error` | Present if Step 1 or 2 failed outright (e.g. network/API error); the record has no usable fields in this case. |

All results across the processed images are collected into a single `ocr_results.json`, keyed by filename.