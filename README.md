# Thai National ID OCR Pipeline

# Setup

## 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

This needs `Pillow` and `opencv-python` (for the preprocessing step). Everything else the scripts use — `base64`, `io`, `json`, `re`, `urllib`, `pathlib` — is in the Python standard library.

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

Put the ID card photos into the `ids/` folder next to the scripts (create it if it doesn't exist):

```bash
mkdir -p ids
# copy .jpg / .jpeg / .png files into ids/
```

## 6. Run it

There are two ways to run this, depending on whether your source photos need preprocessing first.

**Option A — OCR only, no preprocessing**

```bash
python typhoon_test.py
```

Reads images directly from `test/`, OCRs them, and writes `test-pre.json`.

**Option B — full pipeline (grayscale + sharpen, then OCR)**

```bash
python pipeline.py
```

This is the recommended path for real ID card photos. It:
1. Reads images from `ids/`
2. Converts each to grayscale and sharpens it (via `enhance_gray_sharp` in `step_2.py`), saving the result into `gray/`
3. Runs OCR on the images in `gray/`
4. Writes results to `test-pipe-pre.json`

Both scripts save results incrementally (see **Crash resilience & resuming** below) rather than only at the end.

## Overview

This pipeline extracts structured data (name, ID number, dates, address, etc.) from photos of Thai National ID cards using a local vision-language model served by Ollama. It works in up to three stages: optional image preprocessing, raw OCR, then deterministic, regex-based field extraction. There is no second LLM call to "structure" the data — an earlier version tried that and found the model would hallucinate or garble fields, especially dates and names, so extraction is now pure rule-based on top of the raw OCR text.

## Pipeline

### Step 0 — Preprocessing (pipeline.py only)

Each source image in `ids/` is converted to grayscale and sharpened via `enhance_gray_sharp` (in `step_2.py`), then saved into `gray/` with `_crop` stripped from the filename and `_gray_sharp` appended. This is the folder the OCR step actually reads from when using `pipeline.py`.

### Step 1 — Raw OCR (image → text)

Each image is resized (long edge capped at 1024px) and sent to the vision model with a simple instruction: read every visible line of text, Thai and English, and output it raw — no formatting, no JSON. This keeps the model's job in this step narrow and reliable: just transcription, not interpretation. If the model returns garbled/degenerate output (see **Crash resilience & resuming**), this step retries automatically before giving up.

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

- If Step 1 (the OCR call to Ollama) fails outright, e.g. a network or API error, the record gets an `_error` field describing the failure and an empty `raw_ocr_text`.
- If Step 1 returns garbled/degenerate output and retries don't fix it, the record gets an `_error` field noting this, with the (garbled) `raw_ocr_text` preserved for inspection.
- If Step 2 (field extraction) throws an unexpected exception, the record gets an `_error` field, but `raw_ocr_text` is preserved so the raw transcription can still be inspected.
- Any other unhandled exception while processing a given image (e.g. from `pipeline.py`'s preprocessing step) is also caught per-image and recorded as `_error`, rather than stopping the whole batch.

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
| `_error` | Present if a step failed outright (e.g. network/API error, garbled OCR output after retries, or an exception during preprocessing or extraction); other fields on the record may be missing or unreliable in this case. |

All results across the processed images are collected into a single output JSON, keyed by filename — `test-pre.json` for `typhoon_test.py`, or `test-pipe-pre.json` for `pipeline.py`.

# n8n Integration (Optional)

An example n8n workflow (`ID-OCR.json`) is included to automate the OCR pipeline.

## 1. Install n8n

If you don't already have n8n installed:

```bash
npm install -g n8n
```

or run it directly with:

```bash
npx n8n
```

## 2. Import the Workflow

1. Open the n8n editor.
2. Select **Import from File**.
3. Import the provided `ID-OCR.json` workflow.

## 3. ID-OCR Setup Guide

### Folder Structure

Create the following folder structure in the **same directory as your `.n8n` folder** (typically `C:\Users\<your_username>\` on Windows):

```text
C:\
└── Users\
    └── <your_username>\
        ├── .n8n\
        └── .n8n-files\
            └── ids\
                └── YYYY-MM-DD\   ← Only today's dated folder is processed
```

The workflow looks for a folder named with today's date (`YYYY-MM-DD`). Create a new dated folder each day and place the ID card images to be processed inside it.

### Configure Ollama Credentials

Open the **HTTP Request** node in the workflow and configure its **Ollama** credential to point to your local Ollama instance.

Set the **Base URL** to:

```text
http://127.0.0.1:11434
```

Set the **Edit your path here** Node in n8n to your user path

---

## 4. Start n8n

Open **PowerShell** and run:

```powershell
$env:NODE_FUNCTION_ALLOW_BUILTIN="fs,path"
npx n8n
```

The `NODE_FUNCTION_ALLOW_BUILTIN` environment variable allows the workflow's Code nodes to access the required Node.js built-in modules (`fs` and `path`).

---

## Processing Logic

When the workflow is executed, it will:

1. Scan the `.n8n-files/ids/` directory.
2. Find the subfolder whose name matches today's date (`YYYY-MM-DD`).
3. Process only the image files inside that folder whose filenames **do not** contain the word `processed`.
4. Run each image through the OCR pipeline.
5. Save the extracted structured data.
6. Rename or mark processed files so they are not processed again on subsequent runs.

This allows new ID card images to be added each day without reprocessing previously handled files.