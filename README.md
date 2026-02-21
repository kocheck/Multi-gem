# Multi-gem 🎨

**Batch image generation for macOS using Google Gemini API**

A production-ready Python CLI tool that reads a CSV file of image prompts and generates images in bulk using Google Gemini's image generation models. Supports reference images (per-prompt and shared groups), real-time progress tracking, automatic retry/rate limiting, and organized output with an HTML gallery.

---

## Features

- **Batch processing** from a simple CSV file
- **Two Gemini models** supported:
  - `gemini-2.0-flash-preview-image-generation` — fast, cost-effective, great for high volume
  - `gemini-2.5-flash-preview-05-20` — higher quality, supports 2K/4K resolution, up to 14 reference images
  - `gemini-3-pro-image-preview` — highest quality, 4K resolution, deep reasoning, up to 14 reference images
- **Reference images** — attach per-prompt reference images via CSV, or define named groups in config
- **Real-time progress** with Rich progress bars, ETA, and per-row status
- **Resilient batch processing** — one failed prompt never kills the batch
- **Rate limiting + exponential backoff** — respects Gemini free tier (10 RPM)
- **`--dry-run`** to validate everything before spending API quota
- **`--resume`** to continue an interrupted batch
- **`--preview N`** to inspect parsed config before running
- **HTML gallery** generated automatically after each run
- **Manifest CSV** with per-row status, model used, timestamps, and error messages
- **Thumbnails** auto-generated via Pillow
- **Timestamped output folders** — each run is isolated

---

## Requirements

- **Python 3.11+**
- **macOS** (also works on Linux/Windows)
- A **Google Gemini API key** — get one free at [aistudio.google.com](https://aistudio.google.com/app/apikey)

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/kocheck/Multi-gem.git
cd Multi-gem

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp config.yaml.example config.yaml
```

Open `config.yaml` and set your API key:

```yaml
gemini_api_key: "YOUR_API_KEY_HERE"
```

**Or** use an environment variable (takes precedence over the config file):

```bash
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

> **Security**: Never commit `config.yaml` with a real API key. It's listed in `.gitignore` by default.

### 3. Prepare your CSV

Use `batch_prompts_sample.csv` as a starting point. The only required column is `prompt` — all others are optional.

---

## CSV Schema

| Column | Required | Default | Description |
|---|---|---|---|
| `prompt` | ✅ Yes | — | The image generation prompt |
| `aspect_ratio` | No | `1:1` | One of: `1:1`, `3:4`, `4:3`, `9:16`, `16:9`, `2:3`, `3:2`, `4:5`, `5:4` |
| `resolution` | No | `1K` | `1K`, `2K`, or `4K` (2K/4K only for `gemini-2.5-flash-preview-05-20`) |
| `model` | No | from config | `gemini-2.0-flash-preview-image-generation`, `gemini-2.5-flash-preview-05-20`, or `gemini-3-pro-image-preview` |
| `group` | No | — | Named reference group from `config.yaml` |
| `reference_images` | No | — | Semicolon-separated paths to local reference images |
| `output_filename` | No | auto-generated | Custom filename (no extension needed) |
| `style_instructions` | No | — | Additional style guidance appended to the prompt |
| `negative_prompt` | No | — | Things to avoid in the output |

### Example CSV

```csv
prompt,aspect_ratio,resolution,model,group,reference_images,output_filename,style_instructions,negative_prompt
"A serene Japanese garden at sunset",16:9,1K,gemini-2.0-flash-preview-image-generation,,,japanese_garden,"photorealistic, warm golden hour lighting",
"Modern logo for a coffee shop 'Bean Theory'",1:1,1K,gemini-2.0-flash-preview-image-generation,,,bean_theory_logo,"flat design, bold typography",complex background
"Product photo of leather wallet on marble",4:3,1K,gemini-2.0-flash-preview-image-generation,,./refs/wallet.jpg;./refs/marble.jpg,wallet_hero,"soft studio lighting",
```

---

## Reference Images

### Per-prompt (CSV column)

Add semicolon-separated paths in the `reference_images` column:

```csv
"...",1:1,1K,,,./refs/style1.png;./refs/style2.png,my_image,
```

### Per-group (config.yaml)

Define named groups in `config.yaml` under `reference_groups`:

```yaml
reference_groups:
  brand_style:
    - "./references/brand_palette.png"
    - "./references/brand_guide.png"
  character_anna:
    - "./references/anna_face.jpg"
    - "./references/anna_body.jpg"
```

Then reference the group name in the CSV `group` column:

```csv
"Anna hiking in the mountains",16:9,1K,,character_anna,,anna_hiking,
```

**Group refs + per-prompt refs are merged** (group first, then per-prompt), up to 14 total.

---

## Usage

### Validate before running (recommended first step)

```bash
python gemini_batch.py --csv batch_prompts_sample.csv --dry-run
```

### Preview parsed rows

```bash
python gemini_batch.py --csv batch_prompts_sample.csv --preview 5
```

### Run the batch

```bash
python gemini_batch.py --csv batch_prompts_sample.csv
```

### Resume an interrupted batch

```bash
python gemini_batch.py --csv batch_prompts_sample.csv --resume
```

### Use a custom config or output directory

```bash
python gemini_batch.py --csv batch_prompts_sample.csv --config my_config.yaml --output-dir ./my_output
```

### Debug mode

```bash
python gemini_batch.py --csv batch_prompts_sample.csv --verbose
```

---

## Output Structure

Each batch run creates a timestamped subfolder:

```
output/
└── 2026-02-21_143022/
    ├── manifest.csv        ← per-row status, model, timestamps, errors
    ├── gallery.html        ← browse all generated images in your browser
    ├── images/
    │   ├── japanese_garden.png
    │   ├── bean_theory_logo.png
    │   └── wallet_hero.png
    └── thumbnails/
        ├── japanese_garden_thumb.png
        ├── bean_theory_logo_thumb.png
        └── wallet_hero_thumb.png
```

### manifest.csv columns

| Column | Description |
|---|---|
| `row_index` | Line number from the input CSV |
| `prompt` | Original prompt text |
| `output_filename` | Saved filename |
| `status` | `success` or `error` |
| `error` | Error message if failed |
| `model` | Model used |
| `aspect_ratio` | Aspect ratio used |
| `resolution` | Resolution setting |
| `group` | Reference group used |
| `timestamp` | ISO 8601 completion time |

---

## Configuration Reference

All options in `config.yaml`:

```yaml
# API key (or use GEMINI_API_KEY env var)
gemini_api_key: "YOUR_API_KEY_HERE"

# Model defaults
default_model: "gemini-2.0-flash-preview-image-generation"
# Options:
#   gemini-2.0-flash-preview-image-generation  — fast, cost-effective
#   gemini-2.5-flash-preview-05-20             — higher quality, 2K/4K
#   gemini-3-pro-image-preview                 — highest quality, 4K, reasoning
default_aspect_ratio: "1:1"
default_resolution: "1K"

# Output
output_directory: "./output"
output_format: "png"          # png, jpeg, webp
generate_thumbnails: true
thumbnail_size: 256

# Rate limiting
delay_between_requests: 6     # seconds between requests (free tier: 6+)
max_retries: 3                # retry attempts with exponential backoff

# Reference groups
reference_groups:
  my_group:
    - "./references/ref1.png"
    - "./references/ref2.png"
```

---

## Rate Limits

| Tier | Limit | Recommended `delay_between_requests` |
|---|---|---|
| Free | ~10 RPM | `6` seconds |
| Pay-as-you-go | Higher | `2–3` seconds |

The tool automatically retries on `429 Too Many Requests` with exponential backoff.

---

## Troubleshooting

**"Gemini API key is not set"**
→ Set `gemini_api_key` in `config.yaml` or `export GEMINI_API_KEY=your_key`

**"Reference image path not found"**
→ Paths are relative to where you run the script. Use `--dry-run` to validate all paths first.

**"No image returned"**
→ The prompt may have been blocked by safety filters. Check the manifest.csv `error` column for details. Try rephrasing the prompt.

**Thumbnails not generated**
→ Ensure all dependencies are installed: `pip install -r requirements.txt`

**Rate limit errors (429)**
→ Increase `delay_between_requests` in config.yaml. The free tier allows ~10 RPM.

---

## Project Structure

```
Multi-gem/
├── gemini_batch.py          ← Main CLI entry point
├── core/
│   ├── __init__.py
│   ├── config.py            ← Config loading & validation
│   ├── csv_parser.py        ← CSV parsing & validation
│   ├── generator.py         ← Gemini API interaction
│   ├── output.py            ← File saving, manifest, gallery
│   └── rate_limiter.py      ← Rate limiting & retry logic
├── config.yaml.example      ← Template config (copy to config.yaml)
├── batch_prompts_sample.csv ← Sample CSV
├── requirements.txt
└── README.md
```

---

## Contributing

Issues and PRs welcome. Please ensure all code follows the existing style (type hints, docstrings, PEP 8).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
