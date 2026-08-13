# Invoice Verification Service

FastAPI service that verifies a user-entered amount against the total on an
uploaded invoice (PDF / JPEG / PNG) using local Tesseract OCR.

## Endpoints

- `GET /health` - service health.
- `POST /verify` - `multipart/form-data` with `invoice_file` and
  `expected_amount`. Returns `{ matched, expected_amount, actual_amount }`.
  A mismatch is still HTTP 200 (`matched: false`); extraction failures map to
  4xx/5xx.

## Run locally

Requires the `tesseract` binary on `PATH` (e.g. Windows: install Tesseract and
add it to PATH; Debian/Ubuntu: `apt-get install tesseract-ocr`).

```
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Run tests:

```
python -m pytest
```

## Deploy (Railway)

Railway builds this with **Nixpacks** - no Dockerfile needed. `nixpacks.toml`
installs the Tesseract system package and starts the server bound to Railway's
`$PORT`:

```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

No secrets or API keys are required. `PORT` is provided by Railway at runtime.
