# Browser AI Playground

A barcamp demo of small language models running locally in a browser.
The app is one static [index.html](index.html), with no backend or build step.

## Publish on GitHub Pages

1. Commit and push to `main`.
2. In **Settings > Pages**, choose **Deploy from a branch**, `main`, `/ (root)`.
3. Open <https://robbyfrt.github.io/browser-ai-playground/> once deployment finishes.

No Actions workflow or custom domain is needed. GitHub Pages supplies HTTPS;
the app works at the repository subpath without custom headers.

## Local preview

Serve this folder over HTTP, for example:

```bash
uv run python -m http.server 8080 --bind 127.0.0.1
```

Open <http://127.0.0.1:8080/>. Do not use `file://`.

## Downloads and limits

Inference runs on the device. Libraries and models download from external hosts
(including jsDelivr and Hugging Face); cached models do not guarantee offline
startup. The app has no prompt-upload service or analytics. Exports contain
prompts, answers and device details: review before sharing.

WebGPU needs a compatible browser/GPU and enough memory. Built-in browser AI is
feature-detected and may be unavailable. The CPU/WASM fallback uses one thread
because GitHub Pages does not supply cross-origin isolation headers.

## Optional regression tests

```bash
npm ci
npx playwright install chromium
npm test
```

Node.js 22+ is required only for tests. The existing tests start their own server
at the Pages project path and mock model APIs; they do not download models or
verify real inference performance. The frozen monthly-savings prompt lives in
[tests/fixtures/](tests/fixtures/).
