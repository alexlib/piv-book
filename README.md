# Reading PIV

A live book on the principles of particle image velocimetry processing, so
new users can choose parameters and read results with understanding. Each
chapter is a [marimo](https://marimo.io) notebook, published as a static site
with [marimo-book](https://marimobook.org). Chapters run in the browser
(`mode: wasm`), so every control re-runs the analysis. All images are
synthetic with a known displacement, so every step can be broken on purpose.

## Chapters

1. **A PIV Run, Start to Finish** (`content/pipeline.py`): a full OpenPIV
   analysis as six decisions.
2. **The Correlation Peak** (`content/correlation.py`): why one shift per
   window is the measurement, and how it fails.

Planned: interrogation windows and resolution; peak finding and
peak-locking; multi-pass and window deformation; validation; derived
quantities; a parameter-choice checklist.

## Work on it

Everything runs through [uv](https://docs.astral.sh/uv/); dependencies live
in `pyproject.toml`.

```console
uv run marimo-book serve     # live-reload dev server
uv run marimo-book build     # static site into _site/
uv run marimo-book check     # validate book.yml and content
uv run marimo edit content/correlation.py   # edit one chapter as a notebook
```

To add a chapter: write `content/<name>.py`, add it to `toc` in `book.yml`
with `mode: wasm` if it needs live controls, and add any new dependency to
`pyproject.toml` (and to the notebook's PEP 723 header, so it also runs on
molab).

Pushing to `main` deploys to GitHub Pages through
`.github/workflows/deploy.yml`. Set `repo:` in `book.yml` first to enable the
launch buttons.
