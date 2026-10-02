# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20.1",
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "openpiv",
# ]
# ///

import marimo

__generated_with = "0.20.1"
app = marimo.App()

with app.setup:
    import numpy as np
    import marimo as mo
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from openpiv import pyprocess, validation, filters

    # Ground truth for the synthetic pair, in image coordinates (x right,
    # y DOWN, which is what OpenPIV returns): particles move 5 px right and
    # 3 px up, so v is negative.
    TRUE_U, TRUE_V = 5.0, -3.0

    def synthetic_pair(h=256, w=256, n=1500, seed=0):
        """Seeded particle images with a known uniform shift.

        The browser cannot read local image files, so the book uses a
        synthetic pair everywhere. Frame B is frame A rolled by the truth.
        """
        rng = np.random.default_rng(seed)
        img = np.zeros((h, w))
        img[rng.integers(0, h, n), rng.integers(0, w, n)] = 255.0
        a = gaussian_filter(img, 1.0)
        b = np.roll(a, shift=(int(TRUE_V), int(TRUE_U)), axis=(0, 1))
        return a.astype(np.int32), b.astype(np.int32)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # A PIV Run, Start to Finish

    One call to [OpenPIV](https://openpiv.readthedocs.io) gives you a vector
    field. Behind it sit six decisions, and each one can quietly spoil the
    result. This chapter is the map for the rest of the book: we run a
    complete analysis on an image pair we built ourselves, so the true motion
    is known and every error is visible. Later chapters take each step apart.

    ## Six decisions

    1. **Load the frames** as numbers that can be correlated.
    2. **Choose the parameters**: window, search area, overlap, and the time
       between frames.
    3. **Cross-correlate** every window to get a displacement.
    4. **Place the vectors** on the grid the windows define.
    5. **Validate** low-confidence vectors and replace them.
    6. **Plot** the field and compare it with what you expected.

    Our images are 256 × 256 pixels with 1500 particles. Every particle moves
    the same way between the frames: five pixels to the right and three
    pixels up. A correct analysis returns exactly that, everywhere.
    """)
    return


@app.cell(hide_code=True)
def frames():
    frame_a, frame_b = synthetic_pair()
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for _ax, _img, _title in zip(_axes, (frame_a, frame_b), ("Frame A", "Frame B")):
        _ax.imshow(_img, cmap="gray")
        _ax.set_title(_title)
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return frame_a, frame_b


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md(
            "The eye cannot see the shift; the correlation can. The browser "
            "cannot read local files, so this book always uses a synthetic "
            "pair. On your own machine, replace it with `imread` calls on your "
            "own frames."
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Choose the parameters

    The *window* is the patch of frame A that gets one vector. The *search
    area* is the patch of frame B it is compared against, so it cannot be
    smaller than the window; here you set a margin on top of the window
    instead. The *overlap* is how far neighbouring windows share pixels, and
    *dt* turns pixels per frame into a velocity. The controls are built so an
    invalid combination cannot be chosen.
    """)
    return


@app.cell(hide_code=True)
def controls():
    window_size = mo.ui.dropdown(options=[16, 32, 64], value=32, label="Window size (px)")
    search_margin = mo.ui.dropdown(options=[0, 6, 16, 32], value=6, label="Search margin (px)")
    overlap_pct = mo.ui.dropdown(options=["25%", "50%", "75%"], value="50%", label="Overlap (% of window)")
    dt = mo.ui.number(start=0.001, stop=0.1, step=0.005, value=0.02, label="dt (s)")
    mo.hstack([window_size, search_margin, overlap_pct, dt], justify="start", gap=1.5)
    return dt, overlap_pct, search_margin, window_size


@app.cell(hide_code=True)
def effective(overlap_pct, search_margin, window_size):
    # Safe by construction: search >= window, overlap < window.
    eff_window = int(window_size.value)
    eff_search = eff_window + int(search_margin.value)
    eff_overlap = int(eff_window * int(overlap_pct.value.strip("%")) / 100)
    return eff_overlap, eff_search, eff_window


@app.cell
def correlate(dt, eff_overlap, eff_search, eff_window, frame_a, frame_b):
    # Decisions 3 and 4: correlate every window, then place the vectors.
    u0, v0, sig2noise = pyprocess.extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=eff_window,
        overlap=eff_overlap,
        dt=dt.value,
        search_area_size=eff_search,
        sig2noise_method="peak2peak",
        backend="scipy",  # the default "auto" tries a Rust extension
    )
    x, y = pyprocess.get_coordinates(
        image_size=frame_a.shape,
        search_area_size=eff_search,
        overlap=eff_overlap,
    )
    return sig2noise, u0, v0, x, y


@app.cell(hide_code=True)
def _(eff_overlap, eff_search, eff_window, sig2noise):
    mo.md(
        f"Window **{eff_window} px**, search area **{eff_search} px**, overlap "
        f"**{eff_overlap} px** give a grid of **{sig2noise.shape[0]} × "
        f"{sig2noise.shape[1]}** vectors. A bigger window gives fewer, more "
        "reliable vectors; a smaller one gives more vectors, each easier to "
        "get wrong."
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Validate the vectors

    Every window reports a signal-to-noise ratio: the height of the
    correlation peak over the next one. Vectors below the threshold are
    flagged and replaced by an average of their neighbours. Raise the
    threshold and more windows are flagged. On clean images flagging does
    little; on real, noisy images it is what keeps a single bad vector from
    poisoning the field.
    """)
    return


@app.cell(hide_code=True)
def _():
    s2n_threshold = mo.ui.slider(1.0, 2.0, step=0.1, value=1.3, label="Signal-to-noise threshold", show_value=True)
    s2n_threshold
    return (s2n_threshold,)


@app.cell
def validate(s2n_threshold, sig2noise, u0, v0):
    flags = validation.sig2noise_val(sig2noise, threshold=s2n_threshold.value)
    u_clean, v_clean = filters.replace_outliers(
        u0, v0, flags, method="localmean", max_iter=3
    )
    return flags, u_clean, v_clean


@app.cell(hide_code=True)
def s2n_plot(flags, s2n_threshold, sig2noise):
    _fig, _ax = plt.subplots(figsize=(5, 4.4))
    _im = _ax.imshow(sig2noise, cmap="viridis")
    _fy, _fx = np.nonzero(flags)
    _ax.plot(_fx, _fy, "x", color="#ff6b4a", ms=9, mew=2, label="flagged")
    _ax.set_xlabel("window column")
    _ax.set_ylabel("window row")
    _ax.set_title(f"Signal-to-noise (threshold {s2n_threshold.value:.1f}); crosses fail")
    if flags.any():
        _ax.legend(loc="upper right", fontsize=8)
    _fig.colorbar(_im, ax=_ax, shrink=0.8, label="peak-to-peak ratio")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The vector field
    """)
    return


@app.cell(hide_code=True)
def field_plot(dt, flags, frame_a, u_clean, v_clean, x, y):
    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.imshow(frame_a, cmap="gray", alpha=0.35)
    # Arrows are displacements in pixels, drawn 3x longer than life.
    _ax.quiver(
        x, y, u_clean, v_clean,
        angles="xy", scale_units="xy", scale=1 / (3 * dt.value),
        color="#3556b5", width=0.004,
    )
    if flags.any():
        _ax.plot(x[flags], y[flags], "o", mfc="none", mec="#ff6b4a", ms=9)
    _ax.set_xlabel("x (px)")
    _ax.set_ylabel("y (px, down)")
    _ax.set_title("Validated field in image coordinates; circles mark replaced vectors")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(dt, flags, u_clean, v_clean):
    _mu = float(np.nanmean(u_clean)) * dt.value
    _mv = float(np.nanmean(v_clean)) * dt.value
    _err = np.hypot(_mu - TRUE_U, _mv - TRUE_V)
    mo.md(
        f"""
    | | u (px) | v (px) |
    |---|---|---|
    | measured | {_mu:+.2f} | {_mv:+.2f} |
    | truth | {TRUE_U:+.2f} | {TRUE_V:+.2f} |

    Error **{_err:.2f} px**; **{int(flags.sum())} of {flags.size}** vectors
    were replaced. Arrows are drawn three times the true length, with y
    pointing down as in the image. Try a 16-pixel window with 75% overlap,
    then a 64-pixel window with no search margin. Both recover the motion on
    this easy pair, but they pay for it in different ways, and the next
    chapter shows where the difference comes from: a single correlation peak.
    """
    )
    return


if __name__ == "__main__":
    app.run()
