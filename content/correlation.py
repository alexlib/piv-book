# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20.1",
#     "matplotlib",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.20.1"
app = marimo.App()

with app.setup:
    import numpy as np
    import marimo as mo
    import matplotlib.pyplot as plt

    SIGMA = 1.2  # particle image: Gaussian, sigma in pixels (diameter ~ 4*sigma)
    MARGIN = 12  # extra pixels generated around the window, so particles can
    # enter and leave it when the flow moves them

    def render(xp, yp, n):
        """Render Gaussian particle images on an n x n window.

        Particle coordinates are in window pixels, so (0, 0) is the centre of
        the top-left pixel. The image is separable, so it is a single matrix
        product of two small Gaussian profile matrices.
        """
        g = np.arange(n)
        ix = np.exp(-((g[None, :] - xp[:, None]) ** 2) / (2 * SIGMA**2))
        iy = np.exp(-((g[None, :] - yp[:, None]) ** 2) / (2 * SIGMA**2))
        return iy.T @ ix

    def make_pair(n, density, dx, dy, noise, seed):
        """A pair of interrogation windows with a known uniform displacement.

        `density` is particles per 32 x 32 pixels. Frame B is frame A with
        every particle moved by (dx, dy); particles that leave the window
        are lost, and particles from the margin may enter it.
        """
        rng = np.random.default_rng(seed)
        side = n + 2 * MARGIN
        count = rng.poisson(density * side**2 / 32**2)
        xp = rng.uniform(-MARGIN, n + MARGIN, count)
        yp = rng.uniform(-MARGIN, n + MARGIN, count)
        a = render(xp, yp, n)
        b = render(xp + dx, yp + dy, n)
        if noise > 0:
            a = a + rng.normal(0, noise, a.shape)
            b = b + rng.normal(0, noise, b.shape)
        return a, b

    def correlate(a, b, pad):
        """Cross-correlation plane via FFT, zero displacement at the centre.

        Without padding the correlation is circular: the window is treated as
        one tile of an infinite periodic image, which is what a plain FFT
        computes. With padding it is the true (linear) correlation.
        """
        n = a.shape[0]
        size = 2 * n if pad else n
        a = a - a.mean()
        b = b - b.mean()
        fa = np.fft.rfft2(a, s=(size, size))
        fb = np.fft.rfft2(b, s=(size, size))
        c = np.fft.irfft2(np.conj(fa) * fb, s=(size, size))
        return np.fft.fftshift(c), size // 2

    def find_peak(c, centre, method):
        """Locate the highest peak; return (dx, dy) relative to the centre."""
        iy, ix = np.unravel_index(np.argmax(c), c.shape)
        h, w = c.shape
        if method == "integer" or not (0 < ix < w - 1 and 0 < iy < h - 1):
            return ix - centre, iy - centre

        def fit(lo, mid, hi, how):
            if how == "gaussian" and min(lo, mid, hi) > 0:
                lo, mid, hi = np.log(lo), np.log(mid), np.log(hi)
                den = lo - 2 * mid + hi
            elif how == "centroid":
                tot = lo + mid + hi
                return (hi - lo) / tot if tot > 0 else 0.0
            else:  # parabolic, and the fallback when a Gaussian fit has log(<=0)
                den = lo - 2 * mid + hi
            return 0.5 * (lo - hi) / den if den != 0 else 0.0

        sx = fit(c[iy, ix - 1], c[iy, ix], c[iy, ix + 1], method)
        sy = fit(c[iy - 1, ix], c[iy, ix], c[iy + 1, ix], method)
        return ix + sx - centre, iy + sy - centre

    def peak_ratio(c):
        """Height of the highest peak over the next-highest one."""
        iy, ix = np.unravel_index(np.argmax(c), c.shape)
        masked = c.copy()
        masked[max(iy - 2, 0) : iy + 3, max(ix - 2, 0) : ix + 3] = -np.inf
        return c[iy, ix] / max(masked.max(), 1e-12)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # The Correlation Peak

    PIV does not track particles. It cuts a small *interrogation window* out
    of two images taken a moment apart, and asks which single shift makes the
    second window look most like the first.

    Inside a window there are tens of particles, all lit the same way and all
    indistinguishable. You cannot say which particle went where, but you can
    say how far the whole pattern moved. The answer is the position of the
    highest peak of the cross-correlation:

    $$
    R(\mathbf{s}) = \sum_{\mathbf{x}} I_1(\mathbf{x})\, I_2(\mathbf{x} + \mathbf{s})
    \qquad
    \hat{\mathbf{s}} = \arg\max_{\mathbf{s}} R(\mathbf{s})
    $$

    $R(\mathbf{s})$ is large when sliding frame 2 back by $\mathbf{s}$ lines
    its particles up with those in frame 1. The velocity vector for this
    window is the position of the tallest peak, divided by the time between
    frames.

    ## Two windows, one shift

    Here is a synthetic pair with a displacement that we chose, so the right
    answer is known. Every particle in frame 2 is frame 1's particle moved by
    the same amount; some leave the window, and some arrive from outside it.
    """)
    return


@app.cell(hide_code=True)
def controls():
    shift_x = mo.ui.slider(-10, 10, step=0.1, value=3.4, label="Shift x (px)", show_value=True)
    shift_y = mo.ui.slider(-10, 10, step=0.1, value=-1.7, label="Shift y (px)", show_value=True)
    window = mo.ui.dropdown(
        options={"16 × 16": 16, "32 × 32": 32, "64 × 64": 64},
        value="32 × 32",
        label="Window size",
    )
    density = mo.ui.slider(1, 30, step=1, value=8, label="Particles per 32×32 px", show_value=True)
    noise = mo.ui.slider(0.0, 0.5, step=0.01, value=0.05, label="Image noise", show_value=True)
    reseed = mo.ui.button(value=0, on_click=lambda v: v + 1, label="New particles")
    mo.vstack([
        mo.hstack([shift_x, shift_y], justify="start", gap=1.5),
        mo.hstack([window, density, noise, reseed], justify="start", gap=1.5),
    ])
    return density, noise, reseed, shift_x, shift_y, window


@app.cell(hide_code=True)
def pair(density, noise, reseed, shift_x, shift_y, window):
    n = window.value
    img_a, img_b = make_pair(
        n, density.value, shift_x.value, shift_y.value, noise.value, 100 + reseed.value
    )
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for _ax, _img, _title in zip(_axes, (img_a, img_b), ("Frame 1", "Frame 2")):
        _ax.imshow(_img, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        _ax.set_title(_title)
        _ax.set_xticks([0, n - 1])
        _ax.set_yticks([0, n - 1])
    _fig.tight_layout()
    _fig
    return img_a, img_b, n


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Can you see the shift by eye? With a dozen particles you probably can;
    with the density set to 1 you probably cannot, and neither can the
    algorithm.

    ## The correlation plane

    Evaluating $R(\mathbf{s})$ for every shift $\mathbf{s}$ gives an image,
    the correlation plane. It is computed with Fourier transforms, because
    correlation in space is multiplication in frequency. The tallest peak
    sits at the displacement.

    Leave padding off and push the shift past half the window. The peak wraps
    around to the opposite side: a plain FFT treats the window as one tile of
    an endless periodic image. Padding with zeros removes the wrap-around, at
    the price of a bigger transform and a weaker peak.
    """)
    return


@app.cell(hide_code=True)
def _():
    pad = mo.ui.switch(value=False, label="Zero-pad (linear correlation)")
    method = mo.ui.dropdown(
        options=["integer", "centroid", "parabolic", "gaussian"],
        value="gaussian",
        label="Peak estimator (see next section)",
    )
    mo.hstack([pad, method], justify="start", gap=1.5)
    return method, pad


@app.cell
def plane(img_a, img_b, method, pad):
    corr, centre = correlate(img_a, img_b, pad.value)
    est_dx, est_dy = find_peak(corr, centre, method.value)
    ratio = peak_ratio(corr)
    return centre, corr, est_dx, est_dy, ratio


@app.cell(hide_code=True)
def correlation_plot(centre, corr, est_dx, est_dy, shift_x, shift_y):
    _fig, _ax = plt.subplots(figsize=(5.2, 5))
    _size = corr.shape[0]
    _lo = -centre - 0.5
    _hi = _size - centre - 0.5
    _im = _ax.imshow(
        corr, cmap="magma", origin="upper", extent=(_lo, _hi, _hi, _lo)
    )
    _ax.plot(shift_x.value, shift_y.value, "o", mfc="none", mec="#7fd1ff", ms=14, mew=1.5, label="true shift")
    _ax.plot(est_dx, est_dy, "+", color="white", ms=12, mew=2, label="estimated")
    _lim = min(centre, 16)
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(_lim, -_lim)
    _ax.set_xlabel("shift x (px)")
    _ax.set_ylabel("shift y (px)")
    _ax.legend(loc="upper right", fontsize=8)
    _fig.colorbar(_im, ax=_ax, shrink=0.8, label="correlation")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(est_dx, est_dy, ratio, shift_x, shift_y):
    _err = float(np.hypot(est_dx - shift_x.value, est_dy - shift_y.value))
    mo.md(
        f"""
    | true shift | measured | error | peak ratio |
    |---|---|---|---|
    | ({shift_x.value:+.2f}, {shift_y.value:+.2f}) | ({est_dx:+.2f}, {est_dy:+.2f}) | {_err:.2f} px | {ratio:.1f} ({"clear" if ratio > 1.5 else "ambiguous"}) |

    The ring marks the true shift and the cross is what the algorithm
    reports. The *peak ratio* compares the tallest peak with the runner-up.
    It is the algorithm's own sense of how much to trust the answer, and it
    falls before the error rises.

    ## Between the pixels

    The plane is sampled on whole pixels, but the true shift rarely is a
    whole number. Fitting a smooth curve through the peak and its two
    neighbours recovers a fraction of a pixel. Change the estimator above:
    with *integer* the error stays near half a pixel, the rounding of the
    shift, and any of the three fits cuts it several times over. At this
    noise level the parabolic and Gaussian fits are hard to tell apart,
    because random particle pairings distort the peak more than its exact
    shape does. The estimators differ in how they bias the result toward
    whole pixels, which will matter when we get to peak-locking.

    ## One window is one sample

    A single window can be lucky or unlucky. The same shift measured on sixty
    different random particle fields gives a spread of answers, and its shape
    depends on everything you set above.
    """
    )
    return


@app.cell
def trials(density, method, noise, pad, shift_x, shift_y, window):
    # The same measurement repeated on 60 independent particle fields.
    errors = []
    for _seed in range(60):
        _a, _b = make_pair(
            window.value, density.value, shift_x.value, shift_y.value, noise.value, 5000 + _seed
        )
        _c, _ce = correlate(_a, _b, pad.value)
        _dx, _dy = find_peak(_c, _ce, method.value)
        errors.append(np.hypot(_dx - shift_x.value, _dy - shift_y.value))
    errors = np.array(errors)
    return (errors,)


@app.cell(hide_code=True)
def error_histogram(errors):
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.hist(np.clip(errors, 0, 5), bins=np.linspace(0, 5, 26), color="#3556b5")
    _ax.axvline(1.0, color="#b3552e", ls="--", lw=1)
    _ax.set_xlabel("displacement error (px), clipped at 5; dashed line = 1 px")
    _ax.set_ylabel("trials (of 60)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(errors):
    _good = errors[errors < 1.0]
    _rms = f"{np.sqrt(np.mean(_good**2)):.2f} px" if len(_good) else "n/a"
    mo.md(
        f"""
    **{100 * len(_good) / len(errors):.0f}%** of the 60 measurements are valid
    (error under one pixel), with an rms error of **{_rms}** among the valid
    ones. To the right of the dashed line the measurement has locked on to the
    wrong peak.

    Drop the particle density, shrink the window to 16 pixels, or raise the
    shift: the valid fraction falls long before any single measurement looks
    wrong. This is why PIV software validates vectors afterwards, and why the
    next chapters ask how many particles a window needs and how far they may
    travel.
    """
    )
    return


if __name__ == "__main__":
    app.run()
