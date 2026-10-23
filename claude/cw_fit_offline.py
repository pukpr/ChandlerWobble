#!/usr/bin/env python3
"""
cw_fit_offline.py
==================
Standalone script to fit a forced Chandler Wobble model to real cw.dat data,
with a train/holdout split and a correlation-based fitting objective.

WHY A CORRELATION-BASED OBJECTIVE, NOT RAW LEAST-SQUARES:
Minimizing raw sum-of-squared-error against an oscillating target lets the
optimizer "cheat" by shrinking the model's amplitude toward zero whenever
its phase isn't well-aligned -- a flat line near the data's mean often has
LOWER squared error than a full-amplitude, wrong-phase oscillation. This
produces a misleadingly bad-looking, nearly-flat model even when a
decent phase-aligned fit exists. Optimizing correlation directly (then
fitting the amplitude scale separately via simple linear regression) does
not have this escape hatch, since correlation is scale-invariant.

USAGE
-----
    python cw_fit_offline.py --data cw.dat --train-start 1962 --train-end 2000 \\
        --holdout-start 1935 --holdout-end 1962 --annual-wobble

Data format: two whitespace-separated columns, (year, value), one row per
sample, e.g. the pukpr/ChandlerWobble cw.dat format:
    1935.0   -0.0207
    1935.05  -0.0195
    ...

REQUIRES: numpy, scipy, matplotlib
    pip install numpy scipy matplotlib
"""

import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cw_physics import (  # noqa: E402
    CHANDLER_PERIOD_DAYS,
    DEFAULT_Q_FACTOR,
    DRACONIC_PERIOD_DAYS,
    DRACONIC_PERIOD_YEARS,
    TROPICAL_YEAR_DAYS,
    forced_response,
)

NODAL_YEARS = 18.6          # lunar nodal regression period (Zotov & Bizouard 2012
                             # find CW excitation amplitude-modulated at this period --
                             # this is real, independently-published, but note it
                             # supports an AMPLITUDE-envelope claim, not this specific
                             # period-setting mechanism; see the discussion document)

DT = 0.01            # integration step (years) -- ~3.65 days, adequate for both
                     # the CW period and the semiannual impulse cadence
Q_FACTOR = DEFAULT_Q_FACTOR


def simulate(params, t_max, use_envelope):
    """
    Integrate the forced CW model.

    params (no envelope):    [phase0]
    params (with envelope):  [phase0, env_amp, env_phase]

    phase0        : phase offset of the draconic carrier (years)
    env_amp       : (if --envelope) depth of 18.6-yr modulation on the drive
    env_phase     : (if --envelope) phase of the 18.6-yr modulation (years)

    Returns (t_array, x_array) -- raw model output, NOT yet amplitude-scaled
    to the data (that's done separately via linear regression -- see fit()).
    """
    if use_envelope:
        phase0, env_amp, env_phase = params
    else:
        phase0, = params

    times = np.arange(int(t_max / DT) + 1) * DT
    return times, forced_response(
        times,
        phase0,
        envelope_amplitude=env_amp if use_envelope else 0.0,
        envelope_phase_years=env_phase if use_envelope else 0.0,
        q_factor=Q_FACTOR,
    )


def regression_design(model_shape, years, use_annual_wobble):
    """Build the linear terms evaluated after choosing the forcing phase."""
    columns = [model_shape, np.ones_like(model_shape)]
    if use_annual_wobble:
        annual_phase = 2 * np.pi * years
        columns.extend((np.sin(annual_phase), np.cos(annual_phase)))
    return np.column_stack(columns)


def fit_window(t_data, x_data, t0, t1, use_envelope, use_annual_wobble,
               seed=1, maxiter=250, popsize=25):
    """
    Fit model parameters on data in [t0, t1) by maximizing correlation
    (global search via differential_evolution), then determine the
    amplitude scale by simple linear regression (closed-form, not part
    of the search -- correlation is scale-invariant so this is exact
    and doesn't need its own optimization).

    Returns: fitted_params, coefficients, model_func(t_query) -> model values
    """
    mask = (t_data >= t0) & (t_data < t1)
    t_fit, x_fit = t_data[mask], x_data[mask]
    if len(t_fit) < 10:
        raise ValueError(f"Too few points in window [{t0},{t1}): {len(t_fit)}")
    sim_start = t_data[0]

    def neg_corr(params):
        tt, xs = simulate(params, t_fit[-1] - sim_start + 1, use_envelope)
        model_interp = np.interp(t_fit - sim_start, tt, xs)
        if np.std(model_interp) < 1e-10:
            return 1.0
        design = regression_design(model_interp, t_fit, use_annual_wobble)
        fitted = design @ np.linalg.lstsq(design, x_fit, rcond=None)[0]
        return -np.corrcoef(fitted, x_fit)[0, 1]

    if use_envelope:
        bounds = [(0.0, DRACONIC_PERIOD_YEARS), (0.0, 1.0), (0.0, 18.6)]
    else:
        bounds = [(0.0, DRACONIC_PERIOD_YEARS)]

    # NOTE: workers=1 (single-process) because neg_corr is a local closure and can't
    # be pickled for multiprocessing. If you want to parallelize, move the objective
    # to a top-level function (pass t_fit/x_fit/use_envelope as extra args) first.
    res = differential_evolution(neg_corr, bounds, maxiter=maxiter, popsize=popsize,
                                  tol=1e-8, seed=seed, workers=1)

    tt, xs_shape = simulate(res.x, t_fit[-1] - sim_start + 1, use_envelope)
    model_shape_at_fit = np.interp(t_fit - sim_start, tt, xs_shape)
    coefficients = np.linalg.lstsq(
        regression_design(model_shape_at_fit, t_fit, use_annual_wobble),
        x_fit,
        rcond=None,
    )[0]

    def model_func(t_query):
        if t_query.min() < sim_start:
            raise ValueError("Model queries cannot precede the data start.")
        tt2, xs2 = simulate(res.x, t_query.max() - sim_start + 1, use_envelope)
        model_shape = np.interp(t_query - sim_start, tt2, xs2)
        return regression_design(model_shape, t_query, use_annual_wobble) @ coefficients

    corr_fit = -res.fun
    return res.x, coefficients, model_func, corr_fit, (t_fit, x_fit)


def metrics(model_vals, data_vals, label):
    resid = model_vals - data_vals
    rms = np.sqrt(np.mean(resid**2))
    corr = np.corrcoef(model_vals, data_vals)[0, 1]
    print(f"  {label}: correlation={corr:.4f}, RMS={rms:.5f} (data RMS={np.std(data_vals):.5f})")
    return corr, rms


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='Path to cw.dat (year, value columns)')
    p.add_argument('--train-start', type=float, required=True)
    p.add_argument('--train-end', type=float, required=True)
    p.add_argument('--holdout-start', type=float, default=None,
                    help='Optional out-of-sample window start (should not overlap training)')
    p.add_argument('--holdout-end', type=float, default=None)
    p.add_argument('--envelope', action='store_true',
                    help='Add an 18.6-year amplitude-modulation term on the forcing '
                         '(motivated by Zotov & Bizouard 2012 -- see script docstring caveat)')
    p.add_argument('--annual-wobble', action='store_true',
                    help='Add a fitted sine/cosine term constrained to an exact 1-year period')
    p.add_argument('--q', type=float, default=None,
                    help='Override Q factor (default 85, An & Ding 2022; try 130 for Chen et al. 2023)')
    p.add_argument('--seed', type=int, default=1, help='Optimizer random seed')
    p.add_argument('--maxiter', type=int, default=250)
    p.add_argument('--out', default='cw_fit_result.png')
    args = p.parse_args()

    global Q_FACTOR
    if args.q is not None:
        Q_FACTOR = args.q
        print(f"Using overridden Q={Q_FACTOR}")

    data = np.loadtxt(args.data) # , delimiter=',')
    t_data, x_data = data[:, 0], data[:, 1]
    print(f"Loaded {len(t_data)} points, {t_data[0]:.2f} to {t_data[-1]:.2f}")
    print(f"Structural params: T_CW={CHANDLER_PERIOD_DAYS}d, Q={Q_FACTOR}, "
          f"Draconic={DRACONIC_PERIOD_DAYS}d")
    print(f"Envelope term: {'ON (18.6-yr)' if args.envelope else 'OFF'}")

    print(f"\nFitting on [{args.train_start}, {args.train_end})  "
          f"(maximizing correlation, global search)...")
    params, coefficients, model_func, corr_fit, (t_fit, x_fit) = fit_window(
        t_data, x_data, args.train_start, args.train_end, args.envelope, args.annual_wobble,
        seed=args.seed, maxiter=args.maxiter)
    print(f"Fitted params: {np.round(params, 4)}")
    print(f"Forced-response scale: {coefficients[0]:.4f}")
    print(f"Offset: {coefficients[1]:.4f}")
    if args.annual_wobble:
        print(f"Annual sin/cos coefficients: {coefficients[2]:.4f}, {coefficients[3]:.4f}")

    model_at_train = model_func(t_fit)
    print("\nResults:")
    metrics(model_at_train, x_fit, f"TRAIN [{args.train_start},{args.train_end})")

    # holdout
    if args.holdout_start is not None and args.holdout_end is not None:
        hmask = (t_data >= args.holdout_start) & (t_data < args.holdout_end)
        t_hold, x_hold = t_data[hmask], x_data[hmask]
        if len(t_hold) > 0:
            model_at_hold = model_func(t_hold)
            metrics(model_at_hold, x_hold, f"HOLDOUT [{args.holdout_start},{args.holdout_end})")

    # plot everything for visual inspection -- DO NOT trust correlation numbers alone,
    # always look at the curve (a low-amplitude/flat model can still show nonzero
    # correlation on noisy data; a visual check catches this immediately)
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(t_data, x_data, color='blue', lw=1, label='CW data', alpha=0.8)
    full_model = model_func(t_data)
    ax.plot(t_data, full_model, color='red', lw=1, label='Model', alpha=0.8)
    ax.axvspan(args.train_start, args.train_end, color='yellow', alpha=0.15, label='train window')
    if args.holdout_start is not None:
        ax.axvspan(args.holdout_start, args.holdout_end, color='green', alpha=0.1, label='holdout window')
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('Polar motion')
    ax.set_title(
        f'CW forced-response fit (train corr={corr_fit:.3f}, '
        f'envelope={args.envelope}, annual={args.annual_wobble})'
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"\nPlot saved to {args.out} -- look at it before trusting the numbers above.")


if __name__ == '__main__':
    main()
