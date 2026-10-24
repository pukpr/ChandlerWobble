"""Shared Newtonian dynamics for the Chandler Wobble simulations.

The model is a mass-normalized, damped harmonic oscillator:

    x'' + 2*zeta*omega_0*x' + omega_0**2*x = 0

between forcing events.  At each semiannual event, its velocity changes by
an impulse whose magnitude is a unit-RMS squared draconic sinusoid.  This
keeps the forcing discrete while representing the impulse as momentum input,
not as a state-dependent stiffness change.
"""

import numpy as np


TROPICAL_YEAR_DAYS = 365.2422
CHANDLER_PERIOD_DAYS = 432.3
DRACONIC_PERIOD_DAYS = 27.212221
DEFAULT_Q_FACTOR = 85.0
IMPULSE_CADENCE_YEARS = 0.5
DRACONIC_PERIOD_YEARS = DRACONIC_PERIOD_DAYS / TROPICAL_YEAR_DAYS
OMEGA_0 = 2 * np.pi / (CHANDLER_PERIOD_DAYS / TROPICAL_YEAR_DAYS)
OMEGA_DRACONIC = 2 * np.pi / DRACONIC_PERIOD_YEARS
SQUARED_SINUSOID_RMS_SCALE = np.sqrt(8 / 3)


def impulse_magnitude(time, phase_years, envelope_amplitude=0.0,
                      envelope_phase_years=0.0, envelope_period_years=18.6):
    """Return the unit-RMS squared-sinusoid impulse magnitude at ``time``."""
    magnitude = SQUARED_SINUSOID_RMS_SCALE * np.sin(
        OMEGA_DRACONIC * (time + phase_years)
    ) ** 2
    if envelope_amplitude:
        magnitude *= 1 + envelope_amplitude * np.cos(
            2 * np.pi * (time - envelope_phase_years) / envelope_period_years
        )
    return magnitude


def propagate_state(state, elapsed_years, q_factor=DEFAULT_Q_FACTOR):
    """Propagate an underdamped oscillator exactly across a force-free interval."""
    x0, v0 = state
    zeta = 1 / (2 * q_factor)
    decay_rate = zeta * OMEGA_0
    damped_omega = OMEGA_0 * np.sqrt(1 - zeta**2)
    angle = damped_omega * elapsed_years
    decay = np.exp(-decay_rate * elapsed_years)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    sine_coefficient = (v0 + decay_rate * x0) / damped_omega
    x = decay * (x0 * cosine + sine_coefficient * sine)
    v = decay * (
        -decay_rate * (x0 * cosine + sine_coefficient * sine)
        - x0 * damped_omega * sine
        + sine_coefficient * damped_omega * cosine
    )
    return np.array((x, v))


def amplitude_factor(event_index, amp_step, swap=False):
    """Return the inertia-pulse amplitude factor for the event at n/2 years.

    Events at whole years multiply the state by ``1 + amp_step`` and events at
    half years divide it by the same factor (``swap`` reverses this).  The
    alternation modulates the wobble amplitude once a year, which produces the
    1 -/+ CW sidebands of cw.dat as amplitude-modulation products of the CW line.
    Raw IERS polar motion (1973-now) limits such a modulation to about 1%, so
    those cw.dat sidebands appear to come from its processing, not the Earth.
    """
    enlarge = (event_index % 2 == 0) != swap
    return 1 + amp_step if enlarge else 1 / (1 + amp_step)


def forced_response(times, phase_years, impulse_scale=1.0,
                    envelope_amplitude=0.0, envelope_phase_years=0.0,
                    q_factor=DEFAULT_Q_FACTOR, amp_step=0.0, amp_step_swap=False):
    """Evaluate the displacement at nonnegative, monotonically increasing times.

    The oscillator starts from rest at time zero.  Each event at n/2 years
    changes velocity by ``impulse_scale * impulse_magnitude(...)``.  Events
    occur before an observation at the same timestamp.

    With ``amp_step`` nonzero, each event first rescales the state (both
    displacement and velocity) by ``amplitude_factor(...)`` before adding the
    impulse: the inertia pulse acting on the wobble amplitude.  The default 0
    leaves the purely additive model unchanged.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    if len(times) and (times[0] < 0 or np.any(np.diff(times) < 0)):
        raise ValueError("times must be nonnegative and monotonically increasing.")

    response = np.empty_like(times)
    state = np.zeros(2)
    current_time = 0.0
    next_impulse = IMPULSE_CADENCE_YEARS
    for index, target_time in enumerate(times):
        while next_impulse <= target_time:
            state = propagate_state(state, next_impulse - current_time, q_factor)
            if amp_step:
                event_index = int(round(next_impulse / IMPULSE_CADENCE_YEARS))
                state *= amplitude_factor(event_index, amp_step, amp_step_swap)
            state[1] += impulse_scale * impulse_magnitude(
                next_impulse,
                phase_years,
                envelope_amplitude,
                envelope_phase_years,
            )
            current_time = next_impulse
            next_impulse += IMPULSE_CADENCE_YEARS
        state = propagate_state(state, target_time - current_time, q_factor)
        current_time = target_time
        response[index] = state[0]
    return response
