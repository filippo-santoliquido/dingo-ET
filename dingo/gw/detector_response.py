"""Frequency-dependent detector response (Earth rotation, finite arm length) from bilby_xG.

Dingo's standard projection uses the long-wavelength antenna patterns of a non-rotating Earth,
evaluated at one time. For signals that last hours (next-generation detectors from a few Hz),
the beam patterns and the arrival-time delay change while the signal sweeps through the band,
and the finite arm length changes the response at high frequency. This module projects
polarizations with bilby_xG's own response (Nishizawa et al. 2009; Baral et al. 2023, 2025):
``bilby_xG.interferometer.compute_wave_frame`` and
``Interferometer.frequency_dependent_antenna_response``, with the 2PN time to merger of
``bilby_xG.utils.calculate_time_to_merger_for_any_mode`` (dominant mode, ``safety=1``, as in
bilby_xG's ``get_detector_response_for_frequency_dependent_antenna_response``). Dingo does not
reimplement the physics; it only passes the arguments.

Settings (``detector_response`` in the data settings of the training):
    earth_rotation_time_delay, earth_rotation_beam_patterns, finite_size: bool
    orbital_motion: must be False. With the Earth's orbit, bilby_xG's masses are those of a
        moving frame and the response is no longer invariant under a shift in time with the
        corresponding rotation in right ascension, which Dingo uses to train at one reference
        time and analyze events at another (``RAToEventFrame``).

The time is passed as the offset ``geocent_time`` from ``ref_time`` (``elapsed_time``), so the
phase never forms an absolute GPS time in double precision; ``ref_time + geocent_time`` only
sets the Earth's orientation.
"""

import numpy as np

RESPONSE_SWITCHES = (
    "earth_rotation_time_delay",
    "earth_rotation_beam_patterns",
    "finite_size",
)


def check_response_settings(settings: dict) -> dict:
    """Validated copy of the ``detector_response`` settings (all switches present, booleans)."""
    settings = dict(settings)
    unknown = set(settings) - set(RESPONSE_SWITCHES) - {"orbital_motion"}
    if unknown:
        raise KeyError(f"Unknown detector_response settings: {sorted(unknown)}.")
    if settings.get("orbital_motion", False):
        raise NotImplementedError(
            "orbital_motion is not supported: it breaks the time / right-ascension "
            "symmetry that Dingo uses to analyze events at a time other than ref_time."
        )
    out = {k: bool(settings.get(k, False)) for k in RESPONSE_SWITCHES}
    out["orbital_motion"] = False
    return out


def _check_interferometers(ifos):
    for ifo in ifos:
        if not hasattr(ifo, "frequency_dependent_antenna_response"):
            raise TypeError(
                f"{ifo.name} is not a bilby_xG Interferometer; the frequency-dependent "
                "response needs bilby_xG detectors."
            )


def detector_strains(
    ifos,
    h_plus,
    h_cross,
    frequencies,
    ra,
    dec,
    psi,
    ref_time,
    geocent_time,
    mass_1,
    mass_2,
    chi_1,
    chi_2,
    settings,
):
    """Strains {detector name: complex array} of one sample at ``frequencies``.

    Parameters
    ----------
    ifos : list of bilby_xG Interferometer
    h_plus, h_cross : complex arrays at ``frequencies``
    frequencies : array, Hz (no zero frequency)
    ra, dec, psi : float, rad (``ra`` in the frame of ``ref_time``)
    ref_time : float, GPS s; the time origin of ``geocent_time``
    geocent_time : float, s, coalescence time at the geocentre relative to ``ref_time``
    mass_1, mass_2 : float, detector-frame masses in Msun (time to merger)
    chi_1, chi_2 : float, aligned spins (time to merger)
    settings : dict, from :func:`check_response_settings`
    """
    from bilby_xG.interferometer import compute_wave_frame
    from bilby_xG.utils import calculate_time_to_merger_for_any_mode

    _check_interferometers(ifos)
    f = np.asarray(frequencies, dtype=np.float64)
    tau = calculate_time_to_merger_for_any_mode(
        f, mass_1, mass_2, chi_1, chi_2, mode=2, safety=1
    )
    time = ref_time + geocent_time  # orientation only
    flags = dict(
        earth_rotation_time_delay=settings["earth_rotation_time_delay"],
        earth_rotation_beam_patterns=settings["earth_rotation_beam_patterns"],
        finite_size=settings["finite_size"],
    )
    frame = compute_wave_frame(
        ra,
        dec,
        time,
        psi,
        f,
        tau,
        earth_rotation=flags["earth_rotation_time_delay"]
        or flags["earth_rotation_beam_patterns"],
    )
    out = {}
    for ifo in ifos:
        fp, fc = ifo.frequency_dependent_antenna_response(
            ra,
            dec,
            time,
            psi,
            f,
            ref_time,
            tau,
            wave_frame=frame,
            orbital_motion=False,
            elapsed_time=geocent_time,
            **flags,
        )
        out[ifo.name] = fp * h_plus + fc * h_cross
    return out


def component_masses(chirp_mass, mass_ratio):
    """Detector-frame component masses from chirp mass and mass ratio (q <= 1)."""
    from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses

    return chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
