import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from astropy.table import Table
from slsim.ImageSimulation.image_simulation import simulate_image, rgb_image_from_image_list
import matplotlib.pyplot as plt
from matplotlib import lines as mlines
import corner


#################################################################################################################
## DATA EXTRACTION UTILITIES
#################################################################################################################
# Column names and their meaning: data/lensed-agn_catalog_data_column_descriptions.txt

# References for the relations below (and in main.ipynb; full list in the column descriptions file):
#   Auger et al. (2010), ApJ 724, 511, arXiv:1007.2880 (SLACS X)
#   Suberlak, Ivezic & MacLeod (2021), ApJ 907, 96, arXiv:2012.12907
#   MacLeod et al. (2010), ApJ 721, 1014, arXiv:1004.0276
#   Burke et al. (2021), Science 373, 789, arXiv:2108.05389
#   Oguri & Marshall (2010), MNRAS 405, 2579, arXiv:1001.2037
#   Shen et al. (2011), ApJS 194, 45, arXiv:1006.5178
#   Urry & Padovani (1995), PASP 107, 803, arXiv:astro-ph/9506063

# Auger et al. (2010), Table 2, fit without intrinsic scatter, Chabrier IMF:
# log10(sigma / km s^-1) = a log10(M* / 1e11 Msun) + b. Their sigma is sigma_{e/2} (within R_eff / 2).
AUGER10_SIGMA_MSTAR = (0.24, 2.32)

# DRW scaling relation, log10 f = A + B log10(lambda_rest / 4000 A) + C (M_i + 23) + D log10(M_BH / 1e9 Msun),
# with tau in rest-frame days and SF_inf in mag (MacLeod et al. 2010, eq. 7). Coefficients (A, B, C, D) from
# Suberlak et al. (2021), Table 2, rows "S20, SDSS-PS1"; B is fixed there to the MacLeod et al. (2010) wavelength
# fits. Suberlak et al. fit M_i K-corrected to z=2 (from Shen et al. 2011), the same convention as SLSim's M_i(z=2).
VARIABILITY_RELATIONS = {
    "suberlak21": {"tau": (2.597, 0.17, 0.035, 0.141), "sf_inf": (-0.476, -0.479, 0.118, 0.118)},
}
# Burke et al. (2021), eq. 1: tau_rest = 107 d (M_BH / 1e8 Msun)^0.38
BURKE21_TAU = (107.0, 0.38)
# Scatter of log tau and log SF_inf about the Suberlak et al. (2021) mean at fixed (M_BH, M_i, z). Not a published
# value: MacLeod et al. (2010)'s joint i-band Gaussian conditioned on (M_BH, M_i, z), less their fitting-error share.
VARIABILITY_SCATTER = {"tau_dex": 0.15, "sf_inf_dex": 0.04, "correlation": 0.16}
REFERENCE_ANGSTROM = 4000.0

BAND_FILTERS = {
    **{band: f"lsst2016-{band}" for band in "ugrizy"},
    **{band: f"Roman-{band}" for band in ["F062", "F087", "F106", "F129", "F158", "F184", "F146", "F213"]},
    **{band: f"Euclid-{band}" for band in ["VIS", "Y", "J", "H"]},
}


def band_effective_wavelengths(bands):
    """Effective wavelength [Angstrom] of each band's speclite filter."""
    import speclite.filters
    from slsim.Pipelines import roman_speclite

    roman_speclite.configure_roman_filters()
    return {band: float(speclite.filters.load_filter(BAND_FILTERS[band]).effective_wavelength.value)
            for band in bands}


def stellar_mass_from_vel_disp(vel_disp):
    """Chabrier M* implied by sigma through the Auger et al. (2010, Table 2) sigma-M* relation, inverted."""
    a, b = AUGER10_SIGMA_MSTAR
    return 1e11 * 10 ** ((np.log10(vel_disp) - b) / a)


def _variability_mean(relation, quantity, rest_angstrom, M_i, log_m_bh):
    a, b, c, d = VARIABILITY_RELATIONS[relation][quantity]
    return a + b * np.log10(rest_angstrom / REFERENCE_ANGSTROM) + c * (M_i + 23.0) + d * (log_m_bh - 9.0)


def assign_agn_properties(quasars, bands, seed=None, max_inclination_deg=60.0):
    """Add the AGN properties SLSim does not draw to a host-matched quasar catalog, one draw per quasar.

    Disk inclination is isotropic (uniform in cos i) up to ``max_inclination_deg``, an assumption for type-1
    (unobscured) AGN in the unified model (Urry & Padovani 1995, PASP 107, 803), not a fitted relation. The disk
    position angle is uniform and the spin is 0. The DRW damping timescale and SF_inf are drawn jointly about the
    Suberlak et al. (2021) relation at rest-frame 4000 A and scaled to each band with its wavelength dependence;
    the Burke et al. (2021) timescale mean is kept for comparison.
    The columns travel with each quasar into the SLSim source objects, so every lens of a quasar shares them.
    """
    rng = np.random.default_rng(seed)
    n = len(quasars)
    quasars["inclination_angle"] = np.degrees(
        np.arccos(rng.uniform(np.cos(np.radians(max_inclination_deg)), 1.0, n))
    )
    quasars["agn_disk_position_angle_deg"] = rng.uniform(0.0, 180.0, n)
    quasars["black_hole_spin"] = np.zeros(n)

    M_i = np.asarray(quasars["M_i"], dtype=float)
    log_m_bh = np.asarray(quasars["black_hole_mass_exponent"], dtype=float)
    z = np.asarray(quasars["z"], dtype=float)
    for relation in VARIABILITY_RELATIONS:
        for quantity in ("tau", "sf_inf"):
            quasars[f"log10_{quantity}_{relation}"] = _variability_mean(
                relation, quantity, REFERENCE_ANGSTROM, M_i, log_m_bh
            )
    quasars["log10_tau_burke21"] = np.log10(BURKE21_TAU[0]) + BURKE21_TAU[1] * (log_m_bh - 8.0)

    s = VARIABILITY_SCATTER
    cross = s["correlation"] * s["tau_dex"] * s["sf_inf_dex"]
    scatter = rng.multivariate_normal([0.0, 0.0], [[s["tau_dex"] ** 2, cross], [cross, s["sf_inf_dex"] ** 2]], n)
    quasars["tau_rest_days"] = 10 ** (np.asarray(quasars["log10_tau_suberlak21"]) + scatter[:, 0])
    quasars["sf_inf_mag"] = 10 ** (np.asarray(quasars["log10_sf_inf_suberlak21"]) + scatter[:, 1])
    for band, wavelength in band_effective_wavelengths(bands).items():
        shift = _variability_mean("suberlak21", "sf_inf", wavelength / (1 + z), M_i, log_m_bh) - np.asarray(
            quasars["log10_sf_inf_suberlak21"]
        )
        quasars[f"sf_inf_mag_{band}"] = np.asarray(quasars["sf_inf_mag"]) * 10**shift
    quasars["break_timescale_days"] = 2 * np.pi * np.asarray(quasars["tau_rest_days"]) * (1 + z)
    return quasars


AGN_DRAWN_COLUMNS = [
    "agn_disk_position_angle_deg", "tau_rest_days", "sf_inf_mag",
    "log10_tau_suberlak21", "log10_sf_inf_suberlak21", "log10_tau_burke21", "break_timescale_days",
]


def _sersic_index(light_object):
    """Sersic index of a light profile, unwrapping the ExtendedSource wrapper if present."""
    for _ in range(3):
        if light_object is None:
            break
        if hasattr(light_object, "_n_sersic"):
            return light_object._n_sersic
        light_object = getattr(light_object, "_extended_source", None)
    return np.nan


def _agn_parameter(point_source, name):
    """SLSim keeps the AGN parameters as private Quasar attributes; fall back to the source dictionary."""
    value = getattr(point_source, f"_{name}", None)
    if value is None:
        value = point_source.source_dict.get(name, np.nan)
    return np.nan if value is None else value


def _deflector_pl_slope(deflector):
    """Logarithmic slope of the deflector mass profile (2 = isothermal)."""
    return getattr(getattr(deflector.mass, "_mass", None), "_gamma_pl", np.nan)


def _shape(e1, e2):
    """Ellipticity, axis ratio and position angles (SLSim frame, lenstronomy's (-e1, e2) frame) from e1, e2."""
    e = np.hypot(e1, e2)
    return e, (1 - e) / (1 + e), 0.5 * np.degrees(np.arctan2(e2, e1)), 0.5 * np.degrees(np.arctan2(e2, -e1))


def _slots(values, count):
    out = np.full(count, np.nan)
    values = np.asarray(values, dtype=float)[:count]
    out[: len(values)] = values
    return out


def om10_selected(table, band="i", limit=23.3, min_separation_arcsec=0.5, min_flux_ratio=0.1, max_num_images=5):
    """Oguri & Marshall (2010, Section 2.2) selection: the fainter image of a double (with a flux ratio of at least 0.1)
    or the third-brightest of a quad or naked-cusp (three-image) lens brighter than ``limit``, separation >= 0.5".
    ``limit`` defaults to the LSST i-band 10-sigma single-visit depth of their Table 1."""
    n_images = np.asarray(table["num_images"])
    faint = np.where(n_images >= 3, table[f"third_brightest_image_ps_mag_{band}"],
                     table[f"second_brightest_image_ps_mag_{band}"])
    abs_mu = np.abs(np.stack([np.asarray(table[f"ps_magnification_{k}"], dtype=float)
                              for k in range(1, max_num_images + 1)]))
    flux_ratio = np.nanmin(abs_mu, axis=0) / np.nanmax(abs_mu, axis=0)
    return ((faint <= limit) & (np.asarray(table["max_image_separation_arcsec"]) >= min_separation_arcsec)
            & ((n_images != 2) | (flux_ratio >= min_flux_ratio)))


def _lens_row(lens_system, all_bands, max_num_images, microlensing_band):
    source_class = lens_system.source(0)
    deflector_class = lens_system.deflector
    ps_class = source_class.point_source
    es_class = source_class.extended_source
    agn = ps_class.source_dict
    _, mass_kwargs = lens_system.deflector_mass_model_lenstronomy()
    _, light_kwargs = lens_system.deflector_light_model_lenstronomy(band=None)

    # SLSim orders the images by arrival time; every per-image array below shares that order
    x_images, y_images = (np.asarray(v, dtype=float) for v in lens_system.point_source_image_positions()[0])
    magnifications = np.asarray(lens_system.point_source_magnification()[0], dtype=float)
    arrival_times = np.asarray(lens_system.point_source_arrival_times()[0], dtype=float)
    separations = np.hypot(x_images[:, None] - x_images[None], y_images[:, None] - y_images[None])

    row = {
        "z_S": lens_system.source_redshift_list[0],
        "z_D": lens_system.deflector_redshift,
        "theta_E_arcsec": lens_system.einstein_radius[0],
        "theta_E_lens_model_arcsec": mass_kwargs[0].get("theta_E", np.nan),
        "max_time_delay_days": np.ptp(arrival_times) if len(arrival_times) else np.nan,
        "num_images": int(lens_system.image_number[0]),
        "max_image_separation_arcsec": separations.max() if len(x_images) > 1 else np.nan,
    }

    # --- Point source (AGN) ---
    x_ps, y_ps = source_class.point_source_position
    row.update({
        "x_ps_position_arcsec": x_ps,
        "y_ps_position_arcsec": y_ps,
        "black_hole_mass_exponent": _agn_parameter(ps_class, "black_hole_mass_exponent"),
        "agn_disk_eddington_ratio": _agn_parameter(ps_class, "eddington_ratio"),
        "black_hole_spin": _agn_parameter(ps_class, "black_hole_spin"),
        "agn_disk_inclination_angle": _agn_parameter(ps_class, "inclination_angle"),
        "M_i": agn.get("M_i", np.nan),
        "log_bolometric_luminosity": agn.get("log_bolometric_luminosity", np.nan),
    })
    row.update({name: agn.get(name, np.nan) for name in AGN_DRAWN_COLUMNS})
    row.update({f"sf_inf_mag_{band}": agn.get(f"sf_inf_mag_{band}", np.nan) for band in all_bands})

    # --- Host galaxy (source) light [single Sersic] ---
    row["host_galaxy_type"] = str(agn.get("galaxy_type", ""))
    row["host_stellar_mass"] = _agn_parameter(ps_class, "stellar_mass")
    row["host_velocity_dispersion"] = _agn_parameter(ps_class, "vel_disp")
    if es_class is not None:
        x_host, y_host = es_class.extended_source_position
        e1_host, e2_host = es_class.ellipticity
        r_eff_host = es_class.angular_size
    else:
        x_host = y_host = e1_host = e2_host = r_eff_host = np.nan
    e_host, q_host, phi_host, phi_host_lenstronomy = _shape(e1_host, e2_host)
    row.update({
        "x_host_position_arcsec": x_host,
        "y_host_position_arcsec": y_host,
        "host_light_R_eff_arcsec": r_eff_host,
        "host_light_n_sersic": _sersic_index(es_class),
        "host_light_e1": e1_host,
        "host_light_e2": e2_host,
        "host_light_ellipticity": e_host,
        "host_light_axis_ratio": q_host,
        "host_light_position_angle_deg": phi_host,
        "host_light_position_angle_lenstronomy_deg": phi_host_lenstronomy,
    })

    # --- Deflector light [single Sersic] ---
    e1_light, e2_light, e1_mass, e2_mass = lens_system.deflector_ellipticity()
    e_light, q_light, phi_light, phi_light_lenstronomy = _shape(e1_light, e2_light)
    x_deflector, y_deflector = lens_system.deflector_position
    row.update({
        "deflector_light_R_eff_arcsec": deflector_class.angular_size_light,
        "deflector_light_n_sersic": _sersic_index(deflector_class.light.extended_source),
        "deflector_light_e1": e1_light,
        "deflector_light_e2": e2_light,
        "deflector_light_ellipticity": e_light,
        "deflector_light_axis_ratio": q_light,
        "deflector_light_position_angle_deg": phi_light,
        "deflector_light_position_angle_lenstronomy_deg": phi_light_lenstronomy,
        "x_deflector_light_position_arcsec": light_kwargs[0].get("center_x", x_deflector),
        "y_deflector_light_position_arcsec": light_kwargs[0].get("center_y", y_deflector),
    })

    # --- Deflector mass [EPL + external shear] ---
    e_mass, q_mass, phi_mass, phi_mass_lenstronomy = _shape(e1_mass, e2_mass)
    kappa_ext, gamma1_ext, gamma2_ext = lens_system.los_linear_distortions
    gamma_ext, _, phi_ext, phi_ext_lenstronomy = _shape(gamma1_ext, gamma2_ext)
    stellar_mass = lens_system.deflector_stellar_mass()
    vel_disp = lens_system.deflector_velocity_dispersion()
    stellar_mass_auger10 = stellar_mass_from_vel_disp(vel_disp)
    row.update({
        "x_deflector_mass_position_arcsec": x_deflector,
        "y_deflector_mass_position_arcsec": y_deflector,
        "deflector_stellar_mass": stellar_mass_auger10,
        "deflector_stellar_mass_slsim": stellar_mass,
        "deflector_pl_slope": _deflector_pl_slope(deflector_class),
        "deflector_velocity_dispersion": vel_disp,
        "deflector_mass_e1": e1_mass,
        "deflector_mass_e2": e2_mass,
        "deflector_mass_ellipticity": e_mass,
        "deflector_mass_axis_ratio": q_mass,
        "deflector_mass_position_angle_deg": phi_mass,
        "deflector_mass_position_angle_lenstronomy_deg": phi_mass_lenstronomy,
        "external_shear_gamma1": gamma1_ext,
        "external_shear_gamma2": gamma2_ext,
        "external_shear_amplitude": np.hypot(gamma1_ext, gamma2_ext),
        "external_shear_position_angle_deg": phi_ext,
        "external_shear_position_angle_lenstronomy_deg": phi_ext_lenstronomy,
        "external_convergence": kappa_ext,
    })

    # --- Band-dependent magnitudes ---
    per_image = {}
    for band in all_bands:
        mags = np.asarray(lens_system.point_source_magnitude(band=band, lensed=True)[0], dtype=float)
        ranked = np.sort(mags)
        row[f"unlensed_ps_mag_{band}"] = lens_system.point_source_magnitude(band=band, lensed=False)[0]
        row[f"faintest_image_ps_mag_{band}"] = ranked[-1] if len(ranked) else np.nan
        row[f"third_brightest_image_ps_mag_{band}"] = ranked[2] if len(ranked) > 2 else np.nan
        row[f"second_brightest_image_ps_mag_{band}"] = ranked[1] if len(ranked) > 1 else np.nan
        row[f"brightest_image_ps_mag_{band}"] = ranked[0] if len(ranked) else np.nan
        if es_class is not None:
            row[f"unlensed_host_mag_{band}"] = lens_system.extended_source_magnitude(band=band, lensed=False)[0]
            row[f"lensed_host_mag_{band}"] = lens_system.extended_source_magnitude(band=band, lensed=True)[0]
        else:
            row[f"unlensed_host_mag_{band}"] = row[f"lensed_host_mag_{band}"] = np.nan
        row[f"deflector_mag_{band}"] = lens_system.deflector_magnitude(band=band)
        per_image[f"lensed_ps_mag_{band}"] = mags
        per_image[f"deflector_surface_brightness_{band}"] = deflector_class.surface_brightness(
            x_images, y_images, band=band
        )

    # --- Microlensing: kappa_star spreads the Auger et al. (2010) M* implied by sigma like the deflector light, so
    # the stars match the lens mass; kappa_star_slsim is SLSim's, with the SkyPy M*. Both are capped just below the
    # total convergence, as in SLSim. ---
    _, kappa_tot, shear, shear_angle = lens_system._microlensing_parameters_for_image_positions_single_source(
        band=microlensing_band, source_index=0
    )
    kappa_tot = np.asarray(kappa_tot, dtype=float)
    kappa_star_raw = np.asarray(lens_system.kappa_star(x_images, y_images), dtype=float)
    kappa_star_raw_auger10 = kappa_star_raw * stellar_mass_auger10 / stellar_mass
    per_image.update({
        "time_delay_days": arrival_times,
        "ps_magnification": magnifications,
        "x_ps_image_positions_arcsec": x_images,
        "y_ps_image_positions_arcsec": y_images,
        "kappa_star": np.minimum(kappa_star_raw_auger10, kappa_tot - 1e-6),
        "kappa_star_capped": (kappa_star_raw_auger10 >= kappa_tot - 1e-6).astype(float),
        "kappa_star_slsim": np.minimum(kappa_star_raw, kappa_tot - 1e-6),
        "kappa_star_slsim_capped": (kappa_star_raw >= kappa_tot - 1e-6).astype(float),
        "kappa_tot": kappa_tot,
        "shear": shear,
        "shear_angle": shear_angle,
    })
    for name, values in per_image.items():
        for img_num, value in enumerate(_slots(values, max_num_images), start=1):
            row[f"{name}_{img_num}"] = value
    return row


def _lens_rows(lenses, all_bands, max_num_images, microlensing_band):
    return [_lens_row(lens, all_bands, max_num_images, microlensing_band) for lens in lenses]


def extract_lensed_agn_properties(lens_objects, all_bands=None, max_num_images=5, n_jobs=1, id_prefix="lens"):
    """One row per lens system: system, per-image, AGN, host and deflector properties.

    :param lens_objects: SLSim Lens objects
    :param all_bands: bands to extract magnitudes for (default LSST griz y)
    :param max_num_images: number of per-image slots; empty slots are NaN
    :param id_prefix: lens_id is "<id_prefix>_<row index>"; give each catalog its own prefix (e.g. the survey key) so
        ids are unique across catalogs. Row i is lens_objects[i], so the id also indexes the saved Lens objects.
    :param n_jobs: worker processes; -1 uses every available core. Workers are spawned, not forked, because
        forking after JAX has started its threads can deadlock.
    :return: astropy Table, columns described in data/lensed-agn_catalog_data_column_descriptions.txt
    """
    if all_bands is None:
        all_bands = ["g", "r", "i", "z", "y"]
    # kappa_star uses the deflector light profile in one band; any band in the catalog will do
    microlensing_band = "i" if "i" in all_bands else all_bands[0]
    lens_objects = list(lens_objects)
    if n_jobs == -1:
        n_jobs = len(os.sched_getaffinity(0))
    n_jobs = max(1, min(n_jobs, len(lens_objects) // 20))
    if n_jobs == 1:
        rows = _lens_rows(lens_objects, all_bands, max_num_images, microlensing_band)
    else:
        chunks = np.array_split(np.arange(len(lens_objects)), 4 * n_jobs)
        with ProcessPoolExecutor(n_jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_lens_rows, [lens_objects[i] for i in chunk], all_bands, max_num_images,
                                   microlensing_band) for chunk in chunks if len(chunk)]
            rows = [row for future in futures for row in future.result()]
    table = Table(rows=rows)
    if len(table):
        # SLSim's Lens.generate_id is built from the deflector position at 4 decimals and repeats in large samples
        table.add_column([f"{id_prefix}_{index:06d}" for index in range(len(table))], name="lens_id", index=0)
    if len(table) and "i" in all_bands:
        table["om10_selected"] = om10_selected(table, max_num_images=max_num_images)
    return table

#################################################################################################################
## PLOTTING AND IMAGE GENERATION UTILITIES
#################################################################################################################
def make_multiband_images_and_rgb_image(lens_class, bands=['g', 'r', 'i', 'z', 'y'], 
                                        num_pix=41, coadd_years=10, 
                                        add_noise=True,
                                        rgb_bands=['i', 'r', 'g'], rgb_stretch=0.5,
                                        with_point_source=True, with_source=True, with_deflector=True,
                                        observatory="LSST"
                                        ):
    multiband_image_selected_lens = {}

    # Make multiband images
    for i, band in enumerate(bands):
        if observatory == "LSST":    
            simulated_lens_image = simulate_image(
                lens_class=lens_class,
                band=band,
                num_pix=num_pix, coadd_years=coadd_years,
                add_noise=add_noise,
                observatory=observatory,
                with_point_source=with_point_source,
                with_source=with_source,
                with_deflector=with_deflector,
            )
        if observatory == "Roman":    
            simulated_lens_image = simulate_image(
                lens_class=lens_class,
                band=band,
                num_pix=num_pix,
                add_noise=add_noise,
                observatory=observatory,
                with_point_source=with_point_source,
                with_source=with_source,
                with_deflector=with_deflector,
            )
        if observatory == "Euclid":    
            simulated_lens_image = simulate_image(
                lens_class=lens_class,
                band=band,
                num_pix=num_pix,
                add_noise=add_noise,
                observatory=observatory,
                with_point_source=with_point_source,
                with_source=with_source,
                with_deflector=with_deflector,
            )
        multiband_image_selected_lens[band] = simulated_lens_image

    # Create RGB image using specified bands
    rgb_image = rgb_image_from_image_list(
        image_list=[
            multiband_image_selected_lens[rgb_bands[0]],  # R channel
            multiband_image_selected_lens[rgb_bands[1]],  # G channel
            multiband_image_selected_lens[rgb_bands[2]],  # B channel
        ],
        stretch=rgb_stretch,
    )

    return multiband_image_selected_lens, rgb_image

def plot_montage(lenses, 
                 number_to_plot=100,
                 num_cols=10,
                 bands=['g', 'r', 'i', 'z', 'y'], 
                 rgb_bands=['i', 'r', 'g'], rgb_stretch=0.5,
                 with_point_source=True, with_extended_source=True, with_deflector=True,
                 num_pix=41, coadd_years=10, add_noise=True,
                 plot_title=None,
                 random_seed=None,
                 observatory="LSST"):
    
    # Randomly select a subset of lenses to plot
    if len(lenses) > number_to_plot:
        np.random.seed(random_seed)
        random_idxs = np.random.choice(len(lenses), size=number_to_plot, replace=False)
        lenses_to_plot = [lenses[i] for i in random_idxs]
    else:
        lenses_to_plot = lenses

    all_rgb_images = []

    for lens_class in lenses_to_plot:
        _, rgb_image = make_multiband_images_and_rgb_image(
            lens_class,
            bands=bands,
            num_pix=num_pix,
            coadd_years=coadd_years,
            add_noise=add_noise,
            with_point_source=with_point_source,
            with_source=with_extended_source,
            with_deflector=with_deflector,
            rgb_bands=rgb_bands, 
            rgb_stretch=rgb_stretch,
            observatory=observatory
        )

        all_rgb_images.append(rgb_image)

    # montage of all RGB images in a dynamic grid
    num_images = len(all_rgb_images)
    if num_images == 0:
        print(f"No images to plot for {observatory}.")
        return None
        
    num_rows = int(np.ceil(num_images / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    
    # Handle single plot case
    if num_images == 1 and num_cols == 1:
        axes = np.array([axes])
        
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(all_rgb_images[i], origin='lower')
            ax.axis('off')
        else:
            ax.remove()  # Remove unused subplots
            
    plt.tight_layout()
    if plot_title:
        plt.suptitle(plot_title, fontsize=16)
        plt.subplots_adjust(top=0.92)  # Adjust top to make room for title

    return fig


#################################################################################################################
## PLOTTING: CORNER PLOTS
#################################################################################################################
import numpy as np
import corner
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def plot_survey_corner(survey_data, keys_to_plot, params, latex_labels, range_vals, color_map,
                       title="Lensed Quasars Corner Plot", figsize=(18, 18),
                       smooth=2, smooth1d=None, levels=[0.68, 0.95],
                       text_x=0.65, text_y_start=0.85, text_y_step=0.16,
                       label_fontsize=16, tick_fontsize=14,
                       separate_quads_doubles=True,
                       save_path=None):  # <--- Added save_path argument
    """
    Generates a modular corner plot comparing populations across multiple surveys,
    automatically generating summary text boxes with survey statistics and cuts.

    Parameters
    ----------
    survey_data : dict
        The master dictionary containing all survey parameters, cuts, and the extracted 'catalog'.
    keys_to_plot : list of str
        List of specific survey keys to plot (e.g., ['LSST', 'Roman_Wide']).
    params : list of str
        List of column names to plot from the catalog.
    latex_labels : list of str
        List of LaTeX formatted labels corresponding to `params`.
    range_vals : dict
        Dictionary mapping parameter names to their (min, max) plotting range.
    color_map : dict
        Dictionary specifying colors for each survey. 
    title : str
        Title of the overall figure.
    figsize : tuple
        Size of the figure.
    smooth : int or float
        Smoothing factor for the KDE in the corner plot.
    smooth1d : int, float, or None
        Smoothing factor for the 1D histograms. If None, no smoothing is applied.
    levels : list of floats
        Contour levels to draw.
    text_x : float
        X-coordinate (figure relative, 0 to 1) for the text boxes.
    text_y_start : float
        Y-coordinate for the top-most text box.
    text_y_step : float
        Amount to shift down in Y for each subsequent text box.
    label_fontsize : int
        Font size for the X and Y axis labels.
    tick_fontsize : int
        Font size for the tick marks on the axes.
    separate_quads_doubles : bool
        If True, plots quads and doubles as separate contours.
        If False, plots all lenses (quads + doubles) together as a single contour.
    save_path : str or None
        If provided, the figure will be saved to this file path.
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated corner plot figure.
    """
    figure = plt.figure(figsize=figsize)
    handles = []

    for idx, survey_key in enumerate(keys_to_plot):
        if survey_key not in survey_data:
            print(f"Warning: {survey_key} not found in survey_data. Skipping.")
            continue
            
        data = survey_data[survey_key]
        catalog = data.get("catalog", None)
        
        if catalog is None or len(catalog) == 0:
            print(f"Skipping {survey_key} in corner plot (empty or missing catalog).")
            continue

        cat_doubles = catalog[catalog['num_images'] == 2]
        cat_quads = catalog[catalog['num_images'] == 4]
        
        num_doubles = len(cat_doubles)
        num_quads = len(cat_quads)
        total_lenses = len(catalog)
        q_frac = num_quads / total_lenses if total_lenses > 0 else 0.0

        if separate_quads_doubles:
            c_quad = color_map.get(survey_key, {}).get("quads", "blue")
            c_double = color_map.get(survey_key, {}).get("doubles", "orange")

            # --- Plot Quads ---
            data_quads = np.array([cat_quads[param] for param in params]).T
            if len(data_quads) > 0 and data_quads.shape[1] > 0:
                corner.corner(
                    data_quads,
                    range=[range_vals[param] for param in params],
                    labels=latex_labels,
                    label_kwargs={"fontsize": label_fontsize},
                    color=c_quad,
                    fig=figure,
                    plot_datapoints=False,
                    hist_kwargs={"linewidth": 2, "alpha": 0.9}, 
                    contour_kwargs={"linewidths": 2},
                    smooth=smooth,
                    smooth1d=smooth1d,
                    no_fill_contours=True,
                    levels=levels,
                )
                handles.append(mlines.Line2D([], [], color=c_quad, linewidth=2, label=f'{data["name"]} Quads'))

            # --- Plot Doubles ---
            data_doubles = np.array([cat_doubles[param] for param in params]).T
            if len(data_doubles) > 0 and data_doubles.shape[1] > 0:
                corner.corner(
                    data_doubles,
                    range=[range_vals[param] for param in params],
                    labels=latex_labels,
                    label_kwargs={"fontsize": label_fontsize},
                    color=c_double,
                    fig=figure,
                    plot_datapoints=False,
                    hist_kwargs={"linewidth": 1.5, "linestyle": "--", "alpha": 0.9},
                    contour_kwargs={"linewidths": 1.5, "linestyles": "--"},
                    smooth=smooth,
                    smooth1d=smooth1d,
                    no_fill_contours=True,
                    levels=levels,
                )
                handles.append(mlines.Line2D([], [], color=c_double, linestyle='--', linewidth=1.5, label=f'{data["name"]} Doubles'))

        else:
            # --- Plot All Lenses Combined ---
            c_all = color_map.get(survey_key, {}).get("all", color_map.get(survey_key, {}).get("quads", "blue"))
            
            data_all = np.array([catalog[param] for param in params]).T
            if len(data_all) > 0 and data_all.shape[1] > 0:
                corner.corner(
                    data_all,
                    range=[range_vals[param] for param in params],
                    labels=latex_labels,
                    label_kwargs={"fontsize": label_fontsize},
                    color=c_all,
                    fig=figure,
                    plot_datapoints=False,
                    hist_kwargs={"linewidth": 2, "alpha": 0.9},
                    contour_kwargs={"linewidths": 2},
                    smooth=smooth,
                    smooth1d=smooth1d,
                    no_fill_contours=True,
                    levels=levels,
                )
                handles.append(mlines.Line2D([], [], color=c_all, linewidth=2, label=f'{data["name"]} (All)'))

        # --- Auto-Generate Text Box ---
        area = data["sky_area"].value
        cuts = data["kwargs_lens_cuts"]
        min_sep = cuts.get("min_image_separation", "N/A")
        max_sep = cuts.get("max_image_separation", "N/A")
        
        mag_cuts = cuts.get("second_brightest_image_cut", {})
        mag_cut_str = ", ".join([rf"$m^{{\rm 2nd}}_{{{band}}} < {limit}$" for band, limit in mag_cuts.items()])
        
        latex_safe_name = data['name'].replace(' ', r'\ ')
        
        info_str = '\n'.join((
            rf"$\bf{{{latex_safe_name}}}$",
            rf"Area = {area} deg$^2$",
            rf"$N_{{\rm lenses}}={total_lenses}$",
            rf"$N_{{\rm doubles}}={num_doubles}$",
            rf"$N_{{\rm quads}}={num_quads}$",
            rf"$f_{{\rm quad}}={q_frac:.2f}$",
            "",
            rf"${min_sep} < \Delta\theta < {max_sep}$ arcsec",
            mag_cut_str
        ))
        
        y_pos = text_y_start - (idx * text_y_step)
        figure.text(text_x, y_pos, info_str, fontsize=14, color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    # --- Rescale Diagonal 1D Histograms ---
    # Passing `range` makes corner force the diagonal y-limits on every call, so the
    # last survey plotted would clip the taller histograms of the earlier ones.
    n_params = len(params)
    diag_axes = np.array(figure.axes[:n_params ** 2]).reshape((n_params, n_params))
    for i in range(n_params):
        ax = diag_axes[i, i]
        heights = [np.nanmax(p.get_path().vertices[:, 1]) for p in ax.patches
                   if len(p.get_path().vertices) > 0]
        heights += [np.nanmax(line.get_ydata()) for line in ax.lines
                    if len(line.get_ydata()) > 0]
        if heights:
            ax.set_ylim(0, 1.1 * max(heights))

    # --- Increase Tick Sizes ---
    for ax in figure.get_axes():
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.xaxis.get_offset_text().set_fontsize(tick_fontsize)
        ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)

    # Build Legend and Title
    figure.legend(handles=handles, loc='upper right', fontsize=16, bbox_to_anchor=(0.95, 0.95))
    
    plt.subplots_adjust(top=0.95)
    figure.suptitle(title, fontsize=24)

    # --- Save Figure if requested ---
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved figure to: {save_path}")

    return figure

#################################################################################################################
## ETC.
#################################################################################################################
#################################################################################################################
## EUCLID RGB MONTAGE UTILITIES
#################################################################################################################
from slsim.Util.euclid_rgb_util import (
    euclid_rgb_from_image_list,
    euclid_nisp_num_pix_from_vis,
    compute_euclid_band_limits_from_image_lists,
)

EUCLID_RGB_BANDS = ["VIS", "Y", "J", "H"]


def simulate_euclid_band_images(lens_class, bands=EUCLID_RGB_BANDS, num_pix_vis=50,
                                add_noise=True, with_point_source=True,
                                with_extended_source=True, with_deflector=True):
    """Simulate Euclid VIS/NISP images of one lens over a common field of view.

    VIS (0.1"/pix) and NISP (0.3"/pix) have different pixel scales, so the NISP
    pixel count is derived from the VIS one to cover the same sky area.

    Returns a dict mapping band name -> image array.
    """
    num_pix_nisp = euclid_nisp_num_pix_from_vis(num_pix_vis)
    images = {}
    for band in bands:
        images[band] = simulate_image(
            lens_class=lens_class,
            band=band,
            num_pix=num_pix_vis if band == "VIS" else num_pix_nisp,
            add_noise=add_noise,
            observatory="Euclid",
            with_point_source=with_point_source,
            with_source=with_extended_source,
            with_deflector=with_deflector,
        )
    return images


def plot_euclid_montage(lenses, number_to_plot=100, num_cols=10, num_pix_vis=50,
                        colour="VIS_WEIGHTED_Y_J_H", stretch="mtf",
                        uniform_scaling=True, add_noise=True,
                        with_point_source=True, with_extended_source=True, with_deflector=True,
                        plot_title=None, random_seed=None, **rgb_kwargs):
    """Montage of Euclid Q1-style RGB cutouts built from VIS + Y/J/H images.

    With ``uniform_scaling=True`` the per-band black/white limits are measured
    once over the whole sample so all panels share the same stretch.
    """
    if len(lenses) > number_to_plot:
        rng = np.random.default_rng(random_seed)
        idxs = rng.choice(len(lenses), size=number_to_plot, replace=False)
        lenses_to_plot = [lenses[i] for i in idxs]
    else:
        lenses_to_plot = list(lenses)

    if len(lenses_to_plot) == 0:
        print("No lenses to plot.")
        return None

    image_sets = [
        simulate_euclid_band_images(
            lens_class, num_pix_vis=num_pix_vis, add_noise=add_noise,
            with_point_source=with_point_source,
            with_extended_source=with_extended_source,
            with_deflector=with_deflector,
        )
        for lens_class in lenses_to_plot
    ]

    band_limits = None
    if uniform_scaling:
        band_limits = compute_euclid_band_limits_from_image_lists(
            {band: [imgs[band] for imgs in image_sets] for band in EUCLID_RGB_BANDS}
        )

    rgb_images = [
        euclid_rgb_from_image_list(
            [imgs[band] for band in EUCLID_RGB_BANDS],
            colour=colour, stretch=stretch, band_limits=band_limits, **rgb_kwargs
        )
        for imgs in image_sets
    ]

    num_rows = int(np.ceil(len(rgb_images) / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    for i, ax in enumerate(np.atleast_1d(axes).flat):
        if i < len(rgb_images):
            ax.imshow(rgb_images[i], origin="lower")
            ax.axis("off")
        else:
            ax.remove()

    plt.tight_layout()
    if plot_title:
        plt.suptitle(plot_title, fontsize=16)
        plt.subplots_adjust(top=0.92)

    return fig
