import numpy as np
from astropy.table import Table
from slsim.ImageSimulation.image_simulation import simulate_image, rgb_image_from_image_list
import matplotlib.pyplot as plt
from matplotlib import lines as mlines
import corner


#################################################################################################################
## DATA EXTRACTION UTILITIES
#################################################################################################################

# See the file named "data_column_descriptions.txt" for descriptions of each column being extracted below.
def extract_lensed_agn_properties(lens_objects, all_bands=None, max_num_images=5):
    """
    Extracts properties from a list of lens objects and returns an Astropy Table.

    Parameters
    ----------
    lens_objects : list
        A list of lens system objects.
    all_bands : list of str, optional
        List of photometric bands to extract magnitudes for. 
        Defaults to ['g', 'r', 'i', 'z', 'y'].
    max_num_images : int, optional
        Maximum number of images to account for in columns (e.g., 4 for quads). Defaults to 5.

    Returns
    -------
    astropy.table.Table
        A table containing all extracted properties.
    """
    
    # Default bands if none provided
    if all_bands is None:
        all_bands = ['g', 'r', 'i', 'z', 'y']

    # -------------------------------------------------------------------------
    # Initialize table_dict
    # -------------------------------------------------------------------------
    table_dict = {
        # --- System Properties ---
        "z_S": [],
        "z_D": [],
        "theta_E_arcsec": [],
        "max_time_delay_days": [],
        "num_images": [],

        # --- Point Source (AGN) - Single Values ---
        "x_ps_position_arcsec": [],
        "y_ps_position_arcsec": [],
        "black_hole_mass_exponent": [],
        "agn_disk_eddington_ratio": [],
        "black_hole_spin": [],          # Spin
        "agn_disk_inclination_angle": [],        # Inclination

        # --- Host Galaxy (Source) Light [SINGLE SERSIC] ---
        "x_host_position_arcsec": [],
        "y_host_position_arcsec": [],
        "host_light_R_eff_arcsec": [],
        "host_light_n_sersic": [],
        "host_light_e1": [],
        "host_light_e2": [],
        "host_light_ellipticity": [],
        "host_light_axis_ratio": [],
        "host_light_position_angle_deg": [],

        # --- Deflector Light (Lens Galaxy) [SINGLE SERSIC] ---
        "deflector_light_R_eff_arcsec": [],
        "deflector_light_n_sersic": [],
        "deflector_light_e1": [],
        "deflector_light_e2": [],
        "deflector_light_ellipticity": [],
        "deflector_light_axis_ratio": [],
        "deflector_light_position_angle_deg": [],
        "x_deflector_light_position_arcsec": [],
        "y_deflector_light_position_arcsec": [],

        # --- Deflector Mass (Lens Galaxy) [PEMD + SHEAR] ---
        "x_deflector_mass_position_arcsec": [],
        "y_deflector_mass_position_arcsec": [],
        "deflector_stellar_mass": [],
        "deflector_pl_slope": [],               
        "deflector_velocity_dispersion": [],
        "deflector_mass_e1": [],    
        "deflector_mass_e2": [],
        "deflector_mass_ellipticity": [],        
        "deflector_mass_axis_ratio": [],         
        "deflector_mass_position_angle_deg": [], 
        "external_shear_gamma1": [],              
        "external_shear_gamma2": [],
        "external_shear_amplitude": [],          
        "external_shear_position_angle_deg": [], 
        "external_convergence": [],              
    }

    # --- Initialize Band-Dependent Keys ---
    for band in all_bands:
        table_dict[f"unlensed_ps_mag_{band}"] = []
        table_dict[f"faintest_image_ps_mag_{band}"] = []
        table_dict[f"second_brightest_image_ps_mag_{band}"] = []
        table_dict[f"brightest_image_ps_mag_{band}"] = []
        
        table_dict[f"unlensed_host_mag_{band}"] = []
        table_dict[f"lensed_host_mag_{band}"] = []
        
        table_dict[f"deflector_mag_{band}"] = []

        # Lensed PS magnitudes depend on BOTH band and image number
        for img_num in range(1, max_num_images + 1):
             table_dict[f"lensed_ps_mag_{band}_{img_num}"] = []

    # --- Initialize Image-Dependent Keys (Geometry, Time, & Microlensing) ---
    for img_num in range(1, max_num_images + 1):
        table_dict[f"time_delay_days_{img_num}"] = []
        table_dict[f"ps_magnification_{img_num}"] = []
        table_dict[f"x_ps_image_positions_arcsec_{img_num}"] = []
        table_dict[f"y_ps_image_positions_arcsec_{img_num}"] = []
        
        # New Microlensing keys
        table_dict[f"kappa_star_{img_num}"] = []
        table_dict[f"kappa_tot_{img_num}"] = []
        table_dict[f"shear_{img_num}"] = []
        table_dict[f"shear_angle_{img_num}"] = []

    # -------------------------------------------------------------------------
    # Populate properties
    # -------------------------------------------------------------------------
    for lens_system in lens_objects:
        
        source_index = 0  # assuming single source per lens system
        
        # Access internal classes
        source_class = lens_system.source(source_index)
        deflector_class = lens_system.deflector
        ps_class = source_class._source._point_source
        es_class = source_class._source._extended_source
        
        # --- System ---
        table_dict["z_S"].append(lens_system.source_redshift_list[source_index])
        table_dict["z_D"].append(lens_system.deflector_redshift)
        table_dict["theta_E_arcsec"].append(lens_system.einstein_radius[source_index])
        table_dict["num_images"].append(lens_system.image_number[source_index])

        # --- Time Delay ---
        arrival_times = lens_system.point_source_arrival_times()[source_index]
        if len(arrival_times) > 0:
            max_time_delay = np.max(arrival_times) - np.min(arrival_times)
        else:
            max_time_delay = np.nan
        table_dict["max_time_delay_days"].append(max_time_delay)

        # --- Point Source (AGN) ---
        x_ps, y_ps = source_class.point_source_position
        table_dict["x_ps_position_arcsec"].append(x_ps)
        table_dict["y_ps_position_arcsec"].append(y_ps)
        
        # AGN property extraction
        ps_class._init_agn_class()
        agn_kwargs = ps_class.agn_class.kwargs_model
        log_m_bh = agn_kwargs.get('black_hole_mass_exponent', np.nan)
        f_edd = agn_kwargs.get('eddington_ratio', np.nan)
        spin = agn_kwargs.get('black_hole_spin', np.nan)             # Spin
        inc_angle = agn_kwargs.get('inclination_angle', np.nan)      # Inclination
            
        table_dict["black_hole_mass_exponent"].append(log_m_bh)
        table_dict["agn_disk_eddington_ratio"].append(f_edd)
        table_dict["black_hole_spin"].append(spin)
        table_dict["agn_disk_inclination_angle"].append(inc_angle)

        # --- Host Galaxy (Source) Light [SERSIC] ---
        x_host, y_host = es_class.extended_source_position
        table_dict["x_host_position_arcsec"].append(x_host)
        table_dict["y_host_position_arcsec"].append(y_host)
        table_dict["host_light_R_eff_arcsec"].append(es_class.angular_size)
        table_dict["host_light_n_sersic"].append(es_class._n_sersic)
        
        e1_host, e2_host = es_class.ellipticity
        e_host = np.sqrt(e1_host**2 + e2_host**2)
        q_host = (1 - e_host) / (1 + e_host)
        phi_host = 0.5 * np.degrees(np.arctan2(e2_host, e1_host))
        
        table_dict["host_light_e1"].append(e1_host)
        table_dict["host_light_e2"].append(e2_host)
        table_dict["host_light_ellipticity"].append(e_host)
        table_dict["host_light_axis_ratio"].append(q_host)
        table_dict["host_light_position_angle_deg"].append(phi_host)

        # --- Deflector Calculations ---
        e1_light, e2_light, e1_mass, e2_mass = lens_system.deflector_ellipticity()
        
        e_mass = np.sqrt(e1_mass**2 + e2_mass**2)
        q_mass = (1 - e_mass) / (1 + e_mass)
        phi_mass = 0.5 * np.degrees(np.arctan2(e2_mass, e1_mass))
        
        e_light = np.sqrt(e1_light**2 + e2_light**2)
        q_light = (1 - e_light) / (1 + e_light)
        phi_light = 0.5 * np.degrees(np.arctan2(e2_light, e1_light))

        kappa_ext, gamma1_ext, gamma2_ext = lens_system.los_linear_distortions
        gamma_ext = np.sqrt(gamma1_ext**2 + gamma2_ext**2)
        phi_ext = 0.5 * np.degrees(np.arctan2(gamma2_ext, gamma1_ext))

        x_pos_deflector, y_pos_deflector = lens_system.deflector_position

        # --- Deflector Mass (PEMD + EXTERNAL SHEAR) ---
        table_dict["deflector_velocity_dispersion"].append(lens_system.deflector_velocity_dispersion())
        table_dict["deflector_mass_e1"].append(e1_mass)
        table_dict["deflector_mass_e2"].append(e2_mass)
        table_dict["deflector_mass_ellipticity"].append(e_mass)
        table_dict["deflector_mass_axis_ratio"].append(q_mass)
        table_dict["deflector_mass_position_angle_deg"].append(phi_mass)
        
        gamma_pl = np.nan
        if hasattr(deflector_class, 'halo_properties'):
             gamma_pl = deflector_class.halo_properties.get("gamma_pl", np.nan)
        table_dict["deflector_pl_slope"].append(gamma_pl)
        
        table_dict["deflector_stellar_mass"].append(lens_system.deflector_stellar_mass())
        table_dict["x_deflector_mass_position_arcsec"].append(x_pos_deflector)
        table_dict["y_deflector_mass_position_arcsec"].append(y_pos_deflector)
        table_dict["external_convergence"].append(kappa_ext)
        table_dict["external_shear_gamma1"].append(gamma1_ext)
        table_dict["external_shear_gamma2"].append(gamma2_ext)
        table_dict["external_shear_amplitude"].append(gamma_ext)
        table_dict["external_shear_position_angle_deg"].append(phi_ext)

        # --- Deflector Light ---
        table_dict["x_deflector_light_position_arcsec"].append(x_pos_deflector)
        table_dict["y_deflector_light_position_arcsec"].append(y_pos_deflector)
        table_dict["deflector_light_e1"].append(e1_light)
        table_dict["deflector_light_e2"].append(e2_light)
        table_dict["deflector_light_ellipticity"].append(e_light)
        table_dict["deflector_light_axis_ratio"].append(q_light)
        table_dict["deflector_light_position_angle_deg"].append(phi_light)
        table_dict["deflector_light_R_eff_arcsec"].append(lens_system.deflector.angular_size_light)
        
        n_sersic = np.nan
        if hasattr(deflector_class._deflector, '_deflector_dict'):
            n_sersic = deflector_class._deflector._deflector_dict.get("n_sersic", np.nan)
        table_dict["deflector_light_n_sersic"].append(n_sersic)

        # --- Image Properties Setup ---
        num_images = lens_system.image_number[source_index]
        ps_magnifications = lens_system.point_source_magnification()[source_index]
        x_image_positions, y_image_positions = lens_system.point_source_image_positions()[source_index]
        arrival_times = lens_system.point_source_arrival_times()[source_index]

        # --- Microlensing Parameters ---
        try:
            k_star, k_tot, shear_mc, shear_angle_mc = lens_system._microlensing_parameters_for_image_positions_single_source(band="i", source_index=source_index)
        except Exception:
            k_star, k_tot, shear_mc, shear_angle_mc = [], [], [], []

        # --- Band-Dependent Properties ---
        for band in all_bands:
            unlensed_ps_mag = lens_system.point_source_magnitude(band=band, lensed=False)[source_index]
            table_dict[f"unlensed_ps_mag_{band}"].append(unlensed_ps_mag)

            lensed_ps_mags = lens_system.point_source_magnitude(band=band, lensed=True)[source_index]

            if len(lensed_ps_mags) > 0:
                faintest_image_ps_mag = np.max(lensed_ps_mags)
                brightest_image_ps_mag = np.min(lensed_ps_mags)
                sorted_mags = np.sort(lensed_ps_mags)
                second_brightest_image_ps_mag = sorted_mags[1] if len(sorted_mags) > 1 else np.nan
            else:
                faintest_image_ps_mag = np.nan
                second_brightest_image_ps_mag = np.nan
                brightest_image_ps_mag = np.nan

            table_dict[f"faintest_image_ps_mag_{band}"].append(faintest_image_ps_mag)
            table_dict[f"second_brightest_image_ps_mag_{band}"].append(second_brightest_image_ps_mag)
            table_dict[f"brightest_image_ps_mag_{band}"].append(brightest_image_ps_mag)

            unlensed_host_mag = lens_system.extended_source_magnitude(band=band, lensed=False)[source_index]
            lensed_host_mag = lens_system.extended_source_magnitude(band=band, lensed=True)[source_index]
            table_dict[f"unlensed_host_mag_{band}"].append(unlensed_host_mag)
            table_dict[f"lensed_host_mag_{band}"].append(lensed_host_mag)

            deflector_mag = lens_system.deflector_magnitude(band=band)
            table_dict[f"deflector_mag_{band}"].append(deflector_mag)

            for img_num in range(1, max_num_images + 1):
                if img_num <= num_images and img_num <= len(lensed_ps_mags):
                    table_dict[f"lensed_ps_mag_{band}_{img_num}"].append(lensed_ps_mags[img_num-1])
                else:
                    table_dict[f"lensed_ps_mag_{band}_{img_num}"].append(np.nan)
        
        # --- Image-Dependent Properties (Geometry, Time, & Microlensing) ---
        for img_num in range(1, max_num_images + 1):
            if img_num <= num_images:
                # Time Delays
                val_td = arrival_times[img_num-1] if img_num <= len(arrival_times) else np.nan
                table_dict[f"time_delay_days_{img_num}"].append(val_td)

                # Magnifications
                val_mag = ps_magnifications[img_num-1] if img_num <= len(ps_magnifications) else np.nan
                table_dict[f"ps_magnification_{img_num}"].append(val_mag)

                # Positions
                val_x = x_image_positions[img_num-1] if img_num <= len(x_image_positions) else np.nan
                val_y = y_image_positions[img_num-1] if img_num <= len(y_image_positions) else np.nan
                table_dict[f"x_ps_image_positions_arcsec_{img_num}"].append(val_x)
                table_dict[f"y_ps_image_positions_arcsec_{img_num}"].append(val_y)
                
                # Microlensing
                val_k_star = k_star[img_num-1] if img_num <= len(k_star) else np.nan
                val_k_tot = k_tot[img_num-1] if img_num <= len(k_tot) else np.nan
                val_shear = shear_mc[img_num-1] if img_num <= len(shear_mc) else np.nan
                val_shear_ang = shear_angle_mc[img_num-1] if img_num <= len(shear_angle_mc) else np.nan
                
                table_dict[f"kappa_star_{img_num}"].append(val_k_star)
                table_dict[f"kappa_tot_{img_num}"].append(val_k_tot)
                table_dict[f"shear_{img_num}"].append(val_shear)
                table_dict[f"shear_angle_{img_num}"].append(val_shear_ang)

            else:
                table_dict[f"time_delay_days_{img_num}"].append(np.nan)
                table_dict[f"ps_magnification_{img_num}"].append(np.nan)
                table_dict[f"x_ps_image_positions_arcsec_{img_num}"].append(np.nan)
                table_dict[f"y_ps_image_positions_arcsec_{img_num}"].append(np.nan)
                table_dict[f"kappa_star_{img_num}"].append(np.nan)
                table_dict[f"kappa_tot_{img_num}"].append(np.nan)
                table_dict[f"shear_{img_num}"].append(np.nan)
                table_dict[f"shear_angle_{img_num}"].append(np.nan)

    return Table(table_dict)

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
        total_lenses = num_doubles + num_quads
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