"""Stores useful physical functions.
"""
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d, PchipInterpolator

# noinspection PyUnresolvedReferences
from petitRADTRANS.fortran_rebin import fortran_rebin as frebin
import petitRADTRANS.physical_constants as cst
from petitRADTRANS.math import running_mean


def compute_dist(t_irr, dist, t_star, r_star, mode, mode_what):
    # TODO rework/replace this function
    # TODO find better name
    mu_star = 0.
    angle_use = False

    if (mode != 'p') & (mode != 'd'):
        mu_star = float(mode)
        angle_use = True

    if mode_what == 'temp':
        if angle_use:
            t_irr = ((r_star * cst.r_sun / (dist * cst.au)) ** 2. * t_star ** 4. * mu_star) ** 0.25
        elif mode == 'p':
            t_irr = ((r_star * cst.r_sun / (dist * cst.au)) ** 2. * t_star ** 4. / 4.) ** 0.25
        else:
            t_irr = ((r_star * cst.r_sun / (dist * cst.au)) ** 2. * t_star ** 4. / 2.) ** 0.25
        return t_irr
    elif mode_what == 'dist':
        if angle_use:
            dist = np.sqrt((r_star * cst.r_sun) ** 2. * (t_star / t_irr) ** 4. * mu_star) / cst.au
        elif mode == 'p':
            dist = np.sqrt((r_star * cst.r_sun) ** 2. * (t_star / t_irr) ** 4. / 4.) / cst.au
        else:
            dist = np.sqrt((r_star * cst.r_sun) ** 2. * (t_star / t_irr) ** 4. / 2.) / cst.au
        return dist

    return None


def compute_effective_temperature(
    wavelengths: npt.NDArray,
    flux: npt.NDArray,
    orbit_semi_major_axis: float = 1.0,
    planet_radius: float = 1.0,
    use_si_units: bool = False
) -> float:
    """Calculates the effective temperature by integrating the model and using the stefan boltzmann law.

    Args:
        wavelengths : numpy.ndarray
            Wavelength grid
        flux : numpy.ndarray
            Flux density grid
        orbit_semi_major_axis : Optional(float)
            Distance to the object. Must have same units as planet_radius
        planet_radius : Optional(float)
            Object radius. Must have same units as orbit_semi_major_axis
        use_si_units : Optional(bool)
            If the flux is in W/m2/micron, this should be true
    """
    def integrate_flux(_wavelengths, _flux):
        return np.sum(
            _flux[:-1] * ((orbit_semi_major_axis / planet_radius) ** 2.) * np.diff(_wavelengths)
        )

    unit_factor = 1.0

    if use_si_units:
        unit_factor = 1e-3

    energy = integrate_flux(wavelengths, flux)

    return (energy / (cst.sigma*unit_factor)) ** 0.25


# TODO [4.0.0] rename 'press' => 'pressures', and 'T0' => better full lowercase name
def power_law_temperature_profile(press: npt.NDArray, alpha: float, T0: float) -> npt.NDArray:
    """
    Compute a power law profile for temperature; log(T) = a*log(P) + b.

    This function computes a cubic spline profile for temperature using
    pressure and temperature data points, along with a curvature prior.

    Args:
        press (array-like): An array or list of pressure data points.
        alpha (float): power law exponent (how steep is the profile)
        T0 (float): multiplicative factor (offsets the profile)

    Returns:
        temperatures (array): temperature values for each pressure value
    """
    temperatures = T0*(press/press[0])**alpha
    return temperatures


def cubic_spline_profile(
    press: npt.NDArray,
    temperature_points: npt.NDArray,
    gamma: float,
    nnodes: int = 0
) -> tuple[npt.NDArray, float]:
    """
    Compute a cubic spline profile for temperature based on pressure points.

    This function computes a cubic spline profile for temperature using
    pressure and temperature data points, along with a curvature prior.

    Args:
        press (array-like): An array or list of pressure data points.
        temperature_points (array-like): An array or list of temperature data points.
        gamma (float): A parameter controlling the curvature of the spline.
        nnodes (int, optional): Number of nodes to use in the spline interpolation.
            Defaults to 0, which means automatic determination of nodes.

    Returns:
        tuple: A tuple containing two elements:
            - interpolated_temps (array-like): Interpolated temperature values
              based on the cubic spline.
            - prior (array-like): Curvature prior values calculated for the spline.
    """

    cs = PchipInterpolator(
        x=np.linspace(
            np.log10(press[0]),
            np.log10(press[-1]),
            nnodes + 2
        ),
        y=temperature_points
    )

    interpolated_temps: npt.NDArray = cs(np.log10(press))
    prior: float = temperature_curvature_prior(press, interpolated_temps, gamma)

    return interpolated_temps, prior


def doppler_shift(wavelength_0: npt.NDArray | float, velocity: npt.NDArray | float) -> npt.NDArray | float:
    """Calculate the Doppler-shifted wavelength for electromagnetic waves.

    A negative velocity means that the source is going toward the observer. A positive velocity means the source is
    going away from the observer.

    Args:
        wavelength_0: (cm) wavelength of the wave in the referential of the source
        velocity: (cm.s-1) velocity of the source relative to the observer

    Returns:
        (cm) the wavelength of the source as measured by the observer
    """
    return wavelength_0 * np.sqrt((1 + velocity / cst.c) / (1 - velocity / cst.c))


def flux_cm2flux_hz(flux_cm: npt.NDArray | float, wavelength: npt.NDArray | float) -> npt.NDArray | float:
    """
    Convert a flux from [flux units]/cm to [flux units]/Hz at a given wavelength.
    Flux units can be, e.g., erg.s-1.cm-2.

    Steps:
        [cm] = c[cm.s-1] / [Hz]
        => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
        => d[cm]/d[Hz] = c / [Hz]**2
        integral of flux must be conserved: flux_cm * d[cm] = flux_hz * d[Hz]
        flux_hz = flux_cm * d[cm]/d[Hz]
        => flux_hz = flux_cm * wavelength**2 / c

    Args:
        flux_cm: ([flux units]/cm)
        wavelength: (cm)

    Returns:
        ([flux units]/Hz) the radiosity in converted units
    """
    return flux_cm * wavelength ** 2 / cst.c


def flux_hz2flux_cm(flux_hz: npt.NDArray | float, frequency: npt.NDArray | float) -> npt.NDArray | float:
    """Convert a flux from [flux units]/Hz to [flux units]/cm at a given frequency.
    Flux units can be, e.g., erg.s-1.cm-2.

    Steps:
        [cm] = c[cm.s-1] / [Hz]
        => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
        => d[cm]/d[Hz] = c / [Hz]**2
        => d[Hz]/d[cm] = [Hz]**2 / c
        integral of flux must be conserved: flux_cm * d[cm] = flux_hz * d[Hz]
        flux_cm = flux_hz * d[Hz]/d[cm]
        => flux_cm = flux_hz * frequency**2 / c

    Args:
        flux_hz: (erg.s-1.cm-2.sr-1/Hz)
        frequency: (Hz)

    Returns:
        (erg.s-1.cm-2.sr-1/cm) the radiosity in converted units
    """
    return flux_hz * frequency ** 2 / cst.c


def flux2irradiance(flux: npt.NDArray | float, source_radius: float, target_distance: float) -> npt.NDArray | float:
    """Calculate the spectral irradiance of a spherical source on a target from its flux (spectral radiosity).

    Args:
        flux: (M.L-1.T-3) flux of the source
        source_radius: (L) radius of the spherical source
        target_distance: (L) distance from the source to the target

    Returns:
        The irradiance of the source on the target (M.L-1.T-3).
    """
    return flux * (source_radius / target_distance) ** 2


def frequency2wavelength(frequency: npt.NDArray | float) -> npt.NDArray | float:
    """Convert frequencies into wavelength in centimeter.

    Args:
        frequency: frequency: (Hz) the frequency to convert

    Returns:
        (cm) the corresponding wavelengths
    """
    return cst.c / frequency


def hz2um(frequency: npt.NDArray | float) -> npt.NDArray | float:
    """Convert frequencies into wavelengths in micrometer.

    Args:
        frequency: (Hz) the frequency to convert

    Returns:
        (um) the corresponding wavelengths
    """
    return frequency2wavelength(frequency) * 1e4  # cm to um


def linear_spline_profile(
    press: npt.NDArray,
    temperature_points: npt.NDArray,
    gamma: float,
    nnodes: int = 0
) -> tuple[npt.NDArray, float]:
    """
    Compute a linear spline profile for temperature based on pressure points.

    This function computes a linear spline profile for temperature using
    pressure and temperature data points, along with a curvature prior.

    Args:
        press (array-like): An array or list of pressure data points.
        temperature_points (array-like): An array or list of temperature data points.
        gamma (float): A parameter controlling the curvature of the spline.
        nnodes (int, optional): Number of nodes to use in the spline interpolation.
            Defaults to 0, which means automatic determination of nodes.

    Returns:
        tuple: A tuple containing two elements:
            - interpolated_temps (array-like): Interpolated temperature values
              based on the linear spline.
            - prior (array-like): Curvature prior values calculated for the spline.
    """
    interpolated_temps = np.interp(
        np.log10(press),
        np.linspace(
            np.log10(press[0]),
            np.log10(press[-1]),
            int(nnodes) + 2
        ),
        temperature_points
    )

    prior = temperature_curvature_prior(press, interpolated_temps, gamma)

    return interpolated_temps, prior


def make_press_temp(rad_trans_params):
    """Function to make temp"""
    press_many = np.logspace(-8, 5, 260)
    t_no_ave = temperature_profile_function_guillot_modif(
        press_many,
        1e1 ** rad_trans_params['log_delta'],
        1e1 ** rad_trans_params['log_gamma'],
        rad_trans_params['t_int'], rad_trans_params['t_equ'],
        1e1 ** rad_trans_params['log_p_trans'], rad_trans_params['alpha']
    )

    # new
    press_many_new = 1e1 ** running_mean(np.log10(press_many), 25)
    t_new = running_mean(t_no_ave, 25)
    index_new = (press_many_new <= 1e3) & (press_many_new >= 1e-6)
    temp_new = t_new[index_new][::2]
    press_new = press_many_new[index_new][::2]

    return press_new, temp_new


def make_press_temp_iso(rad_trans_params):
    """Function to make temp"""
    press_many = np.logspace(-8, 5, 260)
    t_no_ave = rad_trans_params['t_equ'] * np.ones_like(press_many)

    # new
    press_many_new = 1e1 ** running_mean(np.log10(press_many), 25)
    t_new = running_mean(t_no_ave, 25)
    index_new = (press_many_new <= 1e3) & (press_many_new >= 1e-6)
    temp_new = t_new[index_new][::2]
    press_new = press_many_new[index_new][::2]

    return press_new, temp_new


# TODO [4.0.0] rename 'T_set' => 'temperature_xxx' (xxx => reference? 'set' is not clear)
def madhu_seager_2009(pressures, log_pressure_points, T_set, alpha_points, beta_points):
    """
    Calculate temperatures based on the Madhusudhan and Seager (2009) parameterization.

    This function computes temperatures using the Madhu and Seager (2009) parameterization
    for a given set of pressure values, pressure breakpoints, temperature breakpoints,
    alpha values, and beta values.

    Based off of the POSEIDON implementation:
    https://github.com/MartianColonist/POSEIDON/blob/main/POSEIDON/atmosphere.py

    Parameters:
        pressures : (numpy.ndarray)
            An array of pressure values (in bar) at which to calculate temperatures.
        log_pressure_points : (list)
            A list of log10 pressure breakpoints defining different temperature regimes.
            The zeroth element is the minimum pressure, should be log10(press[0]).
            The first element is the 1-2 boundary
            The second element is the level of the inversion
            The third element is the 2-3 boundary.
            The fourth element is the pressure at which the temperature is set (P_set).
        T_set : (float)
            A temperature at log_pressure_points[4] used to constrain the temperature profile.
        alpha_points : (list)
            A list of alpha values used in the parameterization for different regimes.
            Must have length 2.
        beta_points : (list)
            A list of beta values used in the parameterization for different regimes.
            By default, b[0] == b[1] == 0.5, unclear how well this will work if these aren't used!
            Must have length 2.

    Returns:
        temperatures : (numpy.ndarray)
            An array of calculated temperatures (in K) corresponding to the input pressure values.

    Reference:
    - Madhusudhan, N., & Seager, S. (2009). A Temperature and Abundance Retrieval Method for Exoplanet Atmospheres.
      The Astrophysical Journal, 707(1), 24-39. https://doi.org/10.1088/0004-637X/707/1/24
    """
    temperatures = np.zeros_like(pressures)

    # Set up masks for the different temperature regions
    pressure_points = 10**np.array(log_pressure_points)

    # Change log10 pressure points to log pressure points as used in Madhusudhan and Seager (2009)
    nat_log_pressure_points = np.log(pressure_points)

    mask_1 = pressures < pressure_points[1]
    mask_2 = (pressures >= pressure_points[1]) & (pressures < pressure_points[3])
    mask_3 = pressures >= pressure_points[3]

    # Find index of pressure closest to the set pressure
    i_set = np.argmin(np.abs(pressures - pressure_points[4]))
    p_set_i = pressures[i_set]

    # Store logarithm of various pressure quantities
    nat_log_p = np.log(pressures)
    nat_log_p_min = nat_log_pressure_points[0]
    nat_log_p_set_i = np.log(p_set_i)

    t0 = None
    t2 = None
    t3 = None

    # By default, (P_set = 10 bar), so T(P_set) should be in layer 3
    # By default (P_set = 10 bar), so T(P_set) should be in layer 3
    if nat_log_pressure_points[4] >= nat_log_pressure_points[3]:
        t3 = T_set  # T_deep is the isothermal deep temperature T3 here

        # Use the temperature parameter to compute boundary temperatures
        t2 = t3 - (
            (1.0 / alpha_points[1]) * (nat_log_pressure_points[3] - nat_log_pressure_points[2])
        ) ** (1 / beta_points[1])
        t1 = t2 + (
            (1.0 / alpha_points[1]) * (nat_log_pressure_points[1] - nat_log_pressure_points[2])
        ) ** (1 / beta_points[1])
        t0 = t1 - (
            (1.0 / alpha_points[0]) * (nat_log_pressure_points[1] - nat_log_p_min)
            ) ** (1 / beta_points[0])

    # If a different P_deep has been chosen, solve equations for layer 2...
    elif (
        nat_log_pressure_points[4] >= nat_log_pressure_points[1]
    ):  # Temperature parameter in layer 2
        # Use the temperature parameter to compute the boundary temperatures
        t2 = T_set - (
            (1.0 / alpha_points[1]) * (nat_log_p_set_i - nat_log_pressure_points[2])
        ) ** (1 / beta_points[1])
        t1 = t2 + (
            (1.0 / alpha_points[1]) * (nat_log_pressure_points[1] - nat_log_pressure_points[2])
        ) ** (1 / beta_points[0])
        t3 = t2 + (
            (1.0 / alpha_points[1]) * (nat_log_pressure_points[3] - nat_log_pressure_points[2])
        ) ** (1 / beta_points[1])
        t0 = t1 - ((1.0 / alpha_points[0]) * (nat_log_pressure_points[1] - nat_log_p_min)) ** (
            1 / beta_points[0]
        )

    # ...or for layer 1
    elif (
        nat_log_pressure_points[4] < nat_log_pressure_points[1]
    ):  # Temperature parameter in layer 1

        # Use the temperature parameter to compute the boundary temperatures
        t0 = T_set - ((1.0 / alpha_points[0]) * (nat_log_p_set_i - nat_log_p_min)) ** (
            1 / beta_points[0]
        )
        t1 = t0 + ((1.0 / alpha_points[0]) * (nat_log_pressure_points[1] - nat_log_p_min)) ** (
            1 / beta_points[0]
        )
        t2 = t1 - (
            (1.0 / alpha_points[1]) * (nat_log_pressure_points[1] - nat_log_pressure_points[2])
        ) ** (1 / beta_points[1])
        t3 = t2 + (
            (1.0 / alpha_points[1]) * (nat_log_pressure_points[3] - nat_log_pressure_points[2])
        ) ** (1 / beta_points[1])

    temperatures[mask_1] = t0 + (
        (nat_log_p[mask_1] - nat_log_pressure_points[0]) / (alpha_points[0])
    ) ** (1 / beta_points[0])
    temperatures[mask_2] = t2 + (
        (nat_log_p[mask_2] - nat_log_pressure_points[2]) / (alpha_points[1])
    ) ** (1 / beta_points[1])
    temperatures[mask_3] = t3

    return temperatures


def planck_function_cm(temperature: float, wavelength: npt.NDArray | float) -> npt.NDArray | float:
    """Returns the Planck function :math:`B_{\\lambda}(T)` in units of
    :math:`\\rm erg/s/cm^2/cm/steradian`.

    Args:
        temperature (float):
            Temperature in K.
        wavelength:
            Array containing the wavelength in cm.
    """

    _planck_function = (
            2. * cst.h * cst.c ** 2. / wavelength ** 5.
            / (np.exp(cst.h * cst.c / wavelength / cst.kB / temperature) - 1.)
    )

    return _planck_function


def planck_function_hz(temperature: float, frequency: npt.NDArray | float) -> npt.NDArray | float:
    """Returns the Planck function :math:`B_{\\nu}(T)` in units of
    :math:`\\rm erg/s/cm^2/Hz/steradian`.

    Args:
        temperature (float):
            Temperature in K.
        frequency:
            Array containing the frequency in Hz.
    """

    _planck_function = (
            2. * cst.h * frequency ** 3. / cst.c ** 2. / (np.exp(cst.h * frequency / cst.kB / temperature) - 1.)
    )

    return _planck_function


def planck_function_hz_temperature_derivative(
    temperature: float,
    frequency: npt.NDArray | float
) -> npt.NDArray | float:
    """Returns the derivative of the Planck function with respect to the temperature in units of
    :math:`\\rm erg/s/cm^2/Hz/steradian`.
    # TODO unused?

    Args:
        temperature:
            Temperature in K.
        frequency:
            Array containing the frequency in Hz.
    Returns:

    """
    _planck_function = planck_function_hz(temperature, frequency)
    _planck_function /= np.exp(cst.h * frequency / cst.kB / temperature) - 1.
    _planck_function *= (
            np.exp(cst.h * frequency / cst.kB / temperature) * cst.h * frequency / cst.kB / temperature ** 2.
    )

    return _planck_function


def rebin_spectrum(
    input_wavelengths: npt.NDArray,
    input_spectrum: npt.NDArray,
    rebinned_wavelengths: npt.NDArray
) -> npt.NDArray:
    """Re-bin the spectrum using the Fortran rebin_spectrum function, and catch errors occurring there.
    The fortran rebin function raises non-blocking errors. In that case, the function outputs an array of -1.

    Args:
        input_wavelengths: wavelengths of the input spectrum
        input_spectrum: spectrum to re-bin
        rebinned_wavelengths: wavelengths to re-bin the spectrum to. Must be contained within input_wavelengths

    Returns:
        The re-binned spectrum on the re-binned wavelengths
    """
    rebinned_spectrum = frebin.rebin_spectrum(input_wavelengths, input_spectrum, rebinned_wavelengths)

    if np.all(rebinned_spectrum == -1):
        raise ValueError("something went wrong during re-binning (rebin.f90), check the previous messages")
    elif np.any(rebinned_spectrum < 0):
        raise ValueError(f"negative value in re-binned spectrum, this may be related to the inputs "
                         f"(min input spectrum value: {np.min(input_spectrum)}, "
                         f"min re-binned spectrum value: {np.min(rebinned_spectrum)})")

    return rebinned_spectrum


def rebin_spectrum_bin(
    input_wavelengths: npt.NDArray,
    input_spectrum: npt.NDArray,
    rebinned_wavelengths: npt.NDArray,
    bin_widths: npt.NDArray
) -> npt.NDArray:
    """Re-bin the spectrum using the Fortran rebin_spectrum_bin function, and catch errors occurring there.
    The fortran rebin function raises non-blocking errors. In that case, the function outputs an array of -1.

    Args:
        input_wavelengths: wavelengths of the input spectrum
        input_spectrum: spectrum to re-bin
        rebinned_wavelengths: wavelengths to re-bin the spectrum to. Must be contained within input_wavelengths
        bin_widths: bin widths of the wavelengths to re-bin the spectrum to.

    Returns:
        The re-binned spectrum on the re-binned wavelengths
    """
    rebinned_spectrum = frebin.rebin_spectrum_bin(input_wavelengths, input_spectrum, rebinned_wavelengths, bin_widths)

    if np.all(rebinned_spectrum == -1):
        raise ValueError("something went wrong during re-binning (rebin.f90), check the previous messages")
    elif np.any(rebinned_spectrum < 0):
        raise ValueError(f"negative value in re-binned spectrum, this may be related to the inputs "
                         f"(min input spectrum value: {np.min(input_spectrum)}, "
                         f"min re-binned spectrum value: {np.min(rebinned_spectrum)})")

    return rebinned_spectrum


def temperature_profile_function_guillot(
    pressures: npt.NDArray,
    infrared_mean_opacity: float,
    gamma: float,
    gravities: npt.NDArray,
    intrinsic_temperature: float,
    equilibrium_temperature: float,
    redistribution_coefficient: float = 0.25
) -> npt.NDArray:
    """ Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar).

    For this the temperature model of Guillot (2010) is used (his Equation 29).
    Source: https://doi.org/10.1051/0004-6361/200913396

    Args:
        pressures:
            numpy array of floats, containing the input pressure in bars.
        infrared_mean_opacity:
            The infrared mean opacity in units of :math:`\\rm cm^2/s`.
        gamma:
            The ratio between the visual and infrared mean opacities.
        gravities:
            The planetary gravity at the given pressures in units of :math:`\\rm cm/s^2`.
        intrinsic_temperature:
            The planetary intrinsic temperature (in units of K).
        equilibrium_temperature:
            The planetary equilibrium temperature (in units of K).
        redistribution_coefficient:
            The redistribution coefficient of the irradiance. A value of 1 corresponds to the substellar point, 1/2 for
            the day-side average and 1/4 for the global average.
    """
    # Estimate tau from eq. 24: m is the column mass, dm = rho * dz, dP / dz = -g * rho, so m = P / g
    tau = infrared_mean_opacity * pressures * 1e6 / gravities
    t_irr = equilibrium_temperature * 2.0 ** 0.5  # from eqs. 1 and 2

    temperature = (
                          0.75 * intrinsic_temperature ** 4. * (2. / 3. + tau)
                          + 0.75 * t_irr ** 4. * redistribution_coefficient
                          * (
                                  2. / 3.
                                  + 1. / gamma / 3. ** 0.5
                                  + (gamma / 3. ** 0.5 - 1. / 3. ** 0.5 / gamma) * np.exp(-gamma * tau * 3. ** 0.5)
                          )
                  ) ** 0.25

    return temperature


def temperature_profile_function_guillot_dayside(
    pressures: npt.NDArray,
    infrared_mean_opacity: float,
    gamma: float,
    gravities: npt.NDArray,
    intrinsic_temperature: float,
    equilibrium_temperature: float
) -> npt.NDArray:
    """ Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar). For this the temperature model of Guillot (2010)
    is used (his Equation 29), in the case of averaging the flux over the day side of the planet.

    Args:
        pressures:
            numpy array of floats, containing the input pressure in bars.
        infrared_mean_opacity (float):
            The infrared opacity in units of :math:`\\rm cm^2/s`.
        gamma (float):
            The ratio between the visual and infrared opacity.
        gravities (float):
            The planetary surface gravity in units of :math:`\\rm cm/s^2`.
        intrinsic_temperature (float):
            The planetary internal temperature (in units of K).
        equilibrium_temperature (float):
            The planetary equilibrium temperature (in units of K).
    """
    return temperature_profile_function_guillot(
        pressures=pressures,
        infrared_mean_opacity=infrared_mean_opacity,
        gamma=gamma,
        gravities=gravities,
        intrinsic_temperature=intrinsic_temperature,
        equilibrium_temperature=equilibrium_temperature,
        redistribution_coefficient=0.5
    )


def temperature_profile_function_guillot_global(
    pressures: npt.NDArray,
    infrared_mean_opacity: float,
    gamma: float,
    gravities: npt.NDArray,
    intrinsic_temperature: float,
    equilibrium_temperature: float
) -> npt.NDArray:
    """ Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar). For this the temperature model of Guillot (2010)
    is used (his Equation 29), in the case of averaging the flux over the whole planetary surface.

    Args:
        pressures:
            numpy array of floats, containing the input pressure in bars.
        infrared_mean_opacity (float):
            The infrared opacity in units of :math:`\\rm cm^2/s`.
        gamma (float):
            The ratio between the visual and infrared opacity.
        gravities (float):
            The planetary surface gravity in units of :math:`\\rm cm/s^2`.
        intrinsic_temperature (float):
            The planetary internal temperature (in units of K).
        equilibrium_temperature (float):
            The planetary equilibrium temperature (in units of K).
    """
    return temperature_profile_function_guillot(
        pressures=pressures,
        infrared_mean_opacity=infrared_mean_opacity,
        gamma=gamma,
        gravities=gravities,
        intrinsic_temperature=intrinsic_temperature,
        equilibrium_temperature=equilibrium_temperature,
        redistribution_coefficient=0.25
    )


def temperature_profile_function_guillot_global_ret(
    pressures: npt.NDArray,
    delta: float,
    gamma: float,
    intrinsic_temperature: float,
    equilibrium_temperature: float
) -> npt.NDArray:
    """Global Guillot P-T formula with kappa/gravity replaced by delta."""
    # TODO what is delta?
    delta = np.abs(delta)
    gamma = np.abs(gamma)
    intrinsic_temperature = np.abs(intrinsic_temperature)
    equilibrium_temperature = np.abs(equilibrium_temperature)
    tau = pressures * 1e6 * delta
    t_irr = equilibrium_temperature * np.sqrt(2.)
    temperature = (0.75 * intrinsic_temperature ** 4. * (2. / 3. + tau)
                   + 0.75 * t_irr ** 4. / 4.
                   * (2. / 3. + 1. / gamma / 3. ** 0.5
                      + (gamma / 3. ** 0.5 - 1. / 3. ** 0.5 / gamma)
                      * np.exp(-gamma * tau * 3. ** 0.5))) ** 0.25
    return temperature


def temperature_profile_function_guillot_metallic(
    pressures: npt.NDArray,
    gamma: float,
    reference_gravity: npt.NDArray,
    intrinsic_temperature: float,
    equilibrium_temperature: float,
    infrared_mean_opacity_solar_metallicity: float,
    metallicity: float | None = None
) -> npt.NDArray:
    """Get a Guillot temperature profile depending on metallicity.

    Args:
        pressures: (bar) pressures of the profile
        gamma: ratio between visual and infrared opacity
        reference_gravity: (cm.s-2) surface gravity
        intrinsic_temperature: (K) intrinsic temperature
        equilibrium_temperature: (K) equilibrium temperature
        infrared_mean_opacity_solar_metallicity:
            (cm2.s-1) infrared mean opacity for a solar metallicity (Z = 1) atmosphere
        metallicity: ratio of heavy elements abundance over H abundance with respect to the solar ratio

    Returns:
        temperatures: (K) the temperature at each pressures of the atmosphere
    """
    if metallicity is not None:
        kappa_ir = infrared_mean_opacity_solar_metallicity * metallicity
    else:
        kappa_ir = infrared_mean_opacity_solar_metallicity

    temperatures = temperature_profile_function_guillot_global(
        pressures=pressures,
        infrared_mean_opacity=kappa_ir,
        gamma=gamma,
        gravities=reference_gravity,
        intrinsic_temperature=intrinsic_temperature,
        equilibrium_temperature=equilibrium_temperature
    )

    return temperatures


def temperature_profile_function_guillot_modif(
    pressures: npt.NDArray,
    delta: float,
    gamma: float,
    intrinsic_temperature: float,
    equilibrium_temperature: float,
    ptrans: float,
    alpha: float
) -> npt.NDArray:
    """Modified Guillot P-T formula"""
    # TODO how is it modified? Why for?
    return temperature_profile_function_guillot_global_ret(
        pressures,
        np.abs(delta),
        np.abs(gamma),
        np.abs(intrinsic_temperature), np.abs(equilibrium_temperature)
    ) * (1. - alpha * (1. / (1. + pressures / ptrans)))


def temperature_profile_function_isothermal(pressures: npt.NDArray, temperature: float) -> npt.NDArray:
    # TODO only to temporarily fix methods, change name later
    return np.ones(pressures.size) * temperature


# TODO find better name
# TODO @Evert fix docstrings
def temperature_profile_function_ret_model(rad_trans_params):
    """
    Self-luminous retrieval P-T model.

    Args:
        t3 : np.array([t1, t2, t3])
            temperature points to be added on top
            radiative Eddington structure (above tau = 0.1).
            Use spline interpolation, t1 < t2 < t3 < tconnect as prior.
        delta : float
            proportionality factor in tau = delta * press_cgs**alpha
        alpha : float
            power law index in tau = delta * press_cgs**alpha
            For the tau model: use proximity to kappa_rosseland photosphere
            as prior.
        tint : float
            internal temperature of the Eddington model
        press : np.ndarray
            input pressure profile in bar
        conv : bool
            enforce convective adiabat yes/no
        co_ratio : float
            C/O for the nabla_ad interpolation
        metallicity : float
            metallicity for the nabla_ad interpolation
    Returns:
        Tret : np.ndarray
            The temperature as a function of atmospheric pressure.
    """
    import copy
    from scipy.interpolate import interp1d, CubicSpline
    from petitRADTRANS.chemistry.pre_calculated_chemistry import pre_calculated_equilibrium_chemistry_table

    t3, delta, alpha, tint, press, feh, co_ratio, conv = rad_trans_params
    # Go grom bar to cgs
    pressures_cgs: npt.NDArray[np.floating] = press * 1e6

    # Calculate the optical depth
    tau = delta * pressures_cgs ** alpha

    # This is the eddington temperature
    tedd = (3. / 4. * tint ** 4. * (2. / 3. + tau)) ** 0.25

    ab, _, nabla_ad = (
        pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
            co_ratio * np.ones_like(tedd),
            feh * np.ones_like(tedd),
            tedd,
            press,
            full=True
        )
    )

    tfinal = None  # TODO tmp fix for reference before assignment
    tret = None  # TODO tmp fix for reference before assignment

    # Enforce convective adiabat
    if conv:
        # Calculate the current, radiative temperature gradient
        nab_rad = np.diff(np.log(tedd)) / np.diff(np.log(pressures_cgs))
        # Extend to array of same length as pressure structure
        nabla_rad = np.ones_like(tedd)
        nabla_rad[0] = nab_rad[0]
        nabla_rad[-1] = nab_rad[-1]
        nabla_rad[1:-1] = (nab_rad[1:] + nab_rad[:-1]) / 2.

        # Where is the atmosphere convectively unstable?
        conv_index = nabla_rad > nabla_ad

        # TODO: Check remains convective and convergence
        for i in range(10):
            if i == 0:
                t_take = copy.copy(tedd)
            else:
                t_take = copy.copy(tfinal)  # TODO reference before assignment

            ab, _, nabla_ad = (
                pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
                    co_ratio * np.ones_like(t_take),
                    feh * np.ones_like(t_take),
                    t_take,
                    press,
                    full=True
                )
            )

            # Calculate the average nabla_ad between the layers
            nabla_ad_mean = nabla_ad
            nabla_ad_mean[1:] = (nabla_ad[1:] + nabla_ad[:-1]) / 2.
            # What are the increments in temperature due to convection
            tnew = nabla_ad_mean[conv_index] * np.diff(np.log(pressures_cgs)).mean()
            # What is the last radiative temperature?
            tstart = np.log(t_take[~conv_index][-1])
            # Integrate and translate to temperature from log(temperature)
            tnew = np.exp(np.cumsum(tnew) + tstart)

            # Add upper radiative and
            # lower convective part into one single array
            tfinal = copy.copy(t_take)
            tfinal[conv_index] = tnew

            if np.max(np.abs(t_take - tfinal) / t_take) < 0.01:
                break

    else:
        tfinal = tedd

    # Add the three temperature-point P-T description above tau = 0.1
    def press_tau(_tau):
        # Returns the pressure at a given tau, in cgs
        return (_tau / delta) ** (1. / alpha)

    # Where is the uppermost pressure of the Eddington radiative structure?
    p_bot_spline = press_tau(0.1)

    for i_intp in range(2):

        if i_intp == 0:

            # Create the pressure coordinates for the spline support nodes at low pressure
            support_points_low = np.logspace(np.log10(pressures_cgs[0]),
                                             np.log10(p_bot_spline),
                                             4)

            # Create the pressure coordinates for the spline support nodes at high pressure,
            # the corresponding temperatures for these nodes will be taken from the
            # radiative+convective solution
            support_points_high = 10 ** np.arange(np.log10(p_bot_spline),
                                                  np.log10(pressures_cgs[-1]),
                                                  np.diff(np.log10(support_points_low))[0])

            # Combine into one support node array, don't add the p_bot_spline point twice.
            support_points = np.zeros(len(support_points_low) + len(support_points_high) - 1)
            support_points[:4] = support_points_low
            support_points[4:] = support_points_high[1:]

        else:

            # Create the pressure coordinates for the spline support nodes at low pressure
            support_points_low = np.logspace(np.log10(pressures_cgs[0]),
                                             np.log10(p_bot_spline),
                                             7)

            # Create the pressure coordinates for the spline support nodes at high pressure,
            # the corresponding temperatures for these nodes will be taken from the
            # radiative+convective solution
            support_points_high = np.logspace(np.log10(p_bot_spline), np.log10(pressures_cgs[-1]), 7)

            # Combine into one support node array, don't add the p_bot_spline point twice.
            support_points = np.zeros(len(support_points_low) + len(support_points_high) - 1)
            support_points[:7] = support_points_low
            support_points[7:] = support_points_high[1:]

        # Define the temperature values at the node points.
        t_support = np.zeros_like(support_points)

        if i_intp == 0:
            tfintp = interp1d(pressures_cgs, tfinal, kind='cubic')
            # The temperature at p_bot_spline (from the radiative-convective solution)
            t_support[int(len(support_points_low)) - 1] = tfintp(p_bot_spline)
            # The temperature at pressures below p_bot_spline (free parameters)
            t_support[:(int(len(support_points_low)) - 1)] = t3
            # t_support[:3] = tfintp(support_points_low)
            # The temperature at pressures above p_bot_spline
            # (from the radiative-convective solution)
            t_support[int(len(support_points_low)):] = \
                tfintp(support_points[(int(len(support_points_low))):])

        else:
            tfintp1 = interp1d(pressures_cgs, tret, kind='cubic')  # TODO possible reference before assignment
            t_support[:(int(len(support_points_low)) - 1)] = \
                tfintp1(support_points[:(int(len(support_points_low)) - 1)])

            tfintp = interp1d(pressures_cgs, tfinal)
            # The temperature at p_bot_spline (from the radiative-convective solution)
            t_support[int(len(support_points_low)) - 1] = tfintp(p_bot_spline)
            # print('diff', t_connect_calc - tfintp(p_bot_spline))
            t_support[int(len(support_points_low)):] = \
                tfintp(support_points[(int(len(support_points_low))):])

        # Make the temperature spline interpolation to be returned to the user
        cs = CubicSpline(np.log10(support_points), t_support)
        tret = cs(np.log10(pressures_cgs))

    tret[tret < 0.0] = 1.0
    # Return the temperature, the pressure at tau = 1,
    # and the temperature at the connection point.
    # The last two are needed for the priors on the P-T profile.
    return tret  # , press_tau(1.)/1e6, tfintp(p_bot_spline)


# TODO [4.0.0] remove argument 'press' as it is not used, and rename 'temps' => 'temperatures'
def temperature_curvature_prior(press, temps, gamma):
    """
    Compute a curvature prior for a temperature-pressure profile.

    This function calculates a curvature prior for a temperature-pressure profile,
    penalizing deviations from a smooth, low-curvature profile, based on Line 2015

    Args:
        press (array-like): An array or list of pressure data points.
        temps (array-like): An array or list of temperature data points.
        gamma (float): The curvature penalization factor.

    Returns:
        float: The curvature prior value.
    """
    weighted_temp_prior = -0.5 * np.sum((temps[2:] - 2 * temps[1:-1] + temps[:-2]) ** 2) / gamma
    weighted_temp_prior -= 0.5 * np.log(2 * np.pi * gamma)

    return weighted_temp_prior


def dtdp_temperature_profile(
        press,
        num_layer,
        layer_pt_slopes,
        t_bottom,
        top_of_atmosphere_pressure=-3,
        bottom_of_atmosphere_pressure=3
        ):
    """
    This function takes the temperature gradient at a set number of spline points and interpolates a temperature
    profile as a function of pressure.

    Args:
        press : array_like
            The pressure array.
        num_layer : int
            The number of layers. Default for covering 10^3 to 10^-3 bar as in Zhang 2023 is 6.
            To cover 10^3 to 10^-6 bar 10 is recommended.
        layer_pt_slopes : array_like
            The temperature gradient at the spline points.
        t_bottom : float
            The temperature at the bottom of the atmosphere.
        top_of_atmosphere_pressure : float
            Minimum atmospheric pressure in log bar. If the pressure array extends beyond this value, the
            temperature structure at lower pressures than this value will be isothermal.
        bottom_of_atmosphere_pressure : float
            Maximum atmospheric pressure in log bar.

    Returns:
        temperatures : array_like
            The temperature profile.
    """
    id_sub = np.where(press >= 10**top_of_atmosphere_pressure)
    p_use_sub = press[id_sub]
    num_sub = len(p_use_sub)

    # 1.3 pressures of layers
    layer_pressures = np.logspace(top_of_atmosphere_pressure,
                                  bottom_of_atmosphere_pressure,
                                  int(num_layer))
    # 1.4 assemble the P-T slopes for these layers
    # for index in range(num_layer):
    #    layer_pt_slopes[index] = parameters['PTslope_%d'%(num_layer - index)].value
    # 1.5 interpolate the P-T slopes to compute slopes for all layers
    interp_func = interp1d(np.log10(layer_pressures),
                           layer_pt_slopes,
                           'quadratic')
    pt_slopes_sub = interp_func(np.log10(p_use_sub))
    # 1.6 compute temperatures
    temperatures_sub = np.ones(num_sub) * np.nan
    temperatures_sub[-1] = t_bottom

    # Note: higher index is lower pressure.
    for index in range(1, num_sub):
        temperatures_sub[-1 - index] = np.exp(
            np.log(temperatures_sub[-index]) - pt_slopes_sub[-index]
            * (np.log(p_use_sub[-index]) - np.log(p_use_sub[-1 - index]))
        )

    # 1.7 isothermal in the remaining region, i.e., upper atmosphere
    temperatures = np.ones_like(press) * temperatures_sub[0]
    temperatures[id_sub] = np.copy(temperatures_sub)

    return temperatures


def um2hz(wavelength: npt.NDArray | float) -> npt.NDArray | float:
    """Convert wavelengths in micrometer into frequencies.

    Args:
        wavelength: (um) the wavelengths to convert

    Returns:
        (Hz) the corresponding frequencies
    """
    return frequency2wavelength(wavelength * 1e-4)  # the operation is the same: (c / Hz) -> cm, (c / cm) -> Hz


def wavelength2frequency(wavelength: npt.NDArray | float) -> npt.NDArray | float:
    """Convert wavelengths in centimeter to frequencies.

    Args:
        wavelength: (cm) the wavelengths to convert

    Returns:
        (Hz) the converted frequencies
    """
    return frequency2wavelength(wavelength)  # the operation is the same: (c / Hz) -> cm, (c / cm) -> Hz
