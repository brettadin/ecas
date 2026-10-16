"""Wrapper of ESO's SKYCALC Command Line Interface.
"""
import copy
import json
import os
import tempfile

import numpy as np
from astropy.io import fits
from skycalc_cli.skycalc import AlmanacQuery, SkyModel

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.utils import savez_compressed_record


def find_optimal_airmass_day(ra, dec, observatory='3060m',
                             time_step=cst.s_cst.day, time_range=cst.s_cst.Julian_year, start_time_mjd=6e4):
    modified_julian_dates = np.arange(0.0, time_range, time_step) / cst.s_cst.day + start_time_mjd

    best_airmass = np.inf
    best_time = -1
    airmass_pre = np.inf
    skip_backward = False

    # Forward search
    for i, mjd in enumerate(modified_julian_dates):
        print(
            f"Querying Almanac for (RA {ra} dec {dec}) "
            f"at {mjd} MJD ({i + 1}/{modified_julian_dates.size})...",
            end='\r'
        )

        try:
            airmass = get_airmass(
                ra=ra,
                dec=dec,
                observatory=observatory,
                mjd=mjd
            )
        except json.JSONDecodeError:
            print("Skipping invalid result")
            continue

        if 1.0 <= airmass < best_airmass:
            best_airmass = airmass
            best_time = mjd
        elif airmass > airmass_pre:
            if i != 0:
                skip_backward = True  # the peak airmass has been found, no need to run the backward search

            break
        elif airmass == -1 and np.inf > airmass_pre > -1:
            break

        airmass_pre = airmass

    airmass_pre = np.inf

    # Backward search
    if not skip_backward:
        for i, mjd in enumerate(modified_julian_dates[::-1]):
            print(
                f"Querying Almanac for (RA {ra} dec {dec}) "
                f"at {mjd} MJD ({modified_julian_dates.size - i}/{modified_julian_dates.size})...",
                end='\r'
            )

            try:
                airmass = get_airmass(
                    ra=ra,
                    dec=dec,
                    observatory=observatory,
                    mjd=mjd
                )
            except json.JSONDecodeError:
                print("Skipping invalid result")
                continue

            if 1.0 <= airmass < best_airmass:
                best_airmass = airmass
                best_time = mjd
            elif airmass > airmass_pre:
                break
            elif airmass == -1 and np.inf > airmass_pre > -1:
                break

            airmass_pre = airmass

    print('\nDone')

    return best_time


def get_airmass(ra, dec, mjd, observatory='3060m'):
    """Get the airmass of a sky point at a given date from a given observatory, using ESO's Almanac.

    Args:
        ra: (deg) right ascension
        dec: (deg) declination
        mjd: (day) modified Julian date
        observatory: point of observation ("paranal", "lasilla", "3060m")

    Returns:
        The airmass.
    """
    indic = {
        'ra': ra,
        'dec': dec,
        'observatory': observatory,
        'mjd': mjd
    }

    alm = AlmanacQuery(indic)
    dic = alm.query()

    return dic['airmass']


def get_optimal_airmass_curve(ra, dec, dit, observation_duration, query_dit=120.0, observatory='3060m'):
    """Get the most favourable airmass curve of a sky point from an observation point, using ESO's Almanac.
    First, the day of the year at which the target is best observed is searched for. This corresponds to the day at
    which the airmass of the target at UTC 0 is the lowest. Local time, day and night, the moon position etc. are not
    taken into account.
    Then, the lowest airmass of this day is searched for. The dates are adjusted so that the lowest airmass corresponds
    to the observation mid-time.
    Finally, the curve is queried from the Almanac at the requested dates around the observation mid-time. The requested
    dates use the interval query_dit. The queried curve is then interpolated on dates using the requested DIT.

    Small query DIT gives more accurate results, but a lot of queries (slow) are necessary.

    Args:
        ra: (deg) right ascension
        dec: (deg) declination
        dit: (s) interval of time between observations
        observation_duration: (s) total duration of observations
        query_dit: (s) interval of time between queries
        observatory: point of observation ("paranal", "lasilla", "3060m")

    Returns:
        The times (s) to lowest airmass and the corresponding airmasses.
    """
    best_day = find_optimal_airmass_day(
        ra=ra,
        dec=dec,
        observatory=observatory,
        time_step=cst.s_cst.day,
        time_range=cst.s_cst.Julian_year
    )

    if query_dit is None:
        query_dit = dit
    elif query_dit < dit:
        query_dit = dit

    n_dit = int(np.ceil(observation_duration / dit))

    modified_julian_dates = np.concatenate(
        (np.arange(0.0, n_dit * dit / 2, query_dit), [n_dit * dit / 2])
    ) / cst.s_cst.day
    modified_julian_dates = np.concatenate((-modified_julian_dates[:0:-1], modified_julian_dates)) + best_day
    central_index = int(modified_julian_dates.size / 2)
    shift = 0

    airmasses = -np.ones(modified_julian_dates.size)

    # Find lowest airmass of the day
    # Try after the best day date
    print("Searching for lowest airmass...")

    for i, mjd in enumerate(modified_julian_dates[central_index:]):
        airmasses[central_index + i] = get_airmass(
            ra=ra,
            dec=dec,
            observatory=observatory,
            mjd=mjd
        )

        if central_index + i - 1 > 0:
            if airmasses[central_index + i] > airmasses[central_index + i - 1] > -1:
                shift = i - 1
                break

    if shift == 0:  # lowest airmass is before requested best day
        for i, mjd in enumerate(modified_julian_dates[central_index - 1::-1]):
            airmasses[central_index - 1 - i] = get_airmass(
                ra=ra,
                dec=dec,
                observatory=observatory,
                mjd=mjd
            )

            if central_index - 1 - i > 0:
                if airmasses[central_index - 1 - i] > airmasses[central_index - i] > -1:
                    shift = -i
                    break

    # Shift the dates array center to the lowest airmass
    modified_julian_dates = modified_julian_dates[central_index + shift] - best_day + modified_julian_dates
    airmasses = np.roll(airmasses, -shift)

    # Ensure that rolled over airmasses are -1
    if shift < 0:
        airmasses[:-shift] = -1
    elif shift > 0:
        airmasses[-shift:] = -1

    # Get query airmass curve
    for i, mjd in enumerate(modified_julian_dates):
        print(
            f"Querying Almanac for (RA {ra} dec {dec}) "
            f"at {mjd} MJD ({i + 1}/{modified_julian_dates.size})...",
            end='\r'
        )

        if airmasses[i] != -1:
            continue

        airmasses[i] = get_airmass(
            ra=ra,
            dec=dec,
            observatory=observatory,
            mjd=mjd
        )

    print('\nDone')

    # Interpolate queried airmass curve to requested times
    if query_dit != dit:
        modified_julian_dates_dit = np.concatenate(
            (np.arange(0.0, n_dit * dit / 2, dit), [n_dit * dit / 2])
        ) / cst.s_cst.day
        modified_julian_dates_dit = np.concatenate(
            (-modified_julian_dates_dit[:0:-1], modified_julian_dates_dit)
        ) + modified_julian_dates[central_index]

        airmasses = np.interp(modified_julian_dates_dit, modified_julian_dates, airmasses)
    else:
        modified_julian_dates_dit = modified_julian_dates

    times = (modified_julian_dates_dit - modified_julian_dates[central_index]) * cst.s_cst.day

    return times, airmasses


def get_telluric_data(airmass=1.0, pwv_mode='pwv', season=0, time=0, pwv=3.5, msolflux=130.0, incl_moon='Y',
                      moon_sun_sep=90.0, moon_target_sep=45.0, moon_alt=45.0, moon_earth_dist=1.0,
                      incl_starlight='Y', incl_zodiacal='Y', ecl_lon=135.0, ecl_lat=90.0, incl_loweratm='Y',
                      incl_upperatm='Y', incl_airglow='Y', incl_therm='N', therm_t1=0.0, therm_e1=0.0,
                      therm_t2=0.0, therm_e2=0.0, therm_t3=0.0, therm_e3=0.0, vacair='vac', wmin=300.0,
                      wmax=2000.0, wgrid_mode='fixed_spectral_resolution', wdelta=0.1, wres=1000000,
                      wgrid_user=None, lsf_type='none', lsf_gauss_fwhm=5.0, lsf_boxcar_fwhm=5.0,
                      observatory='3060', temp_flag=0):
    if wgrid_user is None:
        wgrid_user = [500.0, 510.0, 520.0, 530.0, 540.0, 550.0]

    sky_model = SkyModel()

    params = {
        # Airmass. Alt and airmass are coupled through the plane parallel
        # approximation airmass=sec(z), z being the zenith distance
        # z=90°−Alt
        'airmass': airmass,  # float range [1.0,3.0]

        # Season and Period of Night
        'pwv_mode': pwv_mode,  # string  grid ['pwv','season']
        # integer grid [0,1,2,3,4,5,6] (0=all year, 1=dec/jan,2=feb/mar...)
        'season': season,
        # third of night integer grid [0,1,2,3] (0=all year, 1,2,3 = third
        # of night)
        'time': time,

        # Precipitable Water Vapor PWV
        # mm float grid [-1.0,0.5,1.0,1.5,2.5,3.5,5.0,7.5,10.0,20.0]
        'pwv': pwv,

        # Monthly Averaged Solar Flux
        'msolflux': msolflux,  # s.f.u float > 0

        # Scattered Moon Light
        # Moon coordinate constraints: |z – zmoon| ≤ ρ ≤ |z + zmoon| where
        # ρ=moon/target separation, z=90°−target altitude and
        # zmoon=90°−moon altitude.
        # string grid ['Y','N'] flag for inclusion of scattered moonlight.
        'incl_moon': incl_moon,
        # degrees float range [0.0,360.0] Separation of Sun and Moon as
        # seen from Earth ("moon phase")
        'moon_sun_sep': moon_sun_sep,
        # degrees float range [0.0,180.0] Moon-Target Separation ( ρ )
        'moon_target_sep': moon_target_sep,
        # degrees float range [-90.0,90.0] Moon Altitude over Horizon
        'moon_alt': moon_alt,
        # float range [0.91,1.08] Moon-Earth Distance (mean=1)
        'moon_earth_dist': moon_earth_dist,

        # Starlight
        # string  grid ['Y','N'] flag for inclusion of scattered starlight
        'incl_starlight': incl_starlight,

        # Zodiacal light
        # string grid ['Y','N'] flag for inclusion of zodiacal light
        'incl_zodiacal': incl_zodiacal,
        # degrees float range [-180.0,180.0] Heliocentric ecliptic
        # longitude
        'ecl_lon': ecl_lon,
        # degrees float range [-90.0,90.0] Ecliptic latitude
        'ecl_lat': ecl_lat,

        # Molecular Emission of Lower Atmosphere
        # string grid ['Y','N'] flag for inclusion of lower atmosphere
        'incl_loweratm': incl_loweratm,
        # Emission Lines of Upper Atmosphere
        # string grid ['Y','N'] flag for inclusion of upper stmosphere
        'incl_upperatm': incl_upperatm,
        # Airglow Continuum (Residual Continuum)
        # string grid ['Y','N'] flag for inclusion of airglow
        'incl_airglow': incl_airglow,

        # Instrumental Thermal Emission This radiance component represents
        # an instrumental effect. The emission is provided relative to the
        # other model components. To obtain the correct absolute flux, an
        # instrumental response curve must be applied to the resulting
        # model spectrum See section 6.2.4 in the documentation
        # http://localhost/observing/etc/doc/skycalc/
        # The_Cerro_Paranal_Advanced_Sky_Model.pdf
        # string grid ['Y','N'] flag for inclusion of instrumental thermal
        # radiation
        'incl_therm': incl_therm,
        'therm_t1': therm_t1,  # K float > 0
        'therm_e1': therm_e1,  # float range [0,1]
        'therm_t2': therm_t2,  # K float > 0
        'therm_e2': therm_e2,  # float range [0,1]
        'therm_t3': therm_t3,  # float > 0
        'therm_e3': therm_e3,  # K float range [0,1]

        # Wavelength Grid
        'vacair': vacair,  # vac or air
        'wmin': wmin,  # nm float range [300.0,30000.0] < wmax
        'wmax': wmax,  # nm float range [300.0,30000.0] > wmin
        # string grid ['fixed_spectral_resolution','fixed_wavelength_step',
        #              'user']
        'wgrid_mode': wgrid_mode,
        # nm/step float range [0,30000.0] wavelength sampling step dlam
        # (not the res.element)
        'wdelta': wdelta,
        # float range [0,1.0e6] RESOLUTION is misleading, it is rather
        # lam/dlam where dlam is wavelength step (not the res.element)
        'wres': wres,
        'wgrid_user': wgrid_user,
        # convolve by Line Spread Function
        'lsf_type': lsf_type,  # string grid ['none','Gaussian','Boxcar']
        'lsf_gauss_fwhm': lsf_gauss_fwhm,  # wavelength bins float > 0
        'lsf_boxcar_fwhm': lsf_boxcar_fwhm,  # wavelength bins float > 0
        'observatory': observatory,  # paranal
        # compute temperature and bolometric radiance
        'temp_flag': temp_flag
    }

    for key, value in params.items():
        sky_model.params[key] = value

    sky_model.call()

    # Create a temporary file to dump the data into and read it later
    # Data are fits-formatted and cannot be read as it, thus the data has to be saved into a .fits file first, then read
    # with io.fits
    # io.fits cannot work if the file is already opened for writing, so it needs to be:
    # created, written, closed, re-opened with io.fits and read, before it can be deleted
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(sky_model.data)

    with fits.open(f.name) as f_fits:
        data = np.array(copy.deepcopy(f_fits[1].data))

    # Delete the temporary file
    os.unlink(f.name)

    return data


def get_tellurics_npz(file, wavelength_range=None, rewrite=False, **kwargs):
    """Get telluric transmittance and wavelength from a .npz file.
    If the file doesn't exist, a request to the skycalc server is sent to retrieve the data.
    While all the skycalc information is saved into the file, this function extracts only the wavelengths and
    transmittances.
    To prevent issues with large wavelength ranges, the request is split into 2.
    # TODO splitting could be much smarter (no split for small ranges, multiple splits for very large ones)

    Args:
        file: file from which to load the telluric transmittance and corresponding wavelengths
        wavelength_range: (um) list containing the min and max wavelengths
        rewrite: if True, the file is rewritten even if it already exists
        **kwargs: get_telluric_data arguments

    Returns:
        The wavelengths (um) and corresponding telluric transmittances
    """
    if not os.path.isfile(file) or rewrite:
        if wavelength_range is None:
            raise ValueError(f"argument 'wavelength_range' cannot be None: file '{file}' do not exists")

        # Prevent the CLI to complain about requesting too much data by splitting the request into 2
        wavelength_range_ = copy.deepcopy(wavelength_range)
        wavelength_range_[-1] = np.diff(wavelength_range)[0] / 2 + wavelength_range[0]

        telluric_data0 = get_telluric_data(
            wmin=wavelength_range_[0] * 1e3,  # um to nm
            wmax=wavelength_range_[1] * 1e3,  # um to nm
            **kwargs
        )

        wavelength_range_[0] = np.max(telluric_data0['lam'][-2]) * 1e-3
        wavelength_range_[-1] = wavelength_range[-1]

        telluric_data1 = get_telluric_data(
            wmin=wavelength_range_[0] * 1e3,  # um to nm
            wmax=wavelength_range_[1] * 1e3,  # um to nm
            **kwargs
        )

        wh = np.greater(telluric_data1['lam'], np.max(telluric_data0['lam']))
        wavelength_range_ = np.concatenate((
                telluric_data0['lam'],
                telluric_data1['lam'][wh]
            ))
        telluric_data = np.recarray(wavelength_range_.shape, dtype=telluric_data0.dtype)

        for key in telluric_data.dtype.names:
            telluric_data[key] = np.concatenate((
                telluric_data0[key],
                telluric_data1[key][wh]
            ))

        savez_compressed_record(file, telluric_data)
    else:
        telluric_data = np.load(file)

    wavelengths_telluric = telluric_data['lam'] * 1e-3  # nm to um
    telluric_transmittance = telluric_data['trans']

    return wavelengths_telluric, telluric_transmittance
