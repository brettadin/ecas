"""Stores the Planet object.
"""
import os
import warnings

import astropy.io.votable.exceptions
import astropy.units as u
import h5py
import numpy as np
import pyvo
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table.table import Table
from astropy.time import Time

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.config import petitradtrans_config_parser
from petitRADTRANS.math import calculate_uncertainty, longitude2phase, phase2longitude


class Planet:
    r"""Used to store planet- and star-related data and perform useful planet-related operations.

    The data can be automatically downloaded from the NASA Exoplanet Archive.

    Args:
        name:
            Name of the planet. When using the NASA Exoplanet Archive, the given name is case and space sensitive.
        mass:
            (g) Mass of the planet.
        mass_error_upper:
            (g) Upper error on the planet's mass.
        mass_error_lower:
            (g) Lower error on the planet's mass.
        radius:
            (cm) Radius of the planet.
        radius_error_upper:
            (cm) Upper error on the planet's radius.
        radius_error_lower:
            (cm) Lower error on the planet's radius.
        orbit_semi_major_axis:
            (cm) Orbit semi-major-axis of the planet.
        orbit_semi_major_axis_error_upper:
            (cm) Upper error on the planet's orbit semi-major-axis.
        orbit_semi_major_axis_error_lower:
            (cm) Lower error on the planet's orbit semi-major-axis.
        orbital_eccentricity:
            Orbital eccentricity of the planet.
        orbital_eccentricity_error_upper:
            Upper error on the planet's orbital eccentricity.
        orbital_eccentricity_error_lower:
            Lower error on the planet's orbital eccentricity.
        orbital_inclination:
            (deg) Orbital inclination of the planet.
        orbital_inclination_error_upper:
            (deg) Upper error on the planet's orbital inclination.
        orbital_inclination_error_lower:
            (deg) Lower error on the planet's orbital inclination.
        orbital_period:
            (s) Orbital period of the planet.
        orbital_period_error_upper:
            (s)
        orbital_period_error_lower:
            (s)
        argument_of_periastron:
            (deg) Argument of periastron of the planet.
        argument_of_periastron_error_upper:
            (deg)
        argument_of_periastron_error_lower:
            (deg)
        epoch_of_periastron:
            (s) Epoch of periastron of the planet.
        epoch_of_periastron_error_upper:
            (s)
        epoch_of_periastron_error_lower:
            (s)
        ra:
            Right ascension of the planet's star.
        dec:
            Declination of the planet's star.
        x:
            (cm) Position of the planet in a rectangular coordinate system along the x-axis (not used).
        y:
            (cm) Position of the planet in a rectangular coordinate system along the y-axis (not used).
        z:
            (cm) Position of the planet in a rectangular coordinate system along the z-axis (not used).
        reference_pressure:
            (bar) Reference pressure used to set the planet's radius.
        density:
            (g.cm-3) Density of the planet.
        density_error_upper:
            (g.cm-3)
        density_error_lower:
            (g.cm-3)
        reference_gravity:

        reference_gravity_error_upper:
            (cm.s-2)
        reference_gravity_error_lower:
            (cm.s-2)
        equilibrium_temperature:
            (K) Equilibrium temperature of the planet.
        equilibrium_temperature_error_upper:
            (K)
        equilibrium_temperature_error_lower:
            (K)
        insolation_flux:
            (erg.s.cm2) Flux received by the planet from its star.
        insolation_flux_error_upper:
            (erg.s.cm2)
        insolation_flux_error_lower:
            (erg.s.cm2)
        bond_albedo:
            Bond albedo of the planet.
        bond_albedo_error_upper:
        bond_albedo_error_lower:
        transit_depth:
            Transit depth of the planet, relative to its star.
        transit_depth_error_upper:
        transit_depth_error_lower:
        transit_midpoint_time:
            (s) Mid-transit time of the planet.
        transit_midpoint_time_error_upper:
            (s)
        transit_midpoint_time_error_lower:
            (s)
        transit_duration:
            (s) Duration of the planet's transit.
        transit_duration_error_upper:
            (s)
        transit_duration_error_lower:
            (s)
        projected_obliquity:
            (deg) Projected obliquity of the planet's orbit.
        projected_obliquity_error_upper:
            (deg)
        projected_obliquity_error_lower:
            (deg)
        true_obliquity:
            (deg) True obliquity of the planet's orbit.
        true_obliquity_error_upper:
            (deg)
        true_obliquity_error_lower:
            (deg)
        radial_velocity_semi_amplitude:
            (cm.s-1) Semi-amplitude of the planet's radial velocity from its orbital motion.
        radial_velocity_semi_amplitude_error_upper:
            (cm.s-1)
        radial_velocity_semi_amplitude_error_lower:
            (cm.s-1)
        planet_stellar_radius_ratio:
            Ratio between the planet's radius and its star's.
        planet_stellar_radius_ratio_error_upper:
        planet_stellar_radius_ratio_error_lower:
        semi_major_axis_stellar_radius_ratio:
            Ratio between the planet's orbit semi-major axis and its star's radius.
        semi_major_axis_stellar_radius_ratio_error_upper:
        semi_major_axis_stellar_radius_ratio_error_lower:
        reference:
            Reference for the planet's data.
        discovery_year:
            Year of discovery of the planet.
        discovery_method:
            Method of discovery of the planet.
        discovery_reference:
            Reference for the discovery of the planet.
        confirmation_status:
            Whether the planet's existence is confirmed.
        host_name:
            Name of the planet's host (star).
        star_spectral_type:
            Spectral type of the planet's star.
        star_mass:
            (g) Mass of the planet's star.
        star_mass_error_upper:
            (g)
        star_mass_error_lower:
            (g)
        star_radius:
            (cm) Radius of the planet's star.
        star_radius_error_upper:
            (cm)
        star_radius_error_lower:
            (cm)
        star_age:
            (s) Age of the planet's star.
        star_age_error_upper:
            (s)
        star_age_error_lower:
            (s)
        star_metallicity:
            Metallicity of the planet's star, relative to the solar metallicity.
        star_metallicity_error_upper:
        star_metallicity_error_lower:
        star_effective_temperature:
            (K) Effective temperature of the planet's star.
        star_effective_temperature_error_upper:
            (K)
        star_effective_temperature_error_lower:
            (K)
        star_luminosity:
            (erg.s-1) Luminosity of the planet's star.
        star_luminosity_error_upper:
            (erg.s-1)
        star_luminosity_error_lower:
            (erg.s-1)
        star_rotational_period:
            (s) Rotational period of the planet's star.
        star_rotational_period_error_upper:
            (s)
        star_rotational_period_error_lower:
            (s)
        star_radial_velocity:
            (cm.s-1) Radial velocity of the planet's star.
        star_radial_velocity_error_upper:
            (cm.s-1)
        star_radial_velocity_error_lower:
            (cm.s-1)
        star_rotational_velocity:
            (cm.s-1) Rotational velocity of the planet's star.
        star_rotational_velocity_error_upper:
            (cm.s-1)
        star_rotational_velocity_error_lower:
            (cm.s-1)
        star_density:
            (g.cm-3) Density of the planet's star.
        star_density_error_upper:
            (g.cm-3)
        star_density_error_lower:
            (g.cm-3)
        star_reference_gravity:
            (cm.s-2) Reference gravity of the planet's star.
        star_reference_gravity_error_upper:
            (cm.s-2)
        star_reference_gravity_error_lower:
            (cm.s-2)
        star_reference:
            Reference for the data of the planet's star.
        system_star_number:
            Number of stars in the planet's system.
        system_planet_number:
            Number of planets in the planet's system.
        system_moon_number:
            Number of moons in the planet's system.
        system_distance:
            (cm) Distance between the Solar system barycenter and the barycenter of the planet's system.
        system_distance_error_upper:
            (cm)
        system_distance_error_lower:
            (cm)
        system_apparent_magnitude_v:
            Apparent magnitude of the planet's star in band V.
        system_apparent_magnitude_v_error_upper:
        system_apparent_magnitude_v_error_lower:
        system_apparent_magnitude_j:
            Apparent magnitude of the planet's star in band J.
        system_apparent_magnitude_j_error_upper:
        system_apparent_magnitude_j_error_lower:
        system_apparent_magnitude_k:
            Apparent magnitude of the planet's star in band K.
        system_apparent_magnitude_k_error_upper:
        system_apparent_magnitude_k_error_lower:
        system_proper_motion:
            (deg.s-1) Total proper motion of the planet's system.
        system_proper_motion_error_upper:
            (deg.s-1)
        system_proper_motion_error_lower:
            (deg.s-1)
        system_proper_motion_ra:
            (deg.s-1) Right ascension proper motion of the planet's system.
        system_proper_motion_ra_error_upper:
            (deg.s-1)
        system_proper_motion_ra_error_lower:
            (deg.s-1)
        system_proper_motion_dec:
            (deg.s-1) Declination proper motion of the planet's system.
        system_proper_motion_dec_error_upper:
            (deg.s-1)
        system_proper_motion_dec_error_lower:
            (deg.s-1)
        units:
            Units of the attributes.
    """
    def __init__(
            self,
            name,
            mass=0.,
            mass_error_upper=0.,
            mass_error_lower=0.,
            radius=0.,
            radius_error_upper=0.,
            radius_error_lower=0.,
            orbit_semi_major_axis=0.,
            orbit_semi_major_axis_error_upper=0.,
            orbit_semi_major_axis_error_lower=0.,
            orbital_eccentricity=0.,
            orbital_eccentricity_error_upper=0.,
            orbital_eccentricity_error_lower=0.,
            orbital_inclination=0.,
            orbital_inclination_error_upper=0.,
            orbital_inclination_error_lower=0.,
            orbital_period=0.,
            orbital_period_error_upper=0.,
            orbital_period_error_lower=0.,
            argument_of_periastron=0.,
            argument_of_periastron_error_upper=0.,
            argument_of_periastron_error_lower=0.,
            epoch_of_periastron=0.,
            epoch_of_periastron_error_upper=0.,
            epoch_of_periastron_error_lower=0.,
            ra=0.,
            dec=0.,
            x=0.,
            y=0.,
            z=0.,
            reference_pressure=0.01,
            density=0.,
            density_error_upper=0.,
            density_error_lower=0.,
            reference_gravity=0.,
            reference_gravity_error_upper=0.,
            reference_gravity_error_lower=0.,
            equilibrium_temperature=0.,
            equilibrium_temperature_error_upper=0.,
            equilibrium_temperature_error_lower=0.,
            insolation_flux=0.,
            insolation_flux_error_upper=0.,
            insolation_flux_error_lower=0.,
            bond_albedo=0.,
            bond_albedo_error_upper=0.,
            bond_albedo_error_lower=0.,
            transit_depth=0.,
            transit_depth_error_upper=0.,
            transit_depth_error_lower=0.,
            transit_midpoint_time=0.,
            transit_midpoint_time_error_upper=0.,
            transit_midpoint_time_error_lower=0.,
            transit_duration=0.,
            transit_duration_error_upper=0.,
            transit_duration_error_lower=0.,
            projected_obliquity=0.,
            projected_obliquity_error_upper=0.,
            projected_obliquity_error_lower=0.,
            true_obliquity=0.,
            true_obliquity_error_upper=0.,
            true_obliquity_error_lower=0.,
            radial_velocity_semi_amplitude=0.,
            radial_velocity_semi_amplitude_error_upper=0.,
            radial_velocity_semi_amplitude_error_lower=0.,
            planet_stellar_radius_ratio=0.,
            planet_stellar_radius_ratio_error_upper=0.,
            planet_stellar_radius_ratio_error_lower=0.,
            semi_major_axis_stellar_radius_ratio=0.,
            semi_major_axis_stellar_radius_ratio_error_upper=0.,
            semi_major_axis_stellar_radius_ratio_error_lower=0.,
            reference='',
            discovery_year=0,
            discovery_method='',
            discovery_reference='',
            confirmation_status='',
            host_name='',
            star_spectral_type='',
            star_mass=0.,
            star_mass_error_upper=0.,
            star_mass_error_lower=0.,
            star_radius=0.,
            star_radius_error_upper=0.,
            star_radius_error_lower=0.,
            star_age=0.,
            star_age_error_upper=0.,
            star_age_error_lower=0.,
            star_metallicity=0.,
            star_metallicity_error_upper=0.,
            star_metallicity_error_lower=0.,
            star_effective_temperature=0.,
            star_effective_temperature_error_upper=0.,
            star_effective_temperature_error_lower=0.,
            star_luminosity=0.,
            star_luminosity_error_upper=0.,
            star_luminosity_error_lower=0.,
            star_rotational_period=0.,
            star_rotational_period_error_upper=0.,
            star_rotational_period_error_lower=0.,
            star_radial_velocity=0.,
            star_radial_velocity_error_upper=0.,
            star_radial_velocity_error_lower=0.,
            star_rotational_velocity=0.,
            star_rotational_velocity_error_upper=0.,
            star_rotational_velocity_error_lower=0.,
            star_density=0.,
            star_density_error_upper=0.,
            star_density_error_lower=0.,
            star_reference_gravity=0.,
            star_reference_gravity_error_upper=0.,
            star_reference_gravity_error_lower=0.,
            star_reference='',
            system_star_number=0,
            system_planet_number=0,
            system_moon_number=0,
            system_distance=0.,
            system_distance_error_upper=0.,
            system_distance_error_lower=0.,
            system_apparent_magnitude_v=0.,
            system_apparent_magnitude_v_error_upper=0.,
            system_apparent_magnitude_v_error_lower=0.,
            system_apparent_magnitude_j=0.,
            system_apparent_magnitude_j_error_upper=0.,
            system_apparent_magnitude_j_error_lower=0.,
            system_apparent_magnitude_k=0.,
            system_apparent_magnitude_k_error_upper=0.,
            system_apparent_magnitude_k_error_lower=0.,
            system_proper_motion=0.,
            system_proper_motion_error_upper=0.,
            system_proper_motion_error_lower=0.,
            system_proper_motion_ra=0.,
            system_proper_motion_ra_error_upper=0.,
            system_proper_motion_ra_error_lower=0.,
            system_proper_motion_dec=0.,
            system_proper_motion_dec_error_upper=0.,
            system_proper_motion_dec_error_lower=0.,
            units=None
    ):
        self.name = name
        self.mass = mass
        self.mass_error_upper = mass_error_upper
        self.mass_error_lower = mass_error_lower
        self.radius = radius
        self.radius_error_upper = radius_error_upper
        self.radius_error_lower = radius_error_lower
        self.orbit_semi_major_axis = orbit_semi_major_axis
        self.orbit_semi_major_axis_error_upper = orbit_semi_major_axis_error_upper
        self.orbit_semi_major_axis_error_lower = orbit_semi_major_axis_error_lower
        self.orbital_eccentricity = orbital_eccentricity
        self.orbital_eccentricity_error_upper = orbital_eccentricity_error_upper
        self.orbital_eccentricity_error_lower = orbital_eccentricity_error_lower
        self.orbital_inclination = orbital_inclination
        self.orbital_inclination_error_upper = orbital_inclination_error_upper
        self.orbital_inclination_error_lower = orbital_inclination_error_lower
        self.orbital_period = orbital_period
        self.orbital_period_error_upper = orbital_period_error_upper
        self.orbital_period_error_lower = orbital_period_error_lower
        self.argument_of_periastron = argument_of_periastron
        self.argument_of_periastron_error_upper = argument_of_periastron_error_upper
        self.argument_of_periastron_error_lower = argument_of_periastron_error_lower
        self.epoch_of_periastron = epoch_of_periastron
        self.epoch_of_periastron_error_upper = epoch_of_periastron_error_upper
        self.epoch_of_periastron_error_lower = epoch_of_periastron_error_lower
        self.ra = ra
        self.dec = dec
        self.x = x
        self.y = y
        self.z = z
        self.reference_pressure = reference_pressure
        self.density = density
        self.density_error_upper = density_error_upper
        self.density_error_lower = density_error_lower
        self.reference_gravity = reference_gravity
        self.reference_gravity_error_upper = reference_gravity_error_upper
        self.reference_gravity_error_lower = reference_gravity_error_lower
        self.equilibrium_temperature = equilibrium_temperature
        self.equilibrium_temperature_error_upper = equilibrium_temperature_error_upper
        self.equilibrium_temperature_error_lower = equilibrium_temperature_error_lower
        self.insolation_flux = insolation_flux
        self.insolation_flux_error_upper = insolation_flux_error_upper
        self.insolation_flux_error_lower = insolation_flux_error_lower
        self.bond_albedo = bond_albedo
        self.bond_albedo_error_upper = bond_albedo_error_upper
        self.bond_albedo_error_lower = bond_albedo_error_lower
        self.transit_depth = transit_depth
        self.transit_depth_error_upper = transit_depth_error_upper
        self.transit_depth_error_lower = transit_depth_error_lower
        self.transit_midpoint_time = transit_midpoint_time
        self.transit_midpoint_time_error_upper = transit_midpoint_time_error_upper
        self.transit_midpoint_time_error_lower = transit_midpoint_time_error_lower
        self.transit_duration = transit_duration
        self.transit_duration_error_upper = transit_duration_error_upper
        self.transit_duration_error_lower = transit_duration_error_lower
        self.projected_obliquity = projected_obliquity
        self.projected_obliquity_error_upper = projected_obliquity_error_upper
        self.projected_obliquity_error_lower = projected_obliquity_error_lower
        self.true_obliquity = true_obliquity
        self.true_obliquity_error_upper = true_obliquity_error_upper
        self.true_obliquity_error_lower = true_obliquity_error_lower
        self.radial_velocity_semi_amplitude = radial_velocity_semi_amplitude
        self.radial_velocity_semi_amplitude_error_upper = radial_velocity_semi_amplitude_error_upper
        self.radial_velocity_semi_amplitude_error_lower = radial_velocity_semi_amplitude_error_lower
        self.planet_stellar_radius_ratio = planet_stellar_radius_ratio
        self.planet_stellar_radius_ratio_error_upper = planet_stellar_radius_ratio_error_upper
        self.planet_stellar_radius_ratio_error_lower = planet_stellar_radius_ratio_error_lower
        self.semi_major_axis_stellar_radius_ratio = semi_major_axis_stellar_radius_ratio
        self.semi_major_axis_stellar_radius_ratio_error_upper = semi_major_axis_stellar_radius_ratio_error_upper
        self.semi_major_axis_stellar_radius_ratio_error_lower = semi_major_axis_stellar_radius_ratio_error_lower
        self.reference = reference
        self.discovery_year = discovery_year
        self.discovery_method = discovery_method
        self.discovery_reference = discovery_reference
        self.confirmation_status = confirmation_status
        self.host_name = host_name
        self.star_spectral_type = star_spectral_type
        self.star_mass = star_mass
        self.star_mass_error_upper = star_mass_error_upper
        self.star_mass_error_lower = star_mass_error_lower
        self.star_radius = star_radius
        self.star_radius_error_upper = star_radius_error_upper
        self.star_radius_error_lower = star_radius_error_lower
        self.star_age = star_age
        self.star_age_error_upper = star_age_error_upper
        self.star_age_error_lower = star_age_error_lower
        self.star_metallicity = star_metallicity
        self.star_metallicity_error_upper = star_metallicity_error_upper
        self.star_metallicity_error_lower = star_metallicity_error_lower
        self.star_effective_temperature = star_effective_temperature
        self.star_effective_temperature_error_upper = star_effective_temperature_error_upper
        self.star_effective_temperature_error_lower = star_effective_temperature_error_lower
        self.star_luminosity = star_luminosity
        self.star_luminosity_error_upper = star_luminosity_error_upper
        self.star_luminosity_error_lower = star_luminosity_error_lower
        self.star_rotational_period = star_rotational_period
        self.star_rotational_period_error_upper = star_rotational_period_error_upper
        self.star_rotational_period_error_lower = star_rotational_period_error_lower
        self.star_radial_velocity = star_radial_velocity
        self.star_radial_velocity_error_upper = star_radial_velocity_error_upper
        self.star_radial_velocity_error_lower = star_radial_velocity_error_lower
        self.star_rotational_velocity = star_rotational_velocity
        self.star_rotational_velocity_error_upper = star_rotational_velocity_error_upper
        self.star_rotational_velocity_error_lower = star_rotational_velocity_error_lower
        self.star_density = star_density
        self.star_density_error_upper = star_density_error_upper
        self.star_density_error_lower = star_density_error_lower
        self.star_reference_gravity = star_reference_gravity
        self.star_reference_gravity_error_upper = star_reference_gravity_error_upper
        self.star_reference_gravity_error_lower = star_reference_gravity_error_lower
        self.star_reference = star_reference
        self.system_star_number = system_star_number
        self.system_planet_number = system_planet_number
        self.system_moon_number = system_moon_number
        self.system_distance = system_distance
        self.system_distance_error_upper = system_distance_error_upper
        self.system_distance_error_lower = system_distance_error_lower
        self.system_apparent_magnitude_v = system_apparent_magnitude_v
        self.system_apparent_magnitude_v_error_upper = system_apparent_magnitude_v_error_upper
        self.system_apparent_magnitude_v_error_lower = system_apparent_magnitude_v_error_lower
        self.system_apparent_magnitude_j = system_apparent_magnitude_j
        self.system_apparent_magnitude_j_error_upper = system_apparent_magnitude_j_error_upper
        self.system_apparent_magnitude_j_error_lower = system_apparent_magnitude_j_error_lower
        self.system_apparent_magnitude_k = system_apparent_magnitude_k
        self.system_apparent_magnitude_k_error_upper = system_apparent_magnitude_k_error_upper
        self.system_apparent_magnitude_k_error_lower = system_apparent_magnitude_k_error_lower
        self.system_proper_motion = system_proper_motion
        self.system_proper_motion_error_upper = system_proper_motion_error_upper
        self.system_proper_motion_error_lower = system_proper_motion_error_lower
        self.system_proper_motion_ra = system_proper_motion_ra
        self.system_proper_motion_ra_error_upper = system_proper_motion_ra_error_upper
        self.system_proper_motion_ra_error_lower = system_proper_motion_ra_error_lower
        self.system_proper_motion_dec = system_proper_motion_dec
        self.system_proper_motion_dec_error_upper = system_proper_motion_dec_error_upper
        self.system_proper_motion_dec_error_lower = system_proper_motion_dec_error_lower

        if units is None:
            self.units = {
                'name': 'N/A',
                'mass': 'g',
                'mass_error_upper': 'g',
                'mass_error_lower': 'g',
                'radius': 'cm',
                'radius_error_upper': 'cm',
                'radius_error_lower': 'cm',
                'orbit_semi_major_axis': 'cm',
                'orbit_semi_major_axis_error_upper': 'cm',
                'orbit_semi_major_axis_error_lower': 'cm',
                'orbital_eccentricity': 'None',
                'orbital_eccentricity_error_upper': 'None',
                'orbital_eccentricity_error_lower': 'None',
                'orbital_inclination': 'deg',
                'orbital_inclination_error_upper': 'deg',
                'orbital_inclination_error_lower': 'deg',
                'orbital_period': 's',
                'orbital_period_error_upper': 's',
                'orbital_period_error_lower': 's',
                'argument_of_periastron': 'deg',
                'argument_of_periastron_error_upper': 'deg',
                'argument_of_periastron_error_lower': 'deg',
                'epoch_of_periastron': 's',
                'epoch_of_periastron_error_upper': 's',
                'epoch_of_periastron_error_lower': 's',
                'ra': 'deg',
                'dec': 'deg',
                'x': 'cm',
                'y': 'cm',
                'z': 'cm',
                'reference_pressure': 'bar',
                'density': 'g/cm^3',
                'density_error_upper': 'g/cm^3',
                'density_error_lower': 'g/cm^3',
                'reference_gravity': 'cm/s^2',
                'reference_gravity_error_upper': 'cm/s^2',
                'reference_gravity_error_lower': 'cm/s^2',
                'equilibrium_temperature': 'K',
                'equilibrium_temperature_error_upper': 'K',
                'equilibrium_temperature_error_lower': 'K',
                'insolation_flux': 'erg/s/cm^2',
                'insolation_flux_error_upper': 'erg/s/cm^2',
                'insolation_flux_error_lower': 'erg/s/cm^2',
                'bond_albedo': 'None',
                'bond_albedo_error_upper': 'None',
                'bond_albedo_error_lower': 'None',
                'transit_depth': 'None',
                'transit_depth_error_upper': 'None',
                'transit_depth_error_lower': 'None',
                'transit_midpoint_time': 's',
                'transit_midpoint_time_error_upper': 's',
                'transit_midpoint_time_error_lower': 's',
                'transit_duration': 's',
                'transit_duration_error_upper': 's',
                'transit_duration_error_lower': 's',
                'projected_obliquity': 'deg',
                'projected_obliquity_error_upper': 'deg',
                'projected_obliquity_error_lower': 'deg',
                'true_obliquity': 'deg',
                'true_obliquity_error_upper': 'deg',
                'true_obliquity_error_lower': 'deg',
                'radial_velocity_semi_amplitude': 'cm/s',
                'radial_velocity_semi_amplitude_error_upper': 'cm/s',
                'radial_velocity_semi_amplitude_error_lower': 'cm/s',
                'planet_stellar_radius_ratio': 'None',
                'planet_stellar_radius_ratio_error_upper': 'None',
                'planet_stellar_radius_ratio_error_lower': 'None',
                'semi_major_axis_stellar_radius_ratio': 'None',
                'semi_major_axis_stellar_radius_ratio_error_upper': 'None',
                'semi_major_axis_stellar_radius_ratio_error_lower': 'None',
                'reference': 'N/A',
                'discovery_year': 'year',
                'discovery_method': 'N/A',
                'discovery_reference': 'N/A',
                'confirmation_status': 'N/A',
                'host_name': 'N/A',
                'star_spectral_type': 'N/A',
                'star_mass': 'g',
                'star_mass_error_upper': 'g',
                'star_mass_error_lower': 'g',
                'star_radius': 'cm',
                'star_radius_error_upper': 'cm',
                'star_radius_error_lower': 'cm',
                'star_age': 's',
                'star_age_error_upper': 's',
                'star_age_error_lower': 's',
                'star_metallicity': 'dex',
                'star_metallicity_error_upper': 'dex',
                'star_metallicity_error_lower': 'dex',
                'star_effective_temperature': 'K',
                'star_effective_temperature_error_upper': 'K',
                'star_effective_temperature_error_lower': 'K',
                'star_luminosity': 'erg/s',
                'star_luminosity_error_upper': 'erg/s',
                'star_luminosity_error_lower': 'erg/s',
                'star_rotational_period': 's',
                'star_rotational_period_error_upper': 's',
                'star_rotational_period_error_lower': 's',
                'star_radial_velocity': 'cm/s',
                'star_radial_velocity_error_upper': 'cm/s',
                'star_radial_velocity_error_lower': 'cm/s',
                'star_rotational_velocity': 'cm/s',
                'star_rotational_velocity_error_upper': 'cm/s',
                'star_rotational_velocity_error_lower': 'cm/s',
                'star_density': 'g/cm^3',
                'star_density_error_upper': 'g/cm^3',
                'star_density_error_lower': 'g/cm^3',
                'star_reference_gravity': 'cm/s^2',
                'star_reference_gravity_error_upper': 'cm/s^2',
                'star_reference_gravity_error_lower': 'cm/s^2',
                'star_reference': 'N/A',
                'system_star_number': 'None',
                'system_planet_number': 'None',
                'system_moon_number': 'None',
                'system_distance': 'cm',
                'system_distance_error_upper': 'cm',
                'system_distance_error_lower': 'cm',
                'system_apparent_magnitude_v': 'None',
                'system_apparent_magnitude_v_error_upper': 'None',
                'system_apparent_magnitude_v_error_lower': 'None',
                'system_apparent_magnitude_j': 'None',
                'system_apparent_magnitude_j_error_upper': 'None',
                'system_apparent_magnitude_j_error_lower': 'None',
                'system_apparent_magnitude_k': 'None',
                'system_apparent_magnitude_k_error_upper': 'None',
                'system_apparent_magnitude_k_error_lower': 'None',
                'system_proper_motion': 'deg/s',
                'system_proper_motion_error_upper': 'deg/s',
                'system_proper_motion_error_lower': 'deg/s',
                'system_proper_motion_ra': 'deg/s',
                'system_proper_motion_ra_error_upper': 'deg/s',
                'system_proper_motion_ra_error_lower': 'deg/s',
                'system_proper_motion_dec': 'deg/s',
                'system_proper_motion_dec_error_upper': 'deg/s',
                'system_proper_motion_dec_error_lower': 'deg/s',
                'units': 'N/A'
            }
        else:
            self.units = units

    @staticmethod
    def _convert_nasa_exoplanet_archive(value, key, verbose=False, use_best_mass=False):
        skip_unit_conversion = False

        # Heads
        if key[:3] == 'sy_':
            key = 'system_' + key[3:]
        elif key[:3] == 'st_':
            key = 'star_' + key[3:]
        elif key[:5] == 'disc_':
            key = 'discovery_' + key[5:]

        # Tails
        if key[-4:] == 'err1':
            key = key[:-4] + '_error_upper'
        elif key[-4:] == 'err2':
            key = key[:-4] + '_error_lower'
        elif key[-3:] == 'lim':
            key = key[:-3] + '_limit_flag'
            skip_unit_conversion = True
        elif key[-3:] == 'str':
            key = key[:-3] + '_str'
            skip_unit_conversion = True

        # Parameters of interest
        if '_orbper' in key:
            key = key.replace('_orbper', '_orbital_period')

            if not skip_unit_conversion:
                value *= cst.s_cst.day
        elif '_orblper' in key:
            key = key.replace('_orblper', '_argument_of_periastron')
        elif '_orbsmax' in key:
            key = key.replace('_orbsmax', '_orbit_semi_major_axis')

            if not skip_unit_conversion:
                value *= cst.au
        elif '_orbincl' in key:
            key = key.replace('_orbincl', '_orbital_inclination')
        elif '_orbtper' in key:
            key = key.replace('_orbtper', '_epoch_of_periastron')

            if not skip_unit_conversion:
                value *= cst.s_cst.day
        elif '_orbeccen' in key:
            key = key.replace('_orbeccen', '_orbital_eccentricity')
        elif '_eqt' in key:
            key = key.replace('_eqt', '_equilibrium_temperature')
        elif '_occdep' in key:
            key = key.replace('_occdep', '_occultation_depth')
        elif '_insol' in key:
            key = key.replace('_insol', '_insolation_flux')

            if not skip_unit_conversion:
                value *= cst.s_earth
        elif '_dens' in key:
            key = key.replace('_dens', '_density')
        elif '_trandep' in key:
            key = key.replace('_trandep', '_transit_depth')

            if not skip_unit_conversion:
                value *= 1e2
        elif '_tranmid' in key:
            key = key.replace('_tranmid', '_transit_midpoint_time')

            if not skip_unit_conversion:
                value *= cst.s_cst.day
        elif '_trandur' in key:
            key = key.replace('_trandur', '_transit_duration')

            if not skip_unit_conversion:
                value *= cst.s_cst.hour
        elif '_spectype' in key:
            key = key.replace('_spectype', '_spectral_type')
        elif '_rotp' in key:
            key = key.replace('_rotp', '_rotational_period')

            if not skip_unit_conversion:
                value *= cst.s_cst.day
        elif '_projobliq' in key:
            key = key.replace('_projobliq', '_projected_obliquity')
        elif '_rvamp' in key:
            key = key.replace('_rvamp', '_radial_velocity_semi_amplitude')

            if not skip_unit_conversion:
                value *= 1e2
        elif '_radj' in key:
            key = key.replace('_radj', '_radius')

            if not skip_unit_conversion:
                value *= cst.r_jup
        elif '_ratror' in key:
            key = key.replace('_ratror', '_planet_stellar_radius_ratio')
        elif '_trueobliq' in key:
            key = key.replace('_trueobliq', '_true_obliquity')
        elif '_ratdor' in key:
            key = key.replace('_ratdor', '_semi_major_axis_stellar_radius_ratio')
        elif '_imppar' in key:
            key = key.replace('_imppar', '_impact_parameter')
        elif '_msinij' in key:
            key = key.replace('_msinij', '_mass_sini')

            if not skip_unit_conversion:
                value *= cst.m_jup
        elif '_massj' in key:
            if not use_best_mass:
                key = key.replace('_massj', '_mass')

                if not skip_unit_conversion:
                    value *= cst.m_jup
        elif '_bmassj' in key:
            if use_best_mass:
                key = key.replace('_bmassj', '_mass')

                if not skip_unit_conversion:
                    value *= cst.m_jup
        elif '_teff' in key:
            key = key.replace('_teff', '_effective_temperature')
        elif '_met' in key:
            key = key.replace('_met', '_metallicity')
        elif '_radv' in key:
            key = key.replace('_radv', '_radial_velocity')

            if not skip_unit_conversion:
                value *= 1e5
        elif '_vsin' in key:
            key = key.replace('_vsin', '_rotational_velocity')

            if not skip_unit_conversion:
                value *= 1e5
        elif '_lum' in key:
            key = key.replace('_lum', '_luminosity')

            if not skip_unit_conversion:
                value = 10 ** value * cst.l_sun
        elif '_logg' in key:
            key = key.replace('_logg', '_reference_gravity')

            if not skip_unit_conversion:
                value = 10 ** value
        elif '_age' in key:
            if not skip_unit_conversion:
                value *= 1e9 * cst.s_cst.year
        elif 'star_mass' in key:
            if not skip_unit_conversion:
                value *= cst.m_sun
        elif 'star_rad' in key:
            key = key.replace('star_rad', 'star_radius')

            if not skip_unit_conversion:
                value *= cst.r_sun
        elif '_dist' in key:
            key = key.replace('_dist', '_distance')

            if not skip_unit_conversion:
                value *= cst.pc
        elif '_plx' in key:
            key = key.replace('_plx', '_parallax')

            if not skip_unit_conversion:
                value *= 3.6e-6
        elif '_pm' in key:
            if key[-3:] == '_pm':
                key = key.replace('_pm', '_proper_motion')
            else:
                i = key.find('_pm')
                key = key[:i] + '_proper_motion_' + key[i + len('_pm'):]

            if not skip_unit_conversion:
                value *= np.deg2rad(1e-3 / 3600 / cst.s_cst.year)
        elif key == 'hostname':
            key = 'host_name'
        elif key == 'discoverymethod':
            key = 'discovery_method'
        elif key == 'discovery_refname':
            key = 'discovery_reference'
        elif 'controv_flag' in key:
            key = 'controversy_flag'
        elif key == 'star_refname':
            key = 'star_reference'
        elif key == 'soltype':
            key = 'confirmation_status'
        elif key == 'system_snum':
            key = 'system_star_number'
        elif key == 'system_pnum':
            key = 'system_planet_number'
        elif key == 'system_mnum':
            key = 'system_moon_number'
        elif 'mag' in key:
            i = key.find('mag')

            if i + len('mag') == len(key):
                tail = ''
            else:
                tail = key[i + 3:]

                if tail[0] != '_':  # should not be necessary
                    tail = '_' + tail

            # Move magnitude band to the end
            if key[i - 2] == '_':  # one-character band
                letter = key[i - 1]
                key = key[:i - 1] + 'apparent_magnitude_' + letter + tail
            elif key[i - 3] == '_':  # two-characters band
                letters = key[i - 2:i]
                key = key[:i - 2] + 'apparent_magnitude_' + letters + tail
            elif 'kepmag' in key:
                key = key[:i - 3] + 'apparent_magnitude_' + 'kepler' + tail
            elif 'gaiamag' in key:
                key = key[:i - 4] + 'apparent_magnitude_' + 'gaia' + tail
            else:
                raise ValueError(f"unidentified apparent magnitude key '{key}'")
        elif verbose:
            print(f"unchanged key '{key}' with value {value}")

        if key[:3] == 'pl_':
            key = key[3:]

        return value, key

    @staticmethod
    def _select_best_in_column(dictionary):
        parameter_dict = {}
        tails = ['_error_upper', '_error_lower', '_limit_flag', '_str']

        for key in dictionary.keys():
            if tails[0] in key or tails[1] in key or tails[2] in key or tails[3] in key:
                continue  # skip every tailed parameters
            elif dictionary[key].dtype == object or not (key + tails[0] in dictionary and key + tails[1] in dictionary):
                # if object or no error tailed parameters, get the first value that is not masked
                if not hasattr(dictionary[key], '__iter__') or np.size(dictionary[key]) == 0:
                    raise ValueError(f"No value found for parameter '{key}'; "
                                     f"this error is most often caused by the misspelling of a planet name")

                parameter_dict[key] = dictionary[key][0]

                for value in dictionary[key][1:]:
                    if not hasattr(value, 'mask'):
                        parameter_dict[key] = value

                        break
            else:
                value_error_upper = dictionary[key + tails[0]]
                value_error_lower = dictionary[key + tails[1]]
                error_interval = np.abs(value_error_upper) + np.abs(value_error_lower)

                wh = np.where(error_interval == np.min(error_interval))[0]

                parameter_dict[key] = dictionary[key][wh][0]

                for tail in tails:
                    if key + tail in dictionary:
                        parameter_dict[key + tail] = dictionary[key + tail][wh][0]

        return parameter_dict

    @staticmethod
    def bjd_utc2bjd_tdb(times_utc, ra, dec, site_name=None, latitude=None, longitude=None, height=None):
        observer_location, target_coordinates = Planet.get_astropy_coordinates(
            ra=ra,
            dec=dec,
            site_name=site_name,
            latitude=latitude,
            longitude=longitude,
            height=height
        )

        times_utc = Time(times_utc, format='jd', scale='utc')
        times_tdb = times_utc.tdb + times_utc.light_travel_time(target_coordinates, location=observer_location)

        return times_tdb.value

    def calculate_airmass(self, time, site_name=None, latitude=None, longitude=None, height=None,
                          time_format='mjd'):
        return self.compute_airmass(
            ra=self.ra,
            dec=self.dec,
            time=time,
            site_name=site_name,
            latitude=latitude,
            longitude=longitude,
            height=height,
            time_format=time_format
        )

    def calculate_barycentric_velocities(self, time, site_name=None, latitude=None, longitude=None, height=None,
                                         time_format='mjd'):
        return self.compute_barycentric_velocities(
            ra=self.ra,
            dec=self.dec,
            time=time,
            site_name=site_name,
            latitude=latitude,
            longitude=longitude,
            height=height,
            time_format=time_format
        )

    def calculate_equilibrium_temperature(self):
        """Calculate the equilibrium temperature of a planet.
        """

        return self.compute_equilibrium_temperature(
            orbit_semi_major_axis=self.orbit_semi_major_axis,
            star_effective_temperature=self.star_effective_temperature,
            star_radius=self.star_radius,
            bond_albedo=self.bond_albedo,
            orbit_semi_major_axis_error_lower=self.orbit_semi_major_axis_error_lower,
            orbit_semi_major_axis_error_upper=self.orbit_semi_major_axis_error_upper,
            star_effective_temperature_error_lower=self.star_effective_temperature_error_lower,
            star_effective_temperature_error_upper=self.star_effective_temperature_error_upper,
            star_radius_error_lower=self.star_radius_error_lower,
            star_radius_error_upper=self.star_radius_error_upper
        )

    def calculate_intrinsic_temperature(self) -> tuple[float, float, float]:
        """Calculate the intrinsic temperature of a planet.
        """

        return self.compute_intrinsic_temperature(
            mass=self.mass,
            radius=self.radius,
            star_age=self.star_age,
            mass_error_lower=self.mass_error_lower,
            mass_error_upper=self.mass_error_upper,
            radius_error_lower=self.radius_error_lower,
            radius_error_upper=self.radius_error_upper,
            star_age_error_lower=self.star_age_error_lower,
            star_age_error_upper=self.star_age_error_upper
        )

    def calculate_intrinsic_temperature_inflated(self, equilibrium_temperature: float = None) -> float:
        """Calculate the intrinsic temperature of inflated hot Jupiters.
        """
        if not (0.1 * cst.m_jup < self.mass < 10 * cst.m_jup):
            warnings.warn(f"Thorngren et al. 2019 simplified Eq. 3 was calibrated "
                          f"for hot Jupiters, with masses between 0.1 and 10 M_Jup, "
                          f"but this planet's mass is {self.mass / cst.m_jup} M_Jup\n"
                          f"Take the results with caution")

        if equilibrium_temperature is None:
            if self.equilibrium_temperature is not None and self.equilibrium_temperature > 0:
                equilibrium_temperature = self.equilibrium_temperature
            else:
                equilibrium_temperature = self.calculate_equilibrium_temperature()

        return self.compute_intrinsic_temperature_inflated(
            equilibrium_temperature=equilibrium_temperature
        )

    def calculate_intrinsic_temperature_leconte(
        self,
        quality_factor: float,
        love_number_2: float,
        emissivity: float = 1
    ) -> float:
        """Calculate the intrinsic temperature from the tidal heating rate.

        Source: Leconte et al. 2010 http://dx.doi.org/10.1051/0004-6361/201014337 (Eq. 23).

        This makes 3 major assumptions:
            1. the eccentricity is small (<~0.2),
            2. the planet is near synchronization, i.e. angular_speed (omega_p) ~ rotation frequency (n),
            3. the tidal heating rate is evenly deposited near the "surface" of the planet.

        The main unknowns of this equation are `quality_factor` (Q) and `love_number_2` (k2), but what matters most is
        the value of k2/Q, which may be easier to find that the individual parameters.
        The k2/Q value can be given to this function e.g. by setting `love_number_2=1` and `quality_factor=1 / (k2/Q)`.
        Typical k2/Q values are (Lainey et al. 2015, https://arxiv.org/pdf/1604.04184):
            - Earth-like: ~1e-3
            - Neptune-like: ~1e-4
            - Saturn-like: ~2e-4
            - Jupiter-like: ~1e-5
        With typical k2 values ranging from ~0.1 to ~1.0, and typical Q values ranging from 1e2 to 1e6.
        These values are often given with high uncertainties.

        Args:
            quality_factor: float
                Quality factor of the planet (a.k.a. "Q").
                In some works, the reduced quality factor (Q') is used. Its relationship with Q is given by:
                    Q' = 3Q / (2 * k2),
                Where k2 is the Love number of degree 2 of the planet.
            love_number_2: float
                Love number of degree 2 of the planet (a.k.a. "k2"). Default is 3/2, the value for a homogeneous sphere.
            emissivity:
                Emissivity of the planet. Default is 1.

        Returns:
            (K) The intrinsic temperature of the planet, according to Leconte et al. 2010 (Eq. 23).
        """
        return self.compute_intrinsic_temperature_leconte(
            quality_factor=quality_factor,
            radius=self.radius,
            orbit_semi_major_axis=self.orbit_semi_major_axis,
            orbital_velocity=self.calculate_orbital_velocity(),
            orbital_eccentricity=self.orbital_eccentricity,
            star_mass=self.star_mass,
            love_number_2=love_number_2,
            emissivity=emissivity
        )

    def calculate_full_transit_duration(self):
        return self.compute_full_transit_duration(
            total_transit_duration=self.transit_duration,
            planet_radius=self.radius,
            star_radius=self.star_radius,
            impact_parameter=self.calculate_impact_parameter()
        )

    def calculate_impact_parameter(self):
        return self.compute_impact_parameter(
            orbit_semi_major_axis=self.orbit_semi_major_axis,
            orbital_inclination=self.orbital_inclination,
            star_radius=self.star_radius
        )

    def calculate_mid_transit_time(self, observation_day,
                                   source_mid_transit_time=None,
                                   source_mid_transit_time_error_lower=None,
                                   source_mid_transit_time_error_upper=None,
                                   day2second=True):
        if source_mid_transit_time is None:
            source_mid_transit_time = self.transit_midpoint_time

            if source_mid_transit_time_error_lower is not None:
                warnings.warn("overriding source_mid_transit_time_error_lower")

            if source_mid_transit_time_error_upper is not None:
                warnings.warn("overriding source_mid_transit_time_error_upper")

            source_mid_transit_time_error_lower = self.transit_midpoint_time_error_lower
            source_mid_transit_time_error_upper = self.transit_midpoint_time_error_upper

        return self.compute_mid_transit_time_from_source(
            observation_day=observation_day,
            source_mid_transit_time=source_mid_transit_time / cst.s_cst.day,
            source_mid_transit_time_error_lower=source_mid_transit_time_error_lower / cst.s_cst.day,
            source_mid_transit_time_error_upper=source_mid_transit_time_error_upper / cst.s_cst.day,
            orbital_period=self.orbital_period / cst.s_cst.day,
            orbital_period_error_lower=self.orbital_period_error_lower / cst.s_cst.day,
            orbital_period_error_upper=self.orbital_period_error_upper / cst.s_cst.day,
            day2second=day2second
        )

    def calculate_orbital_longitudes(self, times, longitude_start=0, rad2deg=True):
        return self.compute_orbital_longitudes(
            times=times,
            orbital_period=self.orbital_period,
            longitude_start=longitude_start,
            rad2deg=rad2deg
        )

    def calculate_orbital_phases(self, times, phase_start=0):
        return self.compute_orbital_phases(
            times=times,
            orbital_period=self.orbital_period,
            phase_start=phase_start
        )

    def calculate_orbital_velocity(self):
        return self.compute_orbital_velocity(
            star_mass=self.star_mass,
            orbit_semi_major_axis=self.orbit_semi_major_axis
        )

    def calculate_radial_velocity(self, orbital_longitude):
        return self.compute_radial_velocity(
            radial_velocity_semi_amplitude=self.calculate_radial_velocity_semi_amplitude(),
            orbital_longitude=orbital_longitude
        )

    def calculate_radial_velocity_semi_amplitude(self):
        return self.compute_radial_velocity_semi_amplitude(
            orbital_velocity=self.calculate_orbital_velocity(),
            orbital_inclination=self.orbital_inclination
        )

    @staticmethod
    def compute_airmass(ra, dec, time, site_name=None, latitude=None, longitude=None, height=None,
                        time_format='mjd'):
        observer_location, target_coordinates = Planet.get_astropy_coordinates(
            ra=ra,
            dec=dec,
            site_name=site_name,
            latitude=latitude,
            longitude=longitude,
            height=height
        )

        frame = AltAz(
            obstime=Time(time, format=time_format),
            location=observer_location
        )

        target_alt_az = target_coordinates.transform_to(frame)

        return target_alt_az.secz

    @staticmethod
    def compute_barycentric_velocities(ra, dec, time, site_name=None, latitude=None, longitude=None, height=None,
                                       time_format='mjd'):
        observer_location, target_coordinates = Planet.get_astropy_coordinates(
            ra=ra,
            dec=dec,
            site_name=site_name,
            latitude=latitude,
            longitude=longitude,
            height=height
        )

        return target_coordinates.radial_velocity_correction(
            obstime=Time(time, format=time_format),
            location=observer_location
        ).value * 1e2  # m.s-1 to cm.s-1

    @staticmethod
    def compute_equilibrium_temperature(orbit_semi_major_axis, star_effective_temperature, star_radius,
                                        bond_albedo: float = 0,
                                        orbit_semi_major_axis_error_lower: float = 0,
                                        orbit_semi_major_axis_error_upper: float = 0,
                                        star_effective_temperature_error_lower: float = 0,
                                        star_effective_temperature_error_upper: float = 0,
                                        star_radius_error_lower: float = 0,
                                        star_radius_error_upper: float = 0
                                        ):
        """
        Calculate the equilibrium temperature of a planet.
        """
        equilibrium_temperature = \
            star_effective_temperature * np.sqrt(star_radius / (2 * orbit_semi_major_axis)) \
            * (1 - bond_albedo) ** 0.25

        partial_derivatives = np.array([
            equilibrium_temperature / star_effective_temperature,  # dt_eq/dt_eff
            0.5 * equilibrium_temperature / star_radius,  # dt_eq/dr*
            - 0.5 * equilibrium_temperature / orbit_semi_major_axis  # dt_eq/dd
        ])
        uncertainties = np.abs(np.array([
            [star_effective_temperature_error_lower, star_effective_temperature_error_upper],
            [star_radius_error_lower, star_radius_error_upper],
            [orbit_semi_major_axis_error_lower, orbit_semi_major_axis_error_upper]
        ]))

        errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

        return equilibrium_temperature, errors[1], -errors[0]

    @staticmethod
    def compute_full_transit_duration(total_transit_duration, planet_radius, star_radius, impact_parameter):
        k = planet_radius / star_radius

        return total_transit_duration * np.sqrt(
            ((1 - k) ** 2 - impact_parameter ** 2)
            / ((1 + k) ** 2 - impact_parameter ** 2)
        )

    @staticmethod
    def compute_impact_parameter(orbit_semi_major_axis, orbital_inclination, star_radius):
        return orbit_semi_major_axis * np.cos(np.deg2rad(orbital_inclination)) / star_radius

    @staticmethod
    def compute_intrinsic_temperature(mass: float, radius: float, star_age: float,
                                      mass_error_lower: float = 0.0, mass_error_upper: float = 0.0,
                                      radius_error_lower: float = 0.0, radius_error_upper: float = 0.0,
                                      star_age_error_lower: float = 0.0, star_age_error_upper: float = 0.0
                                      ) -> tuple[float, float, float]:
        """Calculate the intrinsic temperature of an irradiated planet.
        The star's age must be greater than 1 Gyr and the planet's mass lower than 1 Jupiter mass.

        Source: Rogers&Seager 2010 https://www.doi.org/10.1088/0004-637X/712/2/974 (Eq. 19 and 20)

        Args:
            mass: (g) mass of the planet
            radius: (cm) radius of the planet
            star_age: (s) age of the planet's star
            mass_error_lower: (g) lower error on the planet's mass
            mass_error_upper: (g) upper error on the planet's mass
            radius_error_lower: (cm) lower error on the planet's radius
            radius_error_upper: (cm) upper error on the planet's radius
            star_age_error_lower: (s) lower error on the planet's star's age
            star_age_error_upper: (s) upper error on the planet's star's age

        Returns:
            The intrinsic temperature in K, and its associated lower and upper uncertainties.
        """
        # Check for model validity
        if mass > cst.m_jup:
            warnings.warn(f"the Rogers&Seager 2010 model is valid for planets < 1 M_Jup, "
                          f"but planet mass is {mass / cst.m_jup} M_jup\n"
                          f"Take the results with caution")

        if star_age < 1e9 * cst.s_cst.year:
            warnings.warn(f"the Rogers&Seager 2010 model is valid for star ages > 1 Gyr, "
                          f"but star age is {star_age / (1e9 * cst.s_cst.year)} Gyr\n"
                          f"Take the results with caution")

        # Power law constants and respective errors
        a1 = -12.46
        a1_error = 0.05

        a_m = 1.74
        a_m_error = 0.03

        a_r = -0.94
        a_r_error = 0.09

        a_t = -1.04
        a_t_error = 0.04

        # Convert parameters to log10 scale
        log_10 = np.log(10)
        log10_mass = np.log10(mass / cst.m_earth)
        log10_radius = np.log10(radius / cst.r_jup)
        log10_star_age = np.log10(star_age / (1e9 * cst.s_cst.year))

        # Log10 of intrinsic luminosity
        intrinsic_luminosity = (
                a1
                + a_m * log10_mass
                + a_r * log10_radius
                + a_t * log10_star_age
        )

        # Log10 of intrinsic luminosity uncertainties
        partial_derivatives = np.array([
            1,  # dL/da1
            log10_mass,  # dL/da_m
            log10_radius,  # dL/da_r
            log10_star_age,  # dL/da_r
            log_10 / mass,  # dL/dmass
            log_10 / radius,  # dL / dradius
            log_10 / star_age,  # dL / dstar_age
        ])
        uncertainties = np.abs(np.array([
            [a1_error, a1_error],
            [a_m_error, a_m_error],
            [a_r_error, a_r_error],
            [a_t_error, a_t_error],
            [mass_error_lower, mass_error_upper],
            [radius_error_lower, radius_error_upper],
            [star_age_error_lower, star_age_error_upper]
        ]))

        errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

        # Intrinsic luminosity and related uncertainties
        intrinsic_luminosity = 10 ** intrinsic_luminosity  # (unitless)
        errors = np.abs(intrinsic_luminosity) * np.abs(log_10 * errors)  # propagation of uncertainties

        intrinsic_luminosity *= cst.l_sun  # (erg.s-1)
        errors *= np.abs(cst.l_sun)

        # Intermediate "temperature" in K4 and related uncertainties
        thermal_area = 4 * np.pi * cst.sigma * radius ** 2  # (erg.s-1.K-4)
        hypercubic_temperature = intrinsic_luminosity / thermal_area  # (K4)

        partial_derivatives = np.array([
            1 / thermal_area,  # dTi/dL
            -2 * hypercubic_temperature / radius  # dTi/dR
        ])
        uncertainties = np.abs(np.array([
            [errors[0], errors[1]],
            [radius_error_lower, radius_error_upper]
        ]))

        errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

        # Intrinsic temperature
        intrinsic_temperature = hypercubic_temperature ** 0.25

        # Intrinsic temperature uncertainties
        partial_derivatives = np.array([
            1 / (4 * hypercubic_temperature ** 0.75)  # dT/dLa
        ])
        uncertainties = np.abs(np.array([
            [errors[0], errors[1]]
        ]))

        errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

        return float(intrinsic_temperature), float(errors[1]), float(-errors[0])

    @staticmethod
    def compute_intrinsic_temperature_inflated(equilibrium_temperature: float) -> float:
        """Calculate the intrinsic temperature of an inflated hot Jupiter.
        This is valid for hot Jupiters with masses between 0.1 and 10 M_Jup, and equilibrium temperatures between ~700 K
        and ~2800 K.

        Source: Thorngren et al. 2019 https://www.doi.org/10.3847/2041-8213/ab43d0 (Eq. 3)
        Corrected source: https://www.doi.org/10.3847/2041-8213/ab6d6c

        The article forget to mention that the flux must be scaled by 1e-9 (in CGS) to match the amplitude of the curve
        in their Fig. 1.

        Args:
            equilibrium_temperature: (K) equilibrium temperature of the planet

        Returns:
            The intrinsic temperature in K, and its associated lower and upper uncertainties.
        """
        if not (700 < equilibrium_temperature < 2800):  # eq. temperature boundaries of the models used in the article
            warnings.warn(f"Thorngren et al. 2019 simplified Eq. 3 was calibrated "
                          f"for equilibrium temperatures between ~700 K and ~2800 K, "
                          f"but the equilibrium temperature is {equilibrium_temperature} K\n"
                          f"Take the results with caution")

        flux = 4 * cst.sigma * equilibrium_temperature ** 4

        scaling_factor = 0.39  # (corrected version)
        mean = 0.14
        standard_deviation = 1.095  # (corrected version)

        return (
            scaling_factor * equilibrium_temperature
            * np.exp(-(np.log10(flux * 1e-9) - mean) ** 2 / standard_deviation)  # flux scaling required to match Fig. 1
        )

    @staticmethod
    def compute_intrinsic_temperature_leconte(
        quality_factor: float,
        radius: float,
        orbit_semi_major_axis: float,
        orbital_velocity: float,
        orbital_eccentricity: float,
        star_mass: float,
        love_number_2: float = 1.5,
        emissivity: float = 1
    ) -> float:
        """Calculate the intrinsic temperature from the tidal heating rate.

        Source: Leconte et al. 2010 http://dx.doi.org/10.1051/0004-6361/201014337 (Eq. 23).

        This makes 3 major assumptions:
            1. the eccentricity is small (<~0.2, see Leconte et al. 2010, fig. 2),
            2. the planet is near synchronization, i.e. angular_speed (omega_p) ~ rotation frequency (n),
            3. the tidal heating rate is evenly deposited near the "surface" of the planet.

        The main unknowns of this equation are `quality_factor` (Q) and `love_number_2` (k2), but what matters most is
        the value of k2/Q, which may be easier to find that the individual parameters.
        The k2/Q value can be given to this function e.g. by setting `love_number_2=1` and `quality_factor=1 / (k2/Q)`.
        Typical k2/Q values are (Lainey et al. 2015, https://arxiv.org/pdf/1604.04184):
            - Earth-like: ~1e-3
            - Neptune-like: ~1e-4
            - Saturn-like: ~2e-4
            - Jupiter-like: ~1e-5
        With typical k2 values ranging from ~0.1 to ~1.0, and typical Q values ranging from 1e2 to 1e6.
        These values are often given with high uncertainties.

        Args:
            quality_factor: float
                Quality factor of the planet (a.k.a. "Q").
                In some works, the reduced quality factor (Q') is used. Its relationship with Q is given by:
                    Q' = 3Q / (2 * k2),
                Where k2 is the Love number of degree 2 of the planet.
            radius: float
                (cm) Radius of the planet.
            orbit_semi_major_axis: float
                (cm) Semi-major axis of the planet's orbit.
            orbital_velocity: float
                (cm.s-1) Orbital velocity of the planet.
            orbital_eccentricity:
                Eccentricity of the planet's orbit.
            star_mass: float
                (g) Mass of the planet's star
            love_number_2: float
                Love number of degree 2 of the planet (a.k.a. "k2"). Default is 3/2, the value for a homogeneous sphere.
            emissivity:
                Emissivity of the planet. Default is 1.

        Returns:
            (K) The intrinsic temperature of the planet, according to Leconte et al. 2010 (Eq. 23).
        """
        # Assuming that the planet is near synchronization, i.e. angular_speed ~ n
        angular_speed = orbital_velocity / orbit_semi_major_axis  # (rad.s-1) omega_p in Leconte et al. 2010

        tidal_heating_rate = (
            21 / 2 * love_number_2 / quality_factor
            * cst.G * star_mass ** 2
            * radius ** 5 / orbit_semi_major_axis ** 6
            * angular_speed * orbital_eccentricity ** 2
        )  # (W) E^dot_tides in Leconte et al. 2010

        # Assuming that the tidal heating rate is evenly deposited near the "surface" of the planet
        area = 4 * np.pi * radius ** 2  # (cm2)

        return (tidal_heating_rate / (area * emissivity * cst.sigma)) ** 0.25  # (K)

    @staticmethod
    def compute_mid_transit_time_from_source(observation_day,
                                             source_mid_transit_time,
                                             source_mid_transit_time_error_lower, source_mid_transit_time_error_upper,
                                             orbital_period, orbital_period_error_lower, orbital_period_error_upper,
                                             day2second=True):
        n_orbits = np.ceil((observation_day - source_mid_transit_time) / orbital_period)
        observation_mid_transit_time = source_mid_transit_time + n_orbits * orbital_period

        derivatives = np.array([
            1,  # dT0 / dT0_source
            n_orbits  # dT0 / dP
        ])

        uncertainties = np.abs(np.array([
            [source_mid_transit_time_error_lower, source_mid_transit_time_error_upper],
            [orbital_period_error_lower, orbital_period_error_upper],
        ]))

        errors = calculate_uncertainty(derivatives, uncertainties)

        if day2second:
            observation_mid_transit_time = np.mod(observation_mid_transit_time, 1) * cst.s_cst.day
            errors[0] *= cst.s_cst.day
            errors[1] *= cst.s_cst.day

        return observation_mid_transit_time, errors[1], -errors[0], n_orbits

    @staticmethod
    def compute_orbital_longitudes(times, orbital_period, longitude_start, rad2deg: bool = True):
        """Calculate orbital longitudes assuming low eccentricity.

        Args:
            times: (s) time array
            orbital_period: (s) orbital period of the planet
            longitude_start: (rad) planet longitude at the start of observations
            rad2deg: if True, convert the longitudes from radians to degrees

        Returns:
            The orbital longitudes at the given times
        """
        orbital_phases = Planet.compute_orbital_phases(
            times=times,
            orbital_period=orbital_period,
            phase_start=longitude2phase(longitude_start)
        )

        return phase2longitude(orbital_phases, rad2deg=rad2deg)

    @staticmethod
    def compute_orbital_phases(phase_start, orbital_period, times):
        """Calculate orbital phases assuming low eccentricity.

        Args:
            phase_start: planet phase at the start of observations
            orbital_period: (s) orbital period of the planet
            times: (s) time array

        Returns:
            The orbital phases for the given time
        """
        phases = phase_start + times / orbital_period
        add = np.zeros(times.size)
        add[np.less(phases, 0)] = - 1

        return add + np.mod(phases, 1.0)

    @staticmethod
    def compute_orbital_velocity(star_mass, orbit_semi_major_axis):
        """Calculate an approximation of the orbital velocity.
        This equation is valid if the mass of the object is negligible compared to the mass of the star, and if the
        eccentricity of the object is close to 0.

        Args:
            star_mass: (g) mass of the star
            orbit_semi_major_axis: (cm) semi-major axis of the orbit of the object

        Returns: (cm.s-1) the mean orbital velocity, assuming 0 eccentricity and mass_object << mass_star
        """
        return np.sqrt(cst.G * star_mass / orbit_semi_major_axis)

    @staticmethod
    def compute_radial_velocity(radial_velocity_semi_amplitude, orbital_longitude):
        """Calculate the planet radial velocity as seen by an observer.

        Args:
            radial_velocity_semi_amplitude: (cm.s-1) radial velocity semi amplitude of the planet (aka Kp).
            orbital_longitude: (degree) angle between the closest point from the observer on the planet orbit and the
                planet position, i.e. if the planet orbital inclination is 0 degree, 0 degree: mid-primary transit
                point, 180 degree: mid-secondary eclipse point

        Returns:

        """
        return radial_velocity_semi_amplitude * np.sin(np.deg2rad(orbital_longitude))

    @staticmethod
    def compute_radial_velocity_semi_amplitude(orbital_velocity, orbital_inclination):
        """

        Args:
            orbital_velocity:
            orbital_inclination: (degree) angle between the normal of the planet orbital plane and the axis of
                observation, i.e. 90 degree: edge view, 0 degree: top view

        Returns:

        """
        return orbital_velocity * np.sin(np.deg2rad(orbital_inclination))

    @staticmethod
    def download_from_nasa_exoplanet_archive(search_request,
                                             tap_service="https://exoplanetarchive.ipac.caltech.edu/TAP"):
        service = pyvo.dal.TAPService(tap_service)
        result_set = service.search(search_request)

        return result_set.to_table()

    @staticmethod
    def download_planet_from_nasa_exoplanet_archive(name):
        astro_table = Planet.download_from_nasa_exoplanet_archive(
            search_request=f"select * from ps where pl_name = '{name}'"
        )
        filename = Planet.generate_filename(name).rsplit('.', 1)[0] + '.vot'

        with warnings.catch_warnings():
            # Temporarily filter the archive-induced units warnings
            warnings.filterwarnings("ignore", category=astropy.io.votable.exceptions.W50)
            astro_table.write(filename, format='votable')

        return astro_table

    @classmethod
    def from_tab_file(cls, filename, use_best_mass=True):
        """Read from a NASA Exoplanet Archive Database .tab file.
        Args:
            filename: file to read
            use_best_mass: if True, use NASA Exoplanet Archive Database 'bmass' argument instead of 'mass'.

        Returns:
            planets: a list of Planet objects
        """
        with open(filename, 'r') as f:
            line = f.readline()
            line = line.strip()

            # Skip header
            while line[0] == '#':
                line = f.readline()
                line = line.strip()

            # Read column names
            columns_name = line.split('\t')

            planet_name_index = columns_name.index('pl_name')

            planets = {}

            # Read data
            for line in f:
                line = line.strip()
                columns = line.split('\t')

                new_planet = cls(columns[planet_name_index])
                keys = []

                for i, value in enumerate(columns):
                    # Clearer keynames
                    keys.append(columns_name[i])

                    if value != '':
                        try:
                            value = float(value)
                        except ValueError:
                            pass

                        value, keys[i] = Planet._convert_nasa_exoplanet_archive(
                            value, keys[i], use_best_mass=use_best_mass
                        )
                    else:
                        value = None

                    if keys[i] in new_planet.__dict__:
                        new_planet.__dict__[keys[i]] = value

                # Try to calculate the planet mass and radius if missing
                if new_planet.radius == 0 and new_planet.mass > 0 and new_planet.density > 0:
                    new_planet.radius = (3 * new_planet.mass / (4 * np.pi * new_planet.density)) ** (1 / 3)

                    partial_derivatives = np.array([
                        new_planet.radius / (3 * new_planet.mass),  # dr/dm
                        - new_planet.radius / (3 * new_planet.density)  # dr/drho
                    ])
                    uncertainties = np.abs(np.array([
                        [new_planet.mass_error_lower, new_planet.mass_error_upper],
                        [new_planet.density_error_lower, new_planet.density_error_upper]
                    ]))

                    new_planet.radius_error_lower, new_planet.radius_error_upper = \
                        calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors
                elif new_planet.mass == 0 and new_planet.radius > 0 and new_planet.density > 0:
                    new_planet.mass = new_planet.density * 4 / 3 * np.pi * new_planet.radius ** 3

                    partial_derivatives = np.array([
                        new_planet.mass / new_planet.density,  # dm/drho
                        3 * new_planet.radius / new_planet.radius  # dm/dr
                    ])
                    uncertainties = np.abs(np.array([
                        [new_planet.density_error_lower, new_planet.density_error_upper],
                        [new_planet.radius_error_lower, new_planet.radius_error_upper]
                    ]))

                    new_planet.mass_error_lower, new_planet.mass_error_upper = \
                        calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

                # Try to calculate the star radius if missing
                if new_planet.star_radius == 0 and new_planet.star_mass > 0:
                    if new_planet.star_reference_gravity > 0:
                        new_planet.star_radius, \
                            new_planet.star_radius_error_upper, new_planet.star_radius_error_lower = \
                            new_planet.reference_gravity2radius(
                                new_planet.star_reference_gravity,
                                new_planet.star_mass,
                                reference_gravity_error_upper=new_planet.star_reference_gravity_error_upper,
                                reference_gravity_error_lower=new_planet.star_reference_gravity_error_lower,
                                mass_error_upper=new_planet.star_mass_error_upper,
                                mass_error_lower=new_planet.star_mass_error_lower
                            )
                    elif new_planet.star_density > 0:
                        new_planet.star_radius = \
                            (3 * new_planet.star_mass / (4 * np.pi * new_planet.star_density)) ** (1 / 3)

                        partial_derivatives = np.array([
                            new_planet.star_radius / (3 * new_planet.star_mass),  # dr/dm
                            - new_planet.star_radius / (3 * new_planet.star_density)  # dr/drho
                        ])
                        uncertainties = np.abs(np.array([
                            [new_planet.star_mass_error_lower, new_planet.star_mass_error_upper],
                            [new_planet.star_density_error_lower, new_planet.star_density_error_upper]
                        ]))

                        new_planet.star_radius_error_lower, new_planet.star_radius_error_upper = \
                            calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

                if 'reference_gravity' not in keys and new_planet.radius > 0 and new_planet.mass > 0:
                    new_planet.reference_gravity, \
                        new_planet.reference_gravity_error_upper, new_planet.reference_gravity_error_lower = \
                        new_planet.mass2reference_gravity(
                            new_planet.mass,
                            new_planet.radius,
                            mass_error_upper=new_planet.mass_error_upper,
                            mass_error_lower=new_planet.mass_error_lower,
                            radius_error_upper=new_planet.radius_error_upper,
                            radius_error_lower=new_planet.radius_error_lower
                        )

                if 'equilibrium_temperature' not in keys \
                        and new_planet.orbit_semi_major_axis > 0 \
                        and new_planet.star_effective_temperature > 0 \
                        and new_planet.star_radius > 0:
                    new_planet.equilibrium_temperature, \
                        new_planet.equilibrium_temperature_error_upper, \
                        new_planet.equilibrium_temperature_error_lower = \
                        new_planet.calculate_equilibrium_temperature()

                planets[new_planet.name] = new_planet

        return planets

    @classmethod
    def from_votable(cls, votable):
        new_planet = cls('new_planet')
        parameter_dict = {}

        for key in votable.keys():
            # Clearer keynames
            value, key = Planet._convert_nasa_exoplanet_archive(votable[key], key)
            parameter_dict[key] = value

        parameter_dict = new_planet._select_best_in_column(parameter_dict)

        for key in parameter_dict:
            if key in new_planet.__dict__:
                new_planet.__dict__[key] = parameter_dict[key]

        if 'reference_gravity' not in parameter_dict and new_planet.radius > 0:
            new_planet.reference_gravity, \
                new_planet.reference_gravity_error_upper, new_planet.reference_gravity_error_lower = \
                new_planet.mass2reference_gravity(
                    new_planet.mass,
                    new_planet.radius,
                    mass_error_upper=new_planet.mass_error_upper,
                    mass_error_lower=new_planet.mass_error_lower,
                    radius_error_upper=new_planet.radius_error_upper,
                    radius_error_lower=new_planet.radius_error_lower
                )

        if 'equilibrium_temperature' not in parameter_dict:
            new_planet.equilibrium_temperature, \
                new_planet.equilibrium_temperature_error_upper, new_planet.equilibrium_temperature_error_lower = \
                new_planet.calculate_equilibrium_temperature()

        return new_planet

    @classmethod
    def from_votable_file(cls, filename):
        with warnings.catch_warnings():
            # Temporarily filter the archive-induced units warnings
            warnings.filterwarnings('ignore', category=u.UnitsWarning)
            astro_table = Table.read(filename)

        return cls.from_votable(astro_table)

    @staticmethod
    def generate_filename(name, directory=None):
        if directory is None:
            directory = Planet.get_default_directory()

        return f"{directory}{os.path.sep}planet_{name.replace(' ', '_')}.h5"

    @classmethod
    def get(cls, name):
        filename = cls.generate_filename(name)

        return cls.get_from(name, filename)

    @staticmethod
    def get_astropy_coordinates(ra, dec, site_name=None, latitude=None, longitude=None, height=None):
        if site_name is not None:
            observer_location = EarthLocation.of_site(site_name)
        else:
            observer_location = EarthLocation.from_geodetic(
                lat=latitude * u.deg,  # type: ignore
                lon=longitude * u.deg,  # type: ignore
                height=height * u.m  # type: ignore
            )

        target_coordinates = SkyCoord(
            ra=ra * u.deg,  # type: ignore
            dec=dec * u.deg  # type: ignore
        )

        return observer_location, target_coordinates

    @staticmethod
    def get_default_directory(path_input_data=None):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        return os.path.abspath(os.path.join(path_input_data, 'planet_data'))

    def get_filename(self):
        return self.generate_filename(self.name)

    @classmethod
    def get_from(cls, name, filename):
        if not os.path.exists(filename):
            filename_vot = filename.rsplit('.', 1)[0] + '.vot'  # search for votable

            if not os.path.exists(filename_vot):
                print(f"file '{filename_vot}' not found, downloading...")

                directory = os.path.dirname(filename_vot)

                if not os.path.isdir(directory):
                    os.makedirs(directory)

                cls.download_planet_from_nasa_exoplanet_archive(name)
            else:
                warnings.warn(f"intermediate vot file found ('{filename_vot}') but not the corresponding final file "
                              f"('{filename}')\n"
                              f"This may indicate an incomplete or corrupted download. If conversion to HDF5 fails, "
                              f"consider removing the vot file.")

            # Save into HDF5 and remove the VO table
            new_planet = cls.from_votable_file(filename_vot)
            new_planet.save()
            os.remove(filename_vot)

            return new_planet
        else:
            return cls.load(name, filename)

    @classmethod
    def load(cls, name, filename=None):
        new_planet = cls(name)

        if filename is None:
            filename = new_planet.get_filename()

        with h5py.File(filename, 'r') as f:
            for key in f:
                if isinstance(f[key][()], bytes):
                    value = str(f[key][()], 'utf-8')
                else:
                    value = f[key][()]

                new_planet.__dict__[key] = value

                if 'units' in f[key].attrs:
                    if key in new_planet.units:
                        if f[key].attrs['units'] != new_planet.units[key]:
                            raise ValueError(f"units of key '{key}' must be '{new_planet.units[key]}', "
                                             f"but is '{f[key].attrs['units']}'")
                    else:
                        new_planet.units[key] = f[key].attrs['units']
                else:
                    new_planet.units[key] = 'N/A'

        return new_planet

    @staticmethod
    def mass2reference_gravity(mass, radius,
                               mass_error_upper=0., mass_error_lower=0., radius_error_upper=0., radius_error_lower=0.):
        """
        Convert the mass of a planet to its surface gravity.
        Args:
            mass: (g) mass of the planet
            radius: (cm) radius of the planet
            mass_error_upper: (g) upper error on the planet mass
            mass_error_lower: (g) lower error on the planet mass
            radius_error_upper: (cm) upper error on the planet radius
            radius_error_lower: (cm) lower error on the planet radius

        Returns:
            (cm.s-2) the reference gravity of the planet, and its upper and lower error
        """
        if radius <= 0:
            warnings.warn(f"unknown or invalid radius ({radius}), reference gravity not calculated")

            return None, None, None

        if mass <= 0:
            warnings.warn(f"unknown or invalid mass ({mass}), reference gravity not calculated")

            return None, None, None

        reference_gravity = cst.G * mass / radius ** 2

        if np.any(np.not_equal((
                radius_error_upper,
                radius_error_lower,
                mass_error_upper,
                mass_error_lower
        ), 0.)):
            partial_derivatives = np.array([
                reference_gravity / mass,  # dg/dm
                - 2 * reference_gravity / radius  # dg/dr
            ])
            uncertainties = np.abs(np.array([
                [mass_error_lower, mass_error_upper],
                [radius_error_lower, radius_error_upper]
            ]))

            errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors
        else:
            errors = (0., 0.)

        return reference_gravity, errors[1], -errors[0]

    @staticmethod
    def reference_gravity2radius(reference_gravity, mass,
                                 reference_gravity_error_upper=0., reference_gravity_error_lower=0.,
                                 mass_error_upper=0., mass_error_lower=0.):
        """
        Convert the mass of a planet to its reference gravity.
        Args:
            reference_gravity: (cm.s-2) reference gravity of the planet
            mass: (g) mass of the planet
            mass_error_upper: (g) upper error on the planet mass
            mass_error_lower: (g) lower error on the planet mass
            reference_gravity_error_upper: (cm.s-2) upper error on the planet radius
            reference_gravity_error_lower: (cm.s-2) lower error on the planet radius

        Returns:
            (cm.s-2) the surface gravity of the planet, and its upper and lower error
        """
        if reference_gravity <= 0:
            warnings.warn(f"unknown or invalid surface gravity ({reference_gravity}), radius not calculated")

            return None, None, None

        radius = np.sqrt(cst.G * mass / reference_gravity)

        if np.any(np.not_equal((
                reference_gravity_error_upper,
                reference_gravity_error_lower,
                mass_error_upper,
                mass_error_lower
        ), 0.)):
            partial_derivatives = np.array([
                radius / (2 * mass),  # dr/dm
                - radius / (2 * reference_gravity)  # dr/dg
            ])
            uncertainties = np.abs(np.array([
                [mass_error_lower, mass_error_upper],
                [reference_gravity_error_lower, reference_gravity_error_upper]
            ]))

            errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors
        else:
            errors = (0., 0.)

        return radius, errors[1], -errors[0]

    @staticmethod
    def reference_gravity2mass(reference_gravity, radius,
                               reference_gravity_error_upper=0., reference_gravity_error_lower=0.,
                               radius_error_upper=0., radius_error_lower=0.):
        """
        Convert the surface gravity of a planet to its mass.
        Args:
            reference_gravity: (cm.s-2) reference gravity of the planet
            radius: (cm) radius of the planet
            reference_gravity_error_upper: (cm.s-2) upper error on the planet surface gravity
            reference_gravity_error_lower: (cm.s-2) lower error on the planet surface gravity
            radius_error_upper: (cm) upper error on the planet radius
            radius_error_lower: (cm) lower error on the planet radius

        Returns:
            (g) the mass of the planet, and its upper and lower error
        """
        mass = reference_gravity / cst.G * radius ** 2

        if np.any(np.not_equal((
                radius_error_upper,
                radius_error_lower,
                reference_gravity_error_upper,
                reference_gravity_error_lower
        ), 0.)):
            partial_derivatives = np.array([
                mass / reference_gravity,  # dm/dg
                2 * mass / radius  # dm/dr
            ])
            uncertainties = np.abs(np.array([
                [reference_gravity_error_lower, reference_gravity_error_upper],
                [radius_error_lower, radius_error_upper]
            ]))

            errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors
        else:
            errors = (0., 0.)

        return mass, errors[1], -errors[0]

    def save(self, filename=None):
        if filename is None:
            filename = self.get_filename()

        with h5py.File(filename, 'w') as f:
            for key in self.__dict__:
                if key == 'units':
                    continue

                data_set = f.create_dataset(
                    name=key,
                    data=self.__dict__[key]
                )

                if self.units[key] != 'N/A':
                    data_set.attrs['units'] = self.units[key]
