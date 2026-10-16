"""petitRADTRANS implementation of ESO's Exposure Time Calculator Command Line Interface.
Source: https://etc.eso.org/observing/etc/home
"""
import copy
import json
import os
from warnings import warn

import numpy as np
import requests

module_dir = os.path.dirname(__file__)
request_file = module_dir + '/eso_etc-form.json'
request_file_base_dict = {
    'application': {'executiondate': '2021-08-20T15:30:31.398Z', 'version': 'P108b'},
    'observatory': {'site': 'paranal'},
    'target': {  # we target a star
        'morphology': {'morphologytype': 'point'},
        'sed': {
            'sedtype': 'spectrum',
            'spectrum': {'spectrumtype': 'upload'}  # upload the stellar spectrum instead of using models
        },
        'brightness': {
            'brightnesstype': 'mag',
            'extinctionav': 0,
            'params': {
                'magband': 'J',  # apparent magnitude band of reference
                'mag': 0.0,  # apparent magnitude of the star in the band of reference
                'magsys': 'vega'  # star used to calculate the reference apparent magnitude (m_band = 0)
            }
        }
    },
    'seeingiqao': {'mode': 'noao', 'aperturepix': 17, 'params': {'seeing': 1.0}},  # TODO find precision on this
    'timesnr': {
        'inputtype': 'ndit',
        'dit': 60,  # (s) integration time duration
        'ndit': {'ndit': 360}  # number of integrations, the exposure time is ndit * dit
    },
    'output': {
        'snr': {'snr': True, 'plotindex': 0},  # output only the signal-to-noise ratio
        'sed': {'sky': False, 'target': False, 'plotindex': 1},
        'sky': {'obsemission': False, 'plotindex': 2},
        'target': {'obstarget': False, 'plotindex': 3},
        'background': {'obsbackground': False, 'plotindex': 4},
        'totalsignal': {'totalsignalcentralpix': False, 'plotindex': 5},
        'blip': {'blip': False, 'plotindex': 6},
        'throughput': {
            'atmosphere': False,
            'telescope': False,
            'instrument': False,
            'blaze': False,
            'enslittedenergy': False,
            'detector': False,
            'totalinclsky': False,
            'plotindex': 7
        },
        'dispersion': {'dispersion': False, 'plotindex': 8},
        'psf': {'psf': False, 'plotindex': 9}
    },
    'sky': {  # assuming J. Birkby conditions
        'airmass': 1.2,  # Jayne did not say anything about airmass...
        'moon_fli': 1,
        'moon_sun_sep': 0,  # ... nor moon-sun-separation
        'pwv': 2.5  # 5 in worst case scenario
    },
    'instrument': {
        'name': 'crires',
        'slit': 0.2,
        'settingkey': '',  # instrument setting ('<band><setting_number>', e.g. 'K2126')
        'order': [],  # list of integers containing the orders of the setting
        'polarimetry': 'free'
    }
}


def call_etc(postdatafile, url, uploadfile=None):
    with open(postdatafile) as f:
        postdata = json.loads(f.read())

    # TODO workaround until we figure out how to handle ssl certificates correctly
    import warnings
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')

    if uploadfile is None:
        return requests.post(url,
                             data=json.dumps(postdata),
                             headers={'Content-Type': 'application/json'},
                             verify=False,
                             )
    else:
        return requests.post(url,
                             data={"data": json.dumps(postdata)},
                             files={"target": open(uploadfile, 'rb')},
                             verify=False)


def collapse(jsondata):
    def go_through(x):
        if isinstance(x, list):
            return go_through_list(x)
        elif isinstance(x, dict):
            return go_through_dict(x)
        else:
            return x

    def go_through_dict(dic):
        for key, value in dic.items():
            if isinstance(value, dict):
                dic[key] = go_through_dict(value)
            elif isinstance(value, list):
                dic[key] = go_through_list(value)
        return dic

    def go_through_list(lst):
        if not any(not isinstance(y, (int, float)) for y in lst):  # pure numeric list
            if len(lst) <= 2:
                return lst
            else:
                return '[' + str(lst[0]) + ' ... ' + str(lst[-1]) + '] (' + str(len(lst)) + ')'
        else:
            return [go_through(y) for y in lst]

    return go_through_dict(jsondata)


def download_snr_data(request_file_name, star_spectrum_file_name, star_apparent_magnitude, star_effective_temperature,
                      exposure_time, integration_time, airmass, setting, setting_orders,
                      star_apparent_magnitude_band='J', instrument='crires', indent=4,
                      output_file=None, save_etc_file=False, star_model='PHOENIX', directory=module_dir):
    write_request_file(request_file_name, star_apparent_magnitude, exposure_time, integration_time, airmass,
                       setting, setting_orders, star_apparent_magnitude_band=star_apparent_magnitude_band)

    if output_file is None and save_etc_file:
        output_file = get_snr_data_file_name(
            instrument=instrument,
            setting=setting,
            exposure_time=exposure_time,
            integration_time=integration_time,
            airmass=airmass,
            star_model=star_model,
            star_effective_temperature=star_effective_temperature,
            star_apparent_magnitude_band=star_apparent_magnitude_band,
            star_apparent_magnitude=star_apparent_magnitude,
            etc_file=True,
            directory=directory
        )

    jsondata = get_data(
        instrument,
        request_file_name,
        uploadfile=star_spectrum_file_name,
        do_collapse=False,  # if True, the file is not readable by the website
        indent=indent,
        outputfile=output_file
    )

    return jsondata


def get_data(etcname, postdatafile, uploadfile=None, do_collapse=False, indent=4, outputfile=None):
    # ETC backend test server with public access
    baseurl = 'https://etctestpub.eso.org/observing/etc/etcapi/'

    etc_name = get_etc_url(etcname)

    url = baseurl + get_etc_url(etc_name)

    jsondata = call_etc(postdatafile, url, uploadfile).json()

    if outputfile is not None:
        output(jsondata, do_collapse, indent, outputfile)

    return jsondata


def get_etc_url(etcname):
    if '4most' in etcname.lower() or 'qmost' in etcname.lower() or 'fourmost' in etcname.lower():
        return 'Qmost/'
    elif 'crires' in etcname.lower():
        return 'Crires2/'
    else:
        print("error: no match for etcname: " + etcname)


def get_snr_data_file_name(instrument, setting, exposure_time, integration_time, airmass, star_model,
                           star_effective_temperature, star_apparent_magnitude_band, star_apparent_magnitude,
                           etc_file=False, directory=module_dir, extension='json'):
    file_name = f"snr_{instrument}_{setting}_exp{exposure_time}s_dit{integration_time}s_airmass{airmass}_" \
                f"{star_model}-{star_effective_temperature}K_" \
                f"m{star_apparent_magnitude_band}{star_apparent_magnitude}"

    if etc_file:
        file_name += '_etc'

    return os.path.join(directory, instrument, file_name + f'.{extension}')


def output(jsondata, do_collapse, indent, outputfile):
    if do_collapse:
        jsondata = collapse(jsondata)

    if outputfile is not None:
        with open(outputfile, "w") as of:
            of.write(json.dumps(jsondata, indent=indent))
    else:
        print(json.dumps(jsondata, indent=indent))


def write_request_file(file_name, star_apparent_magnitude, exposure_time, integration_time, airmass,
                       setting, setting_orders,
                       star_apparent_magnitude_band='V'):
    if airmass < 1:
        warn('airmass cannot be < 1', Warning)
        airmass = 1
    elif airmass >= 3:
        warn('airmass must be < 3', Warning)
        airmass = 3 - 1e-3

    if not isinstance(setting_orders, list):
        raise TypeError(f"setting_orders must be a list, not '{type(setting_orders)}'")

    request_file_ = copy.copy(request_file_base_dict)
    request_file_['target']['brightness']['params']['magband'] = star_apparent_magnitude_band
    request_file_['target']['brightness']['params']['mag'] = star_apparent_magnitude
    request_file_['timesnr']['dit'] = integration_time
    request_file_['timesnr']['ndit']['ndit'] = int(np.ceil(exposure_time / integration_time))
    request_file_['sky']['airmass'] = airmass
    request_file_['instrument']['settingkey'] = setting
    request_file_['instrument']['order'] = setting_orders

    output(request_file_, do_collapse=False, indent=2, outputfile=file_name)
