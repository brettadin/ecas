"""Command line interface to download petitRADTRANS input data files.
# TODO should probably be renamed keeper_cli or prt_keeper_cli in v4.0.0
"""
import datetime
import functools
import http.client
import os
import sys
import time
import urllib.parse
import urllib.request
import warnings

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import SessionNotCreatedException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait

from petitRADTRANS.config import petitradtrans_config_parser

__megabyte = 9.5367431640625e-07  # 1 / 1024 ** 2 (B to MB)


def _get_html(options, url, timeout):
    with webdriver.Chrome(options) as driver:
        print("Done.")
        print(f"Rendering webpage ({url})... ", end='')

        driver.get(url)
        # Wait for the table containing the files to generate
        try:
            WebDriverWait(driver, timeout).until(
                expected_conditions.presence_of_element_located((By.CLASS_NAME, 'table-hover'))
            )
        except TimeoutException as error:
            # Display a more helpful error message than the basic one
            try:
                WebDriverWait(driver, 0.1).until(
                    expected_conditions.presence_of_element_located((By.CLASS_NAME, 'error'))
                )
            except TimeoutException as error:  # no error printed on the webpage, likely because the URL is incorrect
                raise TimeoutException(
                    f"\n{str(error)}\n"
                    f"Spent too much time (> {timeout} s) to wait for the Keeper table's presence.\n"
                    f"This is likely due to an incorrect URL or to an issue with your internet connection.\n"
                    f"If the URL ({url}) is incorrect, open your "
                    f"petitRADTRANS config file ({petitradtrans_config_parser.config_file}) "
                    f"and change the value of the parameter 'prt_input_data_url'.\n"
                    f"If the table is present at the requested URL, increase the delay until timeout."
                )

            # The expected error is printed by the webpage: the URL is correct but the table was not found
            raise TimeoutException(
                f"\n{str(error)}\n"
                f"Spent too much time (> {timeout} s) to wait for the Keeper table's presence.\n"
                f"This is likely due to the absence of the requested table, or to the use of an outdated URL.\n"
                f"Check the URL ({url}) for the presence of a table.\n"
                f"If the URL is incorrect, open your "
                f"petitRADTRANS config file ({petitradtrans_config_parser.config_file}) "
                f"and change the value of the parameter 'prt_input_data_url'."
            )

        # Get the webpage HTML source once the table has generated
        html = driver.page_source

    return html


def _path2keeper_url(path: str, path_input_data: str = None, url_input_data: str = None) -> str:
    """Convert an input_data path to its Keeper equivalent."""
    # Initialization
    if path_input_data is None:
        path_input_data = petitradtrans_config_parser.get_input_data_path()

    if url_input_data is None:
        url_input_data = petitradtrans_config_parser['URLs']['prt_input_data_url']

    # Ensure destination is the absolute path
    path = os.path.abspath(path)

    # Warn if the given path is not within the user's input_data path
    if path_input_data != petitradtrans_config_parser.get_input_data_path():
        warnings.warn(f"path_input_data ('{path}') "
                      f"is not the configured one ('{petitradtrans_config_parser['Paths']['prt_input_data_path']}'); "
                      f"the downloaded file will not be loaded by petitRADTRANS\n"
                      f"Change the destination, move the file manually after download, or re-set the input data path")

    if path[:len(path_input_data)] != path_input_data:
        warnings.warn(f"destination path ('{path}') "
                      f"is not within the input data folder ('{path_input_data}')")
    else:
        path = path.rsplit(path_input_data, 1)[-1]

    # Ensure that forward slashes are used in the path, else the resulting URL will be invalid
    url_path = path.replace(os.path.sep, '/')

    # Ensure that the path starts with a forward slash for the concatenation with the Keeper URL
    if url_path[0] != '/':
        url_path = '/' + url_path

    # Convert the slashes to their URL-safe equivalent, as it is used by Keeper
    return url_input_data + urllib.parse.quote(url_path, safe='')


def _reporthook(count: int, block_size: int, total_size: int, time_start=0.0):
    """Display downloading progress on the terminal.

    Args:
        count: number of the data block being downloaded
        block_size: (B) size of a data block
        total_size: (B) total size of the data
        time_start: (s) time at which the download started
    """
    # Data download progress
    progress_size = count * block_size
    percent = min(int(progress_size * 100 / total_size), 100)

    # Convert from B to MB
    progress_size *= __megabyte
    total_size *= __megabyte

    # Speed and ETA
    duration = max(time.time() - time_start, 1e-6)
    speed = max(progress_size / duration, 1e-6)

    eta = datetime.timedelta(
        seconds=int(total_size / speed - duration)
    )

    _reporthook_sys_output(
        percent=percent,
        progress_size=progress_size,
        total_size=total_size,
        speed=speed,
        eta=eta
    )


def _reporthook_sys_output(percent: int, progress_size: float, total_size: float, speed: float,
                           eta: datetime.timedelta | str):
    """Terminal output of download progress

    Args:
        percent: progression percentage
        progress_size: (MB) amount of data downloaded
        total_size: (MB) amount of data to download
        speed: (MB.s-1) data downloading rate
        eta: (hh:mm:ss) time before download completion
    """

    columns_str = (
        f"\r {percent}% | "
        f"{progress_size:.1f}/{total_size:.1f} MB | "
        f"{speed:.1f} MB/s | "
        f"eta {eta}"
    )

    sys.stdout.write(columns_str)
    sys.stdout.flush()


def download_input_data(destination, source=None, rewrite=False,
                        path_input_data=None, url_input_data=None,
                        byte_amount=8192) -> http.client.HTTPResponse | None:
    """Download a petitRADTRANS input data file.
    If source is None, the source URL is automatically deduced from the destination file.

    Args:
        destination: file to be downloaded (path to the file where the downloaded data are saved)
        source: URL of the source
        rewrite: if True, the data are downloaded even if destination is an already existing local file
        path_input_data: local path to petitRADTRANS' input_data folder
        url_input_data: URL petitRADTRANS' input data
        byte_amount: amount of bytes to be read from the URL response

    Returns:
        If a new file has been downloaded, return a HTTPResponse object, and None otherwise.
    """
    if path_input_data is None:
        path_input_data = petitradtrans_config_parser.get_input_data_path()

    if url_input_data is None:
        url_input_data = petitradtrans_config_parser['URLs']['prt_input_data_url']

    # Ensure destination is the absolute path
    destination = os.path.abspath(destination)

    # Checks before download
    if os.path.isfile(destination) and not rewrite:
        print(f"file '{destination}' already exists, skipping download (set rewrite=True to force re-download)...")
        return None

    if path_input_data != petitradtrans_config_parser['Paths']['prt_input_data_path']:
        warnings.warn(f"path_input_data ('{destination}') "
                      f"is not the configured one ('{petitradtrans_config_parser['Paths']['prt_input_data_path']}'); "
                      f"the downloaded file will not be loaded by petitRADTRANS\n"
                      f"Change the destination, move the file manually after download, or re-set the input data path")

    if destination[:len(path_input_data)] != path_input_data:
        warnings.warn(f"destination path ('{destination}') "
                      f"is not within the input data folder ('{path_input_data}')")
    else:
        destination = destination.rsplit(path_input_data, 1)[-1]

    download_request = '&dl=1'

    # Automatically get the source URL
    if source is None:
        url_path: str = destination.replace(os.path.sep, '/')

        if url_path[0] != '/':
            url_path = '/' + url_path

        source = url_input_data + urllib.parse.quote(url_path, safe='') + download_request
    elif source[:-len(download_request)] != download_request:
        source += download_request

    # Start download
    destination = os.path.join(path_input_data, destination.split(os.path.sep, 1)[1])
    destination_directory = os.path.dirname(destination)

    if not os.path.isdir(destination_directory):
        print(f"Creating directory '{destination_directory}'...")
        os.makedirs(destination_directory)

    print(f"Downloading '{destination}'...")

    with urllib.request.urlopen(source) as response:
        with open(destination, "wb") as f:
            headers = response.info()

            # Get the total size of the downloaded data
            if "content-length" in headers:
                total_size = int(headers["content-length"])
            else:
                total_size = -1

            # Initialize the response reading, using a fixed block size for reporthook
            response_read = functools.partial(response.read, byte_amount)

            # Initialize reporthook, memorizing the download starting time for speed and ETA
            time_start = time.time()
            reporthook = functools.partial(_reporthook, time_start=time_start)

            # Read the response, block by block, and save it to the destination file
            for i, data in enumerate(iter(response_read, b"")):
                f.write(data)
                reporthook(i + 1, byte_amount, total_size)

    # Ensure a clean terminal output after the download
    if (i + 1) * byte_amount >= total_size:
        total_size *= __megabyte
        _reporthook_sys_output(
            percent=100,
            progress_size=total_size,
            total_size=total_size,
            speed=total_size / (time.time() - time_start),
            eta="0:00:00"
        )
    else:
        warnings.warn("not all data has been downloaded")

    sys.stdout.write("\n")

    return response


def get_keeper_files_url_paths(path, ext='h5', timeout=3, path_input_data=None, url_input_data=None):
    url = _path2keeper_url(
        path=path,
        path_input_data=path_input_data,
        url_input_data=url_input_data
    )

    # Initialize Chrome in headless mode to prevent browser window pop-up
    options = webdriver.ChromeOptions()
    options.add_argument("headless")

    print("Starting up Chrome driver... ", end='')
    # Render the Keeper webpage
    try:
        html = _get_html(
            options=options,
            url=url,
            timeout=timeout
        )
    except SessionNotCreatedException as exception:  # this happens in Gitlab's CI
        if os.environ.get('CI', False):  # 'CI' is a Gitlab predefined environment variable
            # This can cause security issues (https://unix.stackexchange.com/a/68951), so it must be used for tests only
            options.add_argument("--no-sandbox")
            html = _get_html(
                options=options,
                url=url,
                timeout=timeout
            )
        else:
            raise exception

    print("Done.")

    # Only keep the table's content to speed up Beautiful soup finding
    table = html.split('<table', 1)[1].rsplit('</table>', 1)[0]

    # Parse the table
    soup = BeautifulSoup(table, 'html.parser')
    elements = soup.find_all('a')

    # Find all the files in the table
    url_path = urllib.parse.urlparse(url).path
    url_domain = url.split(url_path, 1)[0]

    files = {}

    for node in elements:
        href = node.get('href')

        if href.endswith(ext):
            file = href.rsplit(urllib.parse.quote('/', safe=''), 1)[1]
            files[file] = url_domain + href

    print("Ending...")  # it takes some time to return the result, maybe selenium needs to end some process?

    return files
