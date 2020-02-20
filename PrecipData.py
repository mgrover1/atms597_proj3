# Load all dependencies
import xarray as xr
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
from IPython import display
import os


def getPrecip(startYear, endYear):
    """
    Grabs daily precipitation data from the Global Precipitaiton Climatology
    Project 1 daily archive (see https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/)
    for all available days within specified years from 1996-2019.
    Adapted from Julia Kho's script (https://github.com/julia-git/webscraping_ny_mta)

    Parameters:
        - startYear: Integer, >= 1996
        - endYear:   Integer, <= 2019

    Returns:
        xarray DataArray of daily precipitation
    """

    precipList = []

    for year in range(startYear, endYear + 1):
        # Update the cell with status information
        display.clear_output(wait=True)
        display.display(f'Gathering data for {str(year)}')

        url = 'https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/' + str(year) + '/'
        response = requests.get(url)
        tags = BeautifulSoup(response.text, 'html.parser').findAll('a')[5:]  # Returns all hyperlinks of netCDF files

        JJAtags = [tag for tag in tags if int(tag['href'][23:27]) > 600 and int(tag['href'][23:27]) < 900]  # Select only months of JJA

        for tag in JJAtags:
            link = tag['href']
            download_url = url + link

            urllib.request.urlretrieve(download_url, link)  # Save temporary netCDF file
            time.sleep(1)  # We're no spammers!
            temp = xr.open_dataset(link)  # Read netCDF back in
            precipList.append(temp)  # Append to list of all netCDF files

            os.system('rm ' + link)  # Remove the 'test.nc' file

    return xr.concat(precipList, dim='time')  # Merge all into one xr DataArray
