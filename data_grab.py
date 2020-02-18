# Import libraries
import wget
import os
from datetime import datetime, timedelta

# Setup start date
start_date = datetime(1996, 10, 1, 0)

# Setup end date
end_date = datetime(2019, 11, 30)

# Take difference of the times
time_delta = (end_date - start_date)

# Create list of files
files = []

# Loop through different days and create link to file
for day in range(1, time_delta.days):
    file_time = start_date + day * timedelta(days=1)
    files.append(file_time.strftime('https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/%Y/gpcp_v01r03_daily_d%Y%m%d_c20170530.nc'))

# Change directories to the data directory
os.chdir('data/')

# Loop through files and download
for link in files:

    # Print the link so you know what file it is downloading
    print(link)

    # Use wget to download the files
    wget.download(link)
