import requests
import os
import sqlite3
import netCDF4

DOWNLOAD_DIR = 'barra_downloads'
# Ensure the download directory exists
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# create database if not already exists
def create_database():
    # create rows with columns: year, months, variable and download status 
    # year range from 1979 to 2024
    years = list(range(1979, 2025))
    # months Jan, Oct, Nov, DEC in mm
    months = ['01', '10', '11', '12']
    variables = ['vas', 'uas']  # variable to download
    # create rows for each year, month and variable
    rows = []
    for year in years:
        for month in months:
            for variable in variables:
                rows.append((year, month, 'not downloaded', variable))
    # create a database file
    conn = sqlite3.connect('barra_downloads.db')
    cursor = conn.cursor()
    # create a table if it does not exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS barra_downloads (
            year INTEGER,
            month TEXT,
            status TEXT,
            variable TEXT,
            PRIMARY KEY (year, month, variable)
        )
    ''')
    # insert the rows into the table
    cursor.executemany('''
        INSERT OR IGNORE INTO barra_downloads (year, month, status, variable)
        VALUES (?, ?, ?, ?)
    ''', rows)
    conn.commit()
    conn.close()

def get_last_day_of_month(year, month):
    # get the last day of the month for the given year and month
    if month in ['01', '03', '05', '07', '08', '10', '12']:
        return 31
    elif month in ['04', '06', '09', '11']:
        return 30
    elif month == '02':
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 29  # leap year
        else:
            return 28
    else:
        raise ValueError("Invalid month")

def get_download_url(year, month, variable):
    
    last_day = get_last_day_of_month(year, month)
    url = f"https://thredds.nci.org.au/thredds/ncss/grid/ob53/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/day/{variable}/latest/{variable}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_day_{year}{month}-{year}{month}.nc?var={variable}&north=-33&west=146&east=152&south=-43&horizStride=1&time_start={year}-{month}-01T12:00:00Z&time_end={year}-{month}-{last_day}T12:00:00Z&&&accept=netcdf3"
    return url

def update_download_status(year, month, variable, status):
    # update the status of the download for the given year, month, and variable
    conn = sqlite3.connect('barra_downloads.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE barra_downloads
        SET status = ?
        WHERE year = ? AND month = ? AND variable = ?
    ''', (status, year, month, variable))
    conn.commit()
    conn.close()

def get_download_status(year, month, variable):
    # get the status of the download for the given year, month, and variable
    conn = sqlite3.connect('barra_downloads.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT status FROM barra_downloads
        WHERE year = ? AND month = ? AND variable = ?
    ''', (year, month, variable))
    status = cursor.fetchone()
    conn.close()
    return status[0] if status else None

# convert file size to human-readable format
def human_readable_size(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def downloader(url, filename):
    # Build full path with folders
    full_path = os.path.join(DOWNLOAD_DIR, f"{filename}.nc")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    response = requests.get(url)
    if response.status_code == 200:
        with open(full_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded to {full_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        print("Response content:", response.content.decode('utf-8', errors='replace'))
        raise Exception("Download failed")


def get_expected_time_range(month,year):

    expected_time_start = f'{year}-{month}-01T12:00:00'
    # Handle the last day of the month
    last_day = get_last_day_of_month(int(year), month)
    expected_time_end = f'{year}-{month}-{last_day}T12:00:00'
    return expected_time_start, expected_time_end

def check_time_range(dataset, expected_start, expected_end):
    time_var = dataset.variables['time']
    time_units = time_var.units
    time_calendar = time_var.calendar if hasattr(time_var, 'calendar') else 'standard'
    
    # Convert expected time range to datetime objects
    from netCDF4 import num2date
    start_date = num2date(time_var[0], units=time_units, calendar=time_calendar)
    end_date = num2date(time_var[-1], units=time_units, calendar=time_calendar)

    if start_date.strftime('%Y-%m-%dT%H:%M:%S') == expected_start and end_date.strftime('%Y-%m-%dT%H:%M:%S') == expected_end:
        print("The file contains the expected time range.")
    else:
        print("The file does not contain the expected time range.")
        print(f"Expected: {expected_start} to {expected_end}")
        print(f"Actual: {start_date} to {end_date}")

def sanity_check(filename, variable, year, month):
    full_path = os.path.join(DOWNLOAD_DIR, f"{filename}.nc")
    if not os.path.exists(full_path):
        raise Exception("File not downloaded successfully.")

    file_size = os.path.getsize(full_path)
    if file_size == 0 or file_size < 8000000:
        raise Exception(f"File size too small: {file_size} bytes")

    try:
        dataset = netCDF4.Dataset(full_path)
    except Exception as e:
        raise Exception(f"Invalid NetCDF file: {e}")

    if variable not in dataset.variables:
        dataset.close()
        raise Exception(f"Variable '{variable}' not present in file")

    expected_start, expected_end = get_expected_time_range(month, year)
    check_time_range(dataset, expected_start, expected_end)

    dataset.close()
    print(f"{full_path} passed sanity checks.")


def main():
    if not os.path.exists('barra_downloads.db'):
        create_database()

    conn = sqlite3.connect('barra_downloads.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT year, month, variable FROM barra_downloads
        WHERE status != 'downloaded'
    ''')
    rows = cursor.fetchall()
    conn.close()

    for year, month, variable in rows:
        filename = f"{variable}_{year}_{month}"
        print(f"\nProcessing {filename}...")

        full_path = os.path.join(DOWNLOAD_DIR, f"{filename}.nc")

        # If file exists, sanity-check it
        if os.path.exists(full_path):
            try:
                sanity_check(filename, variable, year, month)
                print(f"{filename}.nc already valid. Skipping download.")
                update_download_status(year, month, variable, 'downloaded')
                continue
            except Exception as e:
                print(f"Existing file failed checks: {e}. Re-downloading...")

        try:
            url = get_download_url(year, month, variable)
            downloader(url, filename)
            sanity_check(filename, variable, year, month)
            update_download_status(year, month, variable, 'downloaded')
            print(f"{filename}.nc downloaded and verified.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            update_download_status(year, month, variable, 'failed')

    print("\nAll downloads processed.")


if __name__ == "__main__":
    main()