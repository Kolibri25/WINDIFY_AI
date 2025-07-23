import os
import sqlite3
import cdsapi
# import package for parallel downloads if needed
from concurrent.futures import ThreadPoolExecutor

DOWNLOAD_BASE_DIR = './data/ERA5'  # root directory to hold all yearly folders
MIN_FILE_SIZE_BYTES = 90 * 1024  # 100 KB sanity threshold

VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "mean_wave_direction"
]

YEARS = ['1993','2003','2011']  # 1981 to 2023 inclusive
MONTHS = ['01', '10', '11', '12']

def create_database():
    rows = []
    for year in YEARS:
        for var in VARIABLES:
            rows.append((year, var, 'not downloaded'))
    conn = sqlite3.connect('data/era5_downloads.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS barra_downloads (
            year INTEGER,
            variable TEXT,
            status TEXT,
            PRIMARY KEY (year, variable)
        )
    ''')
    cursor.executemany('''
        INSERT OR IGNORE INTO barra_downloads (year, variable, status)
        VALUES (?, ?, ?)
    ''', rows)
    conn.commit()
    conn.close()

def construct_request(year, variable):
    return {
        "product_type": "reanalysis",
        "variable": variable,
        "year": [year],
        "month": MONTHS,
        "day": [f"{i:02d}" for i in range(1, 32)],
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": [-33, 146, -43, 152],
        "format": "netcdf"
    }

def download_era5_data(year, variable):
    folder = os.path.join(DOWNLOAD_BASE_DIR, f"ERA5_{year}_daily")
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{variable}.nc")

    # Skip if file already exists and is valid
    if os.path.exists(file_path) and os.path.getsize(file_path) > MIN_FILE_SIZE_BYTES:
        print(f"[SKIP] {file_path} already exists and passed size check.")
        return True

    print(f"[DOWNLOAD] {file_path} ...")
    client = cdsapi.Client()
    try:
        client.retrieve(
            "derived-era5-single-levels-daily-statistics",
            construct_request(year, variable),
            file_path
        )
    except Exception as e:
        print(f"[ERROR] Download failed for {year}, {variable}: {e}")
        return False

    # Sanity check
    if os.path.exists(file_path) and os.path.getsize(file_path) > MIN_FILE_SIZE_BYTES:
        print(f"[SUCCESS] Downloaded {file_path}")
        return True
    else:
        print(f"[FAIL] File too small or missing: {file_path}")
        return False

def main():
    create_database()

    conn = sqlite3.connect('data/era5_downloads.db')
    cursor = conn.cursor()
    cursor.execute("SELECT year, variable FROM barra_downloads WHERE status = 'not downloaded'")
    targets = cursor.fetchall()
    conn.close()

    def process_download(target):
        year, variable = target
        success = download_era5_data(year, variable)
        if success:
            conn = sqlite3.connect('data/era5_downloads.db')
            cursor = conn.cursor()
            cursor.execute("UPDATE barra_downloads SET status = 'downloaded' WHERE year = ? AND variable = ?", (year, variable))
            conn.commit()
            conn.close()

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
        executor.map(process_download, targets)

if __name__ == "__main__":
    main()
