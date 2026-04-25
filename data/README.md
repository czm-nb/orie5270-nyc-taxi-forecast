# Data

## `original_data/`

TLC **Yellow Taxi** monthly Parquet files (`yellow_tripdata_*.parquet`) shipped with this repository (January–March 2025). They are the input to:

```bash
nyc-taxi-forecast build-hourly
```

which defaults to `--input-dir data/original_data`. Add or replace files here, or point `--input-dir` at another folder (for example after downloading more months from the [NYC TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) page).
