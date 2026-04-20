import pandas as pd

from nyc_taxi_forecast.aggregate import filter_hour_range, merge_hourly_pickups, trips_to_hourly_pickups


def test_trips_to_hourly_pickups_groups_and_localizes() -> None:
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2025-01-01 00:10:00", "2025-01-01 00:40:00", "2025-01-01 01:05:00"]
            ),
            "PULocationID": [10, 10, 11],
        }
    )
    hourly = trips_to_hourly_pickups(df)
    assert hourly["hour_start"].dt.tz is not None
    counts = hourly.set_index(["zone_id", "hour_start"])["pickup_count"].to_dict()
    ts0 = pd.Timestamp("2025-01-01 00:00:00", tz="America/New_York")
    ts1 = pd.Timestamp("2025-01-01 01:00:00", tz="America/New_York")
    assert counts[(10, ts0)] == 2
    assert counts[(11, ts1)] == 1


def test_merge_hourly_pickups_sums_across_parts() -> None:
    ts = pd.Timestamp("2025-01-01 00:00:00", tz="America/New_York")
    a = pd.DataFrame({"zone_id": [1], "hour_start": [ts], "pickup_count": [2]})
    b = pd.DataFrame({"zone_id": [1], "hour_start": [ts], "pickup_count": [3]})
    out = merge_hourly_pickups([a, b])
    assert len(out) == 1
    assert int(out["pickup_count"].iloc[0]) == 5


def test_filter_hour_range() -> None:
    ts0 = pd.Timestamp("2025-01-01 00:00:00", tz="America/New_York")
    ts1 = pd.Timestamp("2025-01-01 01:00:00", tz="America/New_York")
    df = pd.DataFrame({"zone_id": [1, 1], "hour_start": [ts0, ts1], "pickup_count": [1, 2]})
    start = pd.Timestamp("2025-01-01 00:30", tz="America/New_York")
    end = pd.Timestamp("2025-01-01 01:30", tz="America/New_York")
    out = filter_hour_range(df, start, end)
    assert len(out) == 1
    assert out["hour_start"].iloc[0] == ts1
