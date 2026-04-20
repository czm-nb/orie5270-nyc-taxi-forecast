import pandas as pd
import pytest

from nyc_taxi_forecast.clean import clean_trips


def test_clean_trips_missing_required_columns_raises() -> None:
    with pytest.raises(ValueError):
        clean_trips(pd.DataFrame({"tpep_pickup_datetime": [1]}))
