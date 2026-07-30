from io import BytesIO

import polars as pl
import pytest

from track_dashboard.entry import load_input_data, load_uploaded_data


@pytest.fixture
def input_df() -> pl.DataFrame:
    return pl.DataFrame({"track_id": [1, 2], "value": [3.0, 4.0]})


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_load_input_data(tmp_path, input_df, suffix):
    path = tmp_path / f"tracks{suffix}"
    if suffix == ".csv":
        input_df.write_csv(path)
    else:
        input_df.write_parquet(path)

    assert load_input_data(path).equals(input_df)


def test_load_uploaded_csv(input_df):
    data = input_df.write_csv().encode()

    assert load_uploaded_data(data, "tracks.csv").equals(input_df)


def test_load_uploaded_parquet(input_df):
    buffer = BytesIO()
    input_df.write_parquet(buffer)

    assert load_uploaded_data(buffer.getvalue(), "tracks.parquet").equals(input_df)


def test_rejects_unsupported_input(tmp_path):
    with pytest.raises(ValueError, match="csv or .parquet"):
        load_input_data(tmp_path / "tracks.json")
