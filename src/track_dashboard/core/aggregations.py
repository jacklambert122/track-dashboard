from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import polars as pl

AggFactory = Callable[[str], pl.Expr]

AGGREGATIONS: dict[str, AggFactory] = {
    "mean": lambda column: pl.col(column).mean(),
    "median": lambda column: pl.col(column).median(),
    "min": lambda column: pl.col(column).min(),
    "max": lambda column: pl.col(column).max(),
    "std": lambda column: pl.col(column).std(),
    "sum": lambda column: pl.col(column).sum(),
    "first": lambda column: pl.col(column).first(),
    "last": lambda column: pl.col(column).last(),
}


def is_categorical(dtype: pl.DataType) -> bool:
    return dtype in {pl.String, pl.Boolean, pl.Categorical, pl.Enum}


def numeric_columns(
    df: pl.DataFrame,
    *,
    exclude: Iterable[str] = (),
) -> list[str]:
    excluded = set(exclude)
    return [
        column
        for column, dtype in df.schema.items()
        if column not in excluded and dtype.is_numeric()
    ]


def categorical_columns(
    df: pl.DataFrame,
    *,
    exclude: Iterable[str] = (),
) -> list[str]:
    excluded = set(exclude)
    return [
        column
        for column, dtype in df.schema.items()
        if column not in excluded and is_categorical(dtype)
    ]


def aggregate_tracks(
    df: pl.DataFrame,
    *,
    track_id_col: str,
    methods: list[str] | None = None,
    excluded_numeric_columns: Iterable[str] = (),
    included_numeric_columns: Iterable[str] | None = None,
    methods_by_column: Mapping[str, Iterable[str]] | None = None,
) -> pl.DataFrame:
    """Return one row per track with ``{feature}_{aggregation}`` columns."""
    selected_methods = set(methods or [])
    if methods_by_column is not None:
        selected_methods.update(
            method
            for column_methods in methods_by_column.values()
            for method in column_methods
        )
    unsupported = sorted(selected_methods - AGGREGATIONS.keys())
    if unsupported:
        raise ValueError(f"Unsupported aggregation methods: {unsupported}")

    excluded = {track_id_col, *excluded_numeric_columns}
    numeric = numeric_columns(df, exclude=excluded)
    if included_numeric_columns is not None:
        included = set(included_numeric_columns)
        numeric = [column for column in numeric if column in included]
    categorical = categorical_columns(df, exclude={track_id_col})

    expressions: list[pl.Expr] = [pl.len().alias("track_length")]

    for column in numeric:
        column_methods = (
            methods_by_column.get(column, [])
            if methods_by_column is not None
            else methods or []
        )
        for method in column_methods:
            expressions.append(
                AGGREGATIONS[method](column).alias(f"{column}_{method}")
            )

    for column in categorical:
        expressions.append(pl.col(column).drop_nulls().first().alias(column))

    return df.group_by(track_id_col, maintain_order=True).agg(expressions)
