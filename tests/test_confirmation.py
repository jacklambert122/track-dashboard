import polars as pl
import pytest

from track_dashboard.confirmation import (
    MLConfirmationPath,
    MLModelSpec,
    RangeRule,
    candidate_config,
    evaluate_comparison,
    generic_confirmation_evaluator,
    track_qa_config,
)


def make_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "track_id": [1, 1, 2, 2, 3, 3],
            "time": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "snr": [5.0, 10.0, 4.0, 6.0, 12.0, 13.0],
            "residual": [2.0, 1.0, 0.5, 0.4, 2.0, 2.0],
            "label": [
                "matched",
                "matched",
                "unmatched",
                "unmatched",
                "matched",
                "matched",
            ],
        }
    )


def make_config() -> dict:
    return {
        "dynamic_specific": {
            "track_qa_config": {
                "paths": [
                    {
                        "name": "default_snr",
                        "ranges": {"snr": {"min": 9.0}},
                    }
                ]
            }
        }
    }


def test_track_qa_config_requires_nested_config():
    with pytest.raises(ValueError, match="dynamic_specific"):
        track_qa_config({})


def test_range_rule_validates_bounds():
    with pytest.raises(ValueError, match="greater"):
        RangeRule("path", "snr", minimum=5, maximum=2)


def test_generic_evaluator_creates_binary_path_columns():
    result = generic_confirmation_evaluator(
        track_qa_config(make_config()),
        make_data(),
    )

    assert result.get_column("default_snr").to_list() == [0, 1, 0, 0, 1, 1]


def test_comparison_calculates_timing_label_impact_and_false_alarms():
    comparison = evaluate_comparison(
        make_data(),
        make_config(),
        rules=[RangeRule("low_residual", "residual", maximum=0.5)],
    )

    track_two = comparison.first_times.filter(pl.col("track_id") == 2).row(
        0, named=True
    )
    assert track_two["default_first_confirmation_time"] is None
    assert track_two["candidate_first_confirmation_time"] == 0.0
    assert track_two["newly_confirmed_track"]

    assert comparison.metrics["default_confirmed_tracks"] == 2
    assert comparison.metrics["candidate_confirmed_tracks"] == 3
    assert comparison.metrics["new_confirmed_tracks"] == 1
    assert comparison.metrics["new_confirmed_measurements"] == 2
    assert comparison.metrics["matched_confirmed_measurements"] == 3
    assert comparison.metrics["point_false_alarm_rate"] == pytest.approx(0.4)
    assert comparison.metrics["track_false_alarm_rate"] == pytest.approx(1 / 3)

    unmatched = comparison.label_summary.filter(
        pl.col("label") == "unmatched"
    ).row(0, named=True)
    assert unmatched["new_confirmed_measurements"] == 2
    assert unmatched["new_confirmed_tracks"] == 1

    false_path = comparison.path_summary.filter(
        (pl.col("confirmation_path") == "low_residual")
        & (pl.col("cohort") == "false")
    ).row(0, named=True)
    assert false_path["logic"] == "experimental"
    assert false_path["confirmed_measurements"] == 2
    assert false_path["confirmed_tracks"] == 1


def test_same_named_range_path_overrides_candidate_rule():
    comparison = evaluate_comparison(
        make_data(),
        make_config(),
        rules=[RangeRule("default_snr", "snr", minimum=11.0)],
    )

    track_one = comparison.first_times.filter(pl.col("track_id") == 1).row(
        0, named=True
    )
    assert track_one["default_first_confirmation_time"] == 1.0
    assert track_one["candidate_first_confirmation_time"] is None
    assert comparison.candidate_path_columns == ("default_snr",)
    assert "default_snr" in comparison.path_summary.filter(
        pl.col("logic") == "experimental"
    ).get_column("confirmation_path").to_list()


def test_custom_python_evaluator_and_explicit_paths_are_supported():
    def current_confirmation(_config, data):
        return pl.DataFrame(
            {"manual_path": (data["snr"] >= 12).cast(pl.Int8)}
        )

    comparison = evaluate_comparison(
        make_data(),
        make_config(),
        evaluator=current_confirmation,
        default_path_columns=["manual_path"],
    )

    assert comparison.default_path_columns == ("manual_path",)
    assert comparison.metrics["default_confirmed_tracks"] == 1


def test_all_default_paths_can_be_disabled():
    comparison = evaluate_comparison(
        make_data(),
        make_config(),
        default_path_columns=[],
        rules=[RangeRule("candidate", "residual", maximum=0.5)],
    )

    assert comparison.default_path_columns == ()
    assert comparison.metrics["default_confirmed_tracks"] == 0
    assert comparison.metrics["candidate_confirmed_tracks"] == 1


def test_ml_confirmation_path_scores_thresholds_and_latches():
    class ProbabilityModel:
        def predict_proba(self, features):
            return [
                [1.0 - min(row[0] / 10, 1.0), min(row[0] / 10, 1.0)]
                for row in features
            ]

    model = MLModelSpec(
        name="snr_probability",
        model=ProbabilityModel(),
        features=("snr",),
    )
    comparison = evaluate_comparison(
        make_data(),
        make_config(),
        default_path_columns=[],
        ml_paths=[
            MLConfirmationPath(
                path="ml_snr_path",
                model=model,
                threshold=0.9,
            )
        ],
    )

    assert comparison.candidate_path_columns == ("ml_snr_path",)
    assert comparison.points.get_column("ml_snr_path").to_list() == [
        0,
        1,
        0,
        0,
        1,
        1,
    ]
    assert comparison.points.get_column("candidate_confirmed").to_list() == [
        False,
        True,
        False,
        False,
        True,
        True,
    ]
    ml_summary = comparison.path_summary.filter(
        pl.col("confirmation_path") == "ml_snr_path"
    )
    assert ml_summary.get_column("logic").unique().to_list() == [
        "experimental"
    ]


def test_confirmation_latches_for_all_later_points_in_the_track():
    data = pl.DataFrame(
        {
            "track_id": [1, 1, 1, 1],
            "time": [0.0, 1.0, 2.0, 3.0],
            "snr": [1.0, 10.0, 1.0, 1.0],
            "label": ["matched"] * 4,
        }
    )
    config = {
        "dynamic_specific": {
            "track_qa_config": {
                "paths": [
                    {
                        "name": "snr_path",
                        "ranges": {"snr": {"min": 9.0}},
                    }
                ]
            }
        }
    }

    comparison = evaluate_comparison(data, config)

    assert comparison.points.get_column("default_triggered").to_list() == [
        False,
        True,
        False,
        False,
    ]
    assert comparison.points.get_column("default_confirmed").to_list() == [
        False,
        True,
        True,
        True,
    ]
    assert comparison.metrics["matched_confirmed_measurements"] == 3


def test_confirmation_latching_uses_time_not_input_row_order():
    data = pl.DataFrame(
        {
            "track_id": [1, 1, 1],
            "time": [2.0, 0.0, 1.0],
            "snr": [1.0, 1.0, 10.0],
            "label": ["matched"] * 3,
        }
    )
    config = {
        "dynamic_specific": {
            "track_qa_config": {
                "paths": [
                    {
                        "name": "snr_path",
                        "ranges": {"snr": {"min": 9.0}},
                    }
                ]
            }
        }
    }

    comparison = evaluate_comparison(data, config)

    assert comparison.points.get_column("default_confirmed").to_list() == [
        True,
        False,
        True,
    ]


def test_python_evaluator_receives_each_track_in_time_order():
    data = pl.DataFrame(
        {
            "track_id": [2, 1, 2, 1],
            "time": [1.0, 2.0, 0.0, 0.0],
            "snr": [1.0, 1.0, 1.0, 1.0],
            "label": ["matched"] * 4,
        }
    )
    observed_order = []

    def order_sensitive_evaluator(_config, ordered):
        observed_order.extend(
            ordered.select("track_id", "time").iter_rows()
        )
        return pl.DataFrame({"ordered_path": [0, 0, 1, 0]})

    comparison = evaluate_comparison(
        data,
        make_config(),
        evaluator=order_sensitive_evaluator,
        default_path_columns=["ordered_path"],
    )

    assert observed_order == [(1, 0.0), (1, 2.0), (2, 0.0), (2, 1.0)]
    assert comparison.points.select("track_id", "time").rows() == data.select(
        "track_id", "time"
    ).rows()


def test_candidate_config_preserves_default_and_adds_experimental_paths():
    updated = candidate_config(
        make_config(),
        [RangeRule("candidate", "snr", minimum=7, maximum=20)],
    )

    config = track_qa_config(updated)
    assert config["paths"][0]["name"] == "default_snr"
    assert config["experimental_paths"] == [
        {
            "name": "candidate",
            "ranges": {"snr": {"min": 7, "max": 20}},
        }
    ]
