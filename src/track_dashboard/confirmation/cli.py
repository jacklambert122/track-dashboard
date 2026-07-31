from __future__ import annotations

import argparse

import panel as pn

from ..cli import DEFAULT_PORT, ensure_port_available, replace_default_server
from ..entry import load_input_data
from ..example_data import make_example_data
from ..models.onnx import build_onnx_model_specs
from .dashboard import TrackConfirmationDashboard
from .engine import MLConfirmationPath, MLModelSpec
from .example import make_example_confirmation_config


def build_default_ml_paths(
    registrations: list[list[str]] | None,
    models: list[MLModelSpec],
) -> list[MLConfirmationPath]:
    """Resolve repeated ``PATH MODEL THRESHOLD`` baseline ML paths."""
    model_registry = {model.name: model for model in models}
    paths = []
    for registration in registrations or []:
        if len(registration) != 3:
            raise ValueError(
                "--default-ml-path requires PATH, MODEL, and THRESHOLD."
            )
        path_name, model_name, threshold_text = registration
        if model_name not in model_registry:
            raise ValueError(
                f"Default ML path {path_name!r} references unknown model "
                f"{model_name!r}."
            )
        paths.append(
            MLConfirmationPath(
                path=path_name,
                model=model_registry[model_name],
                threshold=float(threshold_text),
            )
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Track Confirmation Dashboard server."
    )
    parser.add_argument("--data", help="Optional point data in CSV or Parquet format.")
    parser.add_argument(
        "--config",
        help=(
            "Optional JSON config containing "
            "dynamic_specific.track_qa_config."
        ),
    )
    parser.add_argument("--track-id-col", default="track_id")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--matched-value", default="matched")
    parser.add_argument(
        "--model-dir",
        help=(
            "Optional base directory for model files; defaults to the "
            "configuration file's directory."
        ),
    )
    parser.add_argument(
        "--onnx-model",
        action="append",
        nargs="+",
        metavar="VALUE",
        help=(
            "Register an ONNX model as NAME FILE FEATURE [FEATURE ...]. "
            "Repeat this option to register multiple models."
        ),
    )
    parser.add_argument(
        "--default-ml-path",
        action="append",
        nargs=3,
        metavar=("PATH", "MODEL", "THRESHOLD"),
        help=(
            "Use a registered model as an existing baseline confirmation "
            "path. Repeat for multiple paths."
        ),
    )
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    try:
        data = load_input_data(args.data) if args.data else make_example_data()
        config = args.config if args.config else make_example_confirmation_config()
        ml_models = build_onnx_model_specs(args.onnx_model)
        default_ml_paths = build_default_ml_paths(
            args.default_ml_path,
            ml_models,
        )
        dashboard = TrackConfirmationDashboard(
            data,
            config,
            track_id_col=args.track_id_col,
            time_col=args.time_col,
            label_col=args.label_col,
            matched_value=args.matched_value,
            model_base_dir=args.model_dir,
            ml_models=ml_models,
            default_ml_paths=default_ml_paths,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    if args.port == DEFAULT_PORT:
        replace_default_server(args.address, args.port)
    else:
        ensure_port_available(args.address, args.port)

    pn.extension(
        "tabulator",
        sizing_mode="stretch_width",
        design="material",
        theme="dark",
    )
    pn.serve(
        dashboard.view(),
        title="Track Confirmation Dashboard",
        show=True,
        address=args.address,
        port=args.port,
    )
