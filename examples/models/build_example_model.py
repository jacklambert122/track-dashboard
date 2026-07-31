"""Rebuild ``example_quality_model.onnx`` from documented coefficients."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_model(output_path: Path) -> None:
    weights = numpy_helper.from_array(
        np.asarray([[0.18], [-0.9]], dtype=np.float32),
        name="weights",
    )
    intercept = numpy_helper.from_array(
        np.asarray([-2.5], dtype=np.float32),
        name="intercept",
    )
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["features", "weights"], ["linear"]),
            helper.make_node(
                "Add",
                ["linear", "intercept"],
                ["logits"],
            ),
            helper.make_node("Sigmoid", ["logits"], ["probabilities"]),
        ],
        "example_quality_confirmation_model",
        [
            helper.make_tensor_value_info(
                "features",
                TensorProto.FLOAT,
                [None, 2],
            )
        ],
        [
            helper.make_tensor_value_info(
                "probabilities",
                TensorProto.FLOAT,
                [None, 1],
            )
        ],
        initializer=[weights, intercept],
    )
    model = helper.make_model(
        graph,
        producer_name="track-dashboard",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


if __name__ == "__main__":
    build_model(Path(__file__).with_name("example_quality_model.onnx"))
