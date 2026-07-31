from types import SimpleNamespace

import numpy as np

from track_dashboard.models.onnx import ONNXProbabilityModel


class FakeSession:
    def __init__(self, path, providers):
        self.path = path
        self.providers = providers

    def get_inputs(self):
        return [SimpleNamespace(name="features", type="tensor(float)")]

    def get_outputs(self):
        return [
            SimpleNamespace(name="label"),
            SimpleNamespace(name="probabilities"),
        ]

    def run(self, output_names, inputs):
        assert output_names == ["probabilities"]
        matrix = inputs["features"]
        positive = matrix[:, 0]
        return [np.column_stack([1.0 - positive, positive])]


def test_onnx_probability_model_selects_probability_output(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "quality.onnx"
    path.write_bytes(b"test model placeholder")
    fake_runtime = SimpleNamespace(InferenceSession=FakeSession)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_runtime)

    model = ONNXProbabilityModel(path)
    probabilities = model.predict_proba([[0.25], [0.75]])

    assert model.input_name == "features"
    assert model.output_name == "probabilities"
    assert probabilities.tolist() == [[0.75, 0.25], [0.25, 0.75]]
