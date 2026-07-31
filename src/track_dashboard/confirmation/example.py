from __future__ import annotations

from typing import Any


def make_example_confirmation_config() -> dict[str, Any]:
    """Return a runnable nested config for the generated example tracks."""
    return {
        "dynamic_specific": {
            "track_qa_config": {
                "paths": [
                    {
                        "name": "quality_path",
                        "ranges": {
                            "snr": {"min": 20.0},
                            "residual": {"max": 2.0},
                        },
                    },
                    {
                        "name": "limb_path",
                        "ranges": {
                            "snr": {"min": 12.0},
                            "earth_limb_score": {"min": 0.75},
                        },
                    },
                ]
            }
        }
    }
