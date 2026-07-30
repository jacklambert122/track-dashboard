from __future__ import annotations

import random

import polars as pl


def make_example_data(
    *,
    tracks: int = 75,
    points_per_track: int = 20,
    seed: int = 7,
) -> pl.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []

    for track_id in range(1, tracks + 1):
        label = "matched" if rng.random() > 0.35 else "unmatched"
        path = rng.choice(["default", "kinematic", "limb_reject"])
        base_snr = rng.uniform(5, 30)
        base_residual = rng.uniform(0.2, 5.0)
        base_limb = rng.uniform(0, 1)

        for frame in range(points_per_track):
            rows.append(
                {
                    "track_id": track_id,
                    "frame": frame,
                    "time": float(frame),
                    "snr": base_snr + rng.gauss(0, 2),
                    "residual": max(0.0, base_residual + rng.gauss(0, 0.4)),
                    "earth_limb_score": min(
                        1.0, max(0.0, base_limb + rng.gauss(0, 0.08))
                    ),
                    "label": label,
                    "confirmation_path": path,
                }
            )

    return pl.DataFrame(rows)
