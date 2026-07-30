from __future__ import annotations

import math
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
        # Spread track origins across a loose grid, then give each one its own
        # heading, speed, and turn rate. Keeping x/y ahead of frame/time also
        # makes the dashboard's initial scatter plot show the trajectories.
        grid_column = (track_id - 1) % 10
        grid_row = (track_id - 1) // 10
        x = grid_column * 18.0 + rng.uniform(-3.0, 3.0)
        y = grid_row * 18.0 + rng.uniform(-3.0, 3.0)
        heading = rng.uniform(0.0, 2.0 * math.pi)
        speed = rng.uniform(0.6, 1.8)
        turn_rate = rng.uniform(-0.08, 0.08)

        label = "matched" if rng.random() > 0.35 else "unmatched"
        path = rng.choice(["default", "kinematic", "limb_reject"])
        base_snr = rng.uniform(5, 30)
        base_residual = rng.uniform(0.2, 5.0)
        base_limb = rng.uniform(0, 1)

        for frame in range(points_per_track):
            rows.append(
                {
                    "track_id": track_id,
                    "x": x + rng.gauss(0, 0.12),
                    "y": y + rng.gauss(0, 0.12),
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
            heading += turn_rate + rng.gauss(0, 0.01)
            x += speed * math.cos(heading)
            y += speed * math.sin(heading)

    return pl.DataFrame(rows)
