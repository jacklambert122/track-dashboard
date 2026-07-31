from __future__ import annotations

import param

from .data_model import DataModel
from .state import DashboardState


class AnalysisComponent(param.Parameterized):
    """Small common base for components that consume the active dataframe."""

    def __init__(
        self,
        state: DashboardState,
        data_model: DataModel,
        **params,
    ) -> None:
        super().__init__(**params)
        self.state = state
        self.data_model = data_model
        self.state.param.watch(
            self._data_changed,
            [
                "analysis_level",
                "track_agg_features",
                "track_agg_methods_by_feature",
                "data_revision",
            ],
        )

    def _data_changed(self, _event=None) -> None:
        self.refresh_options()

    def refresh_options(self) -> None:
        raise NotImplementedError
