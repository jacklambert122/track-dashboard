from __future__ import annotations

import holoviews as hv
import panel as pn

DARK_THEME_CSS = """
:root {
  --track-bg: #0b1220;
  --track-surface: #121c2e;
  --track-surface-raised: #18253a;
  --track-border: #2b3b54;
  --track-text: #e7edf7;
  --track-muted: #9fb0c8;
  --track-accent: #6dd3ce;
}

html, body {
  background: var(--track-bg);
  color: var(--track-text);
}

.track-dashboard-root {
  background:
    radial-gradient(circle at top right, #17304b 0, transparent 34rem),
    var(--track-bg);
  color: var(--track-text);
  min-height: 100vh;
  padding: 12px;
}

.track-dashboard-root .card,
.track-dashboard-root .bk-card,
.track-dashboard-root .bk-Card {
  background: linear-gradient(145deg, var(--track-surface-raised), var(--track-surface));
  border: 1px solid var(--track-border);
  border-radius: 10px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
}

.track-dashboard-root .bk-tab {
  color: var(--track-muted);
}

.track-dashboard-root .bk-tab.bk-active {
  color: var(--track-accent);
  border-color: var(--track-accent);
}

.track-dashboard-root .tabulator,
.track-dashboard-root .tabulator-tableholder,
.track-dashboard-root .tabulator-row {
  background-color: var(--track-surface);
  color: var(--track-text);
  border-color: var(--track-border);
}

.track-dashboard-root .tabulator-header,
.track-dashboard-root .tabulator-col {
  background-color: var(--track-surface-raised);
  color: var(--track-text);
  border-color: var(--track-border);
}

.track-dashboard-root a {
  color: var(--track-accent);
}

.track-dashboard-root .bk-input,
.track-dashboard-root .bk-input-group,
.track-dashboard-root select,
.track-dashboard-root input,
.track-dashboard-root .choices,
.track-dashboard-root .choices__inner,
.track-dashboard-root .choices__input,
.track-dashboard-root .choices__list,
.track-dashboard-root .choices__list--single,
.track-dashboard-root .choices__list--multiple,
.track-dashboard-root .choices__list--dropdown,
.track-dashboard-root .choices__list[aria-expanded] {
  background-color: var(--track-surface-raised) !important;
  border-color: var(--track-border) !important;
  color: var(--track-text) !important;
}

.track-dashboard-root option,
.track-dashboard-root .choices__item,
.track-dashboard-root .choices__item--selectable,
.track-dashboard-root .choices__placeholder {
  color: var(--track-text) !important;
}

.track-dashboard-root option {
  background-color: var(--track-surface-raised) !important;
}

.track-dashboard-root .choices__item--choice.is-highlighted,
.track-dashboard-root .choices__item--selectable.is-highlighted,
.track-dashboard-root option:checked,
.track-dashboard-root option:hover {
  background-color: #24506a !important;
  color: #ffffff !important;
}

.track-dashboard-root .choices__list--multiple .choices__item {
  background-color: #24506a !important;
  border-color: var(--track-accent) !important;
  color: #ffffff !important;
}

.track-dashboard-root .choices[data-type*="select-one"]::after {
  border-color: var(--track-muted) transparent transparent !important;
}

.track-dashboard-root input::placeholder {
  color: var(--track-muted) !important;
  opacity: 1;
}
"""

DARK_DROPDOWN_STYLESHEET = """
:host {
  color-scheme: dark;
  --track-control-bg: #18253a;
  --track-control-border: #2b3b54;
  --track-control-text: #e7edf7;
  --track-control-muted: #9fb0c8;
  --track-control-highlight: #24506a;
}

.bk-input,
select,
input,
.choices,
.choices__inner,
.choices__input,
.choices__list,
.choices__list--single,
.choices__list--multiple,
.choices__list--dropdown,
.choices__list[aria-expanded] {
  background: var(--track-control-bg) !important;
  background-color: var(--track-control-bg) !important;
  border-color: var(--track-control-border) !important;
  color: var(--track-control-text) !important;
}

option,
.choices__item,
.choices__item--selectable,
.choices__placeholder {
  background-color: var(--track-control-bg) !important;
  color: var(--track-control-text) !important;
}

.choices__item--choice.is-highlighted,
.choices__item--selectable.is-highlighted,
option:checked,
option:hover {
  background-color: var(--track-control-highlight) !important;
  color: #ffffff !important;
}

.choices__list--multiple .choices__item {
  background-color: var(--track-control-highlight) !important;
  border-color: #6dd3ce !important;
  color: #ffffff !important;
}

input::placeholder,
.choices__placeholder {
  color: var(--track-control-muted) !important;
  opacity: 1;
}
"""


def apply_dark_theme() -> None:
    """Apply the shared dark styling without requiring a specific entry point."""
    hv.renderer("bokeh").theme = "dark_minimal"
    if DARK_THEME_CSS not in pn.config.raw_css:
        pn.config.raw_css.append(DARK_THEME_CSS)
    for widget_type in (
        pn.widgets.Select,
        pn.widgets.MultiChoice,
        pn.widgets.MultiSelect,
        pn.widgets.AutocompleteInput,
    ):
        stylesheets = list(widget_type.param.stylesheets.default)
        if DARK_DROPDOWN_STYLESHEET not in stylesheets:
            widget_type.param.stylesheets.default = [
                *stylesheets,
                DARK_DROPDOWN_STYLESHEET,
            ]
