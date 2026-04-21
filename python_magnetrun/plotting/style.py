"""Plot style and color configuration dataclasses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = [
    "PlotStyle",
    "PlotColors",
    "PlotConfig",
    "DEFAULT_STYLE",
    "DEFAULT_COLORS",
    "BUNDLED_CONFIG_PATH",
    "load_plot_config",
    "save_plot_config",
]

# Path to the default config shipped with the package.
BUNDLED_CONFIG_PATH: Path = Path(__file__).parent / "plot_config.json"


@dataclass
class PlotStyle:
    """Configuration for plot styling."""

    figsize: tuple[int, int] = (12, 5)
    dpi: int = 300
    grid: bool = True
    grid_alpha: float = 0.3
    legend_loc: str = "best"
    title_fontsize: int = 12
    label_fontsize: int = 10


@dataclass
class PlotColors:
    """Color configuration for different data sources and regimes."""

    overview: str = "blue"
    archive: str = "red"
    pupitre: str = "green"
    incident: str = "yellow"
    regime_up: str = "green"
    regime_down: str = "red"
    regime_plateau: str = "blue"
    # Matplotlib tab10 palette — cycled per file when same_color_per_type is False
    palette: list[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ])

    def get_file_color(self, file_idx: int, f_extension: str, same_color_per_type: bool) -> str:
        """Return the color to use for a given file.

        When *same_color_per_type* is True all files of the same extension
        share the type color (pupitre / archive / overview).  Otherwise each
        file gets a distinct color from *palette*, cycling when there are more
        files than palette entries.
        """
        if same_color_per_type:
            if f_extension == ".txt":
                return self.pupitre
            if f_extension == ".tdms":
                return self.archive
            return self.overview
        return self.palette[file_idx % len(self.palette)]

    def get_regime_color(self, regime: str) -> str:
        """Return the color for a regime type ('U', 'D', 'P')."""
        regime_map = {
            "U": self.regime_up,
            "D": self.regime_down,
            "P": self.regime_plateau,
        }
        return regime_map.get(regime, "gray")


@dataclass
class PlotConfig:
    """Combined style + color configuration for plots."""

    style: PlotStyle = None  # type: ignore[assignment]
    colors: PlotColors = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.style is None:
            self.style = PlotStyle()
        if self.colors is None:
            self.colors = PlotColors()

    def to_dict(self) -> dict:
        return {"style": asdict(self.style), "colors": asdict(self.colors)}

    @classmethod
    def from_dict(cls, data: dict) -> PlotConfig:
        style_data = data.get("style", {})
        colors_data = data.get("colors", {})
        # figsize is stored as a list in JSON; convert back to tuple
        if "figsize" in style_data:
            style_data["figsize"] = tuple(style_data["figsize"])
        return cls(style=PlotStyle(**style_data), colors=PlotColors(**colors_data))


def load_plot_config(path: str | Path) -> PlotConfig:
    """Load a :class:`PlotConfig` from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return PlotConfig.from_dict(data)


def save_plot_config(config: PlotConfig, path: str | Path) -> None:
    """Save a :class:`PlotConfig` to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


DEFAULT_STYLE = PlotStyle()
DEFAULT_COLORS = PlotColors()
