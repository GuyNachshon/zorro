import sys

from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = 'unknown'
finally:
    del version, PackageNotFoundError

from .utils import format_thousands_label
from .utils import swap_legend_boxes
from .utils import add_axis_label_background
from .utils import add_cbs_logo_to_plot
from .utils import add_values_to_bars
from .utils import add_values_to_bars
from .settings import CBSPlotSettings
from .highcharts import CBSHighChart

__all__ = [
    'format_thousands_label',
    'swap_legend_boxes',
    'add_axis_label_background',
    'add_cbs_logo_to_plot',
    'add_values_to_bars',
    'add_values_to_bars',
    'CBSPlotSettings',
    'CBSHighChart',
]
