"""
Definition of CBS rbg colors. Based on the color rgb definitions from the cbs LaTeX template
"""

import logging
import math

import matplotlib as mpl

from cbsplotlib.colors import get_color_palette, set_cbs_colors

_logger = logging.getLogger(__name__)

RATIO_OPTIONS = {'golden_ratio', 'equal', 'from_rows'}

PT_PER_INCHE = 72.27
INCHES_PER_PT = 1 / PT_PER_INCHE
GOLDEN_MEAN = (math.sqrt(5) - 1) / 2


class CBSPlotSettings(object):
    """
    Class to hold the figure size for a standard document

    Parameters
    ----------
    number_of_figures_rows: int, optional
        Number of figure rows, default = 2
    number_of_figures_cols: int, optional
        Number of figure cols, default = 1
    text_width_in_pt: float, optional
        Width of the text in pt, default = 392.64
    text_height_in_pt: float, optional
        Height of the text in pt: default = 693
    text_margin_bot_in_inch: float, optional
        Space at the bottom in inch. Default = 1 inch
    plot_parameters: dict, optional
        Dictionary with plot settings. If None (default), take the cbs defaults
    color_palette: {"koel", "warm"}, optional
        Pick color palette for the plot. Default is "koel"
    font_size: int, optional
        Size of all fonts. Default = 8

    Notes
    ----------
    * The variables are set to make sure that the figure have the exact same size as the document,
      such that we do not have to rescale them. In this way the fonts will have the same size
      here as in the document
    """

    def __init__(
        self,
        fig_width_in_inch: float = None,
        fig_height_in_inch: float = None,
        number_of_figures_cols: int = 1,
        number_of_figures_rows: int = 2,
        text_width_in_pt: float = 392.64813,
        text_height_in_pt: float = 693,
        text_margin_bot_in_inch: float = 1.0,  # margin in inch
        ratio_option='golden_ratio',
        plot_parameters: dict = None,
        color_palette: str = 'koel',
        font_size: float = 8,
        set_gray_x_tics: bool = False,
        set_gray_y_tics: bool = False,
        reverse: bool = False,
        offset: int = 0,
    ):
        self.number_of_figures_rows = number_of_figures_rows
        self.number_of_figures_cols = number_of_figures_cols
        self.text_width_in_pt = text_width_in_pt
        self.text_height_in_pt = text_height_in_pt
        self.text_margin_bot_in_inch = text_margin_bot_in_inch

        self.text_height = (text_height_in_pt * INCHES_PER_PT,)
        self.text_width = text_width_in_pt * INCHES_PER_PT

        text_width_in_pt = (
            392.64813  # add the line \showthe\columnwidth above you figure in latex
        )
        text_height_in_pt = (
            693  # add the line \showthe\columnwidth above you figure in latex
        )
        text_height = text_height_in_pt * INCHES_PER_PT
        text_width = text_width_in_pt * INCHES_PER_PT
        text_margin_bot = 1.0  # margin in inch

        if fig_width_in_inch is not None:
            self.fig_width = fig_width_in_inch
        else:
            self.fig_width = text_width / number_of_figures_cols

        if fig_height_in_inch is not None:
            self.fig_height = fig_height_in_inch
        elif ratio_option == 'golden_ratio':
            self.fig_height = self.fig_width * GOLDEN_MEAN
        elif ratio_option == 'equal':
            self.fig_height = self.fig_width
        elif ratio_option == 'from_rows':
            self.fig_height = (text_height - text_margin_bot) / number_of_figures_rows
        else:
            raise ValueError(
                f"fig height is not given by 'fig_height_in_inch' and 'ratio_option' "
                f"= {ratio_option} is not in {RATIO_OPTIONS}"
            )

        self.fig_size = (self.fig_width, self.fig_height)

        prop_cycle = get_color_palette(color_palette, reverse=reverse, offset=offset)

        if plot_parameters is not None:
            self.params = plot_parameters
        else:
            self.params = {
                'axes.labelsize': font_size,
                'font.size': font_size,
                'legend.fontsize': font_size,
                'xtick.labelsize': font_size,
                'ytick.labelsize': font_size,
                'figure.figsize': self.fig_size,
                'grid.color': 'cbs:highchartslichtgrijs',
                'grid.linewidth': 1.0,
                'hatch.color': 'cbs:highchartslichtgrijs',
                'axes.prop_cycle': prop_cycle,
                'axes.edgecolor': 'cbs:grijs',
                'axes.linewidth': 1.5,
            }

        set_cbs_colors()
        mpl.rcParams.update(self.params)

        if set_gray_x_tics:
            self.set_tick_color(axis='x')
        if set_gray_y_tics:
            self.set_tick_color(axis='y')

    @staticmethod
    def set_tick_color(axis: str = None):
        """
        Zet de kleur van de ticks grijs.

        Parameters
        ----------
        axis: str

        Notes
        -----
        * Dit is ook hoe highcharts dat heeft.
        * Probleem is dat de kleur van de tick en de tick label voor matplotlib <3.4 gekoppeld zijn,
          zodat dit niet goed mogelijk is.

        """
        if axis is None or axis not in ('x', 'y'):
            msg = "Specificeer de axis waarvan je de ticks wilt kleuren met axis='x' of axis='y'"
            raise ValueError(msg)

        try:
            mpl.rcParams.update({f"{axis}tick.labelcolor": 'black'})
        except KeyError:
            _logger.warning(
                'In matplotlib <3.4 kan je nog niet de tick kleur en tick label kleur'
                'apart instellen omdat xtick.labelcolor nog niet bestaat. Doe'
                'in je script dan gewoon '
                "ax.tick_params(colors='cbs:highchartslichtgrijs', which='both')"
            )
        else:
            # We konden de label kleur op black zetten. Verander dan nu de tick kleur
            mpl.rcParams.update(
                {
                    f"{axis}tick.color": 'cbs:grijs',
                }
            )
