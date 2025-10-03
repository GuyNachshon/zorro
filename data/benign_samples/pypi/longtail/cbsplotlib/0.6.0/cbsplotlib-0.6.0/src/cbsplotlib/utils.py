"""
Utility functions
"""

import logging
from typing import List, Tuple, Dict

import matplotlib
import matplotlib.patches as m_patches
import matplotlib.transforms as trn
import numpy as np
from matplotlib.path import Path as mPath

from matplotlib.axes import Axes
from matplotlib.artist import Artist

_logger = logging.getLogger(__name__)

RATIO_OPTIONS = {"golden_ratio", "equal", "from_rows"}


def add_values_to_bars(
    axis,
    bar_type="bar",
    position="c",
    label_format="{:.0f}",
    x_offset=0,
    y_offset=0,
    color="k",
    horizontalalignment="center",
    verticalalignment="center",
):
    """
    Add the values of the bars as number in the center

    This function adds the numerical value of each bar to the bar itself. By default, it will be
    centered on the bar, but you can change this by using the `position` parameter. The value is
    formatted according to the `format` parameter.

    Parameters
    ----------
    axis: `matplotlib.pyplot.axes.Axes` object
        Axis containing the bar plot
    bar_type: {"bar", "barh"}
        Direction of the bars. Default = "bar", meaning vertical bars. Alternatively, you need to
        specify "barh" for horizontal bars.
    position: {"c", "t", "l", "r", "b"}, optional
        Location of the numbers, where "c" is center, "t" is top, "l" is left, "r" is right and "b"
        is bottom. Default = "c"
    label_format: str, optional
        Formatter to use for the numbers. Default = "{:.0f}" (remove digits from float)
    x_offset: float, optional
        x offset in pt. Default = 0
    y_offset: float, optional
        y offset in pt. Default = 0
    color: "str", optional
        Color of the characters, Default is black
    horizontalalignment: str, optional
        Horizontal alignment of the numbers. Default = "center"
    verticalalignment: str, optional
        Vertical alignment of the numbers Default = "center"
    """

    # voeg percentage to aan bars
    for patch in axis.patches:
        b = patch.get_bbox()
        # calculate the center of the bar
        cx = (b.x1 + b.x0) / 2
        cy = (b.y1 + b.y0) / 2

        # calculate the height and width of the bar
        hh = b.y1 - b.y0
        ww = b.x1 - b.x0

        # determine the position of the text
        if position == "c":
            # center
            (px, py) = (cx, cy)
        elif position == "t":
            # top
            (px, py) = (cx, cy + hh / 2)
        elif position == "b":
            # bottom
            (px, py) = (cx, cy - hh / 2)
        elif position == "l":
            # left
            (px, py) = (cx - ww / 2, cy)
        elif position == "r":
            # right
            (px, py) = (cx + ww / 2, cy)
        else:
            raise ValueError(f"position = {position} not recognised. Please check")

        # add the offsets
        (px, py) = (px + x_offset, py + y_offset)

        # determine the value of the bar
        if bar_type == "bar":
            value = hh
        elif bar_type == "barh":
            value = ww
        else:
            raise ValueError(f"type = {bar_type} not recognised. Please check")

        # make the value string using the format specifier
        value_string = label_format.format(value)

        # add the value to the plot
        axis.annotate(
            value_string,
            (px, py),
            color=color,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
        )


def add_cbs_logo_to_plot(
    fig,
    axes=None,
    margin_x_in_mm=6.0,
    margin_y_in_mm=6.0,
    x0=0,
    y0=0,
    width=None,
    height=None,
    zorder_start=1,
    fillcolor="cbs:highchartslichtgrijs",
    edgecolor="cbs:logogrijs",
):
    """
    Add the CBS logo to the plot.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure` object
        The total canvas of the Figure
    axes : `matplotlib.axes.Axes` object, optional
        The axes of the plot to add a box
    margin_x_in_mm : float, optional
        The margin between the left of the Figure and the logo in mm. Default = 6.0
    margin_y_in_mm : float, optional
        The margin between the bottom of the Figure and the logo in mm. Default = 6.0
    x0 : float, optional
        The x-coordinate of the bottom left corner of the gray square in axis fraction coordinates.
        Default = 0
    y0 : float, optional
        The y-coordinate of the bottom left corner of the gray square in axis fraction coordinates.
        Default = 0
    width : float, optional
        The width of the gray square in axis fraction coordinates. Default = 1
    height : float, optional
        The height of the gray square in axis fraction coordinates. Default = 1
    zorder_start : int, optional
        The zorder of the first path of the logo. The zorders of the other paths are incremented by 1.
        Default = 1
    fillcolor : str, optional
        The fill color of the logo. Default = "cbs:highchartslichtgrijs"
    edgecolor : str, optional
        The edge color of the logo. Default = "cbs:logogrijs"

    Returns
    -------
    None

    """
    if width is None:
        ww = 1
    else:
        ww = width
    if height is None:
        hh = 1
    else:
        hh = height

    if axes is not None:
        tb = trn.Bbox.from_bounds(x0, y0, ww, hh).transformed(axes.transAxes)

        # Calculate the bottom left corner of the figure in Figure coordinates (pt of the bottom left corner)
        x0 = tb.x0 + (margin_x_in_mm / 25.4) * fig.dpi
        y0 = tb.y0 + (margin_y_in_mm / 25.4) * fig.dpi
    else:
        x0 = (margin_x_in_mm / 25.4) * fig.dpi
        y0 = (margin_y_in_mm / 25.4) * fig.dpi

    all_points = _get_cbs_logo_points()

    if axes is not None:
        trans = axes.transAxes
    else:
        trans = fig.transFigure

    zorder = zorder_start
    for points_in_out in all_points:
        for ii, points in enumerate(points_in_out):
            points[:, :2] *= fig.dpi / 25.4
            points[:, 0] += x0
            points[:, 1] += y0
            pl = points[:, :2]
            dr = points[:, 2]
            tr_path = mPath(pl, dr).transformed(trans.inverted())
            if ii == 0:
                color = edgecolor
            else:
                color = fillcolor
            poly = m_patches.PathPatch(
                tr_path, fc=color, linewidth=0, zorder=zorder, transform=trans
            )
            poly.set_clip_on(False)
            if axes is not None:
                axes.add_patch(poly)
            else:
                fig.patches.append(poly)
            zorder += 1


def _get_cbs_logo_points(logo_width_in_mm=3.234, rrcor=0.171):
    """
    Generate the CBS logo points as a list of numpy arrays.

    This function returns the points required to draw the CBS logo,
    consisting of the letters 'C', 'B', and 'S'. Each letter is represented
    by a set of points describing its outline and inner details, using
    matplotlib path codes for path drawing.

    Parameters
    ----------
    logo_width_in_mm : float, optional
        The width of the logo in millimeters. Default is 3.234.
    rrcor : float, optional
        The radius of the rounded corners in millimeters. Default is 0.171.

    Returns
    -------
    list
        A list containing numpy arrays. Each numpy array represents a part
        of a letter and contains point coordinates and path codes for the logo.
    """
    ww = logo_width_in_mm

    # punten C, beginnen links onder, tegen klok in, binnen en buiten kant
    points_c = [
        np.array(
            list(
                [
                    [0.000, 2.663, mPath.MOVETO],
                    [1.430, 2.663, mPath.LINETO],
                    [1.430, 3.308, mPath.LINETO],
                    [0.644, 3.308, mPath.LINETO],
                    [0.644, 3.577, mPath.LINETO],
                    [1.430, 3.577, mPath.LINETO],
                    [1.430, 4.221, mPath.LINETO],
                    [rrcor, 4.221, mPath.LINETO],
                    [0.000, 4.221, mPath.CURVE3],
                    [0.000, 4.221 - rrcor, mPath.CURVE3],
                    [0.000, 2.663, mPath.CLOSEPOLY],
                ]
            )
        ),
        np.array(
            list(
                [
                    [0.188, 2.851, mPath.MOVETO],
                    [1.242, 2.851, mPath.LINETO],
                    [1.242, 3.120, mPath.LINETO],
                    [1.242, 3.120, mPath.LINETO],
                    [0.457, 3.120, mPath.LINETO],
                    [0.457, 3.765, mPath.LINETO],
                    [1.242, 3.765, mPath.LINETO],
                    [1.242, 4.033, mPath.LINETO],
                    [0.188, 4.033, mPath.LINETO],
                    [0.188, 2.851, mPath.CLOSEPOLY],
                ]
            )
        ),
    ]

    points_b1 = [
        np.array(
            list(
                [
                    [1.674, 2.663, mPath.MOVETO],
                    [3.234, 2.663, mPath.LINETO],
                    [3.234, 4.221 - rrcor, mPath.LINETO],
                    [3.234, 4.221, mPath.CURVE3],
                    [3.063, 4.221, mPath.CURVE3],
                    [2.318, 4.221, mPath.LINETO],
                    [2.318, 4.996 - rrcor, mPath.LINETO],
                    [2.318, 4.996, mPath.CURVE3],
                    [2.147, 4.996, mPath.CURVE3],
                    [1.674, 4.996, mPath.LINETO],
                    [1.674, 2.663, mPath.CLOSEPOLY],
                ]
            )
        ),
        np.array(
            list(
                [
                    [1.862, 2.851, mPath.MOVETO],
                    [3.046, 2.851, mPath.LINETO],
                    [3.046, 4.034, mPath.LINETO],
                    [2.130, 4.034, mPath.LINETO],
                    [2.130, 4.808, mPath.LINETO],
                    [1.862, 4.808, mPath.LINETO],
                    [1.862, 2.851, mPath.CLOSEPOLY],
                ]
            )
        ),
    ]

    # in binnen stuk van de b
    points_b2 = [
        np.array(
            list(
                [
                    [2.129, 3.121, mPath.MOVETO],
                    [2.775, 3.121, mPath.LINETO],
                    [2.775, 3.766, mPath.LINETO],
                    [2.129, 3.766, mPath.LINETO],
                    [2.129, 3.121, mPath.CLOSEPOLY],
                ]
            )
        ),
        np.array(
            list(
                [
                    [2.317, 3.309, mPath.MOVETO],
                    [2.588, 3.309, mPath.LINETO],
                    [2.588, 3.578, mPath.LINETO],
                    [2.317, 3.578, mPath.LINETO],
                    [2.317, 3.309, mPath.CLOSEPOLY],
                ]
            )
        ),
    ]

    # De punten van de S, beginnende linksboven, tegen de klok in.
    # Eerst array is de buitenkant, tweede array is de binnenkant
    points_s = [
        np.array(
            list(
                [
                    [0.000, 2.420, mPath.MOVETO],
                    [0.000, 0.888, mPath.LINETO],
                    [2.589, 0.888, mPath.LINETO],
                    [2.589, 0.645, mPath.LINETO],
                    [0.000, 0.645, mPath.LINETO],
                    [0.000, rrcor, mPath.LINETO],
                    [0.000, 0.000, mPath.CURVE3],
                    [rrcor, 0, mPath.CURVE3],
                    [ww - rrcor, 0, mPath.LINETO],
                    [ww, 0, mPath.CURVE3],
                    [ww, rrcor, mPath.CURVE3],
                    [ww, 1.533, mPath.LINETO],
                    [0.646, 1.533, mPath.LINETO],
                    [0.646, 1.772, mPath.LINETO],
                    [3.234, 1.772, mPath.LINETO],
                    [3.234, 2.420, mPath.LINETO],
                    [0.000, 2.420, mPath.CLOSEPOLY],
                ]
            )
        ),
        np.array(
            list(
                [
                    [0.188, 2.232, mPath.MOVETO],
                    [0.188, 1.076, mPath.LINETO],
                    [2.777, 1.076, mPath.LINETO],
                    [2.777, 0.457, mPath.LINETO],
                    [0.188, 0.457, mPath.LINETO],
                    [0.188, 0.188, mPath.LINETO],
                    [3.045, 0.188, mPath.LINETO],
                    [3.045, 1.345, mPath.LINETO],
                    [0.458, 1.345, mPath.LINETO],
                    [0.458, 1.960, mPath.LINETO],
                    [3.045, 1.960, mPath.LINETO],
                    [3.045, 2.232, mPath.LINETO],
                    [0.188, 2.232, mPath.CLOSEPOLY],
                ]
            )
        ),
    ]

    return [points_c, points_b1, points_b2, points_s]


def add_axis_label_background(
    fig,
    axes,
    alpha=1,
    margin=0.05,
    x0=None,
    y0=None,
    loc="east",
    radius_corner_in_mm=1,
    logo_margin_x_in_mm=1,
    logo_margin_y_in_mm=1,
    add_logo=True,
    aspect=None,
    backgroundcolor="cbs:highchartslichtgrijs",
    logo_fillcolor="cbs:highchartslichtgrijs",
    logo_edgecolor="cbs:logogrijs",
):
    """
    Add a background to the axis labels.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure` object
        The total canvas of the Figure
    axes : `matplotlib.axes.Axes` object
        The axes of the plot to add a box
    alpha : float, optional
        The transparency of the background.
        Default = 1
    margin : float, optional
        The margin between the axis labels and the border of the box in axes fraction coordinates.
        Default = 0.05
    x0 : float, optional
        The x-coordinate of the bottom left corner of the gray square in axes fraction coordinates.
        Default = None
    y0 : float, optional
        The y-coordinate of the bottom left corner of the gray square in axes fraction coordinates.
        Default = None
    loc : str, optional
        The location of the gray box. Only "east" and "south" are implemented.
        Default = "east"
    radius_corner_in_mm : float, optional
        The radius of the corner in mm. Default = 1
    logo_margin_x_in_mm : float, optional
        The margin between the left of the Figure and the logo in mm. Default = 1
    logo_margin_y_in_mm : float, optional
        The margin between the bottom of the Figure and the logo in mm. Default = 1
    add_logo : bool, optional
        Whether to add the CBS logo to the plot.
        Default = True
    aspect : float, optional
        The aspect ratio of the plot. If None, the aspect ratio of the Figure is used.
        Default = None
    backgroundcolor : str, optional
        The background color of the plot. Default = "cbs:highchartslichtgrijs"
    logo_fillcolor : str, optional
        The fill color of the logo. Default = "cbs:highchartslichtgrijs"
    logo_edgecolor : str, optional
        The edge color of the logo. Default = "cbs:logogrijs"

    Returns
    -------
    None

    """
    bbox_axis_fig = axes.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    # the bounding box with respect to the axis coordinates
    # (0 is bottom left axis, 1 is top right axis)
    bbox_axi = axes.get_tightbbox(fig.canvas.get_renderer()).transformed(
        axes.transAxes.inverted()
    )

    if loc == "east":
        if x0 is None:
            x0 = bbox_axi.x0 - margin * bbox_axi.width
        x1 = 0

        y0 = 0
        y1 = 1

    elif loc == "south":
        x0 = 0
        x1 = 1

        if y0 is None:
            y0 = bbox_axi.y0 - margin * bbox_axi.height
        y1 = 0
    else:
        raise ValueError(
            f"Location loc = {loc} is not recognised. Only east and south implemented"
        )

    # width and height of the gray box area
    width = x1 - x0
    height = y1 - y0

    _logger.debug(f"Adding rectangle with width {width} and height {height}")

    # eerste vierkant zorgt voor rechte hoeken aan de rechterkant
    if loc == "east":
        rec_p = (x0 + width / 2, y0)
        rec_w = width / 2
        rec_h = height
    elif loc == "south":
        rec_p = (x0, y0 + height / 2)
        rec_w = width
        rec_h = height / 2
    else:
        raise AssertionError("This should not happen")

    p1 = m_patches.Rectangle(
        rec_p,
        width=rec_w,
        height=rec_h,
        alpha=alpha,
        facecolor=backgroundcolor,
        edgecolor=backgroundcolor,
        zorder=0,
    )
    p1.set_transform(axes.transAxes)
    p1.set_clip_on(False)

    # tweede vierkant zorgt voor ronde hoeken aan de linkerkant
    radius_in_inch = radius_corner_in_mm / 25.4
    xshift = radius_in_inch / bbox_axis_fig.width
    yshift = radius_in_inch / bbox_axis_fig.height
    pad = radius_in_inch / bbox_axis_fig.width
    # we moeten corrigeren voor de ronding van de hoeken als we een aspect ratio hebben
    if aspect is None:
        aspect = bbox_axis_fig.height / bbox_axis_fig.width
    _logger.debug(f"Using aspect ratio {aspect}")
    p2 = m_patches.FancyBboxPatch(
        (x0 + xshift, y0 + yshift),
        width=width - 2 * xshift,
        height=height - 2 * yshift,
        mutation_aspect=1 / aspect,
        alpha=alpha,
        facecolor=backgroundcolor,
        edgecolor=backgroundcolor,
        transform=fig.transFigure,
        zorder=0,
    )
    p2.set_boxstyle("round", pad=pad)
    p2.set_transform(axes.transAxes)
    p2.set_clip_on(False)

    axes.add_patch(p1)
    axes.add_patch(p2)

    if add_logo:
        add_cbs_logo_to_plot(
            fig=fig,
            axes=axes,
            x0=x0,
            y0=y0,
            width=width,
            height=height,
            margin_x_in_mm=logo_margin_x_in_mm,
            margin_y_in_mm=logo_margin_y_in_mm,
            edgecolor=logo_edgecolor,
            fillcolor=logo_fillcolor,
        )


def remove_artists(axis: Axes, artists: list[Artist]) -> None:
    """
    Remove artists from an axis.

    Parameters
    ----------
    axis : Axes
        The axis from which to remove artists.
    artists : list[Artist]
        A list of artists to be removed.

    Returns
    -------
    None
    """
    for artist in artists:
        # Try to remove the artist from the axis
        try:
            artist.remove()
        except AttributeError:
            pass

        # Try to remove the artist from the axis collections
        try:
            while artist.collections:
                for collection in artist.collections:
                    artist.collections.remove(collection)
                    try:
                        axis.collections.remove(collection)
                    except ValueError:
                        pass
        except AttributeError:
            pass


def format_thousands_label(value: float, _: object) -> str:
    """
    Format a value with a thousand separator.

    Parameters
    ----------
    value : float
        The value to format.
    _ : object
        Unused parameter, only present to match the signature for a FormatStrFormatter.

    Returns
    -------
    str
        Formatted value with spaces as a thousand separator.
    """
    int_value = int(value)
    return "{:,}".format(int_value).replace(",", " ")


def swap_legend_boxes(
    handles: List[matplotlib.artist.Artist],  # List of legend handles
    labels: List[str],  # List of legend labels
    n_cols: int,  # Number of columns in the legend
) -> Tuple[List[matplotlib.artist.Artist], List[str]]:
    """
    Rearrange legend handles and labels to match the order of the first row.

    In matplotlib, legend handles and labels are filled column-wise, while in highcharts, they are filled row-wise.

    Parameters
    ----------
    handles : List[matplotlib.artist.Artist]
        The list of legend handles.
    labels : List[str]
        The list of legend labels.
    n_cols : int
        The number of columns in the legend.

    Returns
    -------
    reordered_handles : List[matplotlib.artist.Artist]
        The rearranged list of legend handles.
    reordered_labels : List[str]
        The rearranged list of legend labels.
    """
    reordered_handles: List[matplotlib.artist.Artist] = handles.copy()
    reordered_labels: List[str] = labels.copy()

    if len(reordered_labels) != len(reordered_handles):
        raise ValueError("Number of handles and labels must be equal.")

    rows_per_col: Dict[int, int] = {}

    # Calculate number of rows per column
    for idx, _ in enumerate(labels):
        col = idx % n_cols
        rows_per_col[col] = rows_per_col.get(col, 0) + 1

    current_index = 0
    current_row = 0
    for idx, (handle, label) in enumerate(zip(handles, labels)):
        reordered_handles[current_index] = handle
        reordered_labels[current_index] = label

        col = idx % n_cols
        num_rows = rows_per_col[col]
        current_index += num_rows

        if col == n_cols - 1:
            current_row += 1
            current_index = current_row

    return reordered_handles, reordered_labels
