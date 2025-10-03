"""
Script to make a bar plot based on a settings file

Usage:  python make_bar_plot.py settings_file [--write_to_file] [--show_plot] [--export_highcharts] [--debug]

"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import pandas as pd
import seaborn as sns
import yaml
from cbsplotlib import (CBSHighChart,
                        CBSPlotSettings,
                        add_axis_label_background,
                        format_thousands_label,
                        swap_legend_boxes)

sns.set_style("whitegrid", {"axis.grid": False})

logging.basicConfig(
    format="[%(levelname)8s %(lineno)4d] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - settings_file (str): Path to the settings file.
            - loglevel (int): Logging level, either INFO or DEBUG.
    """
    parser = argparse.ArgumentParser(
        "Create a bar plot based on a settings file"
    )
    parser.add_argument("settings_file", type=str, help="Settings file")
    parser.add_argument(
        "--verbose",
        "-v",
        dest="loglevel",
        const=logging.INFO,
        default=logging.INFO,
        action="store_const",
        help="Enable extra output",
    )
    parser.add_argument(
        "--debug",
        "-d",
        dest="loglevel",
        const=logging.DEBUG,
        action="store_const",
        help="Enable extra output",
    )
    parser.add_argument(
        "--write_to_file",
        action="store_true",
        help="Schrijf het pdf plaatje naar een bestand",
    )
    parser.add_argument(
        "--show_plot", action="store_true", help="Laat het plaatje zien"
    )
    parser.add_argument(
        "--export_highcharts",
        action="store_true",
        help="Schrijf de highcharts json naar file",
    )
    args = parser.parse_args()

    return args


# noinspection PyIncorrectDocstring
def make_stacked_vertical_bar_plot(
        data: pd.DataFrame,
        image_name: Path,
        write_to_file: bool = False,
        show_plot: bool = False,
        **kwargs
) -> None:
    """
    Make a stacked vertical bar plot.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.
    image_name : pathlib.Path
        Name of the image to write to disk.
    write_to_file : bool, optional
        Whether to write the image to disk, by default False.
    show_plot : bool, optional
        Whether to show the image, by default False.
    **kwargs : dict
        Additional keyword arguments. See below for a list of valid arguments.

    Other Parameters
    ----------------
    left : float, optional
        Left position of the plot area, by default 0.25.
    top : float, optional
        Top position of the plot area, by default 0.95.
    bottom : float, optional
        Bottom position of the plot area, by default 0.2.
    x_label : str, optional
        X-axis label, by default None.
    y_max : float, optional
        Maximum y-value, by default None.
    x_label_position : list, optional
        Position of the x-axis label, by default None.
    radius_corner_background : float, optional
        Radius of the corner of the background, by default 1.
    background_margin : float, optional
        Margin of the background, by default -0.01.
    logo_margin_x_in_mm : float, optional
        Margin of the logo on the x-axis, by default 1.
    logo_margin_y_in_mm : float, optional
        Margin of the logo on the y-axis, by default 1.
    bar_width : float, optional
        Width of the bars, by default 0.8.
    color_palette : str, optional
        Color palette to use, by default "koelextended".
    fig_height_in_inch : float, optional
        Height of the figure in inches, by default None.
    fig_width_in_inch : float, optional
        Width of the figure in inches, by default None.
    legend_pos : tuple, optional
        Position of the legend, by default (0, -0.03).
    n_cols : int, optional
        Number of columns in the legend, by default 3.

    Returns
    -------
    None
    """
    left: float = kwargs.get("left", 0.25)
    top: float = kwargs.get("top", 0.95)
    bottom: float = kwargs.get("bottom", 0.2)
    x_label: str | None = kwargs.get("x_label")
    y_max: float | None = kwargs.get("y_max")
    x_label_position: list | None = kwargs.get("x_label_position")
    radius_corner_background: float = kwargs.get("radius_corner_background", 1)
    background_margin: float = kwargs.get("background_margin", -0.01)
    logo_margin_x_in_mm: float = kwargs.get("logo_margin_x_in_mm", 1)
    logo_margin_y_in_mm: float = kwargs.get("logo_margin_y_in_mm", 1)
    bar_width: float = kwargs.get("bar_width", 0.8)
    color_palette: str = kwargs.get("color_palette", "koelextended")
    fig_height_in_inch: float | None = kwargs.get("fig_height_in_inch")
    fig_width_in_inch: float | None = kwargs.get("fig_width_in_inch")
    legend_pos: tuple = kwargs.get("legend_pos", (0, -0.03))
    n_cols: int = kwargs.get("n_cols", 3)

    CBSPlotSettings(
        color_palette=color_palette,
        fig_height_in_inch=fig_height_in_inch,
        fig_width_in_inch=fig_width_in_inch,
    )

    fig, axis = plt.subplots()

    fig.subplots_adjust(left=left, top=top, bottom=bottom)
    fig.canvas.draw()

    data.plot(kind="barh", ax=axis, stacked=True, width=bar_width)

    axis.invert_yaxis()
    if y_max is not None:
        axis.set_xlim(0, y_max)
    axis.set_ylabel(None)
    axis.yaxis.grid(False)
    axis.xaxis.grid(True, zorder=0)
    if x_label is not None:
        axis.set_xlabel(x_label, horizontalalignment="right")
        axis.xaxis.set_label_coords(x_label_position)
    sns.despine(ax=axis, bottom=True)
    axis.spines["left"].set_linewidth(1.5)
    axis.spines["left"].set_color("cbs:grijs")

    x_format = tkr.FuncFormatter(format_thousands_label)

    axis.xaxis.set_major_formatter(x_format)

    handles, labels = axis.get_legend_handles_labels()
    handles, labels = swap_legend_boxes(handles, labels, n_cols=n_cols)

    axis.legend(
        handles=handles,
        labels=labels,
        loc="lower left",
        bbox_to_anchor=legend_pos,
        frameon=False,
        bbox_transform=fig.transFigure,
        ncols=n_cols,
    )

    add_axis_label_background(
        fig=fig,
        axes=axis,
        margin=background_margin,
        radius_corner_in_mm=radius_corner_background,
        logo_margin_x_in_mm=logo_margin_x_in_mm,
        logo_margin_y_in_mm=logo_margin_y_in_mm,
    )

    if show_plot:
        plt.show()

    if write_to_file:
        logger.info(f"Writing image to {image_name}")
        fig.savefig(image_name.as_posix())


def main():
    """
    Main entry point for make_bar_plot.

    Reads settings from a YAML file (supplied as an argument on the command line).
    The settings file should contain a dictionary with at least the following keys:

    - general
    - plot_settings

    The general dictionary should contain the following keys:

    - output_directory
    - image_type
    - highcharts_output_directory

    The plot_settings dictionary should contain dictionaries for each plot.
    Each of these dictionaries should have the following keys:

    - input_file
    - x_label
    - y_label
    - title
    - show_legend
    - legend_position
    - highcharts
    - highcharts_title
    - highcharts_bron
    - highcharts_height

    The highcharts dictionary should contain the following keys:

    - output_directory
    - title
    - source_text
    - height

    If the highcharts dictionary is not present for a plot, the plot will not
    be rendered to highcharts format.

    The function reads the settings, reads the input csv file, and
    makes a stacked vertical bar plot using seaborn and matplotlib.

    The function then saves the plot to a file in the output directory.
    If the --export_highcharts argument is passed, the function will also
    save the plot to a highcharts file in the highcharts output directory.

    """
    args = parse_args()

    if args.loglevel is not None:
        logger.setLevel(args.loglevel)

    if not args.show_plot and not args.write_to_file and not args.export_highcharts:
        raise ValueError(
            "At least one of show_plot or write_to_file must be True. \n"
            "Pass option --write_to_file,  --show_plot and/or --export_highcharts"
        )

    logger.info(f"Reading settings file {args.settings_file}")
    with open(args.settings_file, encoding="utf-8") as stream:
        settings = yaml.load(stream, Loader=yaml.FullLoader)

    general_settings = settings["general"]
    plot_settings = settings["plot_settings"]

    image_type = general_settings.get("image_type", "pdf")

    output_directory = Path(general_settings.get("output_directory", "output"))
    output_directory.mkdir(exist_ok=True)
    global_highcharts_output_directory = Path(
        general_settings.get("highcharts_output_directory")
    )
    global_highcharts_output_directory.mkdir(exist_ok=True)

    for plot_key, plot_properties in plot_settings.items():

        input_file = Path(plot_properties["input_file"])

        image_name_base = Path(".".join([plot_key, image_type]))
        image_name = output_directory / image_name_base

        logger.info(f"Reading csv data file {input_file}")
        data = pd.read_csv(input_file, index_col=0)

        if args.show_plot or args.write_to_file:
            make_stacked_vertical_bar_plot(
                data,
                image_name=image_name,
                write_to_file=args.write_to_file,
                show_plot=args.show_plot,
                **plot_settings,
            )

        if args.export_highcharts:
            try:
                highcharts_properties = plot_properties["highcharts"]
            except KeyError as err:
                logger.warning(err)
                raise KeyError(
                    f"highcharts properties not found for {plot_key}, please specify highcharts properties"
                )

            hc_out = highcharts_properties.get("output_directory")
            if hc_out is not None:
                highcharts_output_directory = global_highcharts_output_directory / Path(
                    hc_out
                )
            else:
                highcharts_output_directory = global_highcharts_output_directory

            highcharts_bron = highcharts_properties.get("source_text")
            highcharts_height = highcharts_properties.get("height")
            highcharts_title = highcharts_properties.get("title")
            x_label = highcharts_properties.get("x_label", plot_settings.get("x_label"))

            hc_name = image_name.stem
            logger.debug(f"Saving to highcharts data {hc_name}")
            # highcharts wilt juist een andere volgorde van de bars
            data_for_highcharts = data.reindex(data.index[::-1])
            CBSHighChart(
                data=data_for_highcharts,
                chart_type="bar",
                output_directory=highcharts_output_directory.as_posix(),
                title=highcharts_title,
                chart_height=highcharts_height,
                ylabel=x_label,
                sources_text=highcharts_bron,
                output_file_name=hc_name,
            )


if __name__ == "__main__":
    main()
