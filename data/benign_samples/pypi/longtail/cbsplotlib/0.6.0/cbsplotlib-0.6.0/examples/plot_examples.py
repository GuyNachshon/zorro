import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cbsplotlib.settings import CBSPlotSettings
from cbsplotlib.utils import add_axis_label_background

csv_file = Path("../data/iris.csv")

logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)8s] %(message)s"
                    )
_logger = logging.getLogger(__name__)
figure_properties = CBSPlotSettings()


def make_bar_plot(data_df, orientation="horizontal"):
    """
    Make the bar plot
    
    Parameters
    ----------
    data_df: Dataframe
        pandas dataframe with the data
    orientation: {"horizontal", "vertical"}
        Direction of the bars

    """

    if orientation not in ("horizontal", "vertical"):
        raise ValueError(f"oriental must be 'horizontal' or 'vertical'. Found {orientation}")

    # initieer plot
    fig, axis = plt.subplots(nrows=1, ncols=1)

    if orientation == "horizontal":
        kind = "barh"
    else:
        kind = "bar"

    # plot data. Cbs stijl wil een ruimte tussen de bar van 0.75 pt. Los dit op door een witte rand
    data_df.plot(kind=kind, ax=axis, edgecolor="white", linewidth=0.75, rot=0, zorder=2)

    if orientation == "horizontal":
        # pas marges aan
        fig.subplots_adjust(left=0.2, bottom=0.2)

        xticks = axis.get_xticks()
        # de ccn standaard schrijft voor dat de hoogste tick waarde hoger is dan de hoogste waarde
        # in de grafiek. Lost dat op door de hoogste  xlimit gelijk aan de hoogste xtick te zetten
        # tel die 0.01 op omdat anders de bovenste gridlijn niet goed zichtbaar is
        axis.set_xlim((xticks[0], xticks[-1] + 0.01))

        # stel de grid lijnen in en haal de buiten randen behalve de bodem en links weg
        # zorder is nodig om te zorgen dat de gridlijnen niet boven de bars komen, maar onder
        axis.xaxis.grid(True, zorder=0)
        axis.tick_params(which="both", bottom=False, left=False)
        # xlabel aan de rechter zijde
        axis.set_xlabel("Gemiddelde afmeting [mm]", horizontalalignment="right")
        axis.xaxis.set_label_coords(1.0, -0.1)
        # ylabel boven aan horizontaal geplot
        axis.set_ylabel(data_df.index.name, rotation="horizontal", horizontalalignment="left")
        axis.yaxis.set_label_coords(0, 1.05)
        # de kleur volgorde is per default anders om dan de dataframe volgoeder. Zet hier weer goed
        axis.invert_yaxis()

        # haal de x-as weg maar zet het verticale gris
        sns.despine(ax=axis, bottom=True)
        # haal de tick marks weg
        # pas de y-as dikte en kleur aan
        axis.spines["left"].set_linewidth(1.5)
        axis.spines["left"].set_color("cbs:grijs")

        add_axis_label_background(fig, axes=axis, )
    else:
        fig.subplots_adjust(bottom=0.3)

        yticks = axis.get_yticks()
        axis.set_ylim((yticks[0], yticks[-1]))
        axis.yaxis.grid(True)
        axis.tick_params(which="both", right=False, left=False)

        # ylabel aan de boven
        axis.set_ylabel("Gemiddelde afmeting [mm]", rotation="horizontal",
                        horizontalalignment="left")
        axis.yaxis.set_label_coords(-0.05, 1.05)

        # xlabel boven aan horizontaal geplot
        axis.set_xlabel(data_df.index.name, rotation="horizontal", horizontalalignment="right")
        axis.xaxis.set_label_coords(0.95, -0.15)

        labels = [l.get_text() for l in axis.get_xticklabels()]
        axis.set_xticklabels(labels, ha='center')

        # haal de x-as weg maar zet het verticale gris
        sns.despine(ax=axis, left=True)
        # haal de tick marks weg
        # pas de y-as dikte en kleur aan
        axis.spines["bottom"].set_linewidth(1.5)
        axis.spines["bottom"].set_color("cbs:grijs")

        add_axis_label_background(fig, axes=axis, loc="south")

    # de legend aan de onderkant
    legend = axis.legend(loc="lower left",
                         bbox_to_anchor=(0, 0),
                         ncol=4,
                         bbox_transform=fig.transFigure,
                         frameon=False,
                         title="Afmeting bloemdeel")
    legend._legend_box.align = "left"

    im_name = "plot_example_" + orientation
    # fig.savefig(im_name + ".png")
    fig.savefig(im_name + ".pdf")


def main():
    # laad de dataset
    try:
        iris = pd.read_csv(csv_file)
        _logger.info(f"Read iris data from local cache {csv_file}")
    except FileNotFoundError:
        _logger.info(f"Obtaining iris data from seaborn site")
        iris = sns.load_dataset('iris')
        iris.to_csv(csv_file, index=None)

    _logger.info(f"\n{iris.head()}")

    # hernoem de kolommen
    iris.rename(columns={
        "sepal_length": "Stempellengte",
        "sepal_width": "Stempelbreedte",
        "petal_length": "Bladlengte",
        "petal_width": "Bladbreedte"
    }, inplace=True)

    #  bereken gemiddelde waardes
    geometry_df = iris.groupby("species").mean()
    geometry_df.index.name = "Soort bloem"

    _logger.info(f"\n{geometry_df}")

    make_bar_plot(data_df=geometry_df, orientation="horizontal")

    make_bar_plot(data_df=geometry_df, orientation="vertical")

    plt.show()


if __name__ == "__main__":
    main()
