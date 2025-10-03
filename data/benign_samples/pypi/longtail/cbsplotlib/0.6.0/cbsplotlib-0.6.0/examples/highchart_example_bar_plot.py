import logging
from pathlib import Path

from cbsplotlib.highcharts import CBSHighChart

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

output_directory = "out"
data_input_directory = Path("../data/highcharts_examples")

# create bar plot (met horizontale balken)
input_file_name = data_input_directory / Path("cbs_hc_bar.csv")
defaults_out_file = input_file_name.stem + ".json"
# eerste keer runnen we alleen om de defaults template uit de package directory naar onze eigen
# script folder te schrijven
if not Path(defaults_out_file).exists():
    hc = CBSHighChart(input_file_name=input_file_name.as_posix(),
                      chart_type="bar",
                      output_directory=output_directory,
                      output_file_name=defaults_out_file,
                      defaults_out_file=input_file_name.stem)

# pas nu de template aan en run nogmaals zonder defaults_out_file om de highcharts met onze data te
# maken als je niks aan de template aanpast kan je ook direct de output highcharts maken.
hc = CBSHighChart(input_file_name=input_file_name.as_posix(),
                  defaults_file_name=defaults_out_file,
                  chart_type="bar",
                  output_directory=output_directory,
                  output_file_name="cbs_hc_bar_plot",
                  start=True,
                  xlabel="Land",
                  ylabel="YLABEL",
                  title="Dit is mijn mooie titel",
                  y_lim=(0, 10),
                  y_tick_interval=2,
                  chart_description="Beschrijving van het plaatje",
                  chart_height=430,
                  color_selection="Warm",
                  sources_text="mijn enquete",
                  footnote_text="Dit is een footnote",
                  tooltip_prefix="voor ",
                  tooltip_suffix="na ",
                  )

logger.info("Done")
