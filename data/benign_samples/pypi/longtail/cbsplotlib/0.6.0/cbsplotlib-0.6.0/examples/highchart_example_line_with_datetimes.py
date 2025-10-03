import logging
from pathlib import Path
import pandas as pd

from cbsplotlib.highcharts import CBSHighChart

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

output_directory = "out"
data_input_directory = Path("../data/highcharts_examples")

# create bar plot (met horizontale balken)
input_file_name = data_input_directory / Path("line_with_dates.csv")
data_df = pd.read_csv(input_file_name, index_col=0, parse_dates=True)

# pas het datumtijd-formaat aan, wordt nu maand jaar met 2 digit (mei 21)
data_df.index = pd.to_datetime(data_df.index).strftime("%b '%y")
# data_df.index = pd.to_datetime(data_df.index).strftime("%Y%m")
output_file = input_file_name.stem

# lees de data met de default template en schrijf als json
hc = CBSHighChart(data=data_df,
                  chart_type="line",
                  output_directory=output_directory,
                  output_file_name=output_file,
                  xlabel="Datum",
                  ylabel="Aantal",
                  x_labels_format="{value}",
                  x_tick_interval=4,
                  y_format="{:0.1f}",
                  title="Aantal instellingen",
                  chart_description="Aantal lijnen",
                  sources_text="mijn enquête",
                  footnote_text="Dit is een footnote",
                  )

logger.info("Done")
