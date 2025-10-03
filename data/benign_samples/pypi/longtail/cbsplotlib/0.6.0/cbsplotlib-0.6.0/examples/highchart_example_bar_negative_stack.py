import logging
from pathlib import Path
import pandas as pd

from cbsplotlib.highcharts import CBSHighChart

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

output_directory = "out"
data_input_directory = Path("../data/highcharts_examples")

# create bar plot (met horizontale balken)
input_file_name = data_input_directory / Path("bar_plot_negative_stack.csv")
data_df = pd.read_csv(input_file_name, index_col=0)
output_file = input_file_name.stem

# lees de data met de default template en schrijf als json
hc = CBSHighChart(data=data_df,
                  chart_type="bar_with_negative_stack",
                  output_directory=output_directory,
                  output_file_name=output_file,
                  xlabel="Percentage",
                  ylabel="YLABEL",
                  title="Vergelijking mannen / vrouwen ",
                  chart_description="Beschrijving van het plaatje",
                  sources_text="mijn enquête",
                  footnote_text="Dit is een footnote",
                  )

logger.info("Done")
