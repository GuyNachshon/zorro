import logging
import pandas as pd
from pathlib import Path
from cbsplotlib.highcharts import CBSHighChart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

output_directory = "out"
data_input_directory = Path("../data/highcharts_examples")

# create column plot (met verticale balken)
input_file_name = data_input_directory / Path("cbs_hc_column_2ndcol.csv")
data_df = pd.read_csv(input_file_name, sep=";", index_col=[0, 1], decimal=",")

hc = CBSHighChart(data=data_df,
                  chart_type="column_grouped",
                  output_directory=output_directory,
                  output_file_name="cbs_hc_column_grouped_plot")
print("Done")
