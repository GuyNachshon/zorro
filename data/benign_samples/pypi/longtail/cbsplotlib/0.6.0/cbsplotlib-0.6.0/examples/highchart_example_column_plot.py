import logging
import pandas as pd
from pathlib import Path
from cbsplotlib.highcharts import CBSHighChart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

output_directory = "out"
data_input_directory = Path("../data/highcharts_examples")

# create column plot (met verticale balken)
input_file_name = data_input_directory / Path("cbs_hc_bar.csv")
data_df = pd.read_csv(input_file_name, sep=";", index_col=0, decimal=",")

# we voegen 1 NAN toe om te kijken of deze goed weggelaten worden door highcharts
data_df.loc["Duitsland", "2013"] = pd.NA

series_description = pd.DataFrame(index=data_df.columns,
                                  data=[f"Getallen voor {_}" for _ in data_df.columns.values])

hc = CBSHighChart(data=data_df,
                  chart_type="column",
                  output_directory=output_directory,
                  output_file_name="cbs_hc_column_plot",
                  series_description=series_description,
                  )
print("Done")
