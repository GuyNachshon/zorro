import logging
from pathlib import Path
import pandas as pd

from cbsplotlib.highcharts import CBSHighChart

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# plot of global temperature
# https://datahub.io/core/global-temp#resource-global-temp_zip

output_directory = "out"
data_input_directory = Path("../data")

# create bar plot (met horizontale balken)
input_file_name = data_input_directory / Path("annual.csv")
data_df = pd.read_csv(input_file_name)
df = data_df.groupby("Source")
gist_df = data_df.loc[data_df["Source"] == "GISTEMP", ["Year", "Mean"]].set_index("Year")
gcag_df = data_df.loc[data_df["Source"] == "GCAG", ["Year", "Mean"]].set_index("Year")

temperature_df = pd.concat([gist_df, gcag_df], axis=1, keys=["GIST", "GCAG"])
temperature_df.columns = temperature_df.columns.droplevel(1)
temperature_df.sort_index(ascending=True, inplace=True)

output_file = "global_temperature"

# lees de data met de default template en schrijf als json
hc = CBSHighChart(data=temperature_df,
                  chart_type="line",
                  output_directory=output_directory,
                  output_file_name=output_file,
                  xlabel="Year",
                  ylabel="Mean",
                  x_tick_interval=20,
                  y_format="{:0.1f}",
                  title="Mean global temperature anomaly in degree Celsius",
                  chart_description="Global Temperature Time Series. Data are included from the "
                                    "GISS Surface Temperature (GISTEMP) analysis and the global "
                                    "component of Climate at a Glance (GCAG). Two datasets are "
                                    "provided: 1) global monthly mean and 2) annual mean "
                                    "temperature anomalies in degrees Celsius from 1880 to the "
                                    "present.",
                  sources_text="https://datahub.io",
                  sources_prefix="Bron: "
                  )

logger.info("Done")
