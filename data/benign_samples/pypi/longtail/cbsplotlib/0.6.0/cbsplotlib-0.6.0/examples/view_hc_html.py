import logging
from pathlib import Path
from cbsplotlib.htmlviewer import HtmlViewer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
hc_log = logging.getLogger("cbsplotlib")
hc_log.setLevel(logging.DEBUG)

output_directory = "out"
data_input_directory = Path("../data/htmlviewer_testdata")

# create bar plot (met horizontale balken)
html_filename = data_input_directory / Path("line_with_dates.html")

html_view = HtmlViewer(filename=html_filename,
                       output_directory=output_directory)

html_view.show()

logger.info("Done")
