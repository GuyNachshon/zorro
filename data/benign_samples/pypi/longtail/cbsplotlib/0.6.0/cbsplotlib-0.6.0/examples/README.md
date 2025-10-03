# Examples scripts

#### running the scripts

All the script in this directory can be run using `python <filename>.py`

Only the more advanced scripts 'make_bar_plot.py'  can be launched using the Makefile.
For that, make sure that you have gnu Make installed. Then run 'make' to launch the script.

Alternatively, you can run the script using the following command::

    python  ./make_bar_plot.py grouped_data_settings.yml --verbose  --write_to_file
    python  ./make_bar_plot.py grouped_data_settings.yml --verbose  --export_highcharts


This script demonstrates a more advanced usage of CBSPlotSettings and CBSHighChart. 
It also shows how to use some of the utility functions in cbsplotlib, such as 
*show_axis_label_background*, *swap_legend_boxes* and *format_thousands_label*.