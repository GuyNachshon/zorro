# Make file to process all python scripts in this directory working directory
# this make file will run the following command
# python make_bar_plot.py grouped_data_settings.yml --export_highcharts --write_to_file'

# Define all input variables here
script_directory = .
script_name = make_bar_plot.py
data = grouped_data.csv
settings = grouped_data_settings.yml

image_directory = output
highcharts_directory = highcharts

# Here the output files
pdf_file_naam = grouped_data.pdf
json_file_name = Fig_1_0/grouped_data.json

# From here on no more user input is required
targets_pdf = $(image_directory)/$(pdf_file_naam)
targets_json = $(highcharts_directory)/$(json_file_name)

targets = $(targets_pdf) $(targets_json)

script = $(script_directory)/$(script_name)

# define all  dependencies
# als 1 van deze files verandert, wordt alles opnieuw gegeneerd
depends = $(script) $(data) $(settings)

default_options = --verbose

# geef mogelijkheid om opties via command line mee te geven via OPTIONS="--option1 --option2"
ifdef OPTIONS
options = $(OPTIONS)
else
options = $(default_options)
endif

# geef mogelijkheid om extra opties via command line mee te geven via EXTRA_OPTIONS="--option1 --option2"
ifdef EXTRA_OPTIONS
options += $(EXTRA_OPTIONS)
endif

# alle targets  en dependencies relaties
all: instroom_pdf instroom_json

default: all

instroom_pdf: $(targets_pdf)

instroom_json: $(targets_json)

# hier wordt het volgende commando uitgevoerd:
# python ../../scripts/make_bar_plot.py grouped_data_settings.yml --export_highcharts --write_to_file'
# maar alleen als 1 van de dependencies verandert, dus de settings file, de data of de python script
$(targets_pdf): $(depends)
	@python  $(script) $(settings) $(options)  --write_to_file

$(targets_json): $(depends)
	@python  $(script) $(settings) $(options)  --export_highcharts
