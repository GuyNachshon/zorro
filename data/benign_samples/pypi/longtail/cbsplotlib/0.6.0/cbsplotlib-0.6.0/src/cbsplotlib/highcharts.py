import codecs
import json
import logging
import sys
from pathlib import Path
from typing import Union

import pandas as pd

HC_DEFAULTS_DIRECTORY = 'cbs_hc_defaults'

_logger = logging.getLogger(__name__)

PLOT_TYPES = {
    'line',
    'area',
    'column',
    'bar',
    'pie',
    'polar',
    'choropleth',
    'bubbleChart',
}
PLOT_TEMPLATES = {
    'area_percentage_grouped',
    'area_stacked_grouped',
    'bar',
    'bar_percentage',
    'bar_stacked',
    'bar_stacked_percentage',
    'bar_with_negative_stack',
    'bubblechart_tourist',
    'choropleth_youthcare',
    'column_grouped',
    'column_grouped_stacked',
    'column',
    'column_percentage',
    'column_stacked_percentage',
    'line_column_combination_grouped',
    'line',
    'pie_donut',
    'polar_spider_line',
    'spline_grouped',
}

PALETTES = {
    'Warm': [
        '#e94c0a',
        '#ffcc00',
        '#af0e80',
        '#f39200',
        '#53a31d',
        '#afcb05',
        '#0058b8',
        '#00a1cd',
        '#b23d02',
        '#e1b600',
        '#82045e',
        '#ce7c00',
        '#488225',
        '#899d0c',
        '#163a72',
        '#0581a2',
    ],
    'Cold': [
        '#00a1cd',
        '#0058b8',
        '#afcb05',
        '#53a31d',
        '#f39200',
        '#af0e80',
        '#ffcc00',
        '#e94c0a',
        '#0581a2',
        '#163a72',
        '#899d0c',
        '#488225',
        '#ce7c00',
        '#82045e',
        '#e1b600',
        '#b23d02',
    ],
}


def write_to_json_file(
    output,
    json_indent=2,
    output_directory: Path = None,
    output_file_name: str = None,
    input_file_name: str = None,
    chart_type=None,
):
    """
    Write the dictionary to a json output file

    Parameters
    ----------
    output: dict
        The settings dictionary to write
    json_indent: int
        Number of indenting spaces. If None is give, a compact json is written
    output_directory: Path of None
        Output directory
    output_file_name: str of None
        Naam van de output file
    input_file_name: str of None
        Naam van de input data file. Als we geen output filenaam geven zal de output file hierop
        gebaseerd worden.
    chart_type: str
        Type plot (bar, column, line). Als gegeven wordt dit in de filenaam verwerkt.

    """
    if output_directory is not None:
        output_directory.mkdir(exist_ok=True, parents=True)
    if output_file_name is None:
        if input_file_name is None:
            if chart_type is None:
                chart_label = ''
            else:
                chart_label = chart_type
            default_file_name = Path(
                '_'.join(['highchart', chart_label, 'plot']) + '.json'
            )
        else:
            file_stem = Path(input_file_name).stem
            default_file_name = Path(file_stem).with_suffix('.json')

        outfile = output_directory / default_file_name
    else:
        if output_directory is None:
            outfile = Path(output_file_name).with_suffix('.json')
        else:
            outfile = output_directory / Path(output_file_name).with_suffix('.json')
    _logger.info(f"Writing to {outfile}")
    chart_container = json.dumps(output, indent=json_indent, ensure_ascii=False)
    with codecs.open(outfile.as_posix(), 'w', encoding='utf-8') as stream:
        stream.write(chart_container)

    return outfile


class CBSHighChart:
    def __init__(
        self,
        data: pd.DataFrame = None,
        input_file_name: str = None,
        output_file_name: str = None,
        output_directory: str = None,
        defaults_directory: str = None,
        defaults_file_name: str = None,
        defaults_out_file: str = None,
        chart_type: str = None,
        csv_separator: str = ';',
        decimal: str = ',',
        index_col: int = 0,
        x_format: str = None,
        y_format: str = None,
        x_labels_format: str = None,
        y_labels_format: str = None,
        start: bool = True,
        title: str = None,
        subtitle: str = None,
        xlabel: str = None,
        ylabel: str = None,
        x_lim: tuple = None,
        y_lim: tuple = None,
        x_tick_interval: Union[int, str] = None,
        y_tick_interval: Union[int, str] = None,
        chart_description: str = None,
        chart_height: int = None,
        color_selection: str = None,
        sources_text: str = None,
        sources_prefix: str = None,
        footnote_text: str = None,
        series_description: pd.DataFrame = None,
        tooltip_prefix: str = None,
        tooltip_suffix: str = None,
        has_grouped_categories: bool = None,
        enable_legend: bool = None,
        keep_tick_interval_format: bool = False,
    ):
        self.input_file_name = input_file_name
        self.csv_separator = csv_separator
        self.decimal = decimal
        self.index_col = index_col
        self.enable_legend = enable_legend

        # plot settings
        self.title = title
        self.subtitle = subtitle
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.x_tick_interval = x_tick_interval
        self.y_tick_interval = y_tick_interval
        self.keep_tick_interval_format = keep_tick_interval_format
        self.chart_description = chart_description
        self.chart_height = chart_height
        self.color_selection = color_selection
        self.sources_text = sources_text
        self.sources_prefix = sources_prefix
        self.footnote_text = footnote_text
        self.series_description = series_description
        self.tooltip_prefix = tooltip_prefix
        self.tooltip_suffix = tooltip_suffix
        self.x_format = x_format
        self.y_format = y_format
        self.x_labels_format = x_labels_format
        self.y_labels_format = y_labels_format
        self.has_grouped_categories = has_grouped_categories

        if chart_type is None:
            # defaults chart type is a bar plot
            self.chart_type = 'bar'
        else:
            self.chart_type = chart_type

        if output_directory is None:
            self.output_directory = Path('.')
        else:
            self.output_directory = Path(output_directory)

        self.output_file_name = output_file_name

        self.modification_file_name = defaults_out_file

        # initieer de output dictionary
        self.output = {'template': {}, 'options': {}, 'selectedTemplate': {}}

        # hier worden alle defaults in de self.defaults attribute geladen
        self.defaults = self.read_the_defaults(
            chart_type=chart_type,
            defaults_directory=defaults_directory,
            defaults_file_name=defaults_file_name,
        )

        if defaults_out_file is not None:
            # als we een default output file geven dan schrijven we alleen de huidige template naar
            # deze file.
            write_to_json_file(output=self.defaults, output_file_name=defaults_out_file)
            _logger.info(
                """
                We hebben de defaults template geschreven om het *default_out_file* als argument gegeven
                was. Als je het plaatje met deze template wil maken, geef dan *defaults_out_file* mee
                met het *defaults_file_name* argument. Script stop nu hier.
                    """
            )
            return

        # get the data here or take from the argument
        if data is None:
            _logger.info(f"Reading data from {input_file_name}")
            self.data_df = self.get_data(
                input_file_name,
                index_col=index_col,
                csv_separator=csv_separator,
                decimal=decimal,
            )
        else:
            _logger.debug(f"Using dataframe")
            self.data_df = data

        if self.chart_type == 'bar_with_negative_stack':
            self.negative_stack = True
        else:
            self.negative_stack = False

        # get the categories from the data
        self.categories = self.get_categories()

        _logger.debug(f"categories: {self.categories}")

        if start:
            # als de start optie meegeven is dan gaan we de highcharts file bouwen.
            self.make_highchart()

            self.modify_highchart()

            # Finally, write the result to file
            out_file = write_to_json_file(
                output=self.output,
                output_directory=self.output_directory,
                output_file_name=self.output_file_name,
                input_file_name=self.input_file_name,
                chart_type=self.chart_type,
            )
            _logger.info(f"Successfully wrote to {out_file}")
        else:
            _logger.info(
                'The data was read successfully. To create the highcharts, call the '
                '*make_highcharts()* method or pass the start=True argument'
            )

    def make_highchart(self):
        # now add all the items of the highcarts
        self.add_chart()
        self.add_plot_options()
        self.check_and_modify_data()

        self.add_title(text_key='title')
        self.add_title(text_key='subtitle')

        self.add_axis(
            axis_key='xAxis',
            categories=self.categories,
            labels_format=self.x_labels_format,
        )
        self.add_axis(axis_key='yAxis', labels_format=self.y_labels_format)
        self.add_legend()
        self.add_tooltip()
        self.add_credits()
        self.add_bar_type()

        self.add_options()
        self.add_csv_data()
        self.add_series()
        self.add_axis(
            key='options',
            axis_key='xAxis',
            categories=self.categories,
            labels_format=self.x_labels_format,
        )
        self.add_axis(
            key='options', axis_key='yAxis', labels_format=self.y_labels_format
        )

        self.add_selected_templated()

    def check_and_modify_data(self):
        """
        Voor negative bar type moeten we 2 kolommen hebben waarvan we de eerste hier negatief maken
        """
        if self.negative_stack:
            if len(self.data_df.columns) < 2:
                raise ValueError(
                    "Need at least 2 columns for chart type 'bar_with_negative_stack'"
                )

            # voor negative stack moet je de eerste kolom negatief make
            first_column = self.data_df.columns[0]
            _logger.debug(f"Negating column {first_column}")
            self.data_df[first_column] *= -1

            # ook moet de index een string zijn
            try:
                self.data_df.index = self.data_df.index.astype(str)
            except ValueError:
                _logger.debug(
                    'Kan niet naar string converteren. Hopelijk is dit het al'
                )

    def impose_value(self, value, key_1, key_2=None, key_3=None, output=None):
        if output is None:
            output = self.output
        if key_3 is not None:
            assert key_2 is not None
            try:
                output[key_1][key_2][key_3] = value
            except KeyError:
                # Voor de laatste level kan het zijn dat de entry nog niet bestaat. Maak het gewoon
                _logger.debug(f"Adding new key [{key_2}][{key_3}]")
                output[key_1][key_2] = {}
                output[key_1][key_2][key_3] = value
        elif key_2 is not None:
            assert key_1 is not None
            try:
                output[key_1][key_2] = value
            except KeyError:
                _logger.debug(f"Adding new key [{key_2}]")
                output[key_1] = {}
                output[key_1][key_2] = value
        elif key_1 is not None:
            try:
                output[key_1] = value
            except KeyError:
                _logger.warning(f"Failed imposing value '{value}' to [{key_1}]")
        else:
            raise ValueError('at least one key should be given')

        return output

    def impose_axis_prop(
        self,
        output,
        section='template',
        axis_key='xAxis',
        label=None,
        tick_interval=None,
        lim=None,
        keep_tick_interval_format=False,
    ):
        new_axis = list()
        for axis in self.output[section][axis_key]:
            if label is not None:
                if axis_key == 'xAxis':
                    _logger.debug(
                        f"Imposing {label} to [{section}][{axis_key}][title][text]"
                    )
                    axis = self.impose_value(label, 'title', 'text', output=axis)
                else:
                    _logger.debug(
                        f"Imposing {label} to [{section}][{axis_key}][cbsTitle]"
                    )
                    axis = self.impose_value(label, 'cbsTitle', output=axis)
            if tick_interval is not None:
                _logger.debug(
                    f"Imposing {tick_interval} to [{section}][{axis_key}][tickInterval]"
                )
                if axis_key == 'xAxis' or keep_tick_interval_format:
                    tick_interval_on_axis = tick_interval
                else:
                    tick_interval_on_axis = str(tick_interval)
                # Let op: tick interval moet je soms als string wegschrijven en soms niet
                # Regel dit daarom als gebruiker
                # volgens mijn is xAxis een int, yAxis een float. Introduceer anders vlag 'keep_tick_interval_format
                # als je het als gebruiker zelf wil bepalen
                axis = self.impose_value(
                    tick_interval_on_axis, 'tickInterval', output=axis
                )
            if lim is not None:
                if lim[0] is not None:
                    _logger.debug(f"Imposing {lim[0]} to [{section}][{axis_key}][min]")
                    axis = self.impose_value(str(lim[0]), 'min', output=axis)
                if lim[1] is not None:
                    _logger.debug(f"Imposing {lim[1]} to [{section}][{axis_key}][max]")
                    axis = self.impose_value(str(lim[1]), 'max', output=axis)
            new_axis.append(axis)
        output[section][axis_key] = new_axis

        return output

    def modify_highchart(self):
        """impose the use settings to the highchart"""
        # plot settings
        if self.title is not None:
            _logger.debug(f"Imposing {self.title} to [options][title][text]")
            self.output = self.impose_value(self.title, 'options', 'title', 'text')
        if self.subtitle is not None:
            _logger.debug(f"Imposing {self.subtitle} to [options][title][text]")
            self.output = self.impose_value(
                self.subtitle, 'options', 'subtitle', 'text'
            )

        if (
            self.xlabel is not None
            or self.x_tick_interval is not None
            or self.x_lim is not None
        ):
            for section in ('template', 'options'):
                _logger.debug(f"Imposing xlabel/xtick/xlim to [{section}][xAxis][text]")
                self.output = self.impose_axis_prop(
                    output=self.output,
                    section=section,
                    axis_key='xAxis',
                    label=self.xlabel,
                    tick_interval=self.x_tick_interval,
                    lim=self.x_lim,
                    keep_tick_interval_format=self.keep_tick_interval_format,
                )

        if (
            self.ylabel is not None
            or self.y_tick_interval is not None
            or self.y_lim is not None
        ):
            for section in ('template', 'options'):
                _logger.debug(f"Imposing ylabel/ytick/ylim to [{section}][yAxis][text]")
                self.output = self.impose_axis_prop(
                    output=self.output,
                    section=section,
                    axis_key='yAxis',
                    label=self.ylabel,
                    tick_interval=self.y_tick_interval,
                    lim=self.y_lim,
                    keep_tick_interval_format=self.keep_tick_interval_format,
                )

        if self.chart_description is not None:
            _logger.debug(
                f"Imposing {self.chart_description} to [options][chart][description]"
            )
            self.output = self.impose_value(
                self.chart_description, 'options', 'chart', 'description'
            )
        if self.chart_height is not None:
            _logger.debug(f"Imposing {self.chart_height} to [options][chart][height]")
            self.output = self.impose_value(
                self.chart_height, 'options', 'chart', 'height'
            )
        if self.color_selection is not None:
            pass
        if self.sources_text is not None:
            _logger.debug(f"Imposing {self.sources_text} to [options][sources][text]")
            self.output = self.impose_value(
                self.sources_text, 'options', 'sources', 'text'
            )
        if self.sources_prefix is not None:
            _logger.debug(
                f"Imposing {self.sources_prefix} to [options][sources][prefix]"
            )
            self.output = self.impose_value(
                self.sources_prefix, 'options', 'sources', 'prefix'
            )
        if self.footnote_text is not None:
            _logger.debug(f"Imposing {self.footnote_text} to [options][footNote][text]")
            self.output = self.impose_value(
                self.footnote_text, 'options', 'footNote', 'text'
            )
        if self.tooltip_prefix is not None:
            _logger.debug(
                f"Imposing {self.tooltip_prefix} to [options][tooltip][valuePrefix]"
            )
            self.output = self.impose_value(
                self.tooltip_prefix, 'options', 'tooltip', 'valuePrefix'
            )
        if self.tooltip_suffix is not None:
            _logger.debug(
                f"Imposing {self.tooltip_suffix} to [options][tooltip][valueSuffix]"
            )
            self.output = self.impose_value(
                self.tooltip_suffix, 'options', 'tooltip', 'valueSuffix'
            )

        if self.enable_legend is not None:
            _logger.debug(
                f"Imposing {self.enable_legend} to [options][legend][enabled]"
            )
            self.output = self.impose_value(
                self.enable_legend, 'options', 'legend', 'enabled'
            )

        if self.has_grouped_categories is not None:
            _logger.debug(
                f"Imposing {self.has_grouped_categories}"
                f" to [options][settings][hasGroupedCategories]"
            )
            self.output = self.impose_value(
                self.has_grouped_categories,
                'options',
                'settings',
                'hasGroupedCategories',
            )

        if self.color_selection is not None:
            try:
                colors = PALETTES[self.color_selection]
            except KeyError:
                raise ValueError(
                    f"color_selection argument not valid. Must in {PALETTES.keys()}"
                )

            self.output = self.impose_value(
                self.color_selection, 'options', 'colorSelection'
            )
            self.output = self.impose_value(colors, 'options', 'colors')

        if self.series_description is not None:
            self.series_description.index = self.series_description.index.map(str)
            new_entries = list()
            for entry in self.output['options']['series']:
                name = entry['name']
                description = self.series_description.loc[name].values[0]
                entry['description'] = description
                new_entries.append(entry)
            self.output['options']['series'] = new_entries

    @staticmethod
    def get_data(input_file_name, index_col=0, csv_separator=';', decimal=','):
        if input_file_name is None:
            raise TypeError(
                'Both input data argument *data_df* and input filename '
                '*input_file_name* are None. Please provide at least one.'
            )
        else:
            _logger.debug(f"Reading {input_file_name}")
            data_df = pd.read_csv(
                input_file_name, sep=csv_separator, index_col=index_col, decimal=decimal
            )
        return data_df

    def get_categories(self):
        if self.data_df.index.nlevels == 1:
            categories = ['0' if _ == 0 else _ for _ in self.data_df.index.to_list()]
            if self.has_grouped_categories is None:
                self.has_grouped_categories = False
        elif self.data_df.index.nlevels == 2:
            categories = list()
            if self.has_grouped_categories is None:
                self.has_grouped_categories = True

            for first_level_key, df in self.data_df.groupby(level=0, sort=False):
                categories_strings = [
                    '0' if _ == 0 else _ for _ in df.index.get_level_values(1).to_list()
                ]
                group_categories = {
                    'name': str(first_level_key),
                    'categories': categories_strings,
                }
                categories.append(group_categories)
        else:
            raise TypeError('Multilevel with more than 2 levels not implemented')

        return categories

    @staticmethod
    def read_the_defaults(
        chart_type: str = None,
        defaults_directory: str = None,
        defaults_file_name: str = None,
    ):
        """
        Lees de settings uit een default template json

        Parameters
        ----------
        chart_type: str of None
            Type plot die we maken (bar, column, line). De default template wordt gebaseerd hierop
            gemaakt
        defaults_directory: str of None
            Locatie waar de default template staat. Als niet gegeven en default file naam is niet
            mee gegeven, dan is dit <PACKAGELOCATIE>/Path(cbs_hc_defaults)
        defaults_file_name: str of None
            Naam van de defaults input file. Als niet gegeven wordt deze gebaseerd op *chart_type*
        """

        if defaults_directory is None:
            if defaults_file_name is None:
                # Er is geen default filenaam meegegeven.
                # Ga ervan uit dat we de default uit de package folder halen
                defaults_directory = Path(__file__).parent / Path(HC_DEFAULTS_DIRECTORY)
        else:
            defaults_directory = Path(defaults_directory)

        if defaults_file_name is None:
            # Als de default filename None is, dan is default directory sowieso gezet.
            defaults_file_name = defaults_directory / Path(chart_type + '.json')
        else:
            # Default file naam was door de gebruiker meegegeven. Als ook de directory meegegeven
            # was dan combineren
            # we de naam met de directory, anders nemen we direct de filenaam
            if defaults_directory is None:
                if Path(defaults_file_name).stem not in PLOT_TEMPLATES:
                    _logger.warning(
                        f"default filename {defaults_file_name} not in:\n"
                        f"{PLOT_TEMPLATES}"
                    )
                defaults_file_name = Path(defaults_file_name)
            else:
                defaults_file_name = defaults_directory / Path(defaults_file_name)

        _logger.debug(f"Reading template {defaults_file_name}")
        try:
            with open(defaults_file_name, 'r') as stream:
                defaults = json.load(stream)
        except FileNotFoundError as err:
            try:
                _logger.debug(
                    f"{err}\n"
                    f"Failed reading {defaults_file_name}. Try again with .json suffix"
                )
                defaults_file_name = defaults_file_name.with_suffix('.json')
                with open(defaults_file_name, 'r') as stream:
                    defaults = json.load(stream)
            except FileNotFoundError as err:
                _logger.warning(err)
                template_list = '\n'.join(PLOT_TEMPLATES)
                _logger.warning(
                    f"Je hebt  chart_type={chart_type} geselecteerd maar kan de template "
                    f"{defaults_file_name}\n niet vinden. De volgende templates zijn"
                    f"tot nu geïmplementeerd:\n{template_list}"
                )
                _logger.warning(
                    'Geef een goede template via de chart_type optie of kies een '
                    'custom template via input_file_name. Stop hier'
                )
                sys.exit(-1)

        return defaults

    def add_plot_lines(self, key='template'):
        self.output[key]['plotLines'] = self.defaults[key]['plotLines']

    def add_chart(self, key='template'):
        self.output[key]['chart'] = self.defaults[key]['chart']

    def add_plot_options(self, key='template'):
        self.output[key]['plotOptions'] = self.defaults[key]['plotOptions']

    def add_title(self, key='template', text_key='title'):
        self.output[key][text_key] = self.defaults[key][text_key]

    def add_tooltip(self, key='template'):
        self.output[key]['tooltip'] = self.defaults[key]['tooltip']

    def add_credits(self, key='template'):
        self.output[key]['credits'] = self.defaults[key]['credits']

    def add_bar_type(self, key='template'):
        if self.chart_type == 'bar_with_negative_stack':
            self.output[key]['barType'] = {'negative': True}

    def add_legend(self, key='template'):
        self.output[key]['legend'] = self.defaults[key]['legend']

    def add_axis(
        self, key='template', axis_key='xAxis', categories=None, labels_format=None
    ):
        self.output[key][axis_key] = self.defaults[key][axis_key]
        if labels_format is not None:
            for ax in self.output[key][axis_key]:
                try:
                    labels = ax['labels']
                except KeyError:
                    ax['labels'] = dict()
                    labels = ax['labels']

                labels['format'] = labels_format

        if categories is not None:
            for ax in self.output[key][axis_key]:
                ax['categories'] = []
                for category in categories:
                    if isinstance(category, dict):
                        ax['categories'].append(category)
                    else:
                        ax['categories'].append(str(category))

    def add_options(self):
        self.output['options'] = self.defaults['options']

    def add_csv_data(self, key='options', settings_keys='settings'):
        csv = self.data_df.to_csv(
            sep=self.csv_separator, decimal=self.decimal, float_format='%g'
        )
        self.output[key][settings_keys]['csvData'] = csv.rstrip()

    def add_series(self, key='options', series_key='series'):
        if self.y_format is None:
            self.y_format = '{:g}'

        try:
            # als data_df een Series is maken we er een dataframe van.
            data_df = self.data_df.to_frame()
        except AttributeError:
            # Het was al een dataframe. Geen probleem, ga gewoon door.
            data_df = self.data_df

        series = list()
        for col_name in data_df.columns:
            # Let op dat de name van een serie altijd een string moet zijn.
            item = {
                'name': str(col_name),
                'isSerie': True,
                'borderColor': '#FFFFFF',
                'data': list(),
            }
            for index, row in data_df[[col_name]].iterrows():
                value = row.values[0]
                try:
                    # integer value kan je niet naar json schrijven, dus maak er een float van
                    value = float(value)
                except TypeError:
                    # Als type cast naar float niet lukt, is het een string. Ook goed
                    pass

                if pd.isnull(value):
                    y_string = '.'
                    entry = {
                        'yString': y_string,
                    }
                else:
                    if self.y_format is None:
                        y_string = value
                    else:
                        y_string = self.y_format.format(value)
                    entry = {
                        'y': value,
                        'yString': y_string,
                    }
                if isinstance(index, str):
                    entry['name'] = index
                item['data'].append(entry)

            series.append(item)

        self.output[key][series_key] = series

    def add_selected_templated(self, key='selectedTemplate'):
        self.output[key] = self.defaults[key]
