from cbsplotlib.colors import CBS_COLORS_HEX

__author__ = "Eelco van Vliet"
__copyright__ = "Eelco van Vliet"
__license__ = "MIT"

import pytest

from cbsplotlib.settings import CBSPlotSettings


def test_cbs_settings_default():
    """API Tests"""

    settings = CBSPlotSettings()

    assert settings.fig_width == pytest.approx(5.4330722291407225)
    assert settings.fig_height == pytest.approx(3.357823300942124)


def test_cbs_settings_square():
    """API Tests"""

    settings = CBSPlotSettings(ratio_option="equal")

    assert settings.fig_width == pytest.approx(5.4330722291407225)
    assert settings.fig_height == pytest.approx(settings.fig_width)


def test_cbs_settings_from_rows():
    """API Tests"""

    # call with default rows is 2
    settings = CBSPlotSettings(ratio_option="from_rows")

    assert settings.fig_width == pytest.approx(5.4330722291407225)
    assert settings.fig_height == pytest.approx(4.294520547945205)

    settings = CBSPlotSettings(ratio_option="from_rows", number_of_figures_rows=3)

    assert settings.fig_width == pytest.approx(5.4330722291407225)
    assert settings.fig_height == pytest.approx(2.863013698630137)

    settings = CBSPlotSettings(ratio_option="from_rows", number_of_figures_rows=3, number_of_figures_cols=2)

    assert settings.fig_width == pytest.approx(2.7165361145703613)
    assert settings.fig_height == pytest.approx(2.863013698630137)
