import pytest
from cbsplotlib.colors import CBS_COLORS_HEX
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes

from cbsplotlib.colors import update_color_palette


@pytest.fixture
def default_color_cycle():
    """
    Fixture to store the default color cycle at the start of a test and restore it at the end.

    Returns
    -------
    cycler
        The default color cycle.
    """
    return plt.rcParams['axes.prop_cycle']


def test_cbs_colors_hex_values():
    """Test that the CBS colors are correctly converted to hex values"""
    assert CBS_COLORS_HEX['corporateblauw'] == '#271D6C'
    assert CBS_COLORS_HEX['corporateblauw'] == '#271D6C'
    assert CBS_COLORS_HEX['corporatelichtblauw'] == '#00A1CD'
    assert CBS_COLORS_HEX['donkerblauw'] == '#0058B8'
    assert CBS_COLORS_HEX['donkerblauwvergrijsd'] == '#163A72'
    assert CBS_COLORS_HEX['lichtblauw'] == '#00A1CD'
    assert CBS_COLORS_HEX['lichtblauwvergrijsd'] == '#0581A2'
    assert CBS_COLORS_HEX['geel'] == '#FFCC00'
    assert CBS_COLORS_HEX['geelvergrijsd'] == '#FFB600'
    assert CBS_COLORS_HEX['oranje'] == '#F39200'
    assert CBS_COLORS_HEX['oranjevergrijsd'] == '#CE7C00'
    assert CBS_COLORS_HEX['rood'] == '#E94C0A'
    assert CBS_COLORS_HEX['roodvergrijsd'] == '#B23D02'
    assert CBS_COLORS_HEX['roze'] == '#AF0E80'
    assert CBS_COLORS_HEX['rozevergrijsd'] == '#82045E'
    assert CBS_COLORS_HEX['grasgroen'] == '#53A31D'
    assert CBS_COLORS_HEX['grasgroenvergrijsd'] == '#488225'
    assert CBS_COLORS_HEX['appelgroen'] == '#AFCB05'
    assert CBS_COLORS_HEX['appelgroenvergrijsd'] == '#899D0C'
    assert CBS_COLORS_HEX['violet'] == '#AC218E'
    assert CBS_COLORS_HEX['lichtgrijs'] == '#E0E0E0'
    assert CBS_COLORS_HEX['grijs'] == '#666666'
    assert CBS_COLORS_HEX['logogrijs'] == '#929292'
    assert CBS_COLORS_HEX['codekleur'] == '#585858'


def test_update_color_palette_reverse_flag(default_color_cycle):
    """
    Test that the color palette can be reversed.

    Parameters
    ----------
    default_color_cycle: cycler
        The default color cycle fixture.

    Notes
    -----
    This test sets a color palette with three colors, reverses the color palette,
    and then checks whether the colors are reversed.
    """
    original_colors = ['red', 'green', 'blue']
    expected_colors = ['blue', 'green', 'red']

    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=original_colors)

    update_color_palette(reverse=True)

    assert plt.rcParams['axes.prop_cycle'].by_key()['color'] == expected_colors

    # Restore default color cycle
    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=original_colors)

    update_color_palette(reverse=True)

    assert plt.rcParams['axes.prop_cycle'].by_key()['color'] == expected_colors

    # Restore default color cycle
    plt.rcParams['axes.prop_cycle'] = default_color_cycle
    original_colors = ['red', 'green', 'blue']
    expected_colors = ['blue', 'green', 'red']

    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=original_colors)

    update_color_palette(reverse=True)

    assert plt.rcParams['axes.prop_cycle'].by_key()['color'] == expected_colors

    # Restore default color cycle
    plt.rcParams['axes.prop_cycle'] = default_color_cycle


def test_update_color_palette_offset_parameter(default_color_cycle):
    """
    Test `update_color_palette` with offset parameter.

    This test sets a color palette with three colors, offsets the color palette by 1,
    and then checks whether the colors are offset.

    """
    original_colors = ['red', 'green', 'blue']
    expected_colors = ['green', 'blue', 'red']

    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=original_colors)

    update_color_palette(offset=1)

    assert plt.rcParams['axes.prop_cycle'].by_key()['color'] == expected_colors

    # Restore default color cycle
    plt.rcParams['axes.prop_cycle'] = default_color_cycle
    original_colors = ['red', 'green', 'blue']
    expected_colors = ['green', 'blue', 'red']

    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=original_colors)

    update_color_palette(offset=1)

    assert plt.rcParams['axes.prop_cycle'].by_key()['color'] == expected_colors

    # Restore default color cycle
    plt.rcParams['axes.prop_cycle'] = default_color_cycle


def test_update_color_palette_combined_flags(default_color_cycle):
    """
    Test that the color palette can be reversed and offset simultaneously.

    Parameters
    ----------
    default_color_cycle: cycler
        The default color cycle fixture.

    Notes
    -----
    This test sets a color palette with three colors, reverses the color palette,
    and then checks whether the colors are reversed.
    """
    original_colors = ['red', 'green', 'blue']
    expected_colors = ['green', 'red', 'blue']

    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=original_colors)

    update_color_palette(reverse=True, offset=1)

    assert plt.rcParams['axes.prop_cycle'].by_key()['color'] == expected_colors

    # Restore default color cycle
    plt.rcParams['axes.prop_cycle'] = default_color_cycle
    original_colors = ['red', 'green', 'blue']
    expected_colors = ['green', 'red', 'blue']

    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=original_colors)

    update_color_palette(reverse=True, offset=1)

    assert plt.rcParams['axes.prop_cycle'].by_key()['color'] == expected_colors

    # Restore default color cycle
    plt.rcParams['axes.prop_cycle'] = default_color_cycle
