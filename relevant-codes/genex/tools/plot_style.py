"""
Functions which affect appearance of plots
"""

import matplotlib.pyplot as plt

def set_mpl_rcParams():
    """
    Shared Matplotlib styling parameters througout the diagnostic tools
    """

    plt.rcParams['pcolor.shading'] = 'nearest'
    plt.rcParams['figure.constrained_layout.use'] = True

def get_boxlims(x, y, percent_extra=0.1):
    """
    Get min and max of x and y data, including optional buffer, to e.g. set plot
    data limits to comfortably include all data points
    """
    xspan = x.max() - x.min()
    yspan = y.max() - y.min()
    xmin = x.min() - xspan * percent_extra
    xmax = x.max() + xspan * percent_extra
    ymin = y.min() - yspan * percent_extra
    ymax = y.max() + yspan * percent_extra
    return {'xlim':(xmin, xmax), 'ylim':(ymin, ymax)}
