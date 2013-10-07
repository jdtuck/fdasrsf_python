import matplotlib
import pylab
import matplotlib.pyplot as plt


def rstyle(ax):
    """Styles x,y axes to appear like ggplot2
    Must be called after all plot and axis manipulation operations have been
    carried out (needs to know final tick spacing)
    """
    #Set the style of the major and minor grid lines, filled blocks
    ax.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
    ax.grid(True, 'minor', color='0.99', linestyle='-', linewidth=0.7)
    ax.patch.set_facecolor('0.90')
    ax.set_axisbelow(True)

    #Set minor tick spacing to 1/2 of the major ticks
    ax.xaxis.set_minor_locator((pylab.MultipleLocator((plt.xticks()[0][1]
                                                       - plt.xticks()[0][0]) / 2.0)))
    ax.yaxis.set_minor_locator((pylab.MultipleLocator((plt.yticks()[0][1]
                                                       - plt.yticks()[0][0]) / 2.0)))

    #Remove axis border
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_alpha(0)

    #Restyle the tick lines
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(5)
        line.set_color("gray")
        line.set_markeredgewidth(1.4)

    #Remove the minor tick lines
    for line in (ax.xaxis.get_ticklines(minor=True) +
                 ax.yaxis.get_ticklines(minor=True)):
        line.set_markersize(0)

    #Only show bottom left ticks, pointing out of axis
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def rstyle_bw(ax):
    """Styles x,y axes to appear like ggplot2
    Must be called after all plot and axis manipulation operations have been
    carried out (needs to know final tick spacing)
    """
    #Set the style of the major and minor grid lines, filled blocks
    ax.grid(True, 'major', color='0.88', linestyle='-', linewidth=0.7)
    ax.grid(True, 'minor', color='0.95', linestyle='-', linewidth=0.4)
    ax.set_axisbelow(True)

    #Set minor tick spacing to 1/2 of the major ticks
    ax.xaxis.set_minor_locator((pylab.MultipleLocator((plt.xticks()[0][1]
                                                       - plt.xticks()[0][0]) / 2.0)))
    ax.yaxis.set_minor_locator((pylab.MultipleLocator((plt.yticks()[0][1]
                                                       - plt.yticks()[0][0]) / 2.0)))

    #Remove axis border
    ax.spines['top'].set_alpha(0)
    ax.spines['right'].set_alpha(0)

    #Restyle the tick lines
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(4)
    #     line.set_color("gray")
        line.set_markeredgewidth(.8)

    #Remove the minor tick lines
    for line in (ax.xaxis.get_ticklines(minor=True) +
                 ax.yaxis.get_ticklines(minor=True)):
        line.set_markersize(0)

    #Only show bottom left ticks, pointing out of axis
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def f_plot(time, f, title="Data", bw = False):
    """
    plots function data using matplotlib

    :param time: vector of size N describing the sample points
    :param f: numpy ndarray of shape (M,N) of M SRSFs with N samples
    :param title: string of title

    :return fig: figure definition
    :return ax: axes definition

    """
    CBcdict = {
        'Bu': (0, .45, .7),
        'Or': (.9, .6, 0),
        'SB': (.35, .7, .9),
        'bG': (0, .6, .5),
        'Ye': (.95, .9, .25),
        'Bl': (0, 0, 0),
        'Ve': (.8, .4, 0),
        'rP': (.8, .6, .7),
    }

    fig, ax = plt.subplots()
    ax.set_color_cycle(CBcdict[c] for c in list(CBcdict.keys()))
    ax.plot(time, f)
    plt.title(title)
    if bw:
        rstyle_bw(ax)
    else:
        rstyle(ax)

    return fig, ax