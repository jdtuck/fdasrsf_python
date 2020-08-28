import matplotlib
import pylab
import fdasrsf.curve_functions as cf
from numpy import tile, array, arange
import matplotlib.pyplot as plt


def rstyle(ax, pres=False):
    """
    Styles x,y axes to appear like ggplot2
    Must be called after all plot and axis manipulation operations have been
    carried out (needs to know final tick spacing)
    """
    #Set the style of the major and minor grid lines, filled blocks
    if pres:
        ax.grid(True, 'major', color='w', linestyle='-', linewidth=0.7)
        ax.grid(True, 'minor', color='0.99', linestyle='-', linewidth=0.4)
    else:
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
        if pres:
            line.set_markersize(4)
            line.set_color("gray")
            line.set_markeredgewidth(.8)
        else:
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
    """
    Styles x,y axes to appear like ggplot2
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


def f_plot(time, f, title="Data", bw=False, pres=False):
    """
    plots function data using matplotlib

    :param time: vector of size N describing the sample points
    :param f: numpy ndarray of shape (M,N) of M SRSFs with N samples
    :param title: string of title

    :return fig: figure definition
    :return ax: axes definition

    """

    fig, ax = plt.subplots()
    ax.plot(time, f)
    plt.title(title)
    plt.style.use('ggplot')

    return fig, ax


def plot_curve(beta):
    """
    plots curve

    :param beta: numpy array of shape (2,M) of M samples

    :return fig: figure defintion
    :return ax: axes
    """
    fig, ax = plt.subplots()
    ax.plot(beta[0, :], beta[1, :], 'r', linewidth=2)
    ax.set_aspect('equal')
    ax.axis('off')

    return fig,ax


def plot_reg_open_curve(beta1, beta2n):
    """
    plots registration between two open curves using matplotlib

    :param beta: numpy ndarray of shape (2,M) of M samples
    :param beta: numpy ndarray of shape (2,M) of M samples

    :return fig: figure definition
    :return ax: axes definition

    """
    T = beta1.shape[1]
    centroid1 = cf.calculatecentroid(beta1)
    beta1 = beta1 - tile(centroid1, [T, 1]).T
    centroid2 = cf.calculatecentroid(beta2n)
    beta2n = beta2n - tile(centroid2, [T, 1]).T
    beta2n[0, :] = beta2n[0, :] + 1.3
    beta2n[1, :] = beta2n[1, :] - 0.1

    fig, ax = plt.subplots()
    ax.plot(beta1[0, :], beta1[1, :], 'r', linewidth=2)
    fig.hold()
    ax.plot(beta2n[0, :], beta2n[1, :], 'b-o', linewidth=2)

    for j in range(0, int(T/5)):
        i = j*5
        ax.plot(array([beta1[0, i], beta2n[0, i]]),
                array([beta1[1, i], beta2n[1, i]]), 'k', linewidth=1)

    ax.set_aspect('equal')
    ax.axis('off')
    fig.hold()

    return fig, ax


def plot_geod_open_curve(PsiX):
    """
    plots geodesic between two open curves using matplotlib

    :param PsiX: numpy ndarray of shape (2,M,k) of M samples

    :return fig: figure definition
    :return ax: axes definition

    """
    k = PsiX.shape[2]
    fig, ax = plt.subplots()
    fig.hold()
    for tau in range(0, k):
        ax.plot(.35*tau+PsiX[0, :, tau], PsiX[1, :, tau], 'k', linewidth=2)

    ax.set_aspect('equal')
    ax.axis('off')
    fig.hold()

    return fig, ax


def plot_geod_close_curve(pathsqnc):
    """
    plots geodesic between two closed curves using matplotlib

    :param pathsqnc: numpy ndarray of shape (2,M,k,i) of M samples

    :return fig: figure definition
    :return ax: axes definition

    """
    i = pathsqnc.shape[3]
    k = pathsqnc.shape[2]
    if i > 1:
        plotidx = arange(0, i)
        fig, ax = plt.subplots(plotidx.size, k, sharex=True, sharey=True)
        for j in plotidx:
            for tau in range(0, k):
                beta_tmp = pathsqnc[:, :, tau, j]
                ax[j, tau].plot(beta_tmp[0, :], beta_tmp[1, :], 'r',
                                linewidth=2)
                ax[j, tau].set_aspect('equal')
                ax[j, tau].axis('off')
    else:
        fig, ax = plt.subplots(1, k, sharex=True, sharey=True)
        for tau in range(0, k):
            beta_tmp = pathsqnc[:, :, tau, j]
            ax[tau].plot(beta_tmp[0, :], beta_tmp[1, :], 'r', linewidth=2)
            ax[tau].set_aspect('equal')
            ax[tau].axis('off')

    return fig, ax
