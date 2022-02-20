# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:21:06 2019

@author: hendrick
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tikzplotlib
import datetime
import dateutil

from Photosynthesis import photosynthesis, spec, light_eff
from Calcification import gamma_arag
import TimeSeries
import Supporting_func as support

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# =============================================================================
# %% define figure(s) to be plotted / saved
# =============================================================================
# # file format figures
fileb = 'supportingplot_'
figpath = os.path.join('..', 'Figures', 'Python_figures')
# TeX- or png-file
TikZ = 0  # 1: TeX; 0: png
if TikZ:
    figext = '.tex'
else:
    figext = '.png'
    figqual = 600  # dpi

# # figure dimensions
figheight = 3.  # in
figwidth = 8.  # in

# # font size / line width
if TikZ:
    fs = 16
    lw = 1.
else:
    fs = 8
    lw = .5

# # light response
# PI-curve
plotPI = 0
savePI = 0
# photo-acclimation
plotPA = 0
savePA = 0
# optimal saturation intensity
plotIK = 0
saveIK = 0

# # thermal response
# adapted temperature response
plotF1 = 0
saveF1 = 0
# specialisation term
plotSPEC = 0
saveSPEC = 0
# thermal envelope
plotF2 = 0
saveF2 = 0

# # aragonite dependency
plotARG = 0
saveARG = 0

# # light-availability
plotIZ = 0
saveIZ = 0

# # storm damage
plotCORFAC = 0
saveCORFAC = 0

# # light time-series
plotId = 0
saveId = 0
# # SST time-series
# daily anomalies w.r.t. monthly means
plotSST1 = 0
saveSST1 = 0
# construction of artificial SST time-series
plotSST2 = 0
saveSST2 = 0
# comparison with NOAA data set
plotSST3 = 0
saveSST3 = 0

# =============================================================================
# # # # figure format
# =============================================================================
# # LaTeX format
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# # basic color definitions
bish = (0., 0., 128. / 255.)
rish = (128. / 255., 0., 0.)
gish = (0., 128. / 255., 0.)
# matrix-format
colors = np.array([bish, rish, gish])

# =============================================================================
# =============================================================================
# # # # # light response
# =============================================================================
# =============================================================================

# =============================================================================
# %% PI curve
# =============================================================================
if plotPI or savePI:
    # # data
    PG = 1.
    Ik = 2.
    R = .25
    Iz = np.linspace(0., 8., int(1e3))
    PN = PG * np.tanh(Iz / Ik) - R

    # # figure
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    # zero-line
    ax.plot([Iz.min(), Iz.max()], [0., 0.],
            color='gray', alpha=.5,
            linewidth=lw,
            label='_nolegend_')
    # plot data
    ax.plot(Iz, PN,
            color='black', alpha=.8,
            linewidth=lw,
            label='_nolegend_')
    # axes labels
    ax.set_xlabel(r'light-intensity, $I$ $\rightarrow$')
    ax.set_ylabel(r'photosynthetic rate, $P$ $\rightarrow$')
    # axes ticks
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)
    ax.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)
    # explaining lines / texts / etc.
    ax.plot([Iz.min(), Iz.max()], [-R, -R],
            color='gray', alpha=.8,
            linestyle='dotted', linewidth=lw,
            label='_nolegend_')
    ax.plot([Ik, Ik], [0., PN.max()],
            color='gray', alpha=.8,
            linestyle='dotted', linewidth=lw,
            label='_nolegend_')
    ax.plot([Iz.min(), Iz.max()], [PN.max(), PN.max()],
            color='gray', alpha=.8,
            linestyle='dotted', linewidth=lw,
            label='_nolegend_')
    ax.plot([Iz.min(), Ik], [PN.min(), PN.max()],
            color='gray', alpha=.8,
            linestyle='dotted', linewidth=lw,
            label='_nolegend_')
    ax.plot([Ik, Ik], [-.01, .01],
            color='black',
            linewidth=.5,
            label='_nolegend_')
    ax.text(Ik, -.05, s='$I_{{k}}$',
            fontsize=fs,
            ha='center', va='top')
    ax.annotate(s='',
                xy=(7., 0.), xytext=(7., PN.max()),
                arrowprops=dict(color='black',
                                linewidth=.5,
                                arrowstyle='<|-|>'))
    ax.text(6.9, .5 * PN.max(), s=r'$P_{{N}}^{{\max}}$',
            fontsize=fs,
            ha='right', va='center')
    ax.annotate(s='',
                xy=(7., 0.), xytext=(7., PN.min()),
                arrowprops=dict(color='black',
                                linewidth=.5,
                                arrowstyle='<|-|>'))
    ax.text(6.9, .5 * PN.min(), s=r'$R$',
            fontsize=fs,
            ha='right', va='center')
    ax.annotate(s='',
                xy=(7.5, PN.min()), xytext=(7.5, PN.max()),
                arrowprops=dict(color='black',
                                linewidth=.5,
                                arrowstyle='<|-|>'))
    ax.text(7.6, .5 * PN.max(), s=r'$P_{{G}}^{{\max}}$',
            fontsize=fs,
            ha='left', va='center')
    ax.text(.3, .75 * PN.min(), s=r'$\alpha_{{0}}$',
            fontsize=fs,
            ha='left', va='center')
    # plot limits
    ax.set_xlim([Iz.min(), Iz.max()])
    ax.set_ylim([1.2 * PN.min(), 1.2 * PN.max()])
    # legend / title

    # # save figure
    if savePI:
        figname = fileb + 'PI'
        figfile = figname + figext
        figfull = os.path.join(figpath, figfile)
        if TikZ:
            plt.tight_layout()
            tikzplotlib.save(figfull,
                             figureheight='{0}in'.format(figheight),
                             figurewidth='{0}in'.format(figwidth))
        else:
            fig.savefig(figfull, dpi=figqual,
                        bbox_inches='tight')
        print('Figure saved as: {0}'.format(figfull))

# =============================================================================
# %% photo-acclimation - PI-curve
# =============================================================================
if plotPA or savePA:
    # # data
    Ik = np.array([1., 4.])
    Pm = np.array([.6, .8])
    Iz = np.linspace(0., 10., int(1e3))
    I0 = 10.
    Plo = light_eff(Pm[0], Iz, I0, Ik[0])
    Phi = light_eff(Pm[1], Iz, I0, Ik[1])

    # # figure
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    # zero-line
    ax.plot([Iz.min(), Iz.max()], [0., 0.],
            color='gray', alpha=.5,
            linewidth=lw,
            label='_nolegend_')
    # plot data
    ax.plot(Iz, Plo,
            color=bish, alpha=.8,
            linewidth=lw,
            label=r'low light')
    ax.plot(Iz, Phi,
            color=rish, alpha=.8,
            linewidth=lw,
            label=r'high light')
    # axes labels
    ax.set_xlabel(r'light-intensity, $I$ $\rightarrow$')
    ax.set_ylabel(r'photosynthetic rate, $P$ $\rightarrow$')
    # axes ticks
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)
    ax.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)
    # explaining lines / texts / etc.
    for i in range(len(Ik)):
        ax.plot([Ik[i], Ik[i]], [-2., 2.],
                color='gray', alpha=.8,
                linestyle='dotted', linewidth=lw,
                label='_nolegend_')
        ax.plot([Iz.min(), Iz.max()],
                [Pm[i] * (1. - np.tanh(.01 * I0 / Ik[i])),
                 Pm[i] * (1. - np.tanh(.01 * I0 / Ik[i]))],
                color='gray', alpha=.8,
                linestyle='dotted', linewidth=lw,
                label='_nolegend_')
    ax.annotate(s='',
                xy=(Ik[1], .03), xytext=(Ik[0], .03),
                arrowprops=dict(color=rish,
                                linewidth=.5,
                                arrowstyle='-|>'))
    ax.annotate(s='',
                xy=(Ik[0], .05), xytext=(Ik[1], .05),
                arrowprops=dict(color=bish,
                                linewidth=.5,
                                arrowstyle='-|>'))
    ax.text(Ik.mean(), .07, s=r'$I_{k}$',
            fontsize=fs,
            ha='center', va='bottom')
    ax.annotate(s='',
                xy=(9.65, Pm[1] * (1. - np.tanh(.01 * I0 / Ik[1]))),
                xytext=(9.65, Pm[0] * (1. - np.tanh(.01 * I0 / Ik[0]))),
                arrowprops=dict(color=rish,
                                linewidth=.5,
                                arrowstyle='-|>'))
    ax.annotate(s='',
                xy=(9.5, Pm[0] * (1. - np.tanh(.01 * I0 / Ik[0]))),
                xytext=(9.5, Pm[1] * (1. - np.tanh(.01 * I0 / Ik[1]))),
                arrowprops=dict(color=bish,
                                linewidth=.5,
                                arrowstyle='-|>'))
    ax.text(9.35, Pm.mean(), s=r'$P_{\max}$',
            fontsize=fs,
            ha='right', va='center')
    # plot limits
    ax.set_xlim([Iz.min(), Iz.max()])
    ax.set_ylim([-.05, 1.])
    # legend / title

    # # save figure
    if savePA:
        figname = fileb + 'PA'
        figfile = figname + figext
        figfull = os.path.join(figpath, figfile)
        if TikZ:
            plt.tight_layout()
            tikzplotlib.save(figfull,
                             figureheight='{0}in'.format(figheight),
                             figurewidth='{0}in'.format(figwidth))
        else:
            fig.savefig(figfull, dpi=figqual,
                        bbox_inches='tight')
        print('Figure saved as: {0}'.format(figfull))

# =============================================================================
# %% photo-acclimation - saturation intensity
# =============================================================================
if plotIK or saveIK:
    # # data
    N = 1e4
    I0 = 1.
    # PI(Ik) for given Iz
    Iz1 = np.array([.02, .1, 1.])
    Ik1 = np.linspace(1 / N, 1., int(N))
    PI1 = np.zeros((len(Iz1), len(Ik1)))
    for i in range(len(Iz1)):
        PI1[i, :] = light_eff(Iz1[i], I0, Ik1)
    PI1max = PI1.max(axis=1)
    Ik1_opt = Ik1[PI1.argmax(axis=1)]
    # gross PI
    PIG = np.zeros((len(Iz1), len(Ik1)))
    for i in range(len(Iz1)):
        PIG[i, :] = np.tanh(Iz1[i] / Ik1)
    Rd = np.tanh(.01 / Ik1)
    # optimal Ik per Iz
    Iz = np.linspace(1 / N, 1., int(N))
    Ik = np.linspace(1 / N, 1., int(N))
    PI = np.zeros((len(Iz), len(Ik)))
    for i in range(len(Iz)):
        PI[i, :] = light_eff(Iz[i], I0, Ik)
    PImax = PI.max(axis=1)
    Ik_opt = Ik[PI.argmax(axis=1)]
    # linear fit
    params = np.polyfit(Iz, Ik_opt, 1)
    fit = np.poly1d(params)
    Ik_fit = fit(Iz)
    # Anthony & Hoegh-Guldberg, 2003
    IkA = Ik_opt[-1] * Iz ** .34
    PIA = np.tanh(Iz / IkA) - np.tanh(.01 * I0 / IkA)

    # # figure 1
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    # zero-line
    # plot data
    for i in range(len(Iz1)):
        ax.plot(Ik1, PI1[i, :],
                color=colors[i, :], alpha=.8,
                linewidth=lw,
                label=r'$I_{{z}} / I_{{0}} = {0:.2f}$'.format(Iz1[i]))
        ax.scatter(Ik1_opt[i], PI1max[i], s=5.,
                   color=colors[i, :], alpha=.8,
                   label='_nolegend_')
        ax.plot(Ik1, PIG[i, :],
                color=colors[i, :], alpha=.8,
                linestyle='dashed', linewidth=lw,
                label='_nolegend_')
    ax.plot([], [],
            color='black', alpha=.8,
            linewidth=lw,
            label=r'$P(I)$')
    ax.plot([], [],
            color='black', alpha=.8,
            linestyle='dashed', linewidth=lw,
            label=r'$P_{{G}}$')
    ax.plot(Ik1, Rd,
            color='black', alpha=.8,
            linestyle='dotted', linewidth=lw,
            label=r'$R_{{d}}$')
    # axes labels
    ax.set_xlabel(r'normalised saturation intensity, $I_{{k}} / I_{{0}}$ [-]')
    ax.set_ylabel(r'photo-efficiency $P(I)$ [-]')
    # axes ticks
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([0., 1.])
    ax.set_yticks([0., 1.])
    # explaining lines / texts / etc.
    # plot limits
    ax.set_xlim([0., 1.])
    ax.set_ylim([0., 1.1])
    # legend / title
    ax.legend(ncol=2)

    # # save figure 1
    if saveIK:
        figname = fileb + 'IK1'
        figfile = figname + figext
        figfull = os.path.join(figpath, figfile)
        if TikZ:
            plt.tight_layout()
            tikzplotlib.save(figfull,
                             figureheight='{0}in'.format(figheight),
                             figurewidth='{0}in'.format(figwidth))
        else:
            fig.savefig(figfull, dpi=figqual,
                        bbox_inches='tight')
        print('Figure saved as: {0}'.format(figfull))

    # # figure 2
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    # zero-line
    # plot data
    ax.plot(Iz, PImax,
            color=bish, alpha=.8,
            linewidth=lw,
            label=r'$P(I)$')
    ax.plot(Iz, Ik_opt,
            color=rish, alpha=.8,
            linewidth=lw,
            label=r'$I_{{k}}^{{opt}} / I_{{0}}$')
#    ax.plot(Iz, Ik_fit,
#            color='black', alpha=.8,
#            linestyle='dashed', linewidth=lw,
#            label=r'linear fit')
    ax.plot(Iz, PIA,
            color=bish, alpha=.8,
            linestyle='dashed', linewidth=lw,
            label=r'$P(I)$ - A\&G-H')
    ax.plot(Iz, IkA,
            color=rish, alpha=.8,
            linestyle='dashed', linewidth=lw,
            label=r'$I_{{k}} / I_{{0}}$ - A\&G-H')
    # axes labels
    ax.set_xlabel(r'normalised light-intensity, $I_{{z}} / I_{{0}}$ [-]')
    ax.set_ylabel(r'$P(I)$ [-],  $I_{{k}} / I_{{0}}$ [-]')
    # axes ticks
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([0., 1.])
    ax.set_yticks([0., 1.])
    # explaining lines / texts / etc.
    # plot limits
    ax.set_xlim([0., 1.])
    ax.set_ylim([0., 1.])
    # legend / title
    ax.legend()

    # # save figure 2
    if saveIK:
        figname = fileb + 'IK2'
        figfile = figname + figext
        figfull = os.path.join(figpath, figfile)
        if TikZ:
            plt.tight_layout()
            tikzplotlib.save(figfull,
                             figureheight='{0}in'.format(figheight),
                             figurewidth='{0}in'.format(figwidth))
        else:
            fig.savefig(figfull, dpi=figqual,
                        bbox_inches='tight')
        print('Figure saved as: {0}'.format(figfull))

# =============================================================================
# =============================================================================
# %% thermal response
# =============================================================================
# =============================================================================
diff = 273.15
# =============================================================================
# %% thermal range - adapted temperature response
# =============================================================================
if plotF1 or saveF1:
    # # data
    # thermal key parameters
    T = np.linspace(15., 35., int(1e3)) + diff
    DT = np.array([5., 10., 15.])
    Topt = 300.  # [K]
    Tlo = Topt - (1. / np.sqrt(3.)) * DT
    # photosynthetic response
    PM = np.zeros((len(DT), len(T)))
    for i in range(len(DT)):
        PM[i, :] = photosynthesis(1e3, 1., T, Tlo[i], DT[i],
                                  method='math')
    T -= diff
    Topt -= diff

    # # figure
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    # zero-line
    ax.plot([T.min(), T.max()], [0., 0.],
            color='gray', alpha=.5,
            linewidth=lw,
            label='_nolegend_')
    # plot data
    for i in range(len(DT)):
        ax.plot(T, PM[i, :],
                color=colors[i, :], alpha=.8,
                linewidth=lw,
                label=r'$\Delta T = {0}\ {{}}^{{\circ}}C$'.format(DT[i]))
    # axes labels
    ax.set_xlabel(r'temperature, $T$ $[{{}}^{{\circ}}C]$')
    ax.set_ylabel(r'calcification rate, $\dot{G}$ $\rightarrow$')
    # axes ticks
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([0.])
    # explaining lines / texts / etc.
    ax.scatter(Tlo - diff, np.zeros(len(Tlo)), s=5,
               color=colors, alpha=.8,
               label='_nolegend_')
    for i in range(len(DT)):
        ax.annotate(s='', xy=(Tlo[i] - diff, 0.), xytext=(20., .1),
                    fontsize=fs,
                    arrowprops=dict(color='black',
                                    linewidth=.5,
                                    arrowstyle='-|>'))
    ax.text(20., .1, s=r'$T_{{lo}}$', fontsize=fs,
            ha='center', va='bottom')

    Thi = Tlo + DT
    ax.scatter(Thi - diff, np.zeros(len(Thi)), s=5,
               color=colors, alpha=.8,
               label='_nolegend_')
    for i in range(len(DT)):
        ax.annotate(s='', xy=(Thi[i] - diff, 0.), xytext=(32., .1),
                    fontsize=fs,
                    arrowprops=dict(color='black',
                                    linewidth=.5,
                                    arrowstyle='-|>'))
    ax.text(32., .1, s=r'$T_{{hi}}$', fontsize=fs,
            ha='center', va='bottom')

    ax.scatter(Topt * np.ones(len(DT)), PM.max(axis=1), s=5,
               color=colors, alpha=.8,
               label='_nolegend_')
    ax.plot([Topt, Topt], [0., PM.max()],
            color='gray', alpha=.8,
            linestyle='dotted', linewidth=lw,
            label='_nolegend_')
    ax.plot([Topt, Topt], [-.02 * PM.max(), .02 * PM.max()],
            color='black',
            linewidth=.5,
            label='_nolegend_')
    ax.text(Topt, -.02, s=r'$T_{{opt}}$',
            fontsize=fs,
            ha='center', va='top')
    # plot limits
    ax.set_xlim([T.min(), T.max()])
    ax.set_ylim([-1.1 * PM.max(), 1.1 * PM.max()])
    # legend / title
    ax.legend(loc='lower right',
              prop={'size': fs})

    # # save figure
    if saveF1:
        figname = fileb + 'F1'
        figfile = figname + figext
        figfull = os.path.join(figpath, figfile)
        if TikZ:
            plt.tight_layout()
            tikzplotlib.save(figfull,
                             figureheight='{0}in'.format(figheight),
                             figurewidth='{0}in'.format(figwidth))
        else:
            fig.savefig(figfull, dpi=figqual,
                        bbox_inches='tight')
        print('Figure saved as: {0}'.format(figfull))

# =============================================================================
# %% specialisation term
# =============================================================================
if plotSPEC or saveSPEC:
    # # # area normalisation
    # # data
    # thermal key parameters
    T = np.linspace(15., 35., int(1e3)) + diff
    DT = np.array([5., 10., 15.])
    Topt = 300.  # [K]
    Tlo = Topt - (1. / np.sqrt(3.)) * DT
    # photosynthetic response
    PE = np.zeros((len(DT), len(T)))
    PM = np.zeros((len(DT), len(T)))
    for i in range(len(DT)):
        PE[i, :] = photosynthesis(1e3, 1., T, Tlo[i], DT[i],
                                  method='Evenhuis2015')
        PM[i, :] = photosynthesis(1e3, 1., T, Tlo[i], DT[i],
                                  method='math')
    T -= diff

    # # figure
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    # zero-line
    ax.plot([T.min(), T.max()], [0., 0.],
            color='gray', alpha=.5,
            linewidth=lw,
            label='_nolegend_')
    # plot data
    for i in range(len(DT)):
        ax.plot(T, PE[i, :],
                color=colors[i, :], alpha=.8,
                linewidth=lw,
                label=r'$\Delta T = {0}\ {{}}^{{\circ}}C$'.format(DT[i]))
        ax.plot(T, PM[i, :],
                color=colors[i, :], alpha=.8,
                linestyle='dashed', linewidth=lw,
                label='_nolegend_')
    # axes labels
    ax.set_xlabel(r'temperature, $T$ $[{{}}^{{\circ}}C]$')
    ax.set_ylabel(r'calcification rate, $\dot{G}$ $\rightarrow$')
    # axes ticks
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([0.])
    # explaining lines / texts / etc.
    if TikZ:
        ax.plot(T.min(), PE.min(),
                color='gray', alpha=.8,
                label=r'Eq. \ref{eq:spec}')
        ax.plot(T.min(), PM.min(),
                color='gray', alpha=.8, linestyle='dashed',
                label=r'Eq. \ref{eq:spec_math}')
    else:
        ax.plot(T.min(), PE.min(),
                color='gray', alpha=.8,
                label=r'spec${{}}_{{E}}(\Delta T)$')
        ax.plot(T.min(), PM.min(),
                color='gray', alpha=.8, linestyle='dashed',
                label=r'spec${{}}_{{M}}(\Delta T)$')
    # plot limits
    ax.set_xlim([T.min(), T.max()])
    ax.set_ylim([-1.1 * PM.max(), 1.1 * PM.max()])
    # legend / title
    ax.legend(loc='upper left',
              prop={'size': fs})

    # # save figure
    if saveSPEC:
        figname = fileb + 'SPEC1'
        figfile = figname + figext
        figfull = os.path.join(figpath, figfile)
        if TikZ:
            tikzplotlib.save(figfull,
                             figureheight='{0}in'.format(figheight),
                             figurewidth='{0}in'.format(figwidth))
        else:
            fig.savefig(figfull, dpi=figqual,
                        bbox_inches='tight')
        print('Figure saved as: {0}'.format(figfull))

    # # # effect of spec(DT) on area-normalisation
    # # data
    dt = np.linspace(4., 16., int(1e3))
    # spec(DT)
    fE = spec(dt, method='Evenhuis2015')
    fm = spec(dt, method='math')
    # area-normalisation
    AE = .25 * dt ** 4 * spec(dt, method='Evenhuis2015')
    Am = .25 * dt ** 4 * spec(dt, method='math')

    # # figure
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False,
                           figsize=(figwidth, figheight))
    plt.subplots_adjust(hspace=0)
    # plot data
    if TikZ:
        # plot data - spec(DT)
        ax[0].plot(dt, fE,
                   color=bish, alpha=.8,
                   linewidth=lw,
                   label=r'Eq. (\ref{eq:spec})')
        ax[0].plot(dt, fm,
                   color=rish, alpha=.8,
                   linewidth=lw,
                   label=r'Eq. (\ref{eq:spec_math})')
        # plot data - area
        ax[1].plot(dt, AE,
                   color=bish, alpha=.8,
                   linewidth=lw,
                   label=r'Eq. (\ref{eq:spec})')
        ax[1].plot(dt, Am,
                   color=rish, alpha=.8,
                   linewidth=lw,
                   label=r'Eq. (\ref{eq:spec_math})')
    else:
        # plot data - spec(DT)
        ax[0].plot(dt, fE,
                   color=bish, alpha=.8,
                   linewidth=lw,
                   label=r'spec${{}}_{{E}}(\Delta T)$')
        ax[0].plot(dt, fm,
                   color=rish, alpha=.8,
                   linewidth=lw,
                   label=r'spec${{}}_{{M}}(\Delta T)$')
        # plot data - area
        ax[1].plot(dt, AE,
                   color=bish, alpha=.8,
                   linewidth=lw,
                   label=r'spec${{}}_{{E}}(\Delta T)$')
        ax[1].plot(dt, Am,
                   color=rish, alpha=.8,
                   linewidth=lw,
                   label=r'spec${{}}_{{M}}(\Delta T)$')
    # axes labels
    ax[0].set_ylabel(r'spec($\Delta T$) $\rightarrow$')
    ax[1].set_ylabel(r'normalised area')
    ax[1].set_xlabel(r'thermal range, $\Delta T$ $[{{}}^{{\circ}}C]$')
    fig.align_ylabels()
    # axes ticks
    for i in range(2):
        ax[i].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[1].spines['top'].set_visible(True)
    ax[0].tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labeltop=False)
    ax[1].tick_params(
            axis='x',
            which='both',
            bottom=True,
            top=True,
            labelbottom=True)
    ax[0].tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)
    ax[1].set_yticks([0., 1.])
    # explaining lines / texts / etc.
    # plot limits
    ax[1].set_xlim([dt.min(), dt.max()])
    ax[0].set_ylim([-.1 * fm.max(), 1.1 * fm.max()])
    ax[1].set_ylim([0., 1.2 * AE.max()])
    # legend / title
    ax[0].legend(loc='upper right',
                 prop={'size': fs})

    # # save figure
    if saveSPEC:
        figname = fileb + 'SPEC2'
        figfile = figname + figext
        figfull = os.path.join(figpath, figfile)
        if TikZ:
            plt.tight_layout()
            tikzplotlib.save(figfull,
                             figureheight='{0}in'.format(figheight),
                             figurewidth='{0}in'.format(.95 * figwidth))
        else:
            fig.savefig(figfull, dpi=figqual,
                        bbox_inches='tight')
        print('Figure saved as: {0}'.format(figfull))

# =============================================================================
# %% optimum temperature - thermal envelope
# =============================================================================
if plotF2 or saveF2:
    # # data
    # thermal key parameters
    T = np.linspace(15., 35., int(5e2)) + diff
    DT = 10.
    Topt = np.array([25., 27., 29.]) + diff
    Tlo = Topt - (1. / np.sqrt(3.)) * DT
    # photosynthetic response
    PM = np.zeros((len(Topt), len(T)))
    for i in range(len(Topt)):
        PM[i, :] = photosynthesis(1e3, 1., T, Tlo[i], DT,
                                  method='math')
    T -= diff
    Topt -= diff

    # # figure
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    # zero-line
    ax.plot([T.min(), T.max()], [0., 0.],
            color='gray', alpha=.5,
            linewidth=lw,
            label='_nolegend_')
    # plot data
    for i in range(len(Topt)):
        ax.plot(T, PM[i, :],
                color=colors[i, :], alpha=.8,
                linewidth=lw,
                label=r'$T_{{opt}} = {0}\ {{}}^{{\circ}}C$'.format(Topt[i]))
    # axes labels
    ax.set_xlabel(r'temperature, $T$ $[{{}}^{{\circ}}C]$')
    ax.set_ylabel(r'calcification rate, $\dot{G}$ $\rightarrow$')
    # axes ticks
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([0.])
    # explaining lines / texts / etc.
    TloA = (T - (1. / np.sqrt(3.)) * DT) + diff
    Arr = photosynthesis(1e3, 1., T + diff, TloA, DT,
                         method='math')
    ax.plot(T, Arr,
            color='black', alpha=.8,
            linestyle='dashed', linewidth=lw,
            label='_nolegend_')
    loc = np.array((T[25], Arr[25] + .01))
    df2 = Arr[50] - Arr[0]
    dT = T[50] - T[0]
    rad = np.tan(df2 / dT)
    ang = ax.transData.transform_angles(np.array((3.5 * np.rad2deg(rad),)),
                                        loc.reshape((1, 2)))[0]
    ax.text(loc[0], loc[1], r'Arrhenius equation', fontsize=fs,
            rotation=ang, rotation_mode='anchor')

    ax.scatter(Topt, PM.max(axis=1), s=5,
               color=colors, alpha=.8,
               label='_nolegend_')
    for i in range(len(Topt)):
        ax.plot([Topt[i], Topt[i]], [0., PM[i, :].max()],
                color='gray', alpha=.8,
                linestyle='dotted', linewidth=lw,
                label='_nolegend_')
        ax.plot([Topt[i], Topt[i]], [-.02 * PM.max(), .02 * PM.max()],
                color='black',
                linewidth=.5,
                label='_nolegend_')
        ax.annotate(s='',
                    xy=(Topt[i], 0.), xytext=(24., -.05),
                    arrowprops=dict(color='black',
                                    linewidth=.5,
                                    arrowstyle='-|>'))
    ax.text(24., -.06, s=r'$T_{{opt}}$',
            fontsize=fs,
            ha='center', va='top')
    # plot limits
    ax.set_xlim([T.min(), T.max()])
    ax.set_ylim([-1.1 * PM.max(), 1.1 * PM.max()])
    # legend / title
    ax.legend(loc='lower right',
              prop={'size': fs})

    # # save figure
    if saveF2:
        figname = fileb + 'F2'
        figfile = figname + figext
        figfull = os.path.join(figpath, figfile)
        if TikZ:
            plt.tight_layout()
            tikzplotlib.save(figfull,
                             figureheight='{0}in'.format(figheight),
                             figurewidth='{0}in'.format(figwidth))
        else:
            fig.savefig(figfull, dpi=figqual,
                        bbox_inches='tight')
        print('Figure saved as: {0}'.format(figfull))

# =============================================================================
# =============================================================================
# %% aragonite dependency
# =============================================================================
# =============================================================================
if plotARG or saveARG:
    # # data
    kappa = 1.5
    omega_0 = 1.
    omega = np.linspace(0, 5.5, int(1e3))
    gamma = gamma_arag(omega, ka=kappa, omega_0=omega_0)

    # # figure
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    # zero-line
    ax.plot([omega.min(), omega.max()], [0., 0.],
            color='gray', alpha=.5,
            linewidth=lw,
            label='_nolegend_')
    # plot data
    ax.plot(omega, gamma,
            color='black', alpha=.8,
            linewidth=lw,
            label=r'$\gamma(\Omega_{{a}})$')
    # axes labels
    ax.set_xlabel(r'aragonite saturation state, $\Omega_{{a}}$ $\rightarrow$')
    ax.set_ylabel(r'normalised calcification rate')
    # axes ticks
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(
            axis='x',
            which='both',
            top=False,
            bottom=False,
            labelbottom=False)
    ax.set_yticks([0., .5, 1.])
    # explaining lines / texts / etc.
    ax.plot([omega.min(), omega_0 + kappa, omega_0 + kappa],
            [.5, .5, gamma.min()],
            color='gray', alpha=.8,
            linestyle='dotted', linewidth=lw,
            label='_nolegend_')
    ax.scatter(omega_0 + kappa, .5, s=5,
               color='black', alpha=.8,
               label='_nolegend_')
    ax.plot([omega_0, omega_0], [0., gamma.min()],
            color='gray', alpha=.8,
            linestyle='dotted', linewidth=lw,
            label='_nolegend_')
    ax.annotate(s='',
                xy=(omega_0, -.08), xytext=(omega_0 + kappa, -.08),
                arrowprops=dict(color='black',
                                linewidth=.5,
                                arrowstyle='<|-|>'))
    ax.text(x=(omega_0 + .5 * kappa), y=-.08, s=r'$\kappa_{{a}}$',
            fontsize=fs,
            ha='center', va='bottom')
    ax.annotate(s='',
                xy=(0., -.08), xytext=(omega_0, -.08),
                arrowprops=dict(color='black',
                                linewidth=.5,
                                arrowstyle='<|-|>'))
    ax.text(x=(.5 * omega_0), y=-.08, s=r'$\Omega_{0}$',
            fontsize=fs,
            ha='center', va='bottom')
    ax.annotate(s=r'$\frac{{1}}{{2}}\max\{\gamma\}$',
                xy=(omega_0 + kappa, .5), xytext=(.5 * omega_0 + kappa, .6),
                fontsize=fs,
                arrowprops=dict(color='black',
                                linewidth=.5,
                                arrowstyle='-|>'))
    # plot limits
    ax.set_xlim([omega.min(), omega.max()])
    ax.set_ylim([-.1, 1.])
    # legend / title

    # # save figure
    if saveARG:
        figname = fileb + 'ARG'
        figfile = figname + figext
        figfull = os.path.join(figpath, figfile)
        if TikZ:
            plt.tight_layout()
            tikzplotlib.save(figfull,
                             figureheight='{0}in'.format(figheight),
                             figurewidth='{0}in'.format(figwidth))
        else:
            fig.savefig(figfull, dpi=figqual,
                        bbox_inches='tight')
        print('Figure saved as: {0}'.format(figfull))

# =============================================================================
# =============================================================================
# %% light-availability
# =============================================================================
# =============================================================================
if plotIZ or saveIZ:
    # # data
    Kd = .1
    hc = .7
    z = np.linspace(0., 10., int(1e3))
    HC = z.max() * (1. - hc)
    Iz = np.exp(-Kd * z)
    Ic = np.exp(-Kd * z)
    Ic[z > HC] = np.exp(-Kd * HC) * np.exp(-2 * Kd * (z[z > HC] - HC))

    # # figure
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    # plot data
    ax.plot(Iz, z,
            color='black', alpha=.8,
            linewidth=lw,
            label=r'$I_{{z}}$')
    ax.plot(Ic, z,
            color='black', alpha=.8,
            linestyle='dashed', linewidth=lw,
            label=r'$I_{{z,cor}}$')
    # axes labels
    ax.set_xlabel(r'light-intensity, $I$ $\rightarrow$')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel(r'$\leftarrow$ depth, $z$')
    # axis ticks
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(
            axis='x',
            which='both',
            top=False,
            bottom=False,
            labelbottom=False)
    ax.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)
    # explaining lines / texts / etc.
    ax.plot([0., Iz.max()], [HC, HC],
            color='gray', alpha=.8,
            linestyle='dotted', linewidth=lw,
            label='_nolegend_')
    ax.plot([0., .5 * Ic.min(), .5 * Ic.min()], [HC, HC, z.max()],
            color='gray', alpha=.8,
            linewidth=lw,
            label='_nolegend_')
    ax.annotate(s='',
                xy=(.75 * Ic.min(), z.max()), xytext=(.75 * Ic.min(), HC),
                arrowprops=dict(color='black',
                                linewidth=.5,
                                arrowstyle='<|-'))
    ax.text(.75 * Ic.min() + .01, HC + .5 * (z.max() - HC), s=r'$h_{{c}}$',
            fontsize=fs,
            ha='left', va='center')
    # plot limits
    ax.set_xlim([0., Iz.max()])
    ax.set_ylim([z.min(), z.max()])
    ax.invert_yaxis()
    # legend / title
    ax.legend(loc='lower right',
              prop={'size': fs})

    # # save figure
    if saveIZ:
        figname = fileb + 'IZ'
        figfile = figname + figext
        figfull = os.path.join(figpath, figfile)
        if TikZ:
            plt.tight_layout()
            tikzplotlib.save(figfull,
                             figureheight='{0}in'.format(figheight),
                             figurewidth='{0}in'.format(figwidth))
        else:
            fig.savefig(figfull, dpi=figqual,
                        bbox_inches='tight')
        print('Figure saved as: {0}'.format(figfull))

# =============================================================================
# =============================================================================
# %% storm damage
# =============================================================================
# =============================================================================
if plotCORFAC or saveCORFAC:
    # # data
    t = np.linspace(0, 100, int(1e3))
    L = (np.random.rand(len(t))) ** 25
    Lcr = 1e-60

    # # figure
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    # zero-line
    # plot data
    ax.plot(t, L,
            color=bish, alpha=.8,
            linewidth=lw,
            label=r'$L(t)$')
    # axes labels
    ax.set_xlabel(r'time, $t$ $\rightarrow$')
    ax.set_ylabel(r'hydrodynamic load, $L$ $\rightarrow$')
    # axes ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels([])
    ax.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)
    # explaining lines / texts / etc.
    ax.plot([t.min(), t.max()], [Lcr, Lcr],
            color=rish, alpha=.8,
            linestyle='dashed', linewidth=lw,
            label=r'$L_{{cr}}$')
    a = np.array([20., 40., 60., 80.])
    La1 = 1e-80
    La2 = 1e-70
    for i in range(len(a)):
        ax.annotate(s='',
                    xy=(a[i], La1), xytext=(a[i], La2),
                    arrowprops=dict(color='black',
                                    linewidth=.5,
                                    arrowstyle='<|-'))
    a = t[L < Lcr]
    for i in range(len(a)):
        ax.annotate(s='',
                    xy=(a[i], La1), xytext=(a[i], La2),
                    arrowprops=dict(color=rish,
                                    linewidth=.5,
                                    arrowstyle='<|-'))
    # plot limits
    ax.set_yscale('log')
    ax.set_ylim([1e-80, 1.])
    ax.invert_yaxis()
    ax.set_xlim([t.min(), t.max()])
    # legend / title
    ax.legend(loc=1)

    # # save figure
    if saveCORFAC:
        figname = fileb + 'CORFAC'
        figfile = figname + figext
        figfull = os.path.join(figpath, figfile)
        if TikZ:
            plt.tight_layout()
            tikzplotlib.save(figfull,
                             figureheight='{0}in'.format(figheight),
                             figurewidth='{0}in'.format(figwidth))
        else:
            fig.savefig(figfull, dpi=figqual,
                        bbox_inches='tight')
        print('Figure saved as: {0}'.format(figfull))


# =============================================================================
# =============================================================================
# %% Light time-series
# =============================================================================
# =============================================================================
if plotId or saveId:
    figname = 'I0_TimeSeries'
    # time span
    T = 2.
    Tstart = datetime.date(2000, 1, 1)
    Tend = Tstart + dateutil.relativedelta.relativedelta(years=T, days=-1)
    t = pd.date_range(Tstart, Tend, freq='D')
    n = np.arange(len(t))
    # location(s)
    lat = np.array([-15., 15., 30.])
    NS = np.array(len(lat) * ['N'])
    NS[lat < 0] = 'S'
    S0 = 1600.

    # # calculations
    Id = np.zeros((len(lat), len(n)))
    for i in range(len(lat)):
        Id[i, :] = TimeSeries.TS_PAR(n, lat[i], S0=S0)

    # # figure
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    # zero-line
    # plot data
    for i in range(len(lat)):
        ax.plot(t, Id[i, :],
                color=colors[i, :], alpha=.8,
                linewidth=lw,
                label=r'$\phi = {0}^{{\circ}}{1}$'.format(abs(lat[i]), NS[i]))
    # axes labels
    ax.set_xlabel(r'year')
    ax.set_ylabel(r'incoming light, $I_{0}$ '
                  r'[$\mu mol$ photons $m^{{-2}} s^{{-1}}$]')
    # axes ticks
    ticks = pd.date_range(Tstart,
                          Tend + dateutil.relativedelta.relativedelta(days=1),
                          freq='YS')
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks.year)
    # plot limits
    ax.set_xlim([ticks.min(), ticks.max()])
    # explaining lines / texts / etc.
    # legend / title
    ax.legend(loc='upper right',
              fontsize=fs)

    # # save figure
    if saveId:
        figfile = figname + figext
        figfull = os.path.join(figpath, figfile)
        fig.savefig(figfull, dpi=figqual,
                    bbox_inches='tight')
        print('Figure saved as: {0}'.format(figfull))

# =============================================================================
# =============================================================================
# %% SST time-series
# =============================================================================
# =============================================================================

# =============================================================================
# %% Daily anomalies w.r.t. monthly means
# =============================================================================
if plotSST1 or saveSST1:
    figname = 'SST_daily_anomalies_'
    SSTpath = 'SST_data'
    locs = pd.DataFrame({'fig': ['HelsdonReef', 'RibbonReef9'],
                         'name': ['Helsdon Reef (GBR)',
                                  'Ribbon Reef No. 9 (GBR)'],
                         'lat': [-15., -15.],
                         'lon': [145.5, 145.7]})

    for i in range(len(locs)):
        lat = locs['lat'].iloc[i]
        lon = locs['lon'].iloc[i]
        loc = locs['name'].iloc[i]
        # # NOAA dataset (1981 - 2019)
        # define file
        [latN, lonN], [_, _] = support.data_coor(lat, lon)
        file = ('SST_timeseries_lat{0}lon{1}_Y1981_2019.txt'
                .format(latN, lonN))
        filef = os.path.join(SSTpath, file)
        # check its existence
        if os.path.isfile(filef):
            NOAA = pd.read_csv(filef, sep='\t')
        else:
            latN, lonN = support.SST_file_w(lat, lon, 1981, 2019,
                                            path=SSTpath, latlon=1)
            NOAA = pd.read_csv(filef, sep='\t')
        # filter only full years
        SST = NOAA[np.logical_and(NOAA['year'] > 1981,
                                  NOAA['year'] < 2019)].reset_index(drop=True)

        # annual statistics
        SSTy = SST.groupby([SST.year])['sst'].agg(['mean'])
        SSTy_stats = [SSTy.mean()['mean'], SSTy.std()['mean']]

        # monthly statistics
        SSTmy = SST.groupby([SST.month, SST.year])['sst'].agg(['mean', 'std'])
        SSTmy['anom'] = np.zeros(len(SSTmy))
        SSTmy['anom'] = SSTmy['mean'] - SSTy['mean']
        SSTmy_stats = SSTmy.groupby(level=0)['mean',
                                             'anom'].agg(['mean', 'std'])

        # daily statistics
        SSTymd = SST.groupby([SST.year,
                              SST.month,
                              SST.day])['sst'].agg(['mean'])
        SSTymd['anom'] = SSTymd - SSTy
        SSTymd['anom'] -= SSTmy_stats['anom', 'mean']

        stats = SSTymd.reset_index()
        stats = stats.groupby([stats.month,
                               stats.day])['anom'].agg(['mean', 'std'])

        lmts = np.array([(stats['mean'] - stats['std']).values,
                         (stats['mean'] + stats['std']).values])

        # # # plot area
        # # data
        t = np.arange(1, 367)

        # # figure
        fig, ax = plt.subplots(figsize=(figwidth, figheight))
        # zero line
        ax.plot([t.min(), t.max()], [0., 0.],
                color='gray', alpha=.5,
                linewidth=lw,
                label='_nolegend_')
        # plot data
        ax.fill_between(t, lmts[0, :], lmts[1, :],
                        color=rish, alpha=.2,
                        label=r'$\mu \pm \sigma$')
        ax.plot(t, stats['mean'],
                color=rish, alpha=.8,
                linewidth=lw,
                label=r'$\mu$')
        # axes labels
        ax.set_xlabel(r'day of the year')
        ax.set_ylabel(r'daily anomaly from monthly mean [${{}}^{{\circ}}$C]')
        # axes ticks
        # plot limits
        ax.set_xlim([1, 366])
        ax.set_ylim([-1.5, 1.5])
        # explaining lines / text / etc.
        if lat != 0:
            marker = np.array([-1.3, -1.5])
            Mar1 = 60
            Jun1 = 152
            Sep1 = 244
            Dec1 = 335
            ax.plot([Mar1, Mar1], marker,
                    color='black', alpha=1.,
                    linewidth=lw,
                    label='_nolegend_')
            ax.plot([Jun1, Jun1], marker,
                    color='black', alpha=1.,
                    linewidth=lw,
                    label='_nolegend_')
            ax.plot([Sep1, Sep1], marker,
                    color='black', alpha=1.,
                    linewidth=lw,
                    label='_nolegend_')
            ax.plot([Dec1, Dec1], marker,
                    color='black', alpha=1.,
                    linewidth=lw,
                    label='_nolegend_')
        if lat > 0:
            ax.text(np.mean([t.min(), Mar1]), marker.mean(),
                    r'Winter',
                    fontsize=fs,
                    va='center', ha='center')
            ax.text(np.mean([Mar1, Jun1]), marker.mean(),
                    r'Spring',
                    fontsize=fs,
                    va='center', ha='center')
            ax.text(np.mean([Jun1, Sep1]), marker.mean(),
                    r'Summer',
                    fontsize=fs,
                    va='center', ha='center')
            ax.text(np.mean([Sep1, Dec1]), marker.mean(),
                    r'Autumn',
                    fontsize=fs,
                    va='center', ha='center')
        elif lat < 0:
            ax.text(np.mean([t.min(), Mar1]), marker.mean(),
                    r'Summer',
                    fontsize=fs,
                    va='center', ha='center')
            ax.text(np.mean([Mar1, Jun1]), marker.mean(),
                    r'Autumn',
                    fontsize=fs,
                    va='center', ha='center')
            ax.text(np.mean([Jun1, Sep1]), marker.mean(),
                    r'Winter',
                    fontsize=fs,
                    va='center', ha='center')
            ax.text(np.mean([Sep1, Dec1]), marker.mean(),
                    r'Spring',
                    fontsize=fs,
                    va='center', ha='center')
        # legend / title
        ax.legend(loc='upper right',
                  fontsize=fs)
        ax.set_title(loc)

        # # save figure
        if saveSST1:
            figfile = figname + locs['fig'].iloc[i] + figext
            figfull = os.path.join(figpath, figfile)
            fig.savefig(figfull, dpi=figqual,
                        bbox_inches='tight')
            print('Figure saved as: {0}'.format(figfull))

# =============================================================================
# %% Construction of artificial SST time-series
# =============================================================================
if plotSST2 or saveSST2:
    figname = 'LayeredSST'
    Tstart = datetime.date(2000, 1, 1)
    T = 2.
    lat = -15.
    lon = 145.5
    SSTpath = 'SST_data'
    NOAAstart = 1982
    NOAAend = 2019
    # # NOAA dataset (1981 - 2019)
    # define file
    [latN, lonN], [_, _] = support.data_coor(lat, lon)
    file = ('SST_timeseries_lat{0}lon{1}_Y1981_2019.txt'
            .format(latN, lonN))
    filef = os.path.join(SSTpath, file)
    # check its existence
    if os.path.isfile(filef):
        NOAA = pd.read_csv(filef, sep='\t')
    else:
        latN, lonN = support.SST_file_w(lat, lon, 1981, 2019,
                                        path=SSTpath, latlon=1)
        NOAA = pd.read_csv(filef, sep='\t')

    SST = NOAA[np.logical_and(NOAA['year'] >= NOAAstart,
                              NOAA['year'] < NOAAend)].reset_index(drop=True)

    # # SST statistics
    # annual statistics
    y = SST.groupby([SST.year])['sst'].agg(['mean'])
    y_ms = [y.mean()['mean'], y.std()['mean']]
    # monthly statistics w.r.t. annual statistics - monthly anomalies
    ym = SST.groupby([SST.year, SST.month])['sst'].agg(['mean'])
    ym.reset_index(inplace=True)
    ym.rename(columns={'mean': 'sst'}, inplace=True)
    ym = ym.merge(y.reset_index())
    ym['anom'] = ym['sst'] - ym['mean']
    ym.drop(['mean'], axis=1, inplace=True)
    ym.rename(columns={'sst': 'mean'}, inplace=True)
    m_ms = ym.groupby([ym.month])['anom'].agg(['mean', 'std'])
    # daily statistics w.r.t. monthly statistics - daily anomalies
    ymd = SST.merge(ym.drop(['anom'], axis=1))
    ymd['anom'] = ymd['sst'] - ymd['mean']
    d_ms = ymd.groupby([ymd.month, ymd.day])['anom'].agg(['mean', 'std'])

    # # SST predictions
    # end of time-series
    Tend = Tstart + dateutil.relativedelta.relativedelta(years=T, days=-1)
    # framework
    data = pd.DataFrame({'date': pd.date_range(Tstart, Tend, freq='D'),
                         'year': pd.date_range(Tstart, Tend, freq='D').year,
                         'month': pd.date_range(Tstart, Tend, freq='D').month,
                         'day': pd.date_range(Tstart, Tend, freq='D').day,
                         'sst': 0.})
    # annual means
    SSTyear = np.random.normal(*y_ms, size=int(T))
    # monthly means
    SSTmonth = np.random.normal(m_ms['mean'].values,
                                m_ms['std'].values,
                                size=(int(T), 12)).flatten(order='C')

    # # from annual data to daily data
    data = pd.DataFrame({'year': np.arange(Tstart.year, Tend.year + 1),
                         'month': 1.,
                         'day': 1.,
                         'sst': SSTyear})
    # set dates as index
    data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
    data.set_index('date', inplace=True)
    # # annual > monthly
    # reindex and forward fill - months
    dates = pd.date_range(Tstart, Tend, freq='MS')
    dates.name = 'date'
    data = data.reindex(dates, method='ffill')
    data['month'] = data.index.month
    # add monthly anomalies
    data['sst_m'] = data['sst'] + SSTmonth
    # # monthly > daily
    # reindex and forward fill - days
    dates = pd.date_range(Tstart, Tend, freq='D')
    dates.name = 'date'
    data = data.reindex(dates, method='ffill')
    data['day'] = data.index.day
    # add daily anomalies
    data = data.merge(d_ms.reset_index())
    data['sst_d'] = data['sst_m'] + np.random.normal(data['mean'], data['std'])
    # reset index to 'date'
    data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
    data.set_index('date', inplace=True)
    data.drop(['mean', 'std'], axis=1, inplace=True)
    data.sort_index(inplace=True)

    # # figure
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    # zero-line
    # data
    ax.plot(data['sst'],
            color='black', alpha=.8,
            linestyle='dashed', linewidth=3*lw,
            label='annual')
    ax.plot(data['sst_m'],
            color='black', alpha=.8,
            linewidth=2*lw,
            label='monthly')
    ax.plot(data['sst_d'],
            color=rish, alpha=.6,
            linewidth=lw,
            label=r'daily')
    # axes labels
    ax.set_xlabel(r'date')
    ax.set_ylabel(r'SST [${{}}^{{\circ}}$C]')
    # axes ticks
    ticks = pd.date_range(Tstart,
                          Tend + dateutil.relativedelta.relativedelta(days=1),
                          freq='YS')
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks.year)
    # plot limits
    ax.set_xlim([ticks.min(), ticks.max()])
    # explaining lines / texts / etc.
    # legend / title
    ax.legend(loc='upper right',
              fontsize=fs)

    # # save figure
    if saveSST2:
        figfile = figname + figext
        figfull = os.path.join(figpath, figfile)
        fig.savefig(figfull, dpi=figqual,
                    bbox_inches='tight')
        print('Figure saved as: {0}'.format(figfull))

# =============================================================================
# %% Comparison with NOAA data set
# =============================================================================
if plotSST3 or saveSST3:
    figname = 'NOAAvsTS_SST'
    SSTpath = 'SST_data'
    # DD coordinates
    lat = -15.
    lon = 145.5
    # time specifications
    Tstart = datetime.date(2000, 1, 1)
    T = 10.

    # # NOAA dataset (1981 - 2019)
    # define file
    [latN, lonN], [_, _] = support.data_coor(lat, lon)
    file = ('SST_timeseries_lat{0}lon{1}_Y1981_2019.txt'
            .format(latN, lonN))
    filef = os.path.join(SSTpath, file)
    # check its existence
    if os.path.isfile(filef):
        NOAA = pd.read_csv(filef, sep='\t')
    else:
        latN, lonN = support.SST_file_w(lat, lon, 1981, 2019,
                                        path=SSTpath, latlon=1)
        NOAA = pd.read_csv(filef, sep='\t')
    # 'date' as index
    NOAA['date'] = pd.to_datetime(NOAA[['year', 'month', 'day']])
    NOAA.set_index('date', inplace=True)
    # filter data of interest
    Tend = Tstart + dateutil.relativedelta.relativedelta(years=T, days=-1)
    NOAA = NOAA[np.logical_and(NOAA.index.date >= Tstart,
                               NOAA.index.date <= Tend)]

    # # artificial time-series
    SST = TimeSeries.TS_SST(lat, lon, T, Tstart, SSTpath=SSTpath)

    # # plot both time-series
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    # zero-line
    # plot data
    ax.plot(NOAA.index, NOAA['sst'],
            color='black', alpha=.8,
            linewidth=lw,
            label=r'NOAA')
    ax.plot(SST.index, SST['sst'],
            color=rish, alpha=.5,
            linewidth=lw,
            label=r'$\widetilde{{T}}$')
    # axes labels
    ax.set_xlabel(r'year')
    ax.set_ylabel(r'SST [${{}}^{{\circ}}$C]')
    # axes ticks
    ticks = pd.date_range(Tstart,
                          Tend + dateutil.relativedelta.relativedelta(days=1),
                          freq='YS')
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks.year)
    # plot limits
    ax.set_xlim([ticks.min(), ticks.max()])
    # explaining lines / texts / etc.
    # legend / title
    ax.legend(loc='upper right',
              fontsize=fs)

    # # save figure
    if saveSST3:
        figfile = figname + figext
        figfull = os.path.join(figpath, figfile)
        fig.savefig(figfull, dpi=figqual,
                    bbox_inches='tight')
        print('Figure saved as: {0}'.format(figfull))
