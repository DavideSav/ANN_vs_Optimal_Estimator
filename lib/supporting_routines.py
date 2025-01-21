from copy import deepcopy

import pandas as pn
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import scipy.special
import numpy.linalg as LA


#import pamtra2

niceKeys = {
    'Nw_log10': 'log$_{10}$N$_w$',
    'Dm_log10': 'log$_{10}$D$_m$',
    'Sm_log10': 'log$_{10}\sigma_m$',
    'Nw': 'N$_w$',
    'Dm': 'D$_m$',
    'Sm': '$\sigma_m$',
    'Smprime': "$\sigma_m\!'$",
    'Sm_prime': "$\sigma_m\!'$",
    'Smprime_log10': "log$_{10}\sigma_m\!'$",
    'PCS0': 'PCS 0',
    'PCS1': 'PCS 1',
    'PCS2': 'PCS 2',
}
niceKeysSimple = {
    'Nw': 'N$_w$',
    'Dm': 'D$_m$',
    'Sm': "$\sigma_m$ or $\sigma_m\!'$",
    'Smprime': "$\sigma_m\!'$",
    'PCS0': 'PCS 0',
    'PCS1': 'PCS 1',
    'PCS2': 'PCS 2',
}
niceRuns = {
    'Sm': 'log$_{10}$',
    'SmLin': "linear",
    'Smprime': "linear with $\sigma_m\!'$",
    'SmprimeLog10': "log$_{10}$ with $\sigma_m\!'$",
    'PCS': 'PCS',
}
niceRetrievals = {
    'Z': "$Z_e$ retrieval",
    'ZW': "$Z_e$, $V_d$ retrieval",
    'Zdual': "dual $Z_e$ retrieval",
    'ZWdual': "dual $Z_e$, $V_d$ retrieval",

}


def plotCorrelation(cov, fig, sp, tickLabels=None, isCov=True, cmap='viridis_r'):

    std = pn.Series(np.sqrt(np.diag(cov)), index=cov.index)

    if isCov:
        cor = deepcopy(cov)
        cor[:] = 0
        for xx in cov.index:
            for yy in cov.index:
                cor[xx][yy] = cov[xx][yy] / (std[xx] * std[yy])
    else:
        cor = cov

    sp.set_aspect('equal')

    ind_array = np.arange(cor.shape[0])

    x, y = np.meshgrid(ind_array, ind_array)
    for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
        if x_val < y_val:
            c = "%.g" % cov.iloc[x_val, y_val]
            sp.text(
                x_val + 0.5,
                y_val + 0.5,
                c,
                va='center',
                ha='center',
                fontsize=9)

        if x_val > y_val:
            c = "%.g" % cor.iloc[x_val, y_val]
            sp.text(
                x_val + 0.5,
                y_val + 0.5,
                c,
                va='center',
                ha='center',
                color='w',
                fontsize=9)
            cor.iloc[x_val, y_val] = -1

        if x_val == y_val:
            c = "%.g" % cov.iloc[x_val, y_val]
            sp.text(
                x_val + 0.5,
                y_val + 0.5,
                c,
                va='center',
                ha='center',
                color='k',
                fontsize=9)
            cor.iloc[x_val, y_val] = -1

    if tickLabels != None:
        labels = []
        for ii in range(cor.shape[0]):
            labels.append(tickLabels[cor.index[ii]].split(' [')[0])
    else:
        labels = cov.index
    cor_values = np.ma.masked_equal(cor.values, -1)
    pc = sp.pcolormesh(cor_values, vmin=-1, vmax=1, cmap=cmap)
    sp.tick_params(axis=u'both', which=u'both', length=0)
    sp.set_xticks(np.arange(len(labels)) + 0.5)
    sp.set_xticklabels(labels, rotation=90)
    sp.set_yticks(np.arange(len(labels)) + 0.5)
    sp.set_yticklabels(labels)
    sp.set_xlim(0, len(std))
    sp.set_ylim(0, len(std))
    return pc


def normalizedDSD(D, Nw, Dm, mu):

    fmu = (6*(4 + mu)**(mu + 4)) / (4**4 * scipy.special.gamma(mu + 4))
    N = Nw * fmu * ((D/Dm)**mu) * np.exp(-(mu+4) * (D/Dm))
    return N


def normalizedDSD_sigma_prime(D, Nw, Dm, sigma_prime):

    bm = 1.36
    mu = Dm**(2-2*bm)/sigma_prime**2 - 4  # eq. 25 w14
#     print(mu)
    return normalizedDSD(D, Nw, Dm, mu)


def normalizedDSD_sigma(D, Nw, Dm, sigma):

    mu = (Dm/sigma)**2 - 4  # eq 18 w14
#     print(mu)
    return normalizedDSD(D, Nw, Dm, mu)


def normalizedDSD4Pamtra(sizeCenter, sizeBoundsWidth, Nw, Dm, mu):
    sizeCenter = sizeCenter*1000.

    fmu = (6*(4 + mu)**(mu + 4)) / (4**4 * scipy.special.gamma(mu + 4))
    N = Nw * fmu * ((sizeCenter/Dm)**mu) * np.exp(-(mu+4) * (sizeCenter/Dm))

    N = N*1000

    return N * sizeBoundsWidth


#


def splitTQ(x):
    t_index = [i for i in x.index if i.endswith('t')]
    q_index = [i for i in x.index if i.endswith('q')]
    h_index = [float(i.split('_')[0]) for i in x.index if i.endswith('q')]

    assert len(t_index) == len(q_index)
    assert len(t_index) == len(h_index)
    assert len(t_index)*2 == len(x)

    xt = x[t_index]
    xt.index = h_index

    xq = x[q_index]
    xq.index = h_index

    xt.index.name = 'height'
    xq.index.name = 'height'

    return xt, xq


def plotMwrResults(oe1, title=None, oe2=None, title2=None, oe3=None, title3=None, h=None, hlabel='Height [m]', xlimT=(None, None), xlimH=(None, None)):

    if oe2 is None:
        gridspec = dict(wspace=0.0)
        fig, (axA, axB) = plt.subplots(ncols=2, sharey=True,
                                       gridspec_kw=gridspec, figsize=[5.0, 4.0])
        vals = [oe1], [axA], [axB], [title]
    elif oe3 is None:

        gridspec = dict(wspace=0.0, width_ratios=[1, 1, 0.25, 1, 1])
        fig, (axA, axB, ax0, axC, axD) = plt.subplots(
            ncols=5, sharey=True, figsize=[10.0, 4.0], gridspec_kw=gridspec)
        vals = [oe1, oe2], [axA, axC], [axB, axD], [title, title2]
        ax0.set_visible(False)
    else:

        gridspec = dict(wspace=0.0, width_ratios=[1, 1, 0.1, 1, 1, 0.1, 1, 1])
        fig, (axA, axB, ax0, axC, axD, ax1, axE, axF) = plt.subplots(
            ncols=8, sharey=True, figsize=[12.0, 4.0], gridspec_kw=gridspec)
        vals = [oe1, oe2, oe3], [axA, axC, axE], [
            axB, axD, axF], [title, title2, title3]
        ax0.set_visible(False)
        ax1.set_visible(False)

    for oe, ax1, ax2, tit in zip(*vals):

        t_op, q_op = splitTQ(oe.x_op)
        t_op_err, q_op_err = splitTQ(oe.x_op_err)
        t_a, q_a = splitTQ(oe.x_a)
        t_a_err, q_a_err = splitTQ(oe.x_a_err)
        t_truth, q_truth = splitTQ(oe.x_truth)

        nProf = len(t_op)

        if h is None:
            hvar = t_op.index
        else:
            hvar = h

        ax1.plot(t_op, hvar, color='C0', label='Optimal')
        ax1.fill_betweenx(hvar, t_op+t_op_err, t_op-t_op_err,
                          color='C0', alpha=0.2)

        ax1.plot(t_a, hvar, color='C1', label='Prior')
        ax1.fill_betweenx(hvar, t_a+t_a_err, t_a-t_a_err,
                          color='C1', alpha=0.2)
        ax1.plot(t_truth, hvar, color='C2', label='Truth')

        ax2.plot(q_op, hvar, color='C0')
        ax2.fill_betweenx(hvar, q_op+q_op_err, q_op-q_op_err,
                          color='C0', alpha=0.2)

        ax2.plot(q_a, hvar, color='C1')
        ax2.fill_betweenx(hvar, q_a+q_a_err, q_a-q_a_err,
                          color='C1', alpha=0.2)
        ax2.plot(q_truth, hvar, color='C2')

        ax1.set_xlabel('Temperature [K]')
        ax2.set_xlabel('Specific humidity\n[log$_{10}$(g/kg)]')

        ax1.set_xlim(xlimT)
        ax2.set_xlim(xlimH)

        ax1.set_title(tit, loc='left')

    if h is not None:
        axA.invert_yaxis()

    axA.set_ylabel(hlabel)

    axA.legend(loc='upper right')

    return fig


def q2a(q, p, T):
    '''
    specific to absolute humidty
    '''
    Rair = 287.04  # J/kg/K
    Rvapor = 461.5  # J/kg/K
    rho = p / (Rair * T * (1 + (Rvapor / Rair - 1) * q))  # density kg/m3
    return q*rho


def print_mwr_rms(oe):
    T_optimal, Q_optimal = splitTQ(oe.x_op)
    T_truth, Q_truth = splitTQ(oe.x_truth)

    print('RMS X Temperature: %g [K]' %
          np.sqrt(np.mean((T_optimal - T_truth)**2)))
    print('RMS X Humidity: %g [log$_{10}$(g/kg)]' %
          np.sqrt(np.mean((10**Q_optimal - 10**Q_truth)**2)))
    print('RMS Y %g [K]' % np.sqrt(np.mean((oe.y_obs - oe.y_op)**2)))


def plot_uncertainty_dof(oe1, oe2, label2, pressure, oe3=None, label3=None):

    fig, (axA, axB) = plt.subplots(ncols=2, sharey=True, figsize=(6, 4))

    T, Q = splitTQ(oe1.x_op_err / oe1.x_a_err)
    T_2, Q_2 = splitTQ(oe2.x_op_err / oe2.x_a_err)

    axA.plot(T*100, pressure, color='C2', label='Temperature')
    axA.plot(
        T_2*100,
        pressure,
        color='C2',
        ls='-.',
        label=''
    )

    axA.plot(
        Q*100,
        pressure,
        color='C3',
        label='Specific humidity')
    axA.plot(
        Q_2*100,
        pressure,
        color='C3',
        ls='-.',
        label=''
    )

    if oe3 is not None:
        T_3, Q_3 = splitTQ(oe3.x_op_err / oe3.x_a_err)
        axA.plot(
            T_3 * 100,
            pressure,
            color='C2',
            ls=':',
            label=''
        )
        axA.plot(
            Q_3 * 100,
            pressure,
            color='C3',
            ls=':',
            label=''
        )

    T, Q = [(x) for x in splitTQ(oe1.dgf_x)]
    T_2, Q_2 = [(x) for x in splitTQ(oe2.dgf_x)]

    axB.plot(T, pressure, color='C2', label='Temperature')
    axB.plot(
        T_2,
        pressure,
        color='C2',
        ls='-.',
        label=label2)

    axB.plot(
        Q,
        pressure,
        color='C3',
        label='Specific humidity')
    axB.plot(
        Q_2,
        pressure,
        color='C3',
        ls='-.',
        label=label2)
    if oe3 is not None:
        T_3, Q_3 = [(x) for x in splitTQ(oe3.dgf_x)]
        axB.plot(
            T_3,
            pressure,
            color='C2',
            ls=':',
            label=label3)

        axB.plot(
            Q_3,
            pressure,
            color='C3',
            ls=':',
            label=label3)

    axA.legend(frameon=False, loc='upper left')
#     axB.legend(frameon=False)

    lines = [matplotlib.lines.Line2D(
        [0], [0], color='gray', linestyle=ls) for ls in ['-', '-.', ':']]
    labels = ['reference', label2, label3]
    if oe3 is None:
        lines.pop()
        labels.pop()

    axB.legend(lines, labels, frameon=False, loc='upper right')

    axB.set_xlabel('Degrees of freedom [-]')
    axA.set_xlabel('Optimal to prior uncertainty [%]')

    axA.set_ylabel('Pressure [hPa]')
    axA.invert_yaxis()
    axA.set_xlim(-5, 110)

    axA.text(0.99, 0.99,
             'a)',
             horizontalalignment='right',
             verticalalignment='top',
             transform=axA.transAxes
             )
    axB.text(0.99, 0.99,
             'b)',
             horizontalalignment='right',
             verticalalignment='top',
             transform=axB.transAxes
             )
    fig.subplots_adjust(wspace=0.05)

    return fig
