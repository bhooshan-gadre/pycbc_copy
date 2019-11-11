#!/usr/bin/env python
# coding: utf-8

import numpy as np
import h5py

import pycbc.types as pt
import pycbc.waveform as pw
import pycbc.filter as _filter

def waveform_basis(hp, hc, psd=None, flow=None, ffinal=None):
    '''Given (hp, hc), the function gives orthonormal basis vectors.
    From appendix B (eqns B3, B4) of
    https://journals.aps.org/prd/pdf/10.1103/PhysRevD.95.024010
    '''
    hptilde = _filter.make_frequency_series(hp)
    hctilde = _filter.make_frequency_series(hc)
    sig1 = _filter.sigma(hp, psd, flow, ffinal)
    sig2 = _filter.sigma(hc, psd, flow, ffinal)

    proj = simple_inner(hptilde, hctilde, psd, flow, ffinal, sig1, sig2).real

    if isinstance(hp, pt.TimeSeries):
        hpp = pt.TimeSeries(hp.data/sig1, delta_t=hc.delta_t, epoch=hc._epoch)
        hper = pt.TimeSeries(hc.data/sig2, delta_t=hc.delta_t, epoch=hc._epoch)
    elif isinstance(hp, pt.FrequencySeries):
        hpp = pt.FrequencySeries(hp.data/sig1, delta_f=hc.delta_f, epoch=hc._epoch)
        hper = pt.FrequencySeries(hc.data/sig2, delta_f=hc.delta_f, epoch=hc._epoch)

    hper.data = (hper.data - proj*hpp.data)/np.sqrt(1-proj*proj)
    hper.data /= _filter.sigma(hper, psd, flow, ffinal)

    return hpp, hper

def simple_inner(htilde, stilde, psd=None, flow=None, fhigh=None, norm1=None, norm2=None):
    kmin, kmax = _filter.get_cutoff_indices(flow, fhigh, htilde.delta_f, (len(htilde)-1) * 2)
    indices = slice(kmin, kmax)
    if norm1 and norm2:
        norm = norm1*norm2
    else:
        norm1 = _filter.sigma(htilde, psd, flow, fhigh)
        norm2 = _filter.sigma(stilde, psd, flow, fhigh)
        norm = norm1*norm2

    if psd:
        return (np.conjugate(htilde.data[indices])*stilde.data[indices]/psd.data[indices]).sum()*4.0*psd.delta_f / norm
    else:
        return (np.conjugate(htilde.data[indices])*stilde.data[indices]).sum()*4.0*htilde.delta_f / norm

def minmax_match_with_basis(basis1, basis2, psd=None, flow=None, fhigh=None, norm1=None, norm2=None):
    '''Return phases min-max matches from orthonormal bases.
    From appendix B (eqns B10 - B14) of
    https://journals.aps.org/prd/pdf/10.1103/PhysRevD.95.024010
    '''
    assert len(basis1[0]) == len(basis2[0]), "Length of both basis do not match"
    match11 = _filter.matched_filter(basis1[0], basis2[0], psd, flow, fhigh, norm1)
    match12 = _filter.matched_filter(basis1[0], basis2[1], psd, flow, fhigh, norm2)
    match21 = _filter.matched_filter(basis1[1], basis2[0], psd, flow, fhigh, norm1)
    match22 = _filter.matched_filter(basis1[1], basis2[1], psd, flow, fhigh, norm2)

    a = match11.real().data*match11.real().data + match21.real().data*match21.real().data

    b = match12.real().data*match12.real().data + match22.real().data*match22.real().data

    c = match11.real().data*match12.real().data + match21.real().data*match22.real().data

    delta = np.sqrt((a - b)*(a - b) + 4*c*c)
    min_max = np.sqrt((a+b-delta)/2.0)
    max_max = np.sqrt((a+b+delta)/2.0)
    return min_max, max_max

# Example
params = dict(mass1=50, mass2=5,
              f_lower=20, delta_f=1./64., f_final=1024.,
              spin1x=0.2, spin1y=0.3, spin1z=-0.6,
              spin2x=0., spin2y=-0.45, spin2z=0.4,
              inclination=np.pi/2.,
              approximant='SEOBNRv4P', )
hp, hc = pw.get_td_waveform(**params)

params['mass1'] = 50.1
params['mass2'] = 4.9
hp1, hc1 = pw.get_fd_waveform(**params)

basis = waveform_basis(hp, hc, None) #, 20, 1024)
basis1 = waveform_basis(hp1, hc1, None) #, 20, 1024)
min_max, max_max = minmax_match_with_basis(basis, basis1, None) #, 20, 1024)

print('Time maxed min_max match with delta-t_c:')
print(min_max.max(), min_max.argmax()*hp.delta_t)
print('Time maxed min_max match with delta-t_c:')
print(min_max.max(), min_max.argmax()*hp.delta_t)

