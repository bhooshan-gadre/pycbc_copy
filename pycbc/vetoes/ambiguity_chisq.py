import numpy as np
import itertools
import copy
import logging

import pycbc.pnutils as pnu
import pycbc.noise as pn
import pycbc.filter as pf
import pycbc.psd as pp
import pycbc.noise as pn
import pycbc.waveform as pw
import pycbc.types as pt



def segment_snrs(filters, stilde, psd, flow, fhigh):
    """ This functions calculates the snr of each bank veto template against
    the segment
    Parameters
    ----------
    filters: list of FrequencySeries
        The list of bank veto templates filters.
    stilde: FrequencySeries
        The current segment of data.
    psd: FrequencySeries
    low_frequency_cutoff: float
    Returns
    -------
    snr (list): List of snr time series.
    """
    snrs = []

    for template in filters:
        snr = pf.matched_filter(template, stilde, None, flow, fhigh, pf.sigmasq(template, psd, flow, fhigh))
        snrs.append(snr)
    return snrs

def simple_inner(htilde, stilde, psd, flow, fhigh):
    kmin, kmax = pf.get_cutoff_indices(flow, fhigh, psd.delta_f, 2*len(psd)+1)
    indices = slice(kmin, kmax)
    norm = pf.sigma(htilde, psd, flow, fhigh) * pf.sigma(stilde, psd, flow, fhigh)
    return (np.conjugate(htilde.data[indices])*stilde.data[indices]/psd.data[indices]).sum()*4.0*psd.delta_f / norm

def get_cov_gg(filters, psd, flow, fhigh):
    len_filters = len(filters)
    cov = np.zeros((len_filters, len_filters))
    for i, j in itertools.combinations_with_replacement(range(len_filters), 2):
        cov[i, j] = simple_inner(filters[i], filters[j], psd, flow, fhigh).real
        cov[j,i] = simple_inner(filters[j], filters[i], psd, flow, fhigh).real
        # cov[j,i] = cov[i, j]
    return cov

def get_cov_gh(filters, htilde, psd, flow, fhigh):
    return [simple_inner(htilde, stilde, psd, flow, fhigh) for stilde in filters]

def get_cov_matrix(snr, cov_gg, cov_gh):
    cov = np.zeros_like(cov_gg)
    for i, j in itertools.combinations_with_replacement(range(len(cov_gh)), 2):
        cov[i, j] = cov_gg[i, j] - (snr.real*cov_gh[i].real + snr.imag*cov_gh[i].imag)*(snr.real*cov_gh[j].real + snr.imag*cov_gh[j].imag)
        cov[j,i] = cov[i, j]
    return cov

def get_eval_rot_mat(cov):
    eig, evec = np.linalg.eig(cov)
    sort = eig.argsort()
    return eig[sort], evec[sort, :]

def get_vector(snr, snr_id, seg_snrs, cov_gh):
    vec = np.zeros(len(cov_gh))
    for i in range(len(cov_gh)):
        vec[i] = seg_snrs[i][snr_id].real - snr.real*cov_gh[i].real - snr.imag*cov_gh[i].imag
    return vec

def get_chisq(vec, eig, rot_mat, threshold=1e-2):
    ## Variaous conditions to impose sane chisq and DoF
    # rel_eig = eig/eig[-1] > threshold
    rel_eig = eig/eig[-1] > 1.0/15.0
    # rel_eig = eig > threshold
    dof = rel_eig.sum()
    rot_vec = np.dot(vec, rot_mat)
    chi =(rot_vec[rel_eig]**2.0/eig[rel_eig]).sum()

    if chi / dof > 4:
        print("eig for chi = {}".format(chi/dof))
        print(eig[rel_eig])
        print('min, max, ratio')
        print(eig[rel_eig][0], eig[rel_eig][-1], eig[rel_eig][-1]/eig[rel_eig][0])

    return chi, dof

def compute_chisq(snrs, snr_ids, seg_snrs, cov_gg, cov_gh, threshold=1e-3):
    chis, dofs = [], []
    for snr, snr_id in zip(snrs, snr_ids):
        cov = get_cov_matrix(snr, cov_gg, cov_gh)
#         print "cov:"
#         print cov
        eig, rot_mat = get_eval_rot_mat(cov)
#         print "eig:"
#         print eig
        vec = get_vector(snr, snr_id, seg_snrs, cov_gh)
        chi, dof = get_chisq(vec, eig, rot_mat, threshold)
        chis.append(chi)
        dofs.append(dof)

    return np.array(chis), np.array(dofs)


_snr = None
def inner(vec1, vec2, psd=None,
          low_frequency_cutoff=None, high_frequency_cutoff=None,
          v1_norm=None, v2_norm=None):

    htilde = pf.make_frequency_series(vec1)
    stilde = pf.make_frequency_series(vec2)

    N = (len(htilde)-1) * 2

    global _snr
    _snr = None
    if _snr is None or _snr.dtype != htilde.dtype or len(_snr) != N:
        _snr = pt.zeros(N,dtype=pt.complex_same_precision_as(vec1))
        snr, corr, snr_norm = pf.matched_filter_core(htilde,stilde,psd,low_frequency_cutoff,
                                                  high_frequency_cutoff, v1_norm, out=_snr)
    if v2_norm is None:
        v2_norm = pf.sigmasq(stilde, psd, low_frequency_cutoff, high_frequency_cutoff)

    snr.data = snr.data * snr_norm / np.sqrt(v2_norm)

    return snr

class SingleDetAmbiguityChisq(object):
    returns = {'ambiguity_chisq': np.float32, 'ambiguity_chisq_dof' : np.int}
    def __init__(self, status, snr_threshold, get_relevant_filters, flow, fmax, time_indices=[0], condition_threshold=1.0e-2):
        if status:
            self.do = True

            self.column_name = "ambiguity_chisq"
            self.table_dof_name = "ambiguity_chisq_dof"
            self.snr_threshold = snr_threshold
            self.condition_threshold = condition_threshold
            self.flow = flow
            self.fmax = fmax
            self.get_filters = get_relevant_filters
            self.time_idices = np.array(time_indices) # Currently only trigger time = 0 indices are supported
            self._cache_seg_snrs = {} # seg_snrs
            self._cache_filter_matches = {} # (g, g') matches between filters # Can be time series in future due to 'time_indices'
            self._cache_template_filter_matches = {} # (h, g) matches between template and filter # Can be time series in future 'time_indices'

        else:
            self.do = False

    def cache_seg_snrs(self, htilde, stilde, psd):  # It is assumed ``filters iff htilde''
        key = (id(htilde), id(stilde), id(psd)) # key is always: template, data segment, psd
        if key not in self._cache_seg_snrs:
            filters = self.get_filters(htilde)
            self._cache_seg_snrs[key] = segment_snrs(filters, stilde, psd, self.flow, self.fmax)  ## Assumed data is overwhitened
        return self._cache_seg_snrs[key]

    ## This can be made better by actually caching match between commom filters
    ### But we are using 'id', filters may not live long enough if memory problem. Need to check what happens
    def cache_filter_matches(self, htilde, psd):
        key = (id(htilde), id(psd))
        if key not in self._cache_seg_snrs:
            filters = self.get_filters(htilde)
            self._cache_filter_matches[key] = get_cov_gg(filters, psd, self.flow, self.fmax)
        return self._cache_filter_matches[key]

    def cache_template_filter_matches(self, htilde, psd):
        key = (id(htilde), id(psd))
        if key not in self._cache_template_filter_matches:
            filters = self.get_filters(htilde)
            self._cache_template_filter_matches[key] = get_cov_gh(filters, htilde, psd, self.flow, self.fmax)
        return self._cache_template_filter_matches[key]

    def values(self, snrs, snr_ids, htilde, stilde, psd, snr_threshold=None, condition_threshold=None):
        if self.do:
            logging.info('Ambiguity chi-squares are on! ...')

            thre = snr_threshold if snr_threshold else self.snr_threshold
            logging.info('Gathering relevant triggers')
            if thre:
                rel_ids = np.abs(snrs) > thre
            else:
                rel_ids = np.repeat(True, len(snrs))

            seg_snrs = self.cache_seg_snrs(htilde, stilde, psd)
            cov_gg = self.cache_filter_matches(htilde, psd)
            cov_gh = self.cache_template_filter_matches(htilde, psd)

            cond_thre = condition_threshold if condition_threshold else self.condition_threshold
            logging.info('Computing ambiguity chi-squares ...')
            chisq, dof = compute_chisq(snrs[rel_ids], snr_ids[rel_ids], seg_snrs, cov_gg, cov_gh, cond_thre)

            return chisq/dof, dof
        else:
            return None, None


class filters_for_template(object):
    def __init__(self, bank, min_filters=10, max_filters=20, nudge=0.2):
        self.bank = bank
        bank_tau0, bank_tau3 = pnu.mass1_mass2_to_tau0_tau3(bank.table['mass1'], bank.table['mass2'], bank.table['f_lower'])
        self.bank_tau = bank_tau0 - bank_tau3
        self.min_filters = min_filters
        self.max_filters = max_filters
        self.nudge = nudge
        self._cache_relevant_filters = {}

    def get_relevant_filters(self, htilde, tau0=None, tau3=None, delta_region=1e-3):
        logging.info("Getting relevant templates")
        key = id(htilde)
        if key not in self._cache_relevant_filters:
            try:
                trig_tau0, trig_tau3 = pnu.mass1_mass2_to_tau0_tau3(htilde.params.mass1, htilde.params.mass2, htilde.params.f_lower)
                tau = trig_tau0 - trig_tau3
            except:
                tau = tau0 - tau3

            err = 1e-5
            idx = (np.abs(self.bank_tau - tau) <= delta_region) * (np.abs(self.bank_tau - tau) > err)
            while (idx.sum() > self.max_filters) + (idx.sum() < self.min_filters):
                if idx.sum() > self.max_filters:
                    delta_region *= (1-self.nudge)
                    idx = (np.abs(self.bank_tau - tau) <= delta_region) * (np.abs(self.bank_tau - tau) > err)
                elif idx.sum() < self.min_filters:
                    delta_region *= (1.0 + self.nudge)
                    idx = (np.abs(self.bank_tau - tau) <= delta_region) * (np.abs(self.bank_tau - tau) > err)

            filter_list = copy.copy(self.bank)
            filter_list.table = filter_list.table[idx]
            self._cache_relevant_filters[key] = filter_list # list of filters to be used in chisq
        logging.info("Getting relevant templates... Done!")
        return self._cache_relevant_filters[key]

    def clear_filters(self, htilde):
        key = id(htilde)
        if key in self._cache_relevant_filters:
            del self._cache_relevant_filters[key]
