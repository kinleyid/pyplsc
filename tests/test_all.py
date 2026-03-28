
import pytest
import pyplsc
import numpy as np

from pdb import set_trace

@pytest.fixture
def sample_data():
    np.random.seed(123)
    data = np.random.normal(size=(8, 2))
    covariates = np.random.normal(size=(8, 2))
    between = np.array([0]*4 + [1]*4)
    within = np.array([0, 1]*4)
    participant = np.array(np.cumsum([1, 0]*4))
    return data, covariates, between, within, participant

@pytest.fixture
def fit_bda(sample_data):
    data, _, between, within, participant = sample_data
    bda = pyplsc.BDA(random_state=123)
    bda.fit(X=data, between=between, within=within, participant=participant)
    return bda

def test_bda_basic(fit_bda):
    # Simple testing of model fitting
    bda = fit_bda
    assert len(bda.labels_) == len(bda.design_sals_)
    bda.permute(n_perm=20)
    bda.bootstrap(n_boot=200)
    yerr = bda.get_design_yerr(0)
    assert (yerr >= 0).all()
    assert yerr.shape[0] == 2
    with pytest.raises(Exception):
        yerr = bda.get_design_yerr([0, 1])
    bda.transform(lv_idx=0)
    bda.transform_design(lv_idx=0)
    with pytest.raises(Exception):
        bda.permute(0)
    with pytest.raises(Exception):
        bda.bootstrap(0)    
    bda.bootstrap(n_boot=2, alignment_method='flip')
    
def test_warnings(sample_data):
    data, _, between, within, participant = sample_data
    bda = pyplsc.BDA(pre_subtract='between')
    with pytest.raises(Warning):
        bda.fit(data, between=between)
    bda = pyplsc.BDA(pre_subtract='within')
    with pytest.raises(Warning):
        bda.fit(data, within=within, participant=participant)

def test_errors(sample_data):
    data, _, between, within, participant = sample_data
    bda = pyplsc.BDA()
    with pytest.raises(Exception):
        # Nothing to stratify observations
        bda.fit(data)
    with pytest.raises(Exception):
        # Within condition without a way to differentiate participants
        bda.fit(data, within=within)
    with pytest.raises(Exception):
        # Different length of condition indicator vs data
        bda.fit(data, between=between[:(len(data) - 1)])
    with pytest.raises(Exception):
        # Nothing to stratify observations
        bda.fit(data, between=[0]*len(data))
    with pytest.raises(Exception):
        # yerr without having resampled
        bda.get_design_yerr(0)
    with pytest.raises(Exception):
        # yerr without having resampled
        bda.pre_subtract = 'within'
        bda.fit(data, between=between)
    with pytest.raises(Exception):
        # yerr without having resampled
        bda.pre_subtract = 'between'
        bda.fit(data, within=within, participant=participant)

def test_flips(fit_bda):
    fit_bda.bootstrap(n_boot=2)
    sals_1 = fit_bda.brain_sals_[:, 0].copy()
    recon_1 = fit_bda.design_sals_ @ fit_bda.brain_sals_.T
    fit_bda.flip_signs(0)
    sals_2 = fit_bda.brain_sals_[:, 0]
    recon_2 = fit_bda.design_sals_ @ fit_bda.brain_sals_.T
    assert np.isclose(sals_1, -sals_2).all()
    assert np.isclose(recon_1, recon_2).all()

def test_bda_rng(fit_bda):
    # Test random seeding
    bda = fit_bda
    bda.permute(n_perm=20)
    pvals_1 = bda.pvals_
    bda.permute(n_perm=20)
    pvals_2 = bda.pvals_
    assert all(np.isclose(pvals_1, pvals_2))

def test_bda_pre_subtract(sample_data):
    data, covariates, between, within, participant = sample_data
    bda = pyplsc.BDA(pre_subtract='between')
    bda.fit(X=data, between=between, within=within, participant=participant)
    bda = pyplsc.BDA(pre_subtract='within')
    bda.fit(X=data, between=between, within=within, participant=participant)

def test_plsc_basic(sample_data):
    data, covariates, between, within, participant = sample_data
    plsc = pyplsc.PLSC()
    plsc.fit(X=data, covariates=covariates, between=between, within=within, participant=participant)
    assert len(plsc.labels_) == len(plsc.design_sals_)
    plsc.permute(n_perm=2)
    plsc.bootstrap(n_boot=2)
    plsc.transform(lv_idx=0)
    plsc.transform_design(lv_idx=0)