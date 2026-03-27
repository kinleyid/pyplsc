
import pytest
import pyplsc
import numpy as np

@pytest.fixture
def sample_data():
    np.random.seed(123)
    data = np.random.normal(size=(8, 2))
    covariates = np.random.normal(size=(8, 2))
    between = np.array([0]*4 + [1]*4)
    within = np.array([0, 1]*4)
    participant = np.array(np.cumsum([1, 0]*4))
    return data, covariates, between, within, participant

def test_bda_basic(sample_data):
    # Simple testing of model fitting
    data, _, between, within, participant = sample_data
    bda = pyplsc.BDA()
    bda.fit(X=data, between=between, within=within, participant=participant)
    bda.permute(n_perm=2)
    bda.bootstrap(n_boot=2)

def test_bda_rng(sample_data):
    # Test random seeding
    data, _, between, within, participant = sample_data
    bda = pyplsc.BDA(random_state=123)
    bda.fit(X=data, between=between, within=within, participant=participant)
    bda.permute(n_perm=100)
    pvals_1 = bda.pvals_
    bda.permute(n_perm=100)
    pvals_2 = bda.pvals_
    assert all(np.isclose(pvals_1, pvals_2))

def test_input_validation():
    # a
    pass

def test_plsc_basic(sample_data):
    data, covariates, between, within, participant = sample_data
    plsc = pyplsc.PLSC()
    plsc.fit(X=data, covariates=covariates, between=between, within=within, participant=participant)
    plsc.permute(n_perm=2)
    plsc.bootstrap(n_boot=2)