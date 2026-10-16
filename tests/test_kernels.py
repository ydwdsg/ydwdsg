import os
import numpy as np
import pytest
from phono3py_cupy_accelerator import contract, compare, FourierBatch
from phono3py_cupy_accelerator.fixtures import contraction_fixture, toy_interaction
from phono3py_cupy_accelerator.reference import contract_reference
from phono3py_cupy_accelerator.phono3py_adapter import evaluate_triplet, native_triplet


@pytest.mark.parametrize("n", [1,2,3])
@pytest.mark.parametrize("seed", [17,29])
def test_contraction_independent_oracle(n, seed):
    args = contraction_fixture(n, seed)
    saved = args["eigenvectors"].copy()
    args["frequencies"][0,0] = -0.1
    args["frequencies"][1,1] = args["cutoff"]
    args["frequencies"][2,0] = 0.
    args["bands"] = np.array([3*n-1,0])
    actual, _ = contract(**args)
    expected = contract_reference(**args)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-18)
    np.testing.assert_array_equal(saved, args["eigenvectors"])
    assert not actual[1].any()


@pytest.mark.parametrize("field,value", [("backend","bad"),("g_zero",np.ones((6,6,6))),
                                        ("masses",np.array([0.,1.])),("cutoff",-1.)])
def test_reject_invalid_input(field, value):
    args = contraction_fixture()
    args[field] = value
    with pytest.raises(ValueError):
        contract(**args)


def test_comparison_reports_failure_and_zero():
    assert not compare(np.array([1.]), np.array([0.]))["passed"]
    assert not compare(np.array([np.nan]), np.array([1.]))["passed"]
    assert compare(np.array([0.]), np.array([0.]))["passed"]


@pytest.mark.integration
@pytest.mark.parametrize("compact", [False,True])
def test_fourier_contraction_native_c(compact):
    pytest.importorskip("phono3py")
    _, pp = toy_interaction(compact=compact)
    engine = FourierBatch(pp,"numpy")
    fc3_before = pp._fc3.copy()
    # All triplets at one synthetic non-Gamma q, not a private full-grid claim.
    for triplet in pp.get_triplets_at_q()[0]:
        result, _ = evaluate_triplet(pp, triplet, fourier=engine)
        np.testing.assert_allclose(result, native_triplet(pp,triplet), rtol=1e-8, atol=1e-20)
    np.testing.assert_array_equal(pp._fc3,fc3_before)
    assert engine.calls == len(pp.get_triplets_at_q()[0])


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("ACCELERATOR_TEST_GPU")!="1",reason="GPU tests explicitly opt-in")
@pytest.mark.parametrize("compact", [False, True])
def test_cupy_native_regression(compact):
    pytest.importorskip("cupy")
    _, pp = toy_interaction(compact=compact)
    gpu = FourierBatch(pp, "cupy")
    cpu = FourierBatch(pp, "numpy")
    saved = pp._fc3.copy()
    for triplet in pp.get_triplets_at_q()[0]:
        # Check complex Fourier phases before squaring, then the full interaction.
        np.testing.assert_allclose(gpu(triplet), cpu(triplet), rtol=1e-11, atol=1e-14)
        actual, _ = evaluate_triplet(pp, triplet, backend="cupy", fourier=gpu)
        np.testing.assert_allclose(actual,native_triplet(pp,triplet),rtol=1e-8,atol=1e-20)
    np.testing.assert_array_equal(pp._fc3, saved)
    assert gpu.calls == 2*len(pp.get_triplets_at_q()[0])


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("ACCELERATOR_TEST_GPU")!="1",reason="GPU tests explicitly opt-in")
@pytest.mark.parametrize("n", [1, 2, 3])
def test_cupy_contraction_independent_oracle(n):
    pytest.importorskip("cupy")
    args = contraction_fixture(n, seed=29)
    args["frequencies"][0,0] = -0.1
    args["frequencies"][1,1] = args["cutoff"]
    args["frequencies"][2,0] = 0.
    args["bands"] = np.array([3*n-1, 0])
    expected = contract_reference(**args)
    actual, _ = contract(**args, backend="cupy")
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-18)


@pytest.mark.integration
@pytest.mark.parametrize("compact",[False,True])
def test_complex_fourier_tensor_against_upstream_python(compact):
    pytest.importorskip("phono3py")
    from phono3py.phonon3.real_to_reciprocal import RealToReciprocal
    _,pp=toy_interaction(compact=compact)
    engine=FourierBatch(pp,"numpy")
    # Upstream Python oracle indexes the anchor by p2s even for compact FC3.
    # Expand only those explicitly used anchor rows for that oracle; native C
    # separately receives and tests the original compact array unchanged.
    oracle_fc3=pp.fc3
    if compact:
        ns=len(pp._s2p)
        oracle_fc3=np.zeros((ns,ns,ns,3,3,3))
        oracle_fc3[pp._p2s]=pp.fc3
    oracle=RealToReciprocal(oracle_fc3,pp.primitive,pp.mesh_numbers,
                           make_r0_average=True,all_shortest=pp.all_shortest)
    for triplet in pp.get_triplets_at_q()[0]:
        oracle.run(pp.bz_grid.addresses[triplet])
        atom_cube=oracle.get_fc3_reciprocal()
        expected=atom_cube.transpose(0,3,1,4,2,5).reshape((6,)*3)
        np.testing.assert_allclose(engine(triplet),expected,rtol=1e-11,atol=1e-14)


@pytest.mark.integration
def test_native_nonzero_mask_preserved():
    pytest.importorskip("phono3py")
    _,pp=toy_interaction()
    mask=np.random.default_rng(19).integers(0,2,pp._fc3.shape[:3],dtype=np.byte)
    pp._fc3_nonzero_indices=mask
    engine=FourierBatch(pp,"numpy")
    for triplet in pp.get_triplets_at_q()[0][:3]:
        actual,_=evaluate_triplet(pp,triplet,fourier=engine)
        np.testing.assert_allclose(actual,native_triplet(pp,triplet),rtol=1e-8,atol=1e-20)
