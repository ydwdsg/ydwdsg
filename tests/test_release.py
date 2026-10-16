import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def test_owner_acceptance_preserves_original_gate_and_pending():
    data=json.loads((ROOT/"benchmarks/data/public_results.json").read_text())
    assert data["representation72"]["status"]=="PASS_OWNER_ACCEPTED"
    assert data["representation72"]["original_strict_gate"]=="FAIL"
    assert data["independent_native_c_all_q"].startswith("PENDING")
    with (ROOT/"benchmarks/data/gamma_accuracy_24q.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==24 and all(r["passed"]=="True" for r in rows)
    assert max(float(r["max_abs_THz"]) for r in rows)==data["gamma24"]["max_abs_THz"]
    assert data["pipeline"]["new_gpu_segmented_q_sum_s"]<data["pipeline"]["new_gpu_fresh_process_s"]


def test_bad_triplet_rejected_before_native_call():
    import pytest
    pytest.importorskip("phono3py")
    from phono3py_cupy_accelerator.fixtures import toy_interaction
    from phono3py_cupy_accelerator.phono3py_adapter import native_triplet
    _,pp=toy_interaction()
    for bad in ([0,1,-1],[0,1,100000],[0.,1.,2.],[0,1]):
        with pytest.raises(ValueError):native_triplet(pp,bad)


def test_memory_guard_before_allocation():
    import pytest
    pytest.importorskip("phono3py")
    from phono3py_cupy_accelerator.fixtures import toy_interaction
    from phono3py_cupy_accelerator import FourierBatch
    _,pp=toy_interaction()
    with pytest.raises(MemoryError):FourierBatch(pp,"numpy",max_constant_bytes=1)
