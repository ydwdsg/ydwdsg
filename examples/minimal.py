"""Run a CPU-only synthetic Fourier -> |V3|² -> native-C regression."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
from phono3py_cupy_accelerator import FourierBatch, compare
from phono3py_cupy_accelerator.fixtures import toy_interaction
from phono3py_cupy_accelerator.phono3py_adapter import evaluate_triplet, native_triplet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    args = parser.parse_args()
    _, pp = toy_interaction()
    triplet = pp.get_triplets_at_q()[0][0]
    engine = FourierBatch(pp, args.backend)
    actual, timing = evaluate_triplet(pp, triplet, backend=args.backend, fourier=engine)
    expected = native_triplet(pp, triplet)
    gate = compare(actual, expected, rtol=1e-8, atol=1e-20)
    print(json.dumps(dict(fixture="synthetic; not physical SnP3", gate=gate, timing=timing), indent=2))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
