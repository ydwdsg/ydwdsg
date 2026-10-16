"""Synthetic microbenchmark. Thread binding must be set by the caller before Python starts."""
import argparse
import json
import time
import numpy as np
from phono3py_cupy_accelerator import contract, compare
from phono3py_cupy_accelerator.fixtures import contraction_fixture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atoms",type=int,default=4)
    p.add_argument("--backend",choices=["numpy","cupy"],default="numpy")
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--allow-large",action="store_true")
    args=p.parse_args()
    if args.atoms<1 or args.repeats<1 or (args.atoms>16 and not args.allow_large):
        p.error("Positive sizes/repeats required; >16 atoms needs --allow-large and your own resource approval")
    fixture=contraction_fixture(args.atoms)
    reference,_=contract(**fixture,backend="numpy")
    rows=[]
    start=time.perf_counter()
    for _ in range(args.repeats+1):
        call_start=time.perf_counter()
        actual,timing=contract(**fixture,backend=args.backend)
        timing["complete_call_wall_s"]=time.perf_counter()-call_start
        gate=compare(actual,reference,rtol=1e-8,atol=1e-14)
        if not gate["passed"]:
            raise RuntimeError(gate)
        rows.append(dict(timing=timing,gate=gate))
    print(json.dumps(dict(fixture="synthetic",branches=3*args.atoms,
        first_call=rows[0],warm=rows[1:],warm_median_s=float(np.median([r["timing"]["complete_call_wall_s"] for r in rows[1:]])),
        loop_wall_s=time.perf_counter()-start,
        memory_scope="CuPy reserved pool, not true process peak; device and host peaks require external sampling",
        warning="Not a recreation of the private 219-branch historical benchmark"),indent=2))


if __name__=="__main__":main()
