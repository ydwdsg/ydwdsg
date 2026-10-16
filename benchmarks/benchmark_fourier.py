"""Tiny synthetic Fourier timing: preparation, first call and warm calls separated."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
import argparse
import json
import time
from phono3py_cupy_accelerator import FourierBatch
from phono3py_cupy_accelerator.fixtures import toy_interaction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend",choices=["numpy","cupy"],default="numpy")
    p.add_argument("--repeats",type=int,default=5)
    args=p.parse_args()
    if args.repeats<1:p.error("repeats must be positive")
    _,pp=toy_interaction()
    triplet=pp.get_triplets_at_q()[0][0]
    prepare=time.perf_counter()
    engine=FourierBatch(pp,args.backend)
    prepare=time.perf_counter()-prepare
    durations=[]
    for _ in range(args.repeats+1):
        start=time.perf_counter(); engine(triplet)
        durations.append(time.perf_counter()-start)
    print(json.dumps(dict(fixture="synthetic; no private FC",backend=args.backend,
        prepare_s=prepare,first_call_s=durations[0],warm_call_s=durations[1:],
        metrics=engine.metrics(),warning="Tiny fixture timing does not predict 73-atom speedup"),indent=2))


if __name__=="__main__":main()
