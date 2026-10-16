"""FP64 three-GEMM kernel extracted from the frozen validated implementation.

Consumes native interleaved component cube directly, rather than two inverse
168MB host transpositions. Full219-band first GEMM; one GPU output download
per triplet. No TF32, conjugation, mixed precision or scientific gating change.
Full-M14 regression (not another single triplet benchmark) is mandatory.
"""
# Copyright (C) 2015 Atsushi Togo (upstream conventions).
# Copyright (c) 2026, yangdewei (adaptations).
# BSD-3-Clause; see LICENSE and licenses/PHONO3PY-BSD-3-Clause.txt.
import time
import numpy as np

def contract(fc3,eigenvectors,frequencies,masses,bands,cutoff,unit_conversion,
             *,backend='numpy',block=219,g_zero=None):
    if backend not in ('numpy','cupy'):raise ValueError(backend)
    xp=np
    if backend=='cupy':import cupy as xp
    d=3*len(masses);bands=np.asarray(bands,dtype=np.int64)
    if d == 0 or fc3.shape!=(d,d,d) or fc3.dtype!=np.complex128 or not fc3.flags.c_contiguous:
        raise ValueError('fc3 must be a contiguous complex128 component cube')
    if eigenvectors.shape!=(3,d,d) or frequencies.shape!=(3,d):
        raise ValueError('Expected eigenvectors (3,d,d), frequencies (3,d)')
    if bands.ndim!=1 or not len(bands) or not np.all((bands>=0)&(bands<d)):
        raise ValueError('bands must contain valid component-mode indices')
    if not np.all(np.isfinite(masses)) or not np.all(np.asarray(masses)>0):
        raise ValueError('Masses must be finite and positive')
    if any(not np.isfinite(a).all() for a in (fc3,eigenvectors,frequencies)):
        raise ValueError('Inputs must be finite')
    if not np.isfinite(cutoff) or cutoff<0 or not np.isfinite(unit_conversion) or unit_conversion<0:
        raise ValueError('cutoff and unit_conversion must be finite and nonnegative')
    if g_zero is not None:
        raise ValueError('Full evaluation only: native integration handles g_zero')
    # `block` is retained for source compatibility; the validated fast path is full-band.
    sync=lambda:xp.cuda.get_current_stream().synchronize() if backend=='cupy' else None
    sync();start=time.perf_counter()
    f=xp.asarray(fc3,dtype=xp.complex128)
    e=xp.array(eigenvectors,dtype=xp.complex128,copy=True)
    e/=xp.sqrt(xp.repeat(xp.asarray(masses,dtype=xp.float64),3))[None,:,None]
    freq=xp.asarray(frequencies,dtype=xp.float64);b=xp.asarray(bands)
    sync();upload=time.perf_counter()-start
    t=time.perf_counter()
    event0=event1=None
    if backend=='cupy':
        event0,event1=xp.cuda.Event(),xp.cuda.Event();event0.record()
    # Same three non-conjugated products as accepted V1. Only tile size/layout differs.
    x=(e[0][:,b].T@f.reshape(d,d*d)).reshape(len(bands),d,d)
    y=xp.matmul(x.transpose(0,2,1),e[1]).transpose(0,2,1)
    del x
    z=xp.matmul(y,e[2]);del y
    valid=((freq[0,b,None,None]>cutoff)&(freq[1,None,:,None]>cutoff)&(freq[2,None,None,:]>cutoff))
    denominator=freq[0,b,None,None]*freq[1,None,:,None]*freq[2,None,None,:]
    values=xp.where(valid,(z.real*z.real+z.imag*z.imag)/xp.where(valid,denominator,1.0)*unit_conversion,0.0)
    if backend=='cupy':
        event1.record();event1.synchronize();kernel=xp.cuda.get_elapsed_time(event0,event1)/1000
        peak=xp.get_default_memory_pool().total_bytes()
    else:kernel=time.perf_counter()-t;peak=0
    t=time.perf_counter()
    output=xp.asnumpy(values) if backend=='cupy' else np.ascontiguousarray(values)
    download=time.perf_counter()-t
    return output,dict(backend=backend,dtype='complex128',block=len(bands),
        upload_and_layout_s=upload,kernel_s=kernel,download_s=download,
        end_to_end_s=time.perf_counter()-start,peak_pool_reserved_bytes=peak,
        timing_scope='post-validation transfer/compute/download; excludes import and input checks; caller must time whole call',
        optimization='direct_native_layout_full_first_GEMM_single_D2H',
        peak_memory_scope='CuPy pool reservation; process peak not measured by this function')
