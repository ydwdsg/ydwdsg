"""FP64 Fourier kernel extracted from the frozen r0-average implementation.

Formula audited against official phono3py v3.31.1 c/real_to_reciprocal.c.
Keep all_shortest leg factors, nonzero mask, compact mapping and prephase.
No physical sparsification: only the native nonzero_indices mask is applied.
"""
# Copyright (C) 2015 Atsushi Togo (upstream transformation conventions).
# Copyright (c) 2026, yangdewei (adaptations).
# BSD-3-Clause; see LICENSE and licenses/PHONO3PY-BSD-3-Clause.txt.
import time
from collections import OrderedDict
import numpy as np


class FourierBatch:
    def __init__(self, pp, backend, *, max_constant_bytes=2*1024**3):
        from .phono3py_adapter import validate_interaction
        validate_interaction(pp)
        if backend not in ('numpy','cupy'):
            raise ValueError('backend must be numpy or cupy')
        start=time.perf_counter()
        self.pp=pp; self.backend=backend; self.xp=np
        if backend=='cupy':
            import cupy
            self.xp=cupy
        xp=self.xp
        self.n=len(pp._p2s); self.ns=len(pp._s2p)
        if not pp._make_r0_average:
            raise ValueError('v0.1.0 implements native r0-average=True only')
        self.maps=np.array([np.flatnonzero(pp._s2p==s) for s in pp._p2s],dtype=np.int64)
        if self.maps.ndim!=2 or len(np.unique(self.maps))!=self.ns:
            raise ValueError('Uniform, complete primitive-to-supercell mapping required')
        n=self.n; m=self.maps.shape[1]
        if n*n*n*m*m*(27*8+1)>max_constant_bytes:
            raise MemoryError('Packed constants exceed configured memory budget')
        # Contiguous [anchor,atom_j,atom_k,image_j,image_k,Cartesian27].
        packed=np.empty((n,n,n,m,m,27),dtype=np.float64)
        flags=np.empty((n,n,n,m,m),dtype=bool)
        compact=pp._fc3.shape[0]!=pp._fc3.shape[1]
        flat=self.maps.ravel()
        for i in range(n):
            si=i if compact else int(pp._p2s[i])
            raw=pp._fc3[si][flat[:,None],flat[None,:]].reshape(n,m,n,m,27)
            nz=pp._fc3_nonzero_indices[si][flat[:,None],flat[None,:]].reshape(n,m,n,m)
            packed[i]=raw.transpose(0,2,1,3,4)*nz.transpose(0,2,1,3)[...,None]
            flags[i]=pp._all_shortest[i][flat[:,None],flat[None,:]].reshape(n,m,n,m).transpose(0,2,1,3)!=0
        self.pack_s=time.perf_counter()-start
        t=time.perf_counter(); self.fc=xp.asarray(packed); self.flags=xp.asarray(flags)
        self.sync(); self.upload_s=time.perf_counter()-t
        self.cache=OrderedDict(); self.phase_s=0.; self.sum_s=0.; self.layout_s=0.; self.calls=0
        self.constant_bytes=self.fc.nbytes+self.flags.nbytes
        self.prepare_s=time.perf_counter()-start

    def sync(self):
        if self.backend=='cupy':self.xp.cuda.get_current_stream().synchronize()

    def phase(self, q):
        key=tuple(q)
        if key in self.cache:
            self.cache.move_to_end(key)
        if key not in self.cache:
            t=time.perf_counter(); pp=self.pp
            counts=pp._multi[:,:,0].astype(np.int64)
            starts=pp._multi[:,:,1].astype(np.int64)
            v=pp._svecs
            angle=2*np.pi*(v[:,0]*q[0]+v[:,1]*q[1]+v[:,2]*q[2])
            e=np.cos(angle)+1j*np.sin(angle)
            # Small multiplicity sums avoid subtracting nearly equal prefix sums.
            phase=np.zeros(counts.shape,dtype=np.complex128)
            for k in range(int(counts.max())):
                mask=counts>k
                phase[mask]+=e[starts[mask]+k]
            phase/=counts
            self.cache[key]=self.xp.asarray(phase.T[:,self.maps])
            if len(self.cache)>256:
                self.cache.popitem(last=False)
            self.sync(); self.phase_s+=time.perf_counter()-t
        return self.cache[key]

    def __call__(self, triplet):
        pp=self.pp
        triplet=np.asarray(triplet)
        if triplet.shape!=(3,) or not np.issubdtype(triplet.dtype,np.integer):
            raise ValueError('triplet must contain three integer BZ-grid indices')
        if np.any(triplet<0) or np.any(triplet>=len(pp._bz_grid.addresses)):
            raise ValueError('BZ-grid index outside addresses')
        xp=self.xp; n=self.n
        q=np.ascontiguousarray((pp._bz_grid.addresses[triplet]/pp._bz_grid.D_diag)@pp._bz_grid.Q.T)
        phases=[self.phase(row) for row in q]
        # Exact native prephase uses first smallest vector from primitive atom 0.
        v=pp._svecs[pp._multi[pp._p2s,0,1]]
        total=q[0]+q[1]+q[2]
        angle=2*np.pi*(v[:,0]*total[0]+v[:,1]*total[1]+v[:,2]*total[2])
        pre=xp.asarray(np.cos(angle)+1j*np.sin(angle))
        self.sync(); t=time.perf_counter()
        legs=[]
        for a,b,first in [(1,2,True),(0,2,False),(1,0,False)]:
            # Bounded 8-anchor batches, no giant complex FC copy.
            result=xp.empty((n,n,n,27),dtype=xp.complex128)
            for lo in range(0,n,8):
                hi=min(lo+8,n)
                coef=phases[a][lo:hi,:,None,:,None]*phases[b][lo:hi,None,:,None,:]
                if first:coef=coef*(1+2*self.flags[lo:hi])
                else:coef=coef*(~self.flags[lo:hi])
                result[lo:hi]=xp.einsum('ijkuvc,ijkuv->ijkc',self.fc[lo:hi],coef,optimize=False)
            result*=pre[:,None,None,None]
            legs.append(result.reshape(n,n,n,3,3,3))
        value=(legs[0]+legs[1].transpose(1,0,2,4,3,5)+legs[2].transpose(2,1,0,5,4,3))/3
        self.sync(); self.sum_s+=time.perf_counter()-t
        t=time.perf_counter()
        out=xp.ascontiguousarray(value.transpose(0,3,1,4,2,5)).reshape(3*n,3*n,3*n)
        if self.backend=='cupy':out=xp.asnumpy(out)
        self.sync(); self.layout_s+=time.perf_counter()-t; self.calls+=1
        return np.ascontiguousarray(out)

    def metrics(self):
        return dict(backend=self.backend,calls=self.calls,prepare_s=self.prepare_s,pack_s=self.pack_s,
                    upload_s=self.upload_s,constant_bytes=self.constant_bytes,phase_cache_entries=len(self.cache),
                    phase_s=self.phase_s,sum_s=self.sum_s,layout_and_D2H_s=self.layout_s,
                    formula='native r0 three legs; native masks; float64/complex128; no threshold changes')
