"""Deterministic synthetic algebra fixtures, NOT a material potential or FC model."""
import numpy as np


def contraction_fixture(n=2, seed=2026):
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    d = 3 * n
    cube = np.ascontiguousarray(rng.normal(size=(d,)*3) + 1j*rng.normal(size=(d,)*3))
    eig = np.stack([np.linalg.qr(rng.normal(size=(d,d)) + 1j*rng.normal(size=(d,d)))[0] for _ in range(3)])
    return dict(fc3=cube, eigenvectors=eig, frequencies=rng.uniform(.2, 4, (3,d)),
                masses=rng.uniform(10, 30, n), bands=np.arange(d), cutoff=1e-4, unit_conversion=1.)


def toy_interaction(*, compact=False, seed=1729):
    """Two arbitrary atoms, 2x2x1 supercell, random algebra-only FC3.

    FC2 is an arbitrary translationally invariant spring graph, not a material
    model. This exercises mappings, complex phases and r0 averaging, not transport.
    """
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms
    cell = PhonopyAtoms(symbols=["Si", "Ge"], cell=np.diag([4., 5., 7.]),
                        scaled_positions=[[0.,0.,0.], [.31,.27,.13]])
    ph = Phono3py(cell, [2,2,1], primitive_matrix=np.eye(3),
                   is_symmetry=False, is_mesh_symmetry=False, make_r0_average=True)
    ns = len(ph.supercell)
    graph = ns*np.eye(ns)-np.ones((ns,ns))
    fc2 = graph[:,:,None,None]*np.diag([1.,1.3,1.7])[None,None,:,:]
    rng = np.random.default_rng(seed)
    full = rng.normal(scale=.01, size=(ns, ns, ns, 3, 3, 3))
    ph.fc2 = fc2
    ph.fc3 = np.ascontiguousarray(full[ph.primitive.p2s_map] if compact else full)
    ph.mesh_numbers = [3,3,1]
    ph.init_phph_interaction()
    pp = ph.phph_interaction
    pp.run_phonon_solver()
    pp.set_grid_point(1)
    return ph, pp
