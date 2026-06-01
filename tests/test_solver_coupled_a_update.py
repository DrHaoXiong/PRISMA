import numpy as np

from solver import CoupledTensorSolver


def test_local_a_update_solves_full_coupled_normal_equation():
    rng = np.random.default_rng(4)
    n_snps = 7
    n_tissues = 4
    rank = 3
    solver = CoupledTensorSolver(n_tissues, 1, rank=rank, lambda_reg=0.2, max_iter=1)
    solver.B = np.abs(rng.normal(size=(n_tissues, rank))) + 0.1
    solver.C = np.abs(rng.normal(size=(1, rank))) + 0.1
    assert np.max(np.abs((solver.B.T @ solver.B) * (solver.C.T @ solver.C) - np.diag(np.diag((solver.B.T @ solver.B) * (solver.C.T @ solver.C))))) > 1e-3

    tensor_block = rng.normal(size=(n_snps, n_tissues, 1))
    adjacency = np.eye(n_snps, k=1) + np.eye(n_snps, k=-1)
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency + np.eye(n_snps) * 0.05

    a_hat, kb = solver.solve_local_A(tensor_block, laplacian)
    rhs = tensor_block.reshape(n_snps, -1) @ kb
    m = (solver.C.T @ solver.C) * (solver.B.T @ solver.B)
    residual = solver.lambda_reg * (laplacian @ a_hat) + a_hat @ m - rhs

    assert np.linalg.norm(residual) < 1e-4
