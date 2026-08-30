from invexapi.optim import FISTA, GradientDescent, NonlinearCG, Solver


def test_all_optimizers_subclass_solver():
    assert issubclass(GradientDescent, Solver)
    assert issubclass(NonlinearCG, Solver)
    assert issubclass(FISTA, Solver)


def test_solver_stores_max_iter_and_tol():
    from invexapi import TikhonovPenalty

    penalty = TikhonovPenalty(lamb=0.1)
    solver = GradientDescent(penalty, step=0.1, max_iter=42, tol=1e-3)

    assert solver.max_iter == 42
    assert solver.tol == 1e-3
