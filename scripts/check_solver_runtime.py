"""Exercise the exact variable and bundled CBC API required by advice engines."""

import pulp


def check_solver_runtime():
    if pulp.__version__ != "3.3.2":
        raise RuntimeError(f"Unsupported PuLP {pulp.__version__}; this project requires pulp==3.3.2")
    model = pulp.LpProblem("fpl_startup_solver_check", pulp.LpMaximize)
    selected = pulp.LpVariable("selected", cat=pulp.LpBinary)
    model += selected
    model += selected <= 1
    model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
    if pulp.LpStatus[model.status] != "Optimal" or selected.value() != 1:
        raise RuntimeError("Bundled CBC binary solve failed; advice runtime is not ready")
    return pulp.__version__


if __name__ == "__main__":
    print("PuLP/CBC solve verified:", check_solver_runtime())
