def run(job):
    # The suggested parameters are accessible in job.parameters (dict)
    x = job.parameters["x"]
    b = job.parameters["b"]

    if job.parameters["function"] == "linear":
        y = x + b
    elif job.parameters["function"] == "cubic":
        y = x ** 3 + b

    # Maximization!
    return y


# Necessary IF statement otherwise it will enter in a infinite loop
# when loading the 'run' function from a new process
if __name__ == "__main__":
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO
    from deephyper.evaluator import Evaluator

    # define the variable you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter((-10.0, 10.0), "x")  # real parameter
    problem.add_hyperparameter((0, 10), "b")  # discrete parameter
    problem.add_hyperparameter(["linear", "cubic"], "function")  # categorical parameter

    # define the evaluator to distribute the computation
    evaluator = Evaluator.create(
        run,
        method="process",
        method_kwargs={
            "num_workers": 2,
        },
    )

    # define your search and execute it
    search = CBO(problem, evaluator, random_state=42)

    results = search.search(max_evals=100)
    print(results)
