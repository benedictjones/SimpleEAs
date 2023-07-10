optimizer = CMA(mean=np.mean(bound, axis=1),
    #                 sigma=0.002,
    #                 bounds=np.array(bound),
    #                 seed=2)

    # trial_pops = []
    # for generation in range(num_gens):
    #     solutions = []
    #     for _ in range(optimizer.population_size):
    #         x = optimizer.ask()

    #     exit()
    #     value = fun(x[0], x[1])
    #     solutions.append((x, value))
    #     optimizer.tell(solutions,)
    # animate(optimizer, trial_pops, bound, fun, lab, save="examples/CMA_%s" % (lab), algo="CMA")