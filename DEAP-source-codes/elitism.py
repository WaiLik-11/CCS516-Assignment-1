from deap import tools
from deap import algorithms


def _evaluate(toolbox, individuals):
    """Evaluate the given individuals and return how many fitness evaluations were spent."""
    fitnesses = toolbox.map(toolbox.evaluate, individuals)
    for ind, fit in zip(individuals, fitnesses):
        ind.fitness.values = fit
    return len(individuals)


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, max_evals, stats=None,
                        halloffame=None, verbose=__debug__):
    """Generational EA with elitism and a hard fitness-evaluation budget.

    Based on DEAP's eaSimple(), with two modifications:

    * Elitism: the individuals in ``halloffame`` are copied unchanged into the next
      generation; only ``len(population) - len(halloffame)`` offspring are produced by
      selection + variation (``algorithms.varAnd``).
    * Termination: the loop stops once ``max_evals`` fitness evaluations have been spent.
      Every call to ``toolbox.evaluate`` is counted, including the initial population.
      Offspring that were neither crossed over nor mutated keep their (valid) fitness and
      are not re-evaluated, so they cost nothing. If the last generation would exceed the
      budget, only the offspring that still fit are evaluated; the rest are discarded and
      the population is refilled with the best already-evaluated individuals from the
      previous generation. The budget is therefore never exceeded.

    :returns: the final population and a logbook with per-generation ``nevals`` and the
              cumulative evaluation count ``evals``.
    """
    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'evals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    if len(invalid_ind) > max_evals:
        raise ValueError("max_evals is smaller than the initial population size")
    evals = _evaluate(toolbox, invalid_ind)

    halloffame.update(population)
    hof_size = len(halloffame.items)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), evals=evals, **record)
    if verbose:
        print(logbook.stream)

    gen = 0
    while evals < max_evals:
        gen += 1

        # Parent selection
        offspring = toolbox.select(population, len(population) - hof_size)

        # Variation (clones the parents, then crossover and mutation)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # Do not exceed the evaluation budget in the last generation
        remaining = max_evals - evals
        if len(invalid_ind) > remaining:
            dropped = {id(ind) for ind in invalid_ind[remaining:]}
            invalid_ind = invalid_ind[:remaining]
            offspring = [ind for ind in offspring if id(ind) not in dropped]

        evals += _evaluate(toolbox, invalid_ind)

        # Survivor selection: elites + offspring (generational replacement)
        offspring.extend(halloffame.items)
        halloffame.update(offspring)

        missing = len(population) - len(offspring)
        if missing > 0:
            offspring.extend(toolbox.clone(ind) for ind in tools.selBest(population, missing))

        population[:] = offspring

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), evals=evals, **record)
        if verbose:
            print(logbook.stream)

    return population, logbook
