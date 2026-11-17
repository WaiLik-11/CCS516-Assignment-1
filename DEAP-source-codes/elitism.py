import math
import random

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


# ---------------------------------------------------------------------------------------------
# Parameter control (step 6): the same generational EA with elitism and evaluation budget, but
# the crossover/mutation probabilities are controlled during the run instead of being fixed.
# ---------------------------------------------------------------------------------------------

class AdaptivePursuit:
    """Adaptive pursuit operator selection (Thierens, 2005).

    Each arm is a (p_crossover, p_mutation) setting. Every generation one arm is drawn with
    probability P[i] and used for the whole generation. Its quality estimate Q[i] moves towards
    the observed reward (learning rate alpha); then the probability of the currently best arm is
    pushed towards P_max and all others towards P_min (learning rate beta), with
    P_max = 1 - (K - 1) * P_min, so every arm keeps being tried.
    """

    def __init__(self, arms, alpha, beta, p_min):
        self.arms = list(arms)
        k = len(self.arms)
        self.alpha, self.beta, self.p_min = alpha, beta, p_min
        self.p_max = 1 - (k - 1) * p_min
        self.P = [1.0 / k] * k
        self.Q = [1.0] * k  # optimistic start, so no arm is written off before it was tried

    def choose(self):
        if len(self.arms) == 1:  # no random draw: a single arm reproduces the fixed-parameter EA
            return 0
        return random.choices(range(len(self.arms)), weights=self.P)[0]

    def update(self, arm, reward):
        self.Q[arm] += self.alpha * (reward - self.Q[arm])
        best = max(range(len(self.arms)), key=self.Q.__getitem__)
        for i in range(len(self.arms)):
            target = self.p_max if i == best else self.p_min
            self.P[i] += self.beta * (target - self.P[i])


def perturbRate(rate, gamma, low=0.001, high=1.0):
    """Self-adaptive mutation of a probability (Bäck & Schütz, 1996): a log-normal step on the
    odds p / (1 - p), so the rate stays in (0, 1) and small and large rates change proportionally."""
    rate = 1.0 / (1.0 + (1.0 - rate) / rate * math.exp(-gamma * random.gauss(0.0, 1.0)))
    return min(max(rate, low), high)


def _varyWithCredit(parents, toolbox, cxpb, mutpb, sa_gamma=None):
    """Same as DEAP's algorithms.varAnd (same order of random draws), but also returns, for every
    offspring, the fitness of its better parent, to measure how much variation improved it.

    With sa_gamma (self-adaptation), each offspring's mutation probability is its own `pm`
    attribute: inherited from its parents (mean of both after crossover) and perturbed with
    perturbRate() before it is used.
    """
    offspring = [toolbox.clone(ind) for ind in parents]
    reference = [ind.fitness.values[0] for ind in parents]

    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            if sa_gamma is not None:
                offspring[i - 1].pm = offspring[i].pm = (offspring[i - 1].pm + offspring[i].pm) / 2
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            reference[i - 1] = reference[i] = min(reference[i - 1], reference[i])

    for i in range(len(offspring)):
        if sa_gamma is not None:
            offspring[i].pm = perturbRate(offspring[i].pm, sa_gamma)
            rate = offspring[i].pm
        else:
            rate = mutpb
        if random.random() < rate:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring, reference


def eaControlledWithElitism(population, toolbox, cxpb, mutpb, max_evals, stats=None, halloffame=None,
                            pursuit=None, sa_gamma=None, verbose=__debug__):
    """eaSimpleWithElitism() with parameter control:

    * pursuit (an AdaptivePursuit): every generation picks a (cxpb, mutpb) arm. The reward is the
      mean relative improvement of the evaluated offspring over their better parent,
      sum(max(0, (f_parent - f_child) / f_parent)) / evaluations, i.e. improvement per evaluation
      spent, so cheap settings (few changed, few evaluated offspring) are not penalised.
    * sa_gamma: self-adaptive mutation probability per individual (attribute `pm`; cxpb fixed).

    Elitism, the exact evaluation budget and the logbook are as in eaSimpleWithElitism(); the
    logbook additionally records the arm probabilities (p_arm0, ...) or the mean `pm`.
    """
    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'evals'] + (stats.fields if stats else [])

    def control():
        if pursuit is not None:
            return {f"p_arm{i}": p for i, p in enumerate(pursuit.P)}
        if sa_gamma is not None:
            return {"pm_mean": sum(ind.pm for ind in population) / len(population)}
        return {}

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    if len(invalid_ind) > max_evals:
        raise ValueError("max_evals is smaller than the initial population size")
    evals = _evaluate(toolbox, invalid_ind)

    halloffame.update(population)
    hof_size = len(halloffame.items)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), evals=evals, **record, **control())
    if verbose:
        print(logbook.stream)

    gen = 0
    while evals < max_evals:
        gen += 1

        # Parent selection
        parents = toolbox.select(population, len(population) - hof_size)

        # Variation with the probabilities chosen by the controller
        arm = pursuit.choose() if pursuit is not None else None
        if arm is not None:
            cxpb, mutpb = pursuit.arms[arm]
        offspring, reference = _varyWithCredit(parents, toolbox, cxpb, mutpb, sa_gamma)
        reference = {id(ind): ref for ind, ref in zip(offspring, reference)}

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # Do not exceed the evaluation budget in the last generation
        remaining = max_evals - evals
        if len(invalid_ind) > remaining:
            dropped = {id(ind) for ind in invalid_ind[remaining:]}
            invalid_ind = invalid_ind[:remaining]
            offspring = [ind for ind in offspring if id(ind) not in dropped]

        evals += _evaluate(toolbox, invalid_ind)

        # Credit assignment: improvement over the better parent, per evaluation spent
        if arm is not None and invalid_ind:
            gain = sum(max(0.0, (reference[id(ind)] - ind.fitness.values[0]) / reference[id(ind)])
                       for ind in invalid_ind)
            pursuit.update(arm, gain / len(invalid_ind))

        # Survivor selection: elites + offspring (generational replacement)
        offspring.extend(halloffame.items)
        halloffame.update(offspring)

        missing = len(population) - len(offspring)
        if missing > 0:
            offspring.extend(toolbox.clone(ind) for ind in tools.selBest(population, missing))

        population[:] = offspring

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), evals=evals, **record, **control())
        if verbose:
            print(logbook.stream)

    return population, logbook
