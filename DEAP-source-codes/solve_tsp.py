"""Evolutionary algorithm for the TSP, built only from DEAP operators.

An EA variant is described by a parameter dict (see BASELINE). buildToolbox() turns it into
a DEAP toolbox and runEA() performs one seeded run with a budget of config.MAX_EVALUATIONS
fitness evaluations. Experiments (several variants x TSPs x seeds) are run by experiments.py.

Running this file directly does a single demo run:
    uv run python DEAP-source-codes/solve_tsp.py berlin52
"""

import array
import random
import numpy as np

from deap import base
from deap import creator
from deap import tools

import config
import elitism
import operators
from tsp import TravelingSalesmanProblem

# Step 1 baseline: the operators and parameters of the original template solve_tsp.py,
# only capped at MAX_EVALUATIONS instead of MAX_GENERATIONS = 200.
BASELINE = dict(
    population_size=300,   # POPULATION_SIZE
    elite_size=30,         # HALL_OF_FAME_SIZE: best individuals copied unchanged each generation
    tournament_size=2,     # tools.selTournament(tournsize=2)
    crossover="ox",        # tools.cxOrdered
    p_crossover=0.9,       # P_CROSSOVER: probability that a pair of parents is recombined
    crossover_indpb=None,  # only used by "upmx"
    mutation="swap",       # tools.mutShuffleIndexes
    p_mutation=0.1,        # P_MUTATION: probability that an offspring is mutated
    mutation_indpb=None,   # per-position swap probability for "swap"; None = 1/n (as in the template)
    init_nn_fraction=0.0,  # share of the initial population built by nearest-neighbour (0 = all random)
    # parameter control (step 6): None = fixed p_crossover / p_mutation
    control=None,          # None, "adaptive-pursuit" or "self-adaptive-pm"
    ap_arms=None,          # adaptive pursuit: list of (p_crossover, p_mutation) settings to choose from
    ap_alpha=0.8,          # adaptive pursuit: learning rate of the quality estimates
    ap_beta=0.8,           # adaptive pursuit: learning rate of the probabilities
    ap_p_min=0.1,          # adaptive pursuit: minimum probability of each arm
    sa_gamma=0.22,         # self-adaptive p_mutation: learning rate of the log-normal step
)

# Final design, chosen step by step in experiments.py (see README "Step log" and the report):
# OX crossover (p_c 0.9), inversion mutation (p_m 0.1), tournament size 2, mu = 50 with 5 elites, and an
# initial population of nearest-neighbour tours from 50 distinct random start cities (step 9).
FINAL_DESIGN = {
    **BASELINE,
    "population_size": 50,
    "elite_size": 5,
    "mutation": "inversion",
    "init_nn_fraction": 1.0,
}

# Operators that keep a permutation valid (all from DEAP, except ERX from operators.py)
CROSSOVERS = {
    "ox": tools.cxOrdered,                    # Davis (1985)
    "pmx": tools.cxPartialyMatched,           # Goldberg & Lingle (1985)
    "upmx": tools.cxUniformPartialyMatched,   # Cicirello & Smith (2000)
    "erx": operators.cxEdgeRecombination,     # Whitley, Starkweather & Fuquay (1989)
}
MUTATIONS = {
    "swap": tools.mutShuffleIndexes,          # swap mutation
    "inversion": tools.mutInversion,          # inversion mutation (reverses one segment)
}

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list of integers:
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

def edgeDiversity(population):
    """Distinct (undirected) edges in the population per city: 1.0 when all individuals are the
    same tour, up to the population size when no two individuals share an edge."""
    tours = np.asarray(population)
    a, b = tours, np.roll(tours, -1, axis=1)
    n = tours.shape[1]
    edges = np.minimum(a, b) * n + np.maximum(a, b)
    return np.unique(edges).size / n


class PopulationStatistics(tools.Statistics):
    """DEAP Statistics on the fitness values, plus the edge diversity of the whole population."""

    def __init__(self):
        super().__init__(lambda ind: ind.fitness.values)
        self.fields.append("edges")

    def compile(self, population):
        record = super().compile(population)
        record["edges"] = edgeDiversity(population)
        return record


def tspDistance(individual, problem: TravelingSalesmanProblem):
    """Fitness: TSPLIB tour length (each edge rounded to the nearest integer), minimised."""
    return problem.getTotalDistanceRounded(individual),  # return a tuple

def buildToolbox(problem, params):
    """Register the operators described by `params` (a dict like BASELINE) in a DEAP toolbox."""
    n = len(problem)
    toolbox = base.Toolbox()

    # initialization: uniformly random permutations
    # create an operator that generates randomly shuffled indices:
    toolbox.register("randomOrder", random.sample, range(n), n)
    # create the individual creation operator to fill up an Individual instance with shuffled indices:
    toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)
    # create the population creation operator to generate a list of individuals:
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    toolbox.register("evaluate", tspDistance, problem=problem)

    # parent selection
    toolbox.register("select", tools.selTournament, tournsize=params["tournament_size"])

    # crossover
    if params["crossover"] not in CROSSOVERS:
        raise ValueError(f"unknown crossover {params['crossover']!r}, choose from {list(CROSSOVERS)}")
    if params["crossover"] == "upmx":
        if params["crossover_indpb"] is None:
            raise ValueError("crossover 'upmx' needs crossover_indpb")
        toolbox.register("mate", CROSSOVERS["upmx"], indpb=params["crossover_indpb"])
    else:
        toolbox.register("mate", CROSSOVERS[params["crossover"]])

    # mutation
    if params["mutation"] not in MUTATIONS:
        raise ValueError(f"unknown mutation {params['mutation']!r}, choose from {list(MUTATIONS)}")
    if params["mutation"] == "swap":
        indpb = params["mutation_indpb"] if params["mutation_indpb"] is not None else 1.0 / n
        toolbox.register("mutate", MUTATIONS["swap"], indpb=indpb)
    else:
        toolbox.register("mutate", MUTATIONS[params["mutation"]])

    return toolbox

def runEA(problem, params, seed, max_evals=config.MAX_EVALUATIONS, verbose=False):
    """One independent, seeded EA run.

    :returns: 
        (best individual, logbook). The logbook has per generation: gen, nevals, evals (cumulative), min and avg fitness of the population.
    """
    # every random choice (initialization and DEAP operators) uses the `random` module for repeatable results
    random.seed(seed)
    np.random.seed(seed)

    toolbox = buildToolbox(problem, params)

    # create initial population (generation 0): random permutations, plus optionally some
    # nearest-neighbour tours from distinct random start cities (heuristic initialization)
    n_nn = round(params["population_size"] * params.get("init_nn_fraction", 0.0))
    population = toolbox.populationCreator(n=params["population_size"] - n_nn)
    if n_nn:
        starts = random.sample(range(len(problem)), n_nn)
        population += [creator.Individual(operators.nearestNeighbourTour(problem.distancesRounded, s))
                       for s in starts]

    # prepare the statistics object:
    stats = PopulationStatistics()
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(params["elite_size"])

    # perform the Genetic Algorithm flow with hof feature added:
    control = params.get("control")
    if control is None:
        population, logbook = elitism.eaSimpleWithElitism(
            population, toolbox, cxpb=params["p_crossover"], mutpb=params["p_mutation"],
            max_evals=max_evals, stats=stats, halloffame=hof, verbose=verbose)
    elif control == "adaptive-pursuit":
        pursuit = elitism.AdaptivePursuit(params["ap_arms"], params["ap_alpha"], params["ap_beta"],
                                          params["ap_p_min"])
        population, logbook = elitism.eaControlledWithElitism(
            population, toolbox, cxpb=None, mutpb=None, max_evals=max_evals, stats=stats,
            halloffame=hof, pursuit=pursuit, verbose=verbose)
    elif control == "self-adaptive-pm":
        for ind in population:
            ind.pm = params["p_mutation"]  # every individual starts with the tuned rate
        population, logbook = elitism.eaControlledWithElitism(
            population, toolbox, cxpb=params["p_crossover"], mutpb=None, max_evals=max_evals,
            stats=stats, halloffame=hof, sa_gamma=params["sa_gamma"], verbose=verbose)
    else:
        raise ValueError(f"unknown control {control!r}")

    # validate and return best individual info
    best = hof.items[0]
    assert sorted(best) == list(range(len(problem))), "best tour must be a valid permutation"
    return best, logbook

def main():
    name = config.parse_tsp_names("Single run of the final EA design (reporting seed 1)")[0]
    problem = TravelingSalesmanProblem(name)
    best, logbook = runEA(problem, FINAL_DESIGN, seed=config.RANDOM_SEEDS[0], verbose=True)

    optimum = config.KNOWN_OPTIMA[name]
    length = best.fitness.values[0]
    print(f"-- {name}: best length = {length:.0f}, optimum = {optimum}, "
          f"deviation = {100 * (length - optimum) / optimum:.2f}%, evaluations = {logbook[-1]['evals']}")

if __name__ == "__main__":
    main()
