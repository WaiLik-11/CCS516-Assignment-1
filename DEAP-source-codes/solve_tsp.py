"""Evolutionary algorithm for the TSP (DEAP).

Runs NUM_RUNS independent runs (one seed from config.RANDOM_SEEDS each, max
config.MAX_EVALUATIONS fitness evaluations) on each selected TSP, and reports the
rounded tour lengths and the deviation of the average from the optimum.
The design choices and their justification are in the "EA design" section of README.md.
"""

import array
import csv
import random
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save figures to files; no blocking GUI windows
import matplotlib.pyplot as plt
import seaborn as sns

from deap import base
from deap import creator
from deap import tools

import config
import elitism
import tsp

# EA parameters (tuned on seeds 101-103, see README "EA design")
POPULATION_SIZE = 50     # mu: parents per generation
ELITE_SIZE = 2           # hall-of-fame individuals copied unchanged into each generation
TOURNAMENT_SIZE = 5      # parent selection
CROSSOVER = "ox"         # "ox" (tools.cxOrdered) or "pmx" (tools.cxPartialyMatched)
P_CROSSOVER = 0.5        # probability that a pair of parents is recombined
MUTATION = "inversion"   # "inversion" (mutInversion) or "swap" (tools.mutShuffleIndexes)
P_MUTATION = 0.5         # probability that an offspring is mutated
NEIGHBOURS = 5           # k for guided inversion (0 = uniformly random inversion)
NN_FRACTION = 0.1        # share of the initial population built by nearest-neighbour


# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list of integers:
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)


def nearestNeighbourTour(problem, start):
    """Greedy tour: from `start`, always go to the closest unvisited city."""
    unvisited = set(range(len(problem)))
    unvisited.remove(start)
    tour = [start]
    while unvisited:
        row = problem.distances[tour[-1]]
        nxt = min(unvisited, key=row.__getitem__)
        unvisited.remove(nxt)
        tour.append(nxt)
    return tour


def nearestNeighbourLists(problem, k):
    """For every city, the indices of its k closest other cities."""
    dist = np.asarray(problem.distances)
    return np.argsort(dist, axis=1)[:, 1:k + 1].tolist()


def mutInversion(individual, neighbours=None):
    """Inversion mutation: reverse one segment of the tour (a 2-opt move, changes 2 edges).

    Without `neighbours` the segment is chosen uniformly at random. With neighbour lists,
    a random city c is picked and the segment is chosen so that one of c's k nearest
    cities becomes adjacent to c. This is a neighbour-list-guided 2-opt move. It borrows
    the inversion step of Inver-over (Tao & Michalewicz 1998), but is not Inver-over: the
    target city comes from a fixed nearest-neighbour list rather than from another
    individual, and the move is applied once without an accept-if-better test.
    """
    size = len(individual)
    if neighbours is None:
        i, j = sorted(random.sample(range(size), 2))
        individual[i:j + 1] = individual[i:j + 1][::-1]
        return individual,

    i = random.randrange(size)
    j = individual.index(random.choice(neighbours[individual[i]]))
    if j > i:
        individual[i + 1:j + 1] = individual[i + 1:j + 1][::-1]  # neighbour moves to i + 1
    elif j < i:
        individual[j:i] = individual[j:i][::-1]                  # neighbour moves to i - 1
    return individual,


def initPopulation(problem, n, nn_fraction):
    """Mixed initialization: nearest-neighbour tours from distinct random start cities,
    the rest uniformly random permutations."""
    n_nn = min(round(n * nn_fraction), len(problem))
    starts = random.sample(range(len(problem)), n_nn)
    population = [creator.Individual(nearestNeighbourTour(problem, s)) for s in starts]
    population += [creator.Individual(random.sample(range(len(problem)), len(problem)))
                   for _ in range(n - n_nn)]
    return population


def buildToolbox(problem, tournament_size=TOURNAMENT_SIZE, crossover=CROSSOVER, mutation=MUTATION, neighbours=NEIGHBOURS):
    toolbox = base.Toolbox()

    # fitness: total (float) distance of the closed tour
    toolbox.register("evaluate", lambda ind: (problem.getTotalDistance(ind),))

    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    if crossover == "ox":
        toolbox.register("mate", tools.cxOrdered)
    elif crossover == "pmx":
        toolbox.register("mate", tools.cxPartialyMatched)
    else:
        raise ValueError(f"unknown crossover {crossover!r}")

    if mutation == "inversion":
        toolbox.register("mutate", mutInversion,
                         neighbours=nearestNeighbourLists(problem, neighbours) if neighbours else None)
    elif mutation == "swap":
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0 / len(problem))
    else:
        raise ValueError(f"unknown mutation {mutation!r}")
    return toolbox


def runEA(problem, seed, population_size=POPULATION_SIZE, elite_size=ELITE_SIZE,
          p_crossover=P_CROSSOVER, p_mutation=P_MUTATION, nn_fraction=NN_FRACTION,
          max_evals=config.MAX_EVALUATIONS, **toolbox_kwargs):
    """One independent EA run. Returns (best individual, logbook)."""
    random.seed(seed)
    np.random.seed(seed)

    toolbox = buildToolbox(problem, **toolbox_kwargs)
    population = initPopulation(problem, population_size, nn_fraction)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    hof = tools.HallOfFame(elite_size)

    population, logbook = elitism.eaSimpleWithElitism(
        population, toolbox, cxpb=p_crossover, mutpb=p_mutation, max_evals=max_evals,
        stats=stats, halloffame=hof, verbose=False)
    return hof.items[0], logbook


def saveBestTourPlot(problem, best, length, path):
    plt.figure()
    problem.plotData(best)
    plt.title(f"{problem.name} best EA tour (length {length})")
    plt.savefig(path, dpi=150)
    plt.close()


def saveConvergencePlot(problem, logbooks, path):
    plt.figure()
    sns.set_style("whitegrid")
    for seed, logbook in logbooks.items():
        evals, mins = logbook.select("evals", "min")
        plt.plot(evals, mins, label=f"seed {seed}")
    plt.xlabel("Fitness evaluations")
    plt.ylabel("Best tour length")
    plt.title(f"{problem.name}: best fitness per run")
    plt.legend()
    plt.savefig(path, dpi=150)
    plt.close()


def solve(name):
    """Run the EA NUM_RUNS times on one TSP, save results, and return the summary row."""
    problem = tsp.TravelingSalesmanProblem(name)
    # TSPLIB convention: each edge rounded to the nearest integer (also used for EA tours)
    optimum = problem.getTotalDistanceRounded(problem.loadTour())

    rows, logbooks, best_overall = [], {}, None
    for run, seed in enumerate(config.RANDOM_SEEDS[:config.NUM_RUNS], start=1):
        best, logbook = runEA(problem, seed)
        assert sorted(best) == list(range(len(problem))), "best tour must be a permutation"
        length = problem.getTotalDistanceRounded(best)
        evals = logbook[-1]["evals"]
        rows.append(dict(tsp=name, run=run, seed=seed, length=length, evals=evals,
                         generations=logbook[-1]["gen"],
                         deviation_pct=round(100 * (length - optimum) / optimum, 2)))
        logbooks[seed] = logbook
        if best_overall is None or length < best_overall[1]:
            best_overall = (best, length)
        print(f"{name} run {run} (seed {seed}): length {length}, "
              f"deviation {rows[-1]['deviation_pct']:.2f}%, evaluations {evals}")

    lengths = [r["length"] for r in rows]
    average = statistics.mean(lengths)
    summary = dict(tsp=name, cities=len(problem), optimum=optimum, best=min(lengths),
                   worst=max(lengths), average=round(average, 1),
                   std=round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0.0,
                   deviation_pct=round(100 * (average - optimum) / optimum, 2))

    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(config.RESULTS_DIR / f"{name}_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    saveBestTourPlot(problem, best_overall[0], best_overall[1],
                     config.RESULTS_DIR / f"{name}_best_tour.png")
    saveConvergencePlot(problem, logbooks, config.RESULTS_DIR / f"{name}_convergence.png")
    return summary


def main():
    assert len(config.RANDOM_SEEDS) >= config.NUM_RUNS, "need one seed per run"
    summaries = [solve(name) for name in config.parse_tsp_names("Run the EA")]

    print(f"\n{'TSP':<10}{'optimum':>10}{'best':>10}{'average':>12}{'worst':>10}{'std':>10}{'dev %':>8}")
    for s in summaries:
        print(f"{s['tsp']:<10}{s['optimum']:>10}{s['best']:>10}{s['average']:>12}"
              f"{s['worst']:>10}{s['std']:>10}{s['deviation_pct']:>8.2f}")

    with open(config.RESULTS_DIR / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nResults saved in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
