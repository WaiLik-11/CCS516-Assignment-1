# CCS516 Assignment 1: TSP with DEAP

Solving berlin52, eil101, tsp225, pcb442 and pr1002 using a Distributed Evolutionary Algorithm (DEAP).

## Environment setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.12 is picked from `.python-version` (uv downloads it if missing).

```bash
git clone https://github.com/WaiLik-11/CCS516-Assignment-1.git
cd CCS516-Assignment-1
uv sync
```

`uv sync` creates `.venv/` and installs the exact versions pinned in `uv.lock`.

## Project structure

```
CCS516-Assignment-1/
├── CCS516_Assignment01_S261_v1.pdf   # assignment brief
├── pyproject.toml, uv.lock           # pinned dependencies (uv)
├── .python-version                   # Python 3.12
├── DEAP-source-codes/
│   ├── config.py                     # shared paths + experiment settings (edit here, not in scripts)
│   ├── tsp.py                        # TSP class: load data, tour length, plot optimal tours
│   ├── solve_tsp.py                  # the EA (DEAP toolbox, operators, 5 seeded runs per TSP)
│   └── elitism.py                    # eaSimple with elitism + 50,000-evaluation budget
├── tsp-data/                         # .tsp city coordinates and .opt.tour optimal tours
├── plots/                            # generated optimal-tour PNGs
└── results/                          # generated experiment results (git-ignored)
```

## Running scripts

Run everything from the repo root. Choose instances by name, or leave them out for all five
(`berlin52`, `eil101`, `tsp225`, `pcb442`, `pr1002`):

```bash
uv run python DEAP-source-codes/tsp.py                  # plot all optimal tours into plots/
uv run python DEAP-source-codes/tsp.py berlin52 eil101  # only these
uv run python DEAP-source-codes/solve_tsp.py            # EA: 5 runs on each of the 5 TSPs (~2 min)
uv run python DEAP-source-codes/solve_tsp.py berlin52   # EA on one problem only
uv run python DEAP-source-codes/tsp.py --help
```

No need to activate `.venv` manually; `uv run` uses it automatically.

`solve_tsp.py` writes to `results/`: `<tsp>_runs.csv` (one row per run), `summary.csv`
(best/average/worst/std/deviation per TSP), `<tsp>_best_tour.png` and `<tsp>_convergence.png`.

## EA design

All parameters are constants at the top of `solve_tsp.py`; budget, runs and seeds come from `config.py`.

| Component | Choice | Justification |
|---|---|---|
| Representation | Permutation of city indices (`array('i')`) | Every permutation is a valid tour, so no repair is needed (given by the assignment) |
| Fitness | `getTotalDistance` (float), minimised | Given objective. Reported lengths use `getTotalDistanceRounded` (TSPLIB rounding), the same convention as the optimum |
| Initialization | 10% nearest-neighbour tours from distinct random start cities, 90% random permutations | A purely random start wastes the budget: random init ended 28% (eil101), 418% (pcb442) and 1024% (pr1002) above optimum, against 5–12% with NN seeding. NN tours start at 19–32% above optimum, while the random 90% keeps diversity for crossover (30% NN was worse, 100% NN about equal). NN uses the distance matrix only and costs no fitness evaluations |
| Population model | Generational, μ = 50, with elitism: each generation = 2 elites + 48 offspring | Averaged over the 5 TSPs, μ = 50 beat 30 and 100 (and 200 in round 1). Smaller populations get more generations out of the fixed budget (~1,390 generations per run) |
| Parent selection | Tournament, size 5 (`tools.selTournament`) | Size 2 was clearly worse (pcb442 round 1: 16.8% vs 4.4% with size 5). Size 5 also beat 3 and 7 on average over the 5 TSPs |
| Crossover | Order crossover OX (`tools.cxOrdered`), p_c = 0.5 per pair | OX keeps a segment of one parent and the relative order of the other, which preserves adjacency (edges). OX beat PMX on all 5 instances. p_c = 0.5 beat 0, 0.2 and 0.8 on average |
| Mutation | Neighbour-guided inversion (`mutInversion`, custom: DEAP has no inversion), p_m = 0.5 per offspring | A 2-opt move (only 2 edges change): a random city is reconnected to one of its k = 5 nearest cities by reversing the segment in between. It reuses the inversion step of Inver-over (Tao & Michalewicz 1998) but is **not** Inver-over: the target city comes from a nearest-neighbour list, not from another individual, and there is no accept-if-better test. See the per-instance comparison below. p_m = 0.5 beat 0.3 and 0.8 on average |
| Survivor selection | Offspring replace parents; the 2 best ever (`tools.HallOfFame`) survive unchanged | Elitism guarantees that the best tour is never lost. A small elite keeps selection pressure on offspring |
| Termination | Exactly 50,000 fitness evaluations (`MAX_EVALUATIONS`) | Counted explicitly in `elitism.py`, including the initial population. Unchanged clones are not re-evaluated, so they cost nothing. The final generation is trimmed so the budget is never exceeded |
| Seeds | Run *i* uses `RANDOM_SEEDS[i]` = 1…5 (seeds both `random` and `numpy`) | Runs are reproducible; the seed is stored in `<tsp>_runs.csv` |

**Tuning.** Parameters were tuned one factor at a time with tuning seeds 101–103, kept separate from the
reporting seeds 1–5 so the reported numbers are not tuned on their own seeds. Each variant ran 3 times with the
full 50,000-evaluation budget. Round 1 (eil101, pcb442, pr1002) compared initialization, guided vs random
inversion and the population size. Round 2 (all 5 TSPs) varied each parameter around the final setting.

**Is the mutation suitable for all five TSPs?** Average deviation over tuning seeds 101–103 (round 2 setting, only the mutation changed):

| Mutation | berlin52 | eil101 | tsp225 | pcb442 | pr1002 |
|---|---|---|---|---|---|
| Swap (`tools.mutShuffleIndexes`) | 10.21% | 14.20% | 22.80% | 16.98% | 22.29% |
| Random inversion | 5.07% | 5.99% | 9.78% | 12.95% | 19.88% |
| Guided inversion, k = 3 | 5.11% | 10.02% | 9.63% | 8.77% | 13.90% |
| **Guided inversion, k = 5 (used)** | **2.07%** | 6.25% | **6.89%** | **4.81%** | **9.89%** |
| Guided inversion, k = 8 | 4.12% | 5.56% | 7.57% | 5.07% | 8.82% |

k = 5 is best or near-best on 4 of 5 instances and always far better than swap. On eil101 random inversion and k = 8 are slightly better, and k = 8 is best on pr1002, but with only 3 seeds these gaps are within noise. k = 3 is clearly too restrictive. The weak point is instance size: with a fixed 50,000 evaluations the EA reaches only 9.9–11.6% on pr1002 whatever the mutation, so it is not "suitable" in the sense of reaching near-optimal tours there.

For comparison, a true Inver-over (μ = 50, p = 0.02, one evaluation per child, accept if no worse, same initialization) was also tested on seeds 101–103: berlin52 1.13%, eil101 9.22%, tsp225 20.26%, pcb442 19.08%, pr1002 22.30%. It is better only on berlin52. Its roughly 1,000 generations are too few on the larger instances, so it was not adopted.

**Results** (seeds 1–5, 50,000 evaluations each; deviation % = (average − optimum) / optimum, rounded lengths):

| TSP | Optimum | Best | Average | Worst | Std | Deviation % |
|---|---|---|---|---|---|---|
| berlin52 | 7542 | 7542 | 7898.4 | 8172 | 304.9 | 4.73 |
| eil101 | 629 | 645 | 663.2 | 684 | 14.9 | 5.44 |
| tsp225 | 3916 | 4121 | 4166.0 | 4220 | 41.0 | 6.38 |
| pcb442 | 50778 | 53065 | 53846.6 | 54820 | 725.8 | 6.04 |
| pr1002 | 259045 | 284088 | 288971.6 | 294741 | 3876.2 | 11.55 |

## Conventions

- Paths, TSP names, the 50,000-evaluation cap, number of runs and random seeds live in `config.py`; import them, don't hardcode.
- Use only the libraries in `pyproject.toml` (DEAP environment).
- Generated files (`*.pickle`, `results/`) are git-ignored.
- Work on a branch, open a PR, and link the issue (`Closes #N`).

## Task split

| Area | Owner | Issues |
|---|---|---|
| Plots, optimal lengths, experiment runner | Member A | #2, #3, #5 |
| EA design and tuning (`solve_tsp.py`, `elitism.py`) | Member B | #4, #6 |

## Notes

- Dependencies are the same as the DEAP course environment, except `multidict` is 6.0.5 (6.0.4 fails to build on macOS arm64 with Python 3.12).
- To add a package: `uv add <package>`, then commit `pyproject.toml` and `uv.lock`.
