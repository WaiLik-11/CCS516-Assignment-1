# CCS516 Assignment 1: Solving the TSP with a Distributed Evolutionary Algorithm (DEAP)

Solving berlin52, eil101, tsp225, pcb442 and pr1002 with an evolutionary algorithm (EA) built on DEAP, limited to 50,000 fitness evaluations per run and averaged over 5 runs.

| Member | Student ID |
|--------|-----------|
| AU WAI LIK | 25203542 |
| TIFFANY TIEN NEE PRAGASAM | 25203306 |

Group 02, CCS516 Computational Intelligence, Semester 1, 2026/2027, Universiti Sains Malaysia.

---

## 1. Overview

| Task | Description | Script | Output |
|------|-------------|--------|--------|
| 1 | Plot the optimal tour of each instance | [tsp.py](DEAP-source-codes/tsp.py) | [plots/](plots/) |
| 2 | Compute the optimal tour lengths (TSPLIB rounding) | [tsp.py](DEAP-source-codes/tsp.py) | [results/optimal_length.txt](results/optimal_length.txt) |
| 3 | Design the EA, tune it step by step, and report the final results | [solve_tsp.py](DEAP-source-codes/solve_tsp.py), [elitism.py](DEAP-source-codes/elitism.py), [experiments.py](DEAP-source-codes/experiments.py) | [results/](results/) |

The full write-up (algorithm, experiments, results, problems and pitfalls) is in [CCS516_Group02_A01.pdf](CCS516_Group02_A01.pdf).

## 2. Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| [uv](https://docs.astral.sh/uv/) | latest | Installs Python 3.12 and every dependency pinned in `uv.lock`, so everyone gets identical versions |
| Git | any | Clone the repository |

Install uv:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 3. Libraries

| Library | Role |
|---------|------|
| [DEAP](https://deap.readthedocs.io/) 1.4.1 | Evolutionary algorithm framework: toolbox, crossover, mutation, selection, hall of fame |
| NumPy | Distance matrix, tour lengths, random seeding |
| SciPy | Wilcoxon signed-rank test for comparing variants |
| matplotlib, seaborn | Tour plots and convergence graphs |

`pyproject.toml` also pins the other packages of the course's DEAP environment. The code does not use them.

## 4. Installation

```bash
git clone https://github.com/WaiLik-11/CCS516-Assignment-1.git
cd CCS516-Assignment-1

uv sync        # creates .venv with Python 3.12 and the exact versions in uv.lock
uv run python -c "import deap; print(deap.__version__)"   # should print 1.4.1
```

There is no need to activate `.venv`: `uv run` uses it automatically.

To add a package, run `uv add <package>` and commit both `pyproject.toml` and `uv.lock`.

## 5. Execution Procedure

Run all commands from the repository root. Commands that take instance names use all five (`berlin52`, `eil101`, `tsp225`, `pcb442`, `pr1002`) when none is given. Add `--help` to a script to see its options.

### Tasks 1 and 2: optimal tour plots and lengths

```bash
uv run python DEAP-source-codes/tsp.py                  # all five instances
uv run python DEAP-source-codes/tsp.py berlin52 eil101  # only these
```

Writes `plots/<tsp>_opt.png` and prints the table below, which is also saved to `results/optimal_length.txt`. The script stops with an error if a computed length differs from the known optimum.

| Instance | Cities | Optimal length (TSPLIB nint) | TSP-BEST | Unrounded sum | Edges ending in exactly .5 |
|---|---|---|---|---|---|
| berlin52 | 52 | 7542 | 7542 | 7544.37 | 0 |
| eil101 | 101 | 629 | 629 | 642.31 | 0 |
| tsp225 | 225 | 3916 | 3916 | 3859.00 | 124 |
| pcb442 | 442 | 50778 | 50778 | 50783.55 | 0 |
| pr1002 | 1002 | 259045 | 259045 | 259066.66 | 0 |

**Rounding trap.** TSPLIB rounds each edge with `nint(x) = (int)(x + 0.5)`, which always rounds a half up. Python's `round()` rounds halves to the nearest even number (`round(28.5)` is 28). Only tsp225 is affected, because 124 of its optimal-tour edges end in exactly .5. Python's `round()` gives 3861 there instead of the published 3916. We use the TSPLIB rule everywhere, both for the optimal lengths and as the EA fitness.

### Task 3: the final EA (reporting seeds 1-5)

```bash
uv run python DEAP-source-codes/experiments.py step9 --seeds reporting --jobs 16
```

Runs every step 9 variant 5 times per instance and writes `results/step9-reporting/`. The final design is the `nn-50-fixed` variant. Its rows in `summary.csv` hold the reported numbers, and `best_tour_nn-50-fixed_<tsp>.png` shows its best tour per instance.

Set `--jobs` to the number of CPU cores. It only affects speed, not the results.

To run the final EA once (reporting seed 1) and watch its progress:

```bash
uv run python DEAP-source-codes/solve_tsp.py berlin52
```

### Task 3: reproducing the design steps (tuning seeds 101-105)

Each step compares a few variants of the EA and writes `results/<step>/`. Run them in order:

```bash
for step in step1 step2 step3 step4 step4b step5a step5b step6 step6b step7 step7b step9; do
    uv run python DEAP-source-codes/experiments.py $step --jobs 16
done
```

This takes about 25 minutes with `--jobs 16`. Steps 2, 3 and 7 are the slowest because of ERX. To run one step, run its line on its own.

Step 8 re-checks the design chosen before heuristic initialization on the reporting seeds. It is not used for any decision:

```bash
uv run python DEAP-source-codes/experiments.py step8 --seeds reporting --jobs 16
```

### Checking a re-run

Runs are deterministic, so a re-run must give the same tour lengths as the committed results. Only the timings change. Compare the first 8 columns of every `runs.csv` against the last commit:

```bash
for f in results/*/runs.csv; do
    if diff -q <(git show HEAD:"$f" | cut -d, -f1-8) <(cut -d, -f1-8 "$f") > /dev/null; then
        echo "identical  $f"
    else
        echo "CHANGED    $f"
    fi
done
```

## 6. Assignment Project Structure

```
CCS516-Assignment-1/
├── README.md
├── CCS516_Group02_A01.pdf            # the final report
├── pyproject.toml, uv.lock           # pinned dependencies (uv)
├── .python-version                   # Python 3.12
├── DEAP-source-codes/
│   ├── config.py                     # shared paths and settings (50,000 evaluations, 5 runs, seeds)
│   ├── tsp.py                        # TSP class: load data, tour length, plot optimal tours (Tasks 1, 2)
│   ├── solve_tsp.py                  # the EA: DEAP toolbox built from a parameter dict, one seeded run
│   ├── elitism.py                    # eaSimple with elitism, an exact evaluation budget, parameter control
│   ├── operators.py                  # operators DEAP lacks: enhanced edge recombination (ERX), nearest-neighbour tours
│   └── experiments.py                # reproducible experiment runner, one entry per design step
├── tsp-data/                         # .tsp city coordinates and .opt.tour optimal tours
├── plots/                            # optimal-tour PNGs (Task 1)
└── results/                          # optimal_length.txt and one folder per design step
```

`tsp-data/*.pickle` files are caches of the distance matrices. They are created on the first run and are git-ignored. Delete them after changing the distance code.

Settings that apply to every script (paths, instance names, the 50,000-evaluation cap, number of runs, seeds) live in `config.py`.

## 7. The Final EA Design

| Component | Choice |
|-----------|--------|
| Representation | Permutation of the city indices |
| Fitness | TSPLIB EUC_2D tour length (each edge rounded with `nint`), minimised |
| Initialization | 50 nearest-neighbour tours, each from a different random start city |
| Parent selection | Tournament, size 2 |
| Crossover | Order crossover (OX), probability 0.9 per pair |
| Mutation | Inversion, probability 0.1 per offspring |
| Population model | Generational, population 50, with the 5 best tours copied unchanged each generation (elitism) |
| Termination | 50,000 fitness evaluations, or earlier when the known optimum is reached |
| Seeds | Tuning 101-105, reporting 1-5. Each run seeds `random` and NumPy |

### Final results

Reporting seeds 1-5, 50,000 evaluations per run. Deviation % = (average - optimum) / optimum x 100, so 0% is best.

| Instance | Optimum | Best | Average | Worst | Std | Deviation % |
|---|---|---|---|---|---|---|
| berlin52 | 7542 | 7542 | 7676.8 | 7807 | 115.4 | 1.79 |
| eil101 | 629 | 654 | 666.0 | 681 | 9.9 | 5.88 |
| tsp225 | 3916 | 4369 | 4442.8 | 4520 | 61.3 | 13.45 |
| pcb442 | 50778 | 58421 | 59095.2 | 59620 | 483.5 | 16.38 |
| pr1002 | 259045 | 311547 | 312796.0 | 313703 | 895.0 | 20.75 |

On the same seeds the final design's tours are 73.9% shorter than the original template's (25 of 25 runs, Wilcoxon p = 6e-8). On pcb442 and pr1002 the EA improves its best starting nearest-neighbour tour by less than 1%, so most of the quality there comes from the initialization.

## 8. How the Design Was Chosen

Every design step is an entry in `STEPS` in `experiments.py`: a base parameter dict and a few variants that each change **one** component. Running `experiments.py <step>` runs every variant on every selected instance with every seed, and writes to `results/<step>/`:

- `metadata.json`: full parameters per variant, seeds, budget, command, library versions
- `runs.csv` (one row per run) and `summary.csv` (best, mean, worst, std, deviation %)
- `comparison.csv`: wins per instance and a Wilcoxon signed-rank test on the paired log length ratios, pooled over all instances (`--reanalyse` recomputes it from the saved runs)
- `convergence.csv` (best length every 500 evaluations) and `convergence_<tsp>.png`
- `best_tour_<variant>_<tsp>.png`, only with `--seeds reporting`

All comparisons use the tuning seeds 101-105. The final results use the separate reporting seeds 1-5, so they are not tuned on their own seeds.

| Step | Question | Result |
|---|---|---|
| 1 | Baseline: the template's operators, capped at 50,000 evaluations | 14.8% to 1507% deviation, far from converged on the large instances |
| 2 | Mutation: swap vs inversion | No significant gain from inversion at the baseline's population size. Keep swap, revisit in step 7 |
| 3 | Crossover: OX, PMX, UPMX, ERX, none | PMX and UPMX are worse than OX. ERX is best at population 300. "No crossover" only looks good because unchanged offspring are free (1842 vs 203 generations) |
| 4, 4b | Population size | Population 50 with OX is best. ERX collapses small populations |
| 5a | Tournament size | Size 2 beats 1, 3, 5 and 7 |
| 5b | Crossover and mutation probabilities | No setting is significantly better overall. Keep 0.9 and 0.1 |
| 6, 6b | Parameter control (adaptive pursuit, self-adaptive p_m, uniform mix) | The gain comes from mixing settings, not from learning. Not significant after correcting for 6 comparisons |
| 7, 7b | Verification at population 50 | Inversion now beats swap (-4.9%, p = 0.011): the best mutation depended on the population size |
| 8 | Check on the reporting seeds (random start) | Confirms steps 6b-7b. Not used for any decision |
| 9 | Initialization: 0, 1, 5, 15 or 50 nearest-neighbour tours | Largest effect of the study: 59.5% shorter tours (p = 6e-8). Mixing no longer helps, so the final design uses fixed probabilities (`nn-50-fixed`) |

The reasoning, tables and statistics for every step are in the report.

## 9. References

1. G. Reinelt, "TSPLIB 95", Universitat Heidelberg, 1995: https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/tsp95.pdf
2. Best known TSPLIB solutions (TSP-BEST): https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html
3. F.-A. Fortin et al., "DEAP: Evolutionary algorithms made easy", *J. Mach. Learn. Res.*, vol. 13, pp. 2171-2175, 2012.
4. The full reference list is at the end of the report.

## 10. Notes

- The dependencies match the course's DEAP environment, except `multidict` is 6.0.5 (6.0.4 fails to build on macOS arm64 with Python 3.12).
- Generated results in `results/` and `plots/` are committed as the evidence for the report. Running a script overwrites them, and `git checkout -- results plots` restores the committed copies.
