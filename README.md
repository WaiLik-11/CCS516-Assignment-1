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
│   ├── solve_tsp.py                  # the EA: DEAP toolbox built from a parameter dict, one seeded run
│   ├── elitism.py                    # eaSimple with elitism + 50,000-evaluation budget
│   ├── operators.py                  # operators DEAP lacks: edge recombination crossover (ERX)
│   └── experiments.py                # reproducible experiment runner, one entry per design step
├── tsp-data/                         # .tsp city coordinates and .opt.tour optimal tours
├── plots/                            # generated optimal-tour PNGs
└── results/                          # experiment results, one folder per design step
```

## Running scripts

Run everything from the repo root. No need to activate `.venv` manually; `uv run` uses it
automatically. Most commands take instance names, or all five (`berlin52`, `eil101`, `tsp225`,
`pcb442`, `pr1002`) when left out. Add `--help` to any script for its options.

### Questions 1 and 2: optimal tour plots and optimal tour lengths

```bash
uv run python DEAP-source-codes/tsp.py                  # all five instances
uv run python DEAP-source-codes/tsp.py berlin52 eil101  # only these
```

Writes `plots/<tsp>_opt.png` (Q1) and prints the optimal tour length of each instance (Q2), computed
with TSPLIB's per-edge rounding, next to the two common mistakes (Python `round()` per edge, rounding
the total).

### Question 3: the EA design, step by step (tuning seeds 101–105)

Each step compares a few variants of the EA and writes `results/<step>/` (see [Experiments](#experiments)).
Run them in order; the decision of each step is in the [Step log](#step-log).

```bash
for step in step1 step2 step3 step4 step4b step5a step5b step6 step6b step7 step7b step9; do
    uv run python DEAP-source-codes/experiments.py $step --jobs 16
done
```

| Step | Question it answers |
|---|---|
| `step1` | Baseline: the template's operators, capped at 50,000 evaluations |
| `step2` | Mutation: swap vs inversion |
| `step3` | Crossover: OX vs PMX vs UPMX vs ERX vs none |
| `step4`, `step4b` | Population size (and so the number of generations) |
| `step5a` | Tournament size (selection pressure) |
| `step5b` | Crossover and mutation probabilities |
| `step6`, `step6b` | Parameter control: adaptive pursuit, self-adaptive p_m, uniform mix |
| `step7`, `step7b` | Verification: re-test earlier choices at the design after step 5 |
| `step9` | Initialization: random vs nearest-neighbour seeded start (decides the final design) |

About 25 minutes in total with `--jobs 16`; steps 2, 3 and 7 are the slowest (ERX). Set `--jobs` to
the number of CPU cores; it does not change the results. To run a single step, run its line on its own.

### Question 3: final results (reporting seeds 1–5)

```bash
uv run python DEAP-source-codes/experiments.py step9 --seeds reporting --jobs 16
```

Runs every step 9 variant 5 times per instance on the reporting seeds and writes
`results/step9-reporting/`. **The final design is the `nn-50-fixed` variant**: its rows in
`summary.csv` hold the reported best / average / worst / std and the deviation % of the average from
the optimum, `best_tour_nn-50-fixed_<tsp>.png` holds its best tour per instance, and
`comparison.csv` compares it with random initialization and with the original template (`template`)
on the same seeds.

Step 8 is an intermediate check, not the final results: it runs the design chosen before heuristic
initialization (random start) on the reporting seeds, to confirm the inversion and parameter-control
decisions on fresh seeds:

```bash
uv run python DEAP-source-codes/experiments.py step8 --seeds reporting --jobs 16
```

To run the final EA once (reporting seed 1) and watch its progress:

```bash
uv run python DEAP-source-codes/solve_tsp.py berlin52
```

### Checking a re-run

Runs are deterministic, so a re-run must give the same tour lengths as the committed results; only
the timings change. Compare the first 8 columns of every `runs.csv` against the last commit:

```bash
for f in results/*/runs.csv; do
    if diff -q <(git show HEAD:"$f" | cut -d, -f1-8) <(cut -d, -f1-8 "$f") > /dev/null; then
        echo "identical  $f"
    else
        echo "CHANGED    $f"
    fi
done
```

## Experiments

Every design step is an entry in `STEPS` in `experiments.py`: a base parameter dict and a few
variants that each change **one** component. `experiments.py <step>` runs every variant on every
selected TSP with every seed, each run with exactly 50,000 fitness evaluations, and writes
`results/<step>/`:

- `metadata.json`: full parameters per variant, seeds, budget, command, library versions
- `runs.csv` (one row per run), `summary.csv` (best/mean/worst/std, deviation % of the mean)
- `convergence.csv` (best length so far every 500 evaluations, per run) and `convergence_<tsp>.png`
- `best_tour_<variant>_<tsp>.png`, only with `--seeds reporting`

Results do not depend on `--jobs`: each run seeds `random`/`numpy` itself.

Comparisons (`comparison.csv`): wins per instance, and a Wilcoxon signed-rank test on the paired log length
ratios pooled over all instances. `--reanalyse` recomputes them from a step's saved `runs.csv`.

Seeds: `TUNING_SEEDS` (101–105) for all comparisons, `RANDOM_SEEDS` (1–5) only for the final
reported results, so the reported numbers are not tuned on their own seeds.

Fitness is the TSPLIB EUC_2D tour length (each edge rounded with `nint(x) = (int)(x + 0.5)`), the
same measure as the known optima. Deviation % = (mean − optimum) / optimum × 100.

### Step log

| Step | Question | Variants | Decision |
|---|---|---|---|
| 1 | Baseline: original template operators, capped at 50,000 evaluations | baseline | Reference for step 2 (results below) |
| 2 | Mutation: swap vs inversion (at p_m 0.1 and 0.5, and with ERX as an early order check) | swap, inversion, swap-pm0.5, inversion-pm0.5, swap-erx, inversion-erx | No significant difference between swap and inversion in any setting (swap significantly better in the baseline setting, p = 0.026; p = 0.29 and 0.87 in the others), and the winner changes with the setting. p_m 0.5 is worse than 0.1 for both. **Keep swap, p_m 0.1** (no evidence to change); revisit mutation in the verification step. ERX gave the largest gain, so crossover is step 3 |
| 3 | Crossover: OX vs PMX vs UPMX (indpb 0.1, 0.5) vs ERX vs none | ox, pmx, upmx-0.1, upmx-0.5, erx, no-crossover | PMX and UPMX are significantly worse than OX (0–1/25 wins, p < 1e-6): drop them. ERX beats OX on 4/5 instances (all seeds) but loses on pr1002 (pooled: 9.6% shorter tours, p = 0.0004). No crossover is best on the 3 largest instances, but only because its unchanged offspring are free: it gets 1842 generations vs 203. **Keep OX and ERX (no crossover as control); decide in step 4 together with the population size** |
| 4 | Population size x crossover: mu in {300, 100, 50} (elites 10%), crossover in {OX, ERX, none} | ox/erx/none-300, -100, -50 | Smaller mu is the biggest gain so far (OX mu 50 vs 300: 22/25 wins, p = 1.5e-6). ERX is significantly worse than OX at mu 100 and 50 (2–3/25 wins): it collapses small populations. OX beats no crossover on 4/5 instances at mu 50 (p = 0.0006, pr1002 the exception) |
| 4b | Extend mu below 50 (best value was at the edge): OX, mu in {50, 30, 20, 10} | ox-50, ox-30, ox-20, ox-10 | mu 50 best on 4/5 instances; mu 30 only better on pr1002 (not significant overall, p = 0.16); mu ≤ 20 clearly worse. **Use OX, mu = 50, 5 elites**; drop ERX |
| 5a | Tournament size k in {1, 2, 3, 5, 7} | k1, k2, k3, k5, k7 | k = 2 significantly better than all others (k3: +8.9% longer tours, p = 0.0006; k1 random parents: +67%). **Keep k = 2** |
| 5b | p_c in {0.6, 0.9, 1.0} x p_m in {0.05, 0.1, 0.2} | pc*-pm* | No setting significantly different overall (all p ≥ 0.096): small instances prefer more variation (p_c 1.0, p_m 0.2), pr1002 less (p_c 0.6, p_m 0.05, 5/5 wins). **Keep p_c 0.9, p_m 0.1**; motivates parameter control |
| 6 | Parameter control vs tuned: adaptive pursuit over (p_c, p_m) in {(0.6, 0.05), (0.9, 0.1), (1.0, 0.2)}; self-adaptive p_m | tuned, adaptive-pursuit, self-adaptive-pm | Neither significant overall (pursuit −1.0%, p = 0.30; self-adaptive +1.6%, p = 0.28). Self-adaptive p_m helps berlin52/eil101 (5/5) but hurts pcb442/pr1002 (0/5): p_m rises on every instance, so it cannot learn pr1002's preference for less variation. Pursuit probabilities oscillate |
| 6b | Pursuit learning rates {0.8, 0.3, 0.1, 0.03} + uniform mix (beta = 0, no learning) | tuned, ap-*, uniform-mix | ap-0.1 best vs tuned (−2.76%, p = 0.012) but uniform mix also helps (−1.41%, p = 0.039) and no pursuit variant beats uniform mix (best p = 0.28): the gain is from mixing settings, not learning. Not significant after Bonferroni (5 comparisons). **Candidates for the final runs: tuned, uniform-mix, ap-0.1** |
| 7 | Verification at the final design (mu = 50): inversion, PMX, UPMX, ERX, no crossover | tuned, inversion, pmx, upmx-0.1, erx, no-crossover | **Order effect: inversion now significantly better than swap** (−4.9%, p = 0.011; swap had won at mu = 300). OX confirmed (all alternatives significantly worse) |
| 7b | With inversion: p_m in {0.05, 0.1, 0.2, 0.5}, uniform mix, pursuit 0.1 | inv-* | p_m 0.1 kept; uniform mix again better than fixed (−1.85%, p = 0.022); pursuit not better than uniform mix (p = 0.67). **Final design: OX, inversion, k 2, mu 50, uniform mix of (p_c, p_m)** (decision rule fixed before step 8) |
| 8 | Check of the random-start design on the reporting seeds 1–5 (`--seeds reporting`) | random-start, inv-fixed, swap-fixed, template, per-instance | Confirmed: random-start vs fixed −2.0% (p = 0.045), vs swap design −5.5% (p = 0.0015), vs template −35.5% (25/25). Per-instance tuning no better (p = 0.85). Deviation % (random start, superseded by step 9): berlin52 8.80, eil101 20.06, tsp225 139.37, pcb442 382.26, pr1002 948.83 |
| 9 | Initialization: 0, 1, 5, 15 or 50 of the 50 initial tours by nearest neighbour (distinct random starts); fixed vs mixed probabilities | random, nn-1, nn-5, nn-15, nn-50, nn-5-fixed, nn-50-fixed | Largest effect of the study: seeding gives 58–60% shorter tours (p < 1e-6); 50 NN better than 5 (p = 0.0004); mixing no longer helps (p = 0.26). **Final design: nn-50-fixed** (OX p_c 0.9, inversion p_m 0.1, k 2, mu 50, 5 elites, all-NN start). Reporting seeds (`step9-reporting`, final results): berlin52 1.79, eil101 5.88, tsp225 13.45, pcb442 16.38, pr1002 20.75 deviation %; 73.9% shorter tours than the template (25/25). On pcb442/pr1002 the EA improves its best starting tour by < 1% |

Step 1, tuning seeds 101–105 (OX p_c = 0.9, swap mutation p_m = 0.1, tournament 2, μ = 300, 30 elites, random initialization):

| TSP | Optimum | Best | Mean | Worst | Std | Deviation % |
|---|---|---|---|---|---|---|
| berlin52 | 7542 | 8460 | 8659.8 | 9022 | 243.0 | 14.82 |
| eil101 | 629 | 1069 | 1155.4 | 1259 | 77.8 | 83.69 |
| tsp225 | 3916 | 17607 | 18023.6 | 18291 | 281.0 | 360.26 |
| pcb442 | 50778 | 414058 | 429289.6 | 436732 | 9275.6 | 745.42 |
| pr1002 | 259045 | 4111551 | 4163176 | 4224493 | 44063.5 | 1507.12 |

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
