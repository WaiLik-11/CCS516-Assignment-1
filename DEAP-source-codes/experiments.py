"""Reproducible experiment runner for the step-by-step EA design.

Each step compares a few EA variants. A variant is a set of overrides on the step's base
parameters (a dict like solve_tsp.BASELINE). Every variant is run once per seed on every
selected TSP. A run terminates after config.MAX_EVALUATIONS fitness evaluations, or earlier
when it reaches the known optimal tour length (which only happens on berlin52, in step 9).
All outputs of a step are written to results/<step>/ (results/<step>-reporting/ with --seeds reporting):

    metadata.json          what was run: full parameters per variant, seeds, budget, command, versions
    runs.csv               one row per run: variant, tsp, seed, length, deviation %, evaluations used, ...
    summary.csv            per variant and TSP: best / mean / worst / std, deviation % of the mean
    comparison.csv         variant vs reference: wins per TSP, Wilcoxon signed-rank p-value over all runs
    convergence.csv        per run, every CONVERGENCE_STEP evaluations: best length so far, population average
    convergence_<tsp>.png  best length vs evaluations (mean over seeds, shaded min-max)
    best_tour_<variant>_<tsp>.png   only with --seeds reporting

Usage (from the repo root):
    uv run python DEAP-source-codes/experiments.py step1                 # all 5 TSPs, tuning seeds
    uv run python DEAP-source-codes/experiments.py step1 berlin52 --jobs 4
    uv run python DEAP-source-codes/experiments.py step1 --seeds reporting   # reporting seeds
"""

import argparse
import csv
import json
import multiprocessing
import platform
import statistics
import sys
import time
from bisect import bisect_right
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save figures to files; no blocking GUI windows
import matplotlib.pyplot as plt
import seaborn as sns
import deap
from scipy.stats import wilcoxon

import config
import solve_tsp
import tsp

# One entry per design step. Change ONE component per step and keep everything else at the
# best setting found so far (put the previous winner in "base"). comparison.csv compares the
# (variant, reference) pairs in "comparisons"; by default every variant vs the first one.
STEPS = {
    "step1": dict(
        description="Baseline: operators and parameters of the original template capped at 50,000 evaluations",
        base=solve_tsp.BASELINE,
        variants={"baseline": {}},
    ),
    "step2": dict(
        description="Mutation operator: swap vs inversion, checked at a higher rate and with ERX "
                    "to see whether the choice depends on the crossover studied later",
        base=solve_tsp.BASELINE,
        variants={
            "swap": {},                                            # = baseline (OX, p_m = 0.1)
            "inversion": {"mutation": "inversion"},
            "swap-pm0.5": {"p_mutation": 0.5},
            "inversion-pm0.5": {"mutation": "inversion", "p_mutation": 0.5},
            "swap-erx": {"crossover": "erx"},
            "inversion-erx": {"mutation": "inversion", "crossover": "erx"},
        },
        # inversion vs swap in the same setting
        comparisons=[("inversion", "swap"), ("inversion-pm0.5", "swap-pm0.5"), ("inversion-erx", "swap-erx")],
    ),
    # step 2 decision: keep swap mutation, p_m = 0.1 (= BASELINE)
    "step3": dict(
        description="Crossover operator: OX vs PMX vs UPMX vs ERX vs no crossover (swap mutation, p_m = 0.1)",
        base=solve_tsp.BASELINE,
        variants={
            "ox": {},                                              # = baseline, reference
            "pmx": {"crossover": "pmx"},
            "upmx-0.1": {"crossover": "upmx", "crossover_indpb": 0.1},
            "upmx-0.5": {"crossover": "upmx", "crossover_indpb": 0.5},
            "erx": {"crossover": "erx"},
            "no-crossover": {"p_crossover": 0.0},
        },
    ),
    # step 3 decision: drop PMX/UPMX; OX, ERX and no crossover depend on the number of generations
    "step4": dict(
        description="Population size (and so the number of generations) x crossover: "
                    "mu in {300, 100, 50}, elites = 10% of mu, crossover in {OX, ERX, none}",
        base=solve_tsp.BASELINE,
        variants={
            f"{cx}-{mu}": {"population_size": mu, "elite_size": mu // 10, **overrides}
            for mu in (300, 100, 50)
            for cx, overrides in (("ox", {}), ("erx", {"crossover": "erx"}), ("none", {"p_crossover": 0.0}))
        },
        comparisons=[
            # crossover vs OX at the same population size
            *[(f"{cx}-{mu}", f"ox-{mu}") for mu in (300, 100, 50) for cx in ("erx", "none")],
            # smaller population vs mu = 300, for the same crossover
            *[(f"{cx}-{mu}", f"{cx}-300") for cx in ("ox", "erx", "none") for mu in (100, 50)],
        ],
    ),
    # step 4 decision: OX (ERX converges prematurely in small populations); best mu = 50 lies at
    # the edge of the tested range, so extend the range downwards
    "step4b": dict(
        description="Population size below 50 with OX (elites = 10% of mu, at least 1)",
        base=solve_tsp.BASELINE,
        variants={f"ox-{mu}": {"population_size": mu, "elite_size": max(1, mu // 10)} for mu in (50, 30, 20, 10)},
    ),
}

# step 4/4b decision: OX, mu = 50, 5 elites (swap mutation, p_m = 0.1, tournament 2 unchanged)
STEP4_DESIGN = {**solve_tsp.BASELINE, "population_size": 50, "elite_size": 5}

STEPS["step5a"] = dict(
    description="Tournament size (parent selection pressure)",
    base=STEP4_DESIGN,
    variants={f"k{k}": {"tournament_size": k} for k in (2, 1, 3, 5, 7)},  # k = 1: random parents
)

# step 5a decision: keep tournament size 2
STEPS["step5b"] = dict(
    description="Crossover and mutation probabilities: p_c in {0.6, 0.9, 1.0} x p_m in {0.05, 0.1, 0.2}",
    base=STEP4_DESIGN,
    variants={  # current setting (p_c 0.9, p_m 0.1) first, as the reference
        f"pc{pc}-pm{pm}": {"p_crossover": pc, "p_mutation": pm}
        for pc, pm in [(0.9, 0.1)] + [(pc, pm) for pc in (0.6, 0.9, 1.0) for pm in (0.05, 0.1, 0.2)
                                      if (pc, pm) != (0.9, 0.1)]
    },
)

# step 5b decision: keep p_c 0.9, p_m 0.1 -> tuned design = STEP4_DESIGN. The instances prefer
# opposite probabilities, so step 6 lets the EA control them during the run.
PURSUIT_ARMS = [(0.6, 0.05), (0.9, 0.1), (1.0, 0.2)]  # low / tuned / high variation, from step 5b
STEPS["step6"] = dict(
    description="Parameter control: adaptive pursuit over (p_c, p_m) settings and self-adaptive p_m, "
                "vs the tuned fixed probabilities",
    base=STEP4_DESIGN,
    variants={
        "tuned": {},
        "adaptive-pursuit": {"control": "adaptive-pursuit", "ap_arms": PURSUIT_ARMS},
        "self-adaptive-pm": {"control": "self-adaptive-pm"},
    },
    comparisons=[("adaptive-pursuit", "tuned"), ("self-adaptive-pm", "tuned"),
                 ("self-adaptive-pm", "adaptive-pursuit")],
)

# step 6 check: the pursuit probabilities oscillate from generation to generation. Is the controller
# chasing noise in the per-generation reward? Slower learning rates average the reward over more generations.
STEPS["step6b"] = dict(
    description="Adaptive pursuit learning rates alpha = beta in {0.8 (step 6), 0.3, 0.1, 0.03}, and uniform random mixing of the arms (no learning)",
    base={**STEP4_DESIGN, "control": "adaptive-pursuit", "ap_arms": PURSUIT_ARMS},
    variants={"tuned": {"control": None}, "ap-0.8": {}, "ap-0.3": {"ap_alpha": 0.3, "ap_beta": 0.3},
              "ap-0.1": {"ap_alpha": 0.1, "ap_beta": 0.1}, "ap-0.03": {"ap_alpha": 0.03, "ap_beta": 0.03},
              # control: beta = 0 keeps every arm at probability 1/3 -> random mixing, no learning
              "uniform-mix": {"ap_beta": 0.0}},
    comparisons=[(v, "tuned") for v in ("ap-0.8", "ap-0.3", "ap-0.1", "ap-0.03", "uniform-mix")]
                + [(v, "uniform-mix") for v in ("ap-0.8", "ap-0.3", "ap-0.1", "ap-0.03")],
)

# ---- Verification and final runs --------------------------------------------------------------
# Final design = STEP4_DESIGN (OX, swap, k = 2, mu = 50, 5 elites, p_c 0.9, p_m 0.1). Tournament size,
# mu and the probabilities were already varied around exactly this design (steps 4b, 5a, 5b); the
# mutation (step 2) and crossover (step 3) were chosen at mu = 300, so they are re-checked here.
STEPS["step7"] = dict(
    description="Verification: mutation and crossover alternatives re-tested at the final design (mu = 50)",
    base=STEP4_DESIGN,
    variants={
        "tuned": {},
        "inversion": {"mutation": "inversion"},
        "pmx": {"crossover": "pmx"},
        "upmx-0.1": {"crossover": "upmx", "crossover_indpb": 0.1},
        "erx": {"crossover": "erx"},
        "no-crossover": {"p_crossover": 0.0},
    },
)

# step 7 result: inversion is now significantly better than swap (-4.9%, p = 0.011), unlike at
# mu = 300 in step 2 -> the order of the steps mattered. Re-check what was tuned with swap.
INVERSION_DESIGN = {**STEP4_DESIGN, "mutation": "inversion"}
STEPS["step7b"] = dict(
    description="Verification round 2: with inversion mutation, re-check p_m and the parameter controllers",
    base=INVERSION_DESIGN,
    variants={
        "inv-pm0.1": {},
        "inv-pm0.05": {"p_mutation": 0.05},
        "inv-pm0.2": {"p_mutation": 0.2},
        "inv-pm0.5": {"p_mutation": 0.5},
        "inv-uniform-mix": {"control": "adaptive-pursuit", "ap_arms": PURSUIT_ARMS, "ap_beta": 0.0},
        "inv-ap-0.1": {"control": "adaptive-pursuit", "ap_arms": PURSUIT_ARMS, "ap_alpha": 0.1, "ap_beta": 0.1},
    },
    comparisons=[(v, "inv-pm0.1") for v in ("inv-pm0.05", "inv-pm0.2", "inv-pm0.5", "inv-uniform-mix", "inv-ap-0.1")]
                + [("inv-pm0.05", "inv-uniform-mix"), ("inv-ap-0.1", "inv-uniform-mix")],
)

# Step 8: the design chosen after step 7b, on the reporting seeds (--seeds reporting). It is not the final
# design (step 9 adds heuristic initialization). Decision rule, fixed BEFORE running: the chosen
# design is the simplest variant that is not significantly worse than the best and significantly better
# than the fixed design. On the tuning seeds (step 7b) that is "inv-uniform-mix": best mean, better than
# inv-pm0.1 (p = 0.022), and adaptive pursuit is not better (p = 0.67). inv-pm0.05 is not worse than
# it (p = 0.16) but not better than inv-pm0.1 either (p = 0.35).
# Compared on the fresh seeds: the same design with fixed probabilities, the step 5 design (swap), the
# original template, and per-instance tuned p_m (best fixed p_m per TSP in step 7b).
STEP8_DESIGN = {**INVERSION_DESIGN, "control": "adaptive-pursuit", "ap_arms": PURSUIT_ARMS, "ap_beta": 0.0}
STEP7B_BEST_PM = {"berlin52": 0.1, "eil101": 0.2, "tsp225": 0.05, "pcb442": 0.05, "pr1002": 0.05}
STEPS["step8"] = dict(
    description="Random-start design after step 7b (OX, inversion, k = 2, mu = 50, uniform mix of (p_c, p_m)) vs fixed probabilities, "
                "the step 5 design (swap), the original template and per-instance tuned p_m",
    base=STEP8_DESIGN,
    variants={
        "random-start": {},
        "inv-fixed": {"control": None},
        "swap-fixed": {"control": None, "mutation": "swap"},
        "template": dict(solve_tsp.BASELINE),
        "per-instance": {"control": None, "per_tsp": {t: {"p_mutation": pm} for t, pm in STEP7B_BEST_PM.items()}},
    },
    comparisons=[("random-start", "inv-fixed"), ("random-start", "swap-fixed"), ("random-start", "template"),
                 ("inv-fixed", "swap-fixed"), ("per-instance", "inv-fixed"), ("per-instance", "random-start")],
)

# Step 9: initialization, the one design component not tested so far (the template's random start was
# kept). Heuristic initialization: a share of the initial population is built by nearest-neighbour from
# distinct random start cities (operators.nearestNeighbourTour); the rest stays random.
STEPS["step9"] = dict(
    description="Initialization: share of nearest-neighbour tours in the initial population "
                "(0 = random, as in step 8), at the step 8 design",
    base=STEP8_DESIGN,
    variants={
        "random": {},
        "nn-1": {"init_nn_fraction": 0.02},     # 1 of 50
        "nn-5": {"init_nn_fraction": 0.1},      # 5 of 50
        "nn-15": {"init_nn_fraction": 0.3},
        "nn-50": {"init_nn_fraction": 1.0},     # whole population
        # does uniform mixing still help with a seeded start?
        "nn-5-fixed": {"init_nn_fraction": 0.1, "control": None},
        # fixed probabilities; the (unused) pursuit settings reset to their defaults
        "nn-50-fixed": {"init_nn_fraction": 1.0, "control": None, "ap_arms": None, "ap_beta": 0.8},
        # the original template on the same seeds, so the final comparison is within this step
        "template": dict(solve_tsp.BASELINE),
    },
    comparisons=[(v, "random") for v in ("nn-1", "nn-5", "nn-15", "nn-50", "nn-5-fixed")]
                + [("nn-1", "nn-5"), ("nn-15", "nn-5"), ("nn-50", "nn-5"), ("nn-5-fixed", "nn-5"),
                   ("nn-50-fixed", "nn-50"), ("nn-50-fixed", "random"), ("nn-50-fixed", "template")],
)

# step 9 decision (tuning seeds), same rule as step 8 - the simplest variant not significantly worse
# than the best: nn-50-fixed (vs nn-50: +0.47%, p = 0.26). Final results: step9 --seeds reporting
assert {**STEP8_DESIGN, **STEPS["step9"]["variants"]["nn-50-fixed"]} == solve_tsp.FINAL_DESIGN, \
    "solve_tsp.FINAL_DESIGN must be the design chosen in step 9"

# convergence is recorded every CONVERGENCE_STEP evaluations (not every generation),
# so all runs share the same x-axis and convergence.csv stays small
CONVERGENCE_STEP = 500


def deviation(length, optimum):
    """Deviation percentage from the known optimum; 0% is best."""
    return 100 * (length - optimum) / optimum

# problems are cached per process (each worker loads them once)
_problems = {}

def getProblem(name):
    if name not in _problems:
        _problems[name] = tsp.TravelingSalesmanProblem(name)
    return _problems[name]


def runOne(job):
    """One run of one variant on one TSP. Returns (run row, convergence rows, best tour)."""
    variant, params, name, seed = job
    problem = getProblem(name)

    start = time.perf_counter()
    optimum = config.KNOWN_OPTIMA[name]
    # termination: MAX_EVALUATIONS evaluations, or earlier once the known optimum is reached
    best, logbook = solve_tsp.runEA(problem, params, seed, target=optimum)
    seconds = time.perf_counter() - start

    length = problem.getTotalDistanceRounded(best)
    assert logbook[-1]["evals"] == config.MAX_EVALUATIONS or (
        length == optimum and logbook[-1]["evals"] < config.MAX_EVALUATIONS), \
        "run must use exactly the evaluation budget, unless it reached the optimum earlier"
    run = dict(variant=variant, tsp=name, seed=seed, length=length, optimum=optimum,
               deviation_pct=round(deviation(length, optimum), 2), evals=logbook[-1]["evals"],
               generations=logbook[-1]["gen"], seconds=round(seconds, 2))

    # population at each grid point = the last generation finished within that many evaluations
    # (with elitism its min is the best length found so far; after an early stop at the optimum,
    # the curve stays at the optimum)
    evals = logbook.select("evals")
    curve = []
    for e in range(0, config.MAX_EVALUATIONS + 1, CONVERGENCE_STEP):
        if e >= evals[0]:
            record = logbook[bisect_right(evals, e) - 1]
            # parameter control state (arm probabilities or mean self-adaptive p_m), if any
            control = {k: round(v, 4) for k, v in record.items() if k.startswith(("p_arm", "pm_mean"))}
            curve.append(dict(variant=variant, tsp=name, seed=seed, evals=e,
                              best=int(record["min"]), avg=round(float(record["avg"]), 1),
                              edges=round(float(record["edges"]), 3), **control))
    return run, curve, list(best)


def summarise(runs):
    """Per variant and TSP: best / mean / worst / std of the final lengths."""
    groups = {}
    for r in runs:
        groups.setdefault((r["variant"], r["tsp"]), []).append(r)
    rows = []
    for (variant, name), group in groups.items():
        lengths = [r["length"] for r in group]
        mean = statistics.mean(lengths)
        rows.append(dict(variant=variant, tsp=name, runs=len(lengths), optimum=config.KNOWN_OPTIMA[name],
                         best=min(lengths), mean=round(mean, 1), worst=max(lengths),
                         std=round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0.0,
                         deviation_pct=round(deviation(mean, config.KNOWN_OPTIMA[name]), 2),
                         # runs that found the optimum (and stopped), and their mean evaluations used
                         optimum_hits=len(hits := [int(r["evals"]) for r in group if r["length"] == r["optimum"]]),
                         mean_evals_to_optimum=round(statistics.mean(hits)) if hits else "",
                         mean_seconds=round(statistics.mean(r["seconds"] for r in group), 1)))
    return rows


def compare(runs, pairs):
    """Variant vs reference, paired by (TSP, seed): same seed = same initial population.

    Per TSP: mean deviation % of both and how many of the seeds the variant wins.
    rel_diff_pct: how much shorter (<0) or longer (>0) the variant's tours are than the reference's
    (geometric mean of the paired length ratios).
    Over all TSPs: two-sided Wilcoxon signed-rank test [Derrac et al. 2011] on the paired
    log length ratios log(variant / reference). Relative differences weigh every instance equally;
    differences in deviation % would not (pr1002 deviations are ~100x those of berlin52, so its
    differences would always get the largest ranks). With 5 seeds a per-TSP test is pointless
    (the exact two-sided p-value cannot go below 0.0625), so the test pools all TSPs x seeds.
    """
    length = {(r["variant"], r["tsp"], r["seed"]): int(r["length"]) for r in runs}
    keys = list(dict.fromkeys((r["tsp"], r["seed"]) for r in runs))
    names = list(dict.fromkeys(t for t, _ in keys))
    rows = []
    for variant, reference in pairs:
        for name in names + ["all"]:
            pair_keys = [k for k in keys if name in ("all", k[0])]
            a = np.array([length[(variant, *k)] for k in pair_keys], dtype=float)
            b = np.array([length[(reference, *k)] for k in pair_keys], dtype=float)
            optima = np.array([config.KNOWN_OPTIMA[t] for t, _ in pair_keys])
            log_ratio = np.log(a / b)
            row = dict(variant=variant, reference=reference, tsp=name, pairs=len(a),
                       mean_dev_variant=round(deviation(a, optima).mean(), 2),
                       mean_dev_reference=round(deviation(b, optima).mean(), 2),
                       rel_diff_pct=round(100 * np.expm1(log_ratio.mean()), 2),
                       variant_wins=int((a < b).sum()), reference_wins=int((a > b).sum()), p_value="")
            if name == "all":
                row["p_value"] = f"{wilcoxon(log_ratio).pvalue:.2g}" if log_ratio.any() else "1"
            rows.append(row)
    return rows


def writeCsv(path, rows):
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))  # rows may have extra columns
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def writeMetadata(out_dir, **info):
    meta = dict(command="uv run python " + " ".join(sys.argv),
                created=datetime.now().isoformat(timespec="seconds"),
                max_evaluations=config.MAX_EVALUATIONS, **info,
                versions=dict(python=platform.python_version(), deap=deap.__version__, numpy=np.__version__))
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

def plotConvergence(curves, name, path):
    """Best length vs evaluations per variant: mean over seeds, shaded between min and max."""
    plt.figure()
    sns.set_style("whitegrid")
    for variant in dict.fromkeys(c["variant"] for c in curves):
        per_seed = {}
        for c in curves:
            if c["variant"] == variant and c["tsp"] == name:
                per_seed.setdefault(c["seed"], []).append(c)
        # all seeds of a variant share the same grid of evaluation counts
        grid = [c["evals"] for c in next(iter(per_seed.values()))]
        lines = np.array([[c["best"] for c in seed_curve] for seed_curve in per_seed.values()])
        plt.plot(grid, lines.mean(axis=0), label=variant)
        plt.fill_between(grid, lines.min(axis=0), lines.max(axis=0), alpha=0.2)
    plt.axhline(config.KNOWN_OPTIMA[name], color="black", linestyle="--", linewidth=1, label="optimum")
    plt.xlabel("Fitness evaluations")
    plt.ylabel("Best tour length")
    plt.title(f"{name}: convergence (mean over seeds, shaded min-max)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plotControl(curves, names, path):
    """Parameter control over the run, mean over seeds, for ONE variant: arm probabilities of
    adaptive pursuit (one panel per TSP) or the mean self-adaptive p_m (one line per TSP)."""
    pursuit = [c for c in curves if "p_arm0" in c and c["p_arm0"] != ""]
    adaptive = [c for c in curves if "pm_mean" in c and c["pm_mean"] != ""]
    if not pursuit and not adaptive:
        return
    sns.set_style("whitegrid")
    panels = len(names) if pursuit else 1
    fig, axes = plt.subplots(1, panels, figsize=(3.2 * panels, 3.2), sharex=True)
    axes = np.atleast_1d(axes)

    def meanCurve(rows, key):
        by_evals = {}
        for c in rows:
            by_evals.setdefault(c["evals"], []).append(float(c[key]))
        grid = sorted(by_evals)
        return grid, [statistics.mean(by_evals[e]) for e in grid]

    arms = sorted({k for c in pursuit for k in c if k.startswith("p_arm")})
    labels = PURSUIT_ARMS
    for ax, name in zip(axes if pursuit else [], names):
        for arm, (pc, pm) in zip(arms, labels):
            ax.plot(*meanCurve([c for c in pursuit if c["tsp"] == name], arm), label=f"p_c {pc}, p_m {pm}")
        ax.set_title(f"{name}: pursuit")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Evaluations")
    if pursuit:
        axes[0].set_ylabel("Arm probability")
        axes[0].legend(fontsize=7)
    if adaptive:
        ax = axes[-1]
        for name in names:
            ax.plot(*meanCurve([c for c in adaptive if c["tsp"] == name], "pm_mean"), label=name)
        ax.set_title("self-adaptive p_m")
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Mean p_m in population")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plotBestTour(name, tour, length, variant, path):
    plt.figure()
    getProblem(name).plotData(tour)
    plt.title(f"{name} best tour, {variant} (length {length})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def runStep(step, names, seed_set, jobs):
    spec = STEPS[step]
    seeds = config.TUNING_SEEDS if seed_set == "tuning" else config.RANDOM_SEEDS[:config.NUM_RUNS]
    out_dir = config.RESULTS_DIR / (step if seed_set == "tuning" else f"{step}-reporting")
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {v: {**spec["base"], **overrides} for v, overrides in spec["variants"].items()}
    writeMetadata(out_dir, step=step, description=spec["description"], tsps=names, seeds=seeds,
                  known_optima={n: config.KNOWN_OPTIMA[n] for n in names}, variants=variants)

    # a variant may override parameters for single TSPs: {"per_tsp": {"pr1002": {...}, ...}}
    def paramsFor(params, name):
        return {**{k: v for k, v in params.items() if k != "per_tsp"}, **params.get("per_tsp", {}).get(name, {})}
    jobs_list = [(v, paramsFor(params, name), name, seed)
                 for name in names for v, params in variants.items() for seed in seeds]
    print(f"{step}: {spec['description']}\n{len(jobs_list)} runs "
          f"({len(variants)} variants x {len(names)} TSPs x {len(seeds)} seeds), results in {out_dir}")

    runs, curves, best_tours = [], [], {}
    with multiprocessing.Pool(jobs) as pool:
        # imap keeps the job order, so the CSV files are identical for any number of jobs
        for run, curve, tour in pool.imap(runOne, jobs_list):
            runs.append(run)
            curves.extend(curve)
            key = (run["variant"], run["tsp"])
            if key not in best_tours or run["length"] < best_tours[key][1]:
                best_tours[key] = (tour, run["length"])
            print(f"  {run['variant']:<18}{run['tsp']:<10} seed {run['seed']:>4}: length {run['length']:>7}  "
                  f"({run['deviation_pct']:6.2f}%)  {run['evals']} evals  {run['seconds']:6.1f}s")

    summary = summarise(runs)
    writeCsv(out_dir / "runs.csv", runs)
    writeCsv(out_dir / "summary.csv", summary)
    writeCsv(out_dir / "convergence.csv", curves)
    for name in names:
        plotConvergence(curves, name, out_dir / f"convergence_{name}.png")
    for variant in dict.fromkeys(c["variant"] for c in curves):
        plotControl([c for c in curves if c["variant"] == variant], names, out_dir / f"control_{variant}.png")
    if seed_set == "reporting":  # tour plots are only needed for the reported results
        for (variant, name), (tour, length) in best_tours.items():
            plotBestTour(name, tour, length, variant, out_dir / f"best_tour_{variant}_{name}.png")

    report(step, out_dir, runs, summary, names)


def report(step, out_dir, runs, summary, names):
    """Print the summary, the best variant per instance and the comparisons; write comparison.csv."""
    spec = STEPS[step]
    variants = list(spec["variants"])
    print(f"\n{'variant':<18}{'TSP':<10}{'optimum':>9}{'best':>9}{'mean':>11}{'worst':>9}{'std':>9}{'dev %':>8}")
    for s in summary:
        print(f"{s['variant']:<18}{s['tsp']:<10}{s['optimum']:>9}{s['best']:>9}{s['mean']:>11}"
              f"{s['worst']:>9}{s['std']:>9}{s['deviation_pct']:>8.2f}")

    if len(variants) > 1:
        # would a separate choice per instance differ from one shared design?
        print("\nBest variant per instance (lowest mean):")
        for name in names:
            best = min((s for s in summary if s["tsp"] == name), key=lambda s: float(s["mean"]))
            print(f"  {name:<10}{best['variant']:<18}{float(best['deviation_pct']):>8.2f}%")

        pairs = spec.get("comparisons") or [(v, variants[0]) for v in variants[1:]]
        comparison = compare(runs, pairs)
        writeCsv(out_dir / "comparison.csv", comparison)
        print(f"\n{'variant':<18}{'vs':<14}{'TSP':<10}{'dev % variant':>14}{'dev % ref':>11}"
              f"{'rel. diff %':>12}{'wins':>8}{'p':>9}")
        for c in comparison:
            print(f"{c['variant']:<18}{c['reference']:<14}{c['tsp']:<10}{c['mean_dev_variant']:>14}"
                  f"{c['mean_dev_reference']:>11}{c['rel_diff_pct']:>12}{c['variant_wins']:>5}/{c['pairs']:<2}"
                  f"{c['p_value']:>9}")


def reanalyse(step, seed_set):
    """Recompute summary, comparisons and plots from a step's saved CSV files, without re-running the EA."""
    out_dir = config.RESULTS_DIR / (step if seed_set == "tuning" else f"{step}-reporting")
    with open(out_dir / "runs.csv") as f:
        runs = [{**r, "seed": int(r["seed"]), "length": int(r["length"]), "optimum": int(r["optimum"]),
                 "seconds": float(r["seconds"])} for r in csv.DictReader(f)]
    names = list(dict.fromkeys(r["tsp"] for r in runs))
    with open(out_dir / "convergence.csv") as f:
        curves = [{**c, "seed": int(c["seed"]), "evals": int(c["evals"]), "best": int(c["best"])}
                  for c in csv.DictReader(f)]
    for name in names:
        plotConvergence(curves, name, out_dir / f"convergence_{name}.png")
    for variant in dict.fromkeys(c["variant"] for c in curves):
        plotControl([c for c in curves if c["variant"] == variant], names, out_dir / f"control_{variant}.png")
    report(step, out_dir, runs, summarise(runs), names)


def main():
    parser = argparse.ArgumentParser(description="Run one EA design step")
    parser.add_argument("step", choices=list(STEPS))
    parser.add_argument("tsp", nargs="*", choices=config.TSP_NAMES,
                        help="one or more of: " + ", ".join(config.TSP_NAMES) + " (default: all)")
    parser.add_argument("--seeds", choices=["tuning", "reporting"], default="tuning",
                        help="tuning: config.TUNING_SEEDS (default); reporting: config.RANDOM_SEEDS, for the report")
    parser.add_argument("--jobs", type=int, default=1, help="parallel runs (results do not depend on it)")
    parser.add_argument("--reanalyse", action="store_true",
                        help="recompute comparison.csv from the saved runs.csv instead of running the EA")
    args = parser.parse_args()
    names = args.tsp or config.TSP_NAMES

    if args.reanalyse:
        reanalyse(args.step, args.seeds)
    else:
        runStep(args.step, names, args.seeds, args.jobs)


if __name__ == "__main__":
    main()
