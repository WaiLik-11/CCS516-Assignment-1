<!--
DRAFT. Final version: Times New Roman 12 pt, single spacing, max 15 pages incl. figures.
Markers:  [TODO] = still to do / fill in after the experiment,  [CHECK] = verify before submitting.
Citations are IEEE style [n]; see References. [CHECK] every reference (pages, years) in Google Scholar.
-->

# Cover page

**CCS516 Computational Intelligence — Assignment 01**
**Solving the Traveling Salesman Problem (TSP) Using Distributed Evolutionary Algorithms in Python (DEAP)**

School of Computer Sciences, Universiti Sains Malaysia, Semester 1, 2026/2027

Group: [TODO GroupXX]

| Name | Matric No. | Contribution |
|---|---|---|
| [TODO] | [TODO] | [TODO] |
| [TODO] | [TODO] | [TODO] |

Submission package: `CCS516_[GroupNumber]_A01.zip`. Source code: [TODO repository link / commit hash].

---

# 1. Introduction

The Traveling Salesman Problem (TSP) asks for the shortest closed tour that visits each of *n* cities exactly once. It is NP-hard, so exact methods become impractical as *n* grows, and metaheuristics such as evolutionary algorithms (EAs) are commonly used to find good tours in limited time [1], [2].

This report designs an EA with the DEAP framework [3] for five symmetric TSPLIB instances [4]: berlin52, eil101, tsp225, pcb442 and pr1002 (52 to 1002 cities). Following the assignment, every run is limited to **50,000 fitness evaluations**, results are **averaged over 5 runs**, and quality is reported as the deviation of the average tour length from the known optimum [5]:

$$\text{Deviation (\%)} = \frac{\text{Average} - \text{Optimum}}{\text{Optimum}} \times 100, \qquad 0\% \text{ is best.}$$

The given files `tsp.py`, `solve_tsp.py` and `elitism.py` are the starting point. Section 2 covers the optimal tour plots and lengths (Tasks 1 and 2). Section 3 describes the EA and the candidate operators. Section 4 explains how the experiments were designed to be reproducible. Section 5 reports the step-by-step design decisions and final results. Section 6 discusses problems and pitfalls.

# 2. Optimal tours and optimal tour lengths (Tasks 1 and 2)

## 2.1 Reading the TSPLIB files

Each `.tsp` file has a header (e.g. `EDGE_WEIGHT_TYPE : EUC_2D`) followed by a `NODE_COORD_SECTION` with one line per city: `index x y` [4]. `tsp.py` skips the header, reads the coordinates until `EOF` (pr1002 has no `EOF` line, so the end of the file also stops it), and computes the full distance matrix once. The matrix is cached on disk with `pickle`.

Each `.opt.tour` file lists the optimal tour in its `TOUR_SECTION` as 1-based city IDs ending with `-1`. `loadTour()` converts them to 0-based indices and checks that the tour is a permutation of all cities.

## 2.2 Plotting

`plotData()` draws every city as a red dot and connects consecutive cities of the tour with blue lines, including the edge from the last city back to the first. Figure 1 shows the five optimal tours.

[TODO Figure 1: `plots/berlin52_opt.png`, `eil101_opt.png`, `tsp225_opt.png`, `pcb442_opt.png`, `pr1002_opt.png` in a grid like Figure 1 of the assignment.]

## 2.3 Euclidean distance as defined by TSPLIB

All five instances are of type `EUC_2D`. TSPLIB defines the distance between cities *i* and *j* as [5]:

```
xd = x[i] - x[j];  yd = y[i] - y[j];
dij = nint( sqrt( xd*xd + yd*yd ) );
```

where `nint` rounds to the nearest integer and "can be implemented as `(int)(x + 0.5)`" in C [5]. Two details matter:

1. **Each edge is rounded before summing.** The tour length is a sum of integers, not the rounded sum of real distances.
2. **Halves are rounded up.** `(int)(x + 0.5)` turns 28.5 into 29.

Our implementation precomputes the integer matrix once:

```python
self.distancesRounded = (np.asarray(self.distances) + 0.5).astype(np.int64)
```

`astype` truncates like a C cast, which is correct here because distances are never negative.

## 2.4 The rounding trap in Python

Python's built-in `round()` does **not** behave like C/C++. Since Python 3, `round()` uses *round half to even* ("banker's rounding") [6]: `round(0.5) = 0`, `round(1.5) = 2`, `round(2.5) = 2`, `round(28.5) = 28`. C++ `std::round` rounds halves *away from zero* [7] (`std::round(2.5) = 3`), and TSPLIB's `(int)(x + 0.5)` rounds them up. NumPy's `np.round`/`np.rint` also round halves to even. A second surprise is binary floating point: `round(2.675, 2)` gives `2.67`, because 2.675 cannot be stored exactly.

Table 1 shows the effect on the optimal tours.

**Table 1.** Length of the optimal tour under different rounding methods (output of `tsp.py`).

| Instance | TSP-BEST [5] | TSPLIB `nint` per edge | Python `round()` per edge | `round(total)` | Unrounded | Edges ending in exactly .5 |
|---|---|---|---|---|---|---|
| berlin52 | 7542 | **7542** | 7542 | 7544 | 7544.37 | 0 |
| eil101 | 629 | **629** | 629 | 642 | 642.31 | 0 |
| tsp225 | 3916 | **3916** | 3861 | 3859 | 3859.00 | 124 |
| pcb442 | 50778 | **50778** | 50778 | 50784 | 50783.55 | 0 |
| pr1002 | 259045 | **259045** | 259045 | 259067 | 259066.66 | 0 |

Only rounding each edge with `nint` reproduces all five known optima. The error from Python's `round()` appears only in tsp225. For the other four instances the coordinates are integers, so $\sqrt{xd^2+yd^2}$ is either an integer or irrational and can never end in exactly .5. tsp225 is a rectilinear drawing of the letters "TSP" whose coordinates are offset by half units (e.g. 183.92 − 155.42 = 28.5). 124 of its 225 optimal-tour edges end in exactly .5, and `round()` rounds every one with an even integer part down, producing 3861 instead of 3916. Rounding only the total is wrong for every instance, by up to 13 units on eil101 (642 vs 629).

**Answer to Task 2.** The optimal tour lengths are berlin52 = 7542, eil101 = 629, tsp225 = 3916, pcb442 = 50778, pr1002 = 259045, identical to TSP-BEST [5].

# 3. Evolutionary algorithm design (Task 3)

## 3.1 Framework, representation and fitness

The EA is built with DEAP [3]. As given in the assignment:

- **Representation:** an individual is a permutation of the city indices 0…*n*−1 (`array('i')`), so every individual is a valid tour.
- **Objective:** minimise the closed tour length.

We use the **TSPLIB integer length** (Section 2.3) as the fitness. The original template used the unrounded length, which would make the EA optimise a slightly different quantity from the one compared with the optimum (on tsp225 the two differ by 1.5% for the optimal tour; Table 1).

All operators come from DEAP (`deap.tools`) except edge recombination (Section 3.3), which DEAP does not provide. It is written in DEAP's operator style and registered in the toolbox in the same way.

## 3.2 Evolutionary cycle and termination

`elitism.py` extends DEAP's `eaSimple` [3] in three ways:

1. **Elitism.** The `HallOfFame` individuals are copied unchanged into every generation, so the best tour is never lost.
2. **An exact evaluation budget.** The run stops after at most 50,000 fitness evaluations, counting the initial population. Offspring that were neither crossed over nor mutated keep their fitness and cost nothing. If the last generation would overshoot the budget, only the offspring that still fit are evaluated, and the population is refilled with the best individuals of the previous generation.

The template's generation limit (`MAX_GENERATIONS = 200`) only controls the budget indirectly. With its parameters a run happens to use about 300 + 200 × 270 × (1 − 0.1 × 0.9) ≈ 49,440 evaluations, but any change to the population size or the operator probabilities changes this number. Counting evaluations directly makes every variant use the same budget, which is required for a fair comparison.

3. **Stopping at the optimum.** A run also stops as soon as its best tour reaches the known optimal length.

**Termination as a design choice.** The run terminates on whichever of two criteria comes first: (1) 50,000 evaluations, the assignment's budget, or (2) the best tour reaching the known optimum. Criterion (2) is a *benchmarking* rule: it requires the optimum to be known, which is never the case in a real application. It cannot change any reported result, because elitism keeps the best tour and nothing is better than the optimum, so a run that stops early ends with the same tour it would have had after 50,000 evaluations. What it adds is a measure of speed: the number of evaluations needed to find the optimum. Only berlin52 ever reaches its optimum. Two other criteria were considered and rejected:

- **Stopping on stagnation** (no improvement for *G* generations). With a fixed budget and the result measured at the end, stopping early can never improve the result; it would only save time. The convergence curves of the large instances are also still falling at 50,000 evaluations.
- **Restarting on stagnation** (re-initializing part of a converged population). Given how strongly the populations converge (Section 5.7), this could help, but it is a new design feature rather than a termination rule, and is left as future work.

## 3.3 Candidate variation operators

TSP fitness depends on which cities are **adjacent** (the edges), not on the absolute position of a city in the tour. A good operator should therefore pass edges from parents to children (crossover) and change few edges at a time (mutation) [1], [2].

**Crossover** (applied to a pair of parents with probability *p_c*):

| Operator | DEAP function | Idea | Ref. |
|---|---|---|---|
| Order crossover (OX) | `tools.cxOrdered` | Copy a random segment from parent 1; fill the remaining positions with the missing cities in the order they appear in parent 2. Keeps the segment's edges and the relative order of the rest. | [8] |
| Partially mapped crossover (PMX) | `tools.cxPartialyMatched` | Swap a segment between the parents and repair duplicates through the mapping defined by the segment. Preserves absolute positions more than adjacency. | [9] |
| Uniform PMX (UPMX) | `tools.cxUniformPartialyMatched` | Like PMX, but each position is exchanged independently with probability `indpb`. | [10] |
| Enhanced edge recombination (ERX) | `operators.cxEdgeRecombination` | Build an edge map of each city's neighbours in either parent, marking the edges that both parents share. Starting from a city, follow a shared edge if one is left; otherwise move to the neighbour with the fewest remaining neighbours (ties at random); if none is left, jump to a random unvisited city. | [11], [12] |

ERX is designed specifically for the TSP: it builds children almost entirely from parental edges. We use the enhanced version [12], which gives priority to edges present in both parents. The original ERX [11] merges the two parents' edges into one list and loses this information, so it can break an edge that both parents agree on. With the enhancement, every shared edge was kept in our tests with random parents. With near-identical parents, 99–99.9% were kept: when the child enters a city in the middle of a shared path from outside, one of the city's two shared edges must break. Only about 3% of child edges were in neither parent (random parents, *n* = 52 and 1002). These "edge failures" are the random jumps, a known weakness of the operator [12]. ERX produces one child per pair; to fit DEAP's `mate` interface (two parents in, two children out), the second child starts from the other parent's first city.

**Mutation** (applied to each offspring with probability *p_m*):

| Operator | DEAP function | Edges changed | Ref. |
|---|---|---|---|
| Swap (exchange) | `tools.mutShuffleIndexes`, each position swapped with probability `indpb` (template: 1/*n*) | up to 4 per swap | [2] |
| Inversion | `tools.mutInversion` | 2 (reverses one segment; this is a 2-opt move [13]) | [2], [14] |

## 3.4 Selection, population model, initialization and seeds

- **Parent selection:** tournament selection (`tools.selTournament`). The tournament size *k* sets the selection pressure [15]. Fitness-proportional selection (`selRoulette`) was excluded: DEAP's documentation warns it is not suitable for minimisation, and it is sensitive to the scale of the fitness values [1].
- **Survivor selection:** generational replacement with elitism (Section 3.2).
- **Population size** μ and elite size: studied in the experiments, because with a fixed budget a smaller population gets more generations.
- **Initialization:** the template uses uniformly random permutations (`tools.initIterate` with `random.sample`). Step 9 tests **heuristic initialization** [27]: part or all of the initial population is built by the greedy nearest-neighbour rule (start at a city and always move to the closest unvisited city), each tour from a different, randomly chosen start city. Building these tours reads the distance matrix but evaluates no tour, so it costs no fitness evaluations. The tours themselves are evaluated as part of the initial population, and those evaluations are counted.
- **Random seeds:** each run seeds Python's `random` (used by all DEAP operators) and NumPy at the start, so every run can be repeated exactly (Section 4).

**Table 1b.** Final EA design and the experiments that justify each choice.

| Component | Final choice | Evidence |
|---|---|---|
| Representation, fitness | Permutation of city indices; TSPLIB EUC_2D tour length | Given; Section 2.3 |
| Initialization | All 50 individuals are nearest-neighbour tours from distinct random start cities | Step 9: 59.5% shorter tours than random initialization on the reporting seeds (25 of 25 runs, *p* = 6·10⁻⁸); better than seeding 5 of 50 (*p* = 0.0002) |
| Parent selection | Tournament, size *k* = 2 (`tools.selTournament`) | Step 5a: better than *k* = 1, 3, 5, 7 (all *p* < 0.001) |
| Crossover | Order crossover OX (`tools.cxOrdered`) | Step 3: better than PMX and UPMX; Steps 4 and 7: better than ERX and no crossover at μ = 50 |
| Mutation | Inversion (`tools.mutInversion`) | Step 7: better than swap at μ = 50 (−4.9%, *p* = 0.011); confirmed on the reporting seeds (−3.6%, *p* = 0.048) |
| Probabilities *p_c*, *p_m* | Fixed, *p_c* = 0.9, *p_m* = 0.1 | Step 5b: no setting better overall. With random initialization, uniform mixing of three settings was better (Steps 6b, 7b, 8), but with nearest-neighbour initialization it is not (Step 9: *p* = 0.26 tuning, 0.063 reporting), so the simpler fixed setting is kept |
| Population model | Generational, μ = 50, of which 5 elites (`tools.HallOfFame`) copied unchanged | Steps 4 and 4b: μ = 50 better than 300, 100, 20, 10; not significantly different from 30 |
| Survivor selection | Offspring replace the parents; the 5 best-ever tours survive | Elitism keeps the best tour (Section 3.2) |
| Termination | 50,000 fitness evaluations, or earlier when the known optimum is reached | Assignment budget, counted in `elitism.py`; the optimum rule cannot change the result (Section 3.2) |
| Random seeds | Tuning: 101–105; reporting: 1–5; each run seeds `random` and NumPy | Section 4.1 |

# 4. Experimental methodology

## 4.1 Protocol

- **Budget:** at most 50,000 fitness evaluations per run; exactly 50,000 unless the run reached the known optimum earlier (checked in every run).
- **Tuning vs. reporting seeds:** all design comparisons use tuning seeds 101–105. The final results use separate seeds 1–5, so the reported numbers are not tuned on the seeds that produced them.
- **Metrics:** best, mean, worst and standard deviation of the final tour length over the runs, and the deviation of the mean from the optimum (Section 1). Convergence curves show the best length found against the number of evaluations used.
- **Reproducibility:** each design step is defined in `experiments.py` as a set of variants. Running a step writes `metadata.json` with the full parameters of every variant, the seeds, the exact command and the library versions, together with the per-run results (`runs.csv`), the summary (`summary.csv`) and the convergence data. Results do not depend on how many runs execute in parallel, because each run seeds itself. Any table in this report can be regenerated with the command in its `metadata.json`.
- **Statistics:** each variant is compared with a reference variant on the same seeds, so each pair of runs starts from the same initial population. For each instance we report the number of seeds on which the variant wins. Over all instances we apply the two-sided Wilcoxon signed-rank test [16] to the 25 paired **log length ratios** log(L_variant / L_reference), and report their geometric mean as the relative difference in tour length (negative = shorter tours). Relative differences give every instance equal weight; differences in deviation % would not (Section 6, item 12). A per-instance test is not meaningful with 5 seeds, because the exact two-sided *p*-value cannot go below 0.0625. `experiments.py --reanalyse` recomputes all comparisons from the saved runs.

## 4.2 Design strategy: one component at a time

The design is built step by step: each step changes **one** component, keeps everything else fixed at the best setting so far, and adopts the winner. This "one factor at a time" (OFAT) approach is easy to follow and to explain. Its known weakness is that it **ignores interactions between components** [17], [18]. For example, which crossover works best can depend on the mutation operator used with it. Because of this, the **order of the steps matters**: each step's winner is chosen in the context fixed by the earlier steps.

**Chosen order and justification.** Mutation is studied first, then crossover, then selection pressure and population size, then the probabilities:

1. **The baseline analysis points to mutation** (Section 5.1). Swap mutation changes up to four edges at random, so it mostly destroys good adjacencies, while inversion changes only two (a 2-opt move [13]). Operators that change fewer, related edges are said to have high *locality*, which generally makes the search more effective [19]. Larrañaga et al. also found inversion-type mutations among the best for the TSP [2]. [CHECK the exact ranking in their results tables before keeping this sentence.]
2. **A destructive mutation can hide crossover differences.** If mutation keeps breaking the edges that crossover passes on, the crossovers are compared under conditions that will not hold in the final design.
3. **Mutation is the operator that does the fine-tuning.** Here, mutation makes small changes to tours that are already good, while crossover recombines tours that are already reasonable. Fixing the fine-tuning operator first gives crossover a fairer test.

**Countering the order effect.** After the last step, a **verification step** varies each component once more around the final design (e.g. the other crossovers with the final mutation). If a different choice now wins, the order affected the result, and this is reported.

# 5. Results

## 5.1 Step 1: baseline

The baseline keeps the operators and parameters of the original template (OX with *p_c* = 0.9, swap mutation with *p_m* = 0.1 and `indpb` = 1/*n*, tournament size 2, μ = 300, 30 elites, random initialization) and only replaces the generation limit with the 50,000-evaluation budget. It is the reference that every later step must beat.

**Table 2.** Step 1 baseline, tuning seeds 101–105.

| Instance | Optimum | Best | Mean | Worst | Std | Deviation % | Random start, deviation %* |
|---|---|---|---|---|---|---|---|
| berlin52 | 7542 | 8460 | 8659.8 | 9022 | 243.0 | 14.82 | 223 |
| eil101 | 629 | 1069 | 1155.4 | 1259 | 77.8 | 83.69 | 380 |
| tsp225 | 3916 | 17607 | 18023.6 | 18291 | 281.0 | 360.26 | 874 |
| pcb442 | 50778 | 414058 | 429289.6 | 436732 | 9275.6 | 745.42 | 1333 |
| pr1002 | 259045 | 4111551 | 4163176 | 4224493 | 44063.5 | 1507.12 | 2288 |

\*Best tour in the random initial population (300 tours), mean over the 5 seeds (`results/step1/convergence.csv`, first checkpoint).

The last column shows where the search starts. On berlin52 the EA reduces the deviation from 223% to 15%. On pr1002 it only reduces it from 2288% to 1507%, so the improvement shrinks quickly as the instance grows. The convergence curves are still falling steeply at 50,000 evaluations [TODO Figure 2: `results/step1/convergence_pcb442.png`], so the search is far from converged when the budget runs out.

Three reasons are likely, each tested in a later step:

1. Swap mutation is destructive (Section 4.2).
2. μ = 300 leaves only about 202 generations within the budget.
3. Tournament size 2 gives weak selection pressure [15].

## 5.2 Step 2: mutation operator

**Question.** Is inversion better than swap mutation, as Section 4.2 argued?

To make the answer robust, the two mutations were compared in three settings:

- the baseline (OX, *p_m* = 0.1);
- a higher mutation rate (*p_m* = 0.5), in case 0.1 is too low for the operator to matter;
- with ERX instead of OX, an **early order check**: if the mutation ranking changes with the crossover, then the order of the steps matters.

All other parameters are as in the baseline.

**Table 3.** Step 2, mean deviation % over tuning seeds 101–105 (lower is better). "Wins" = number of seeds on which inversion beat swap in the same setting, using the same seed and therefore the same initial population.

| Setting | Mutation | berlin52 | eil101 | tsp225 | pcb442 | pr1002 | Inversion wins (of 25) | Rel. diff. | Wilcoxon *p* |
|---|---|---|---|---|---|---|---|---|---|
| OX, *p_m* = 0.1 | swap | **14.82** | **83.69** | **360.26** | 745.42 | **1507.12** | | | |
| | inversion | 19.99 | 96.92 | 379.95 | **729.06** | 1524.51 | 9 | +2.95% | **0.026** |
| OX, *p_m* = 0.5 | swap | **25.81** | **120.54** | **393.90** | 771.80 | 1585.37 | | | |
| | inversion | 33.28 | 127.19 | 406.85 | **754.04** | **1551.28** | 11 | +1.45% | 0.29 |
| ERX, *p_m* = 0.1 | swap | 9.77 | 33.77 | **193.90** | **575.58** | **1394.55** | | | |
| | inversion | **8.33** | **32.34** | 196.31 | 585.81 | 1397.74 | 14 | +0.07% | 0.98 |

Rel. diff.: geometric mean of the paired tour length ratios inversion/swap (positive = inversion's tours are longer). *p*: two-sided Wilcoxon signed-rank test on the 25 paired log ratios (Section 4.1).

**Findings.**

1. **The hypothesis of Section 4.2 is not supported.** In no setting is inversion significantly better than swap. In the baseline setting it is even significantly *worse* (tours 2.95% longer, *p* = 0.026); in the other two settings there is no significant difference. The differences between the two mutations (a few percent) are much smaller than the differences between settings.
2. **The ranking depends on the setting and the instance.** With OX, swap is better on the three smaller instances but inversion wins on pcb442 (5 of 5 seeds at *p_m* = 0.1) and on pr1002 (5 of 5 at *p_m* = 0.5). With ERX, inversion is better on the two smallest instances. So the components **interact**, which confirms that the order of the one-component-at-a-time steps matters (Section 4.2). It is also a first sign that the best operator differs between instances (Section 5.6).
3. **More mutation is worse.** *p_m* = 0.5 is worse than 0.1 for both operators on all five instances (10 of 10 comparisons). With tournament size 2, selection is too weak to remove the many offspring that mutation makes worse. This shows in the population: on berlin52 the final population average is 17.1% above its best tour with *p_m* = 0.5, against 6.0% with *p_m* = 0.1.
4. **Crossover matters much more than mutation at this stage.** Replacing OX with ERX, with no other change, reduced the mean deviation on every instance: from 14.8% to 9.8% on berlin52, from 83.7% to 33.8% on eil101, from 360% to 194% on tsp225, from 745% to 576% on pcb442 and from 1507% to 1395% on pr1002. [TODO Figure 3: `results/step2/convergence_eil101.png`, which shows the two ERX curves clearly separated from the four OX curves.]

**Decision.** Keep **swap mutation with *p_m* = 0.1**. Inversion was significantly worse in the baseline setting and not significantly better in the others, so there is no evidence to justify changing the baseline. Mutation is revisited in the verification step (Section 5.8), when the rest of the design is final. Because the crossover had the largest effect, it is studied next.

**Lesson.** The argument for studying mutation first (Section 4.2) was reasonable from the literature, but the experiment showed that at this design point (weak selection, only about 200 generations) mutation is not the bottleneck. Testing the assumption instead of adopting it changed the plan.

## 5.3 Step 3: crossover operator

**Question.** Which crossover works best: OX (the baseline), PMX, UPMX or ERX? Is crossover useful at all? All variants use swap mutation with *p_m* = 0.1, chosen in Step 2. UPMX was run with two values of its exchange probability (`indpb` = 0.1 and 0.5), so that a poor result cannot be blamed on one arbitrary setting. The no-crossover variant (*p_c* = 0) is a control.

**Table 4.** Step 3, mean deviation % over tuning seeds 101–105. Wins and *p* are against OX, as in Table 3. The last column is the mean number of generations completed within the 50,000 evaluations.

| Crossover | berlin52 | eil101 | tsp225 | pcb442 | pr1002 | Wins vs OX (of 25) | Rel. diff. | Wilcoxon *p* | Generations |
|---|---|---|---|---|---|---|---|---|---|
| OX (reference) | 14.82 | 83.69 | 360.26 | 745.42 | 1507.12 | | | | 203 |
| PMX | 58.57 | 130.91 | 428.79 | 823.09 | 1684.76 | 0 | +19.34% | 6·10⁻⁸ | 203 |
| UPMX, `indpb` 0.1 | 48.45 | 127.63 | 411.65 | 765.13 | 1587.42 | 1 | +13.86% | 3·10⁻⁷ | 203 |
| UPMX, `indpb` 0.5 | 73.11 | 234.59 | 583.15 | 1073.06 | 1989.82 | 0 | +48.56% | 6·10⁻⁸ | 203 |
| ERX | **9.77** | **33.77** | **193.90** | 575.58 | 1394.55 | 24 | −19.89% | **2·10⁻⁷** | 203 |
| No crossover | 31.52 | 72.53 | 276.55 | **564.82** | **1259.51** | 19 | −10.19% | **0.002** | **1842** |

**Findings.**

1. **Position-based crossovers are clearly worse.** PMX and both UPMX settings lose to OX on almost every run (0–1 wins of 25, tours 14–49% longer, *p* < 10⁻⁶). This matches the literature [2], [12]: PMX and UPMX preserve the *absolute positions* of cities, but the TSP fitness depends on *adjacency*, which they disrupt. OX keeps whole segments, and therefore their edges. The more positions UPMX exchanges (`indpb` 0.5 vs 0.1), the worse it gets.
2. **ERX, the edge-based operator, is the best crossover on every instance.** It beats OX on 24 of 25 runs (all 5 seeds of every instance except one seed of berlin52), with the largest gains on eil101 (83.7% → 33.8%) and tsp225 (360% → 194%). Pooled over the instances ERX is significantly better (tours 19.9% shorter, *p* = 2·10⁻⁷). It also beats "no crossover" on the three smaller instances (all seeds), and pooled (−10.8%, *p* = 0.002), even though no crossover completes 9× more generations. On the large instances ERX keeps the population more diverse (on pr1002 the final population average is 4.3% above its best tour with ERX, against 1.3% with OX), so it is still far from converged after about 200 generations. ERX is also about 7× slower per run (35 s vs 4.6 s); this does not count against it, because the budget is measured in evaluations, but it matters in practice.
3. **"No crossover" is the best variant on the two largest instances, but the comparison is confounded.** With *p_c* = 0 and *p_m* = 0.1, 90% of the offspring are unchanged copies of their parents. DEAP does not re-evaluate them, so they cost no evaluations, and the run completes **1842 generations instead of 203**. The gain comes from 9× more generations of selection, not from leaving out crossover. Under the assignment's budget (evaluations) this is a legitimate design, but it shows that **the number of generations, set by the population size and the probabilities, is a bigger factor than the choice of crossover**. This is consistent with Step 2, where weak selection over few generations was already suspected.
4. **Best variant per instance:** ERX on berlin52, eil101 and tsp225, no crossover on pcb442 and pr1002. Given finding 3, this split mainly reflects how many generations each instance needs, not a true difference in which crossover suits it.

**Decision.**

- **Drop PMX and UPMX.** They are significantly worse than OX.
- **Keep ERX and OX as candidates, and keep "no crossover" as a control.** None of them can be declared best yet, because their ranking depends on the number of generations.
- Step 4 therefore varies the **population size** (and with it the number of generations) **together with the crossover**. If crossover beats no crossover once the generation counts are comparable, crossover adds value; if not, a mutation-only EA is the better design under this budget.

[TODO Figure 4: `results/step3/convergence_pcb442.png`. The no-crossover curve keeps falling while the crossover curves flatten.]

## 5.4 Step 4: population size × crossover

**Question.** Does crossover add value once every variant gets enough generations? Which crossover is then best?

The population size μ was varied together with the three crossover options kept from Step 3. Under a fixed evaluation budget, a smaller population means more generations. The elite size was scaled to 10% of μ, the template's ratio (30 of 300). With a fixed 30 elites, a population of 50 would be 60% elites, mixing two effects.

**Table 5.** Step 4, mean deviation % over tuning seeds 101–105 (swap mutation, *p_m* = 0.1, tournament size 2). Generations = mean completed within 50,000 evaluations.

| μ | Crossover | berlin52 | eil101 | tsp225 | pcb442 | pr1002 | Generations |
|---|---|---|---|---|---|---|---|
| 300 | OX | 14.82 | 83.69 | 360.26 | 745.42 | 1507.12 | 203 |
| | ERX | **9.77** | **33.77** | 193.90 | 575.58 | 1394.55 | 203 |
| | none | 31.52 | 72.53 | 276.55 | 564.82 | 1259.51 | 1842 |
| 100 | OX | 13.99 | 39.81 | 192.73 | 485.34 | 1156.94 | 610 |
| | ERX | 15.80 | 58.76 | 199.42 | 465.83 | 1327.29 | 610 |
| | none | 30.71 | 72.97 | 221.57 | 438.98 | 991.80 | 5550 |
| 50 | OX | 14.97 | 45.76 | **155.06** | **377.30** | 974.96 | 1245 |
| | ERX | 22.67 | 66.49 | 233.69 | 474.91 | 1102.26 | 1245 |
| | none | 36.50 | 62.42 | 202.08 | 408.12 | **889.85** | 11103 |

Pooled Wilcoxon tests (25 paired runs each; relative difference in tour length):

- **Smaller μ:** μ = 50 vs 300 with OX: 22 wins, −30.1%, *p* = 1.5·10⁻⁶.
- **ERX vs OX:** μ = 300: 24 wins, −19.9%, *p* = 1.8·10⁻⁷; μ = 100: 7 wins, +5.3%, *p* = 0.003; μ = 50: 0 wins, +16.5%, *p* = 6·10⁻⁸.
- **No crossover vs OX at μ = 50:** 5 wins, +8.9%, *p* = 0.0006.

**Findings.**

1. **A smaller population is the largest improvement so far.** With OX, going from μ = 300 to 50 roughly halves the deviation on tsp225 (360% → 155%) and pcb442 (745% → 377%). Six times more generations of selection outweighs the loss of diversity. This confirms the suspicion from Steps 2 and 3 that the baseline had too few generations.
2. **ERX only wins in a large population, and loses in a small one.** At μ = 300 it is the best crossover on all five instances. At μ = 100 and 50 it is significantly worse than OX (at μ = 100 it still wins on pcb442, on 4 of 5 seeds), and on berlin52, eil101 and tsp225 it even gets *worse* as μ shrinks (9.8% → 22.7% on berlin52). The final population shows why. With ERX the population average ends within 0.4–1.0% of its best tour on berlin52 and eil101, at both μ = 300 and 50 (OX at μ = 300: 6.0–6.5%), so the population has collapsed to nearly identical tours. ERX only uses edges present in the parents, and the enhanced version always keeps the edges they share, so two near-identical parents produce a near-identical child, and the search stalls. A large population delays this collapse; a small one does not. This is another **interaction**: which crossover is best depends on the population size.
3. **Crossover adds value once generations are comparable.** At μ = 50, OX beats no crossover on four instances (e.g. tsp225 155% vs 202%), even though no crossover completes 9× more generations. Only pr1002 still prefers no crossover (890% vs 975%). Pooled, OX is significantly better (no crossover gives 8.9% longer tours, *p* = 0.0006).

**Step 4b: extending the range.** μ = 50 was the smallest size tested and still improving, so the best value might lie below the tested range. Step 4b tested OX with μ ∈ {50, 30, 20, 10} (elites 10% of μ, at least 1):

| μ | berlin52 | eil101 | tsp225 | pcb442 | pr1002 | Wins vs μ = 50 | Rel. diff. | *p* |
|---|---|---|---|---|---|---|---|---|
| 50 | **14.97** | **45.76** | **155.06** | **377.30** | 974.96 | | | |
| 30 | 18.59 | 50.40 | 194.43 | 385.62 | **889.13** | 9 | +2.78% | 0.16 |
| 20 | 34.73 | 67.22 | 242.82 | 456.14 | 982.73 | 1 | +16.16% | 6·10⁻⁷ |
| 10 | 45.16 | 106.10 | 310.88 | 596.63 | 1209.94 | 0 | +38.55% | 6·10⁻⁸ |

Below μ = 30 the population is too small: it loses diversity faster than the extra generations help. μ = 50 is best on four instances. μ = 30 is better only on pr1002, the largest instance, which benefits most from extra generations; pooled, the difference is not significant (*p* = 0.16).

**Decision.** Use **OX with μ = 50 (5 elites)**. ERX is dropped: it is significantly worse than OX at this population size, and about 7× slower per run. "No crossover" is not adopted, because OX is better on four of five instances. The preference of pr1002 for no crossover and for μ = 30 is noted as instance dependence (Section 5.6).

## 5.5 Step 5: selection pressure and operator probabilities

**Step 5a: tournament size.** With the Step 4 design (OX, μ = 50, 5 elites, swap, *p_c* = 0.9, *p_m* = 0.1), the tournament size *k* was varied over {1, 2, 3, 5, 7}. *k* = 1 means parents are chosen at random, so only the elites exert selection.

**Table 6.** Step 5a, mean deviation % over tuning seeds 101–105. Comparisons are against *k* = 2.

| *k* | berlin52 | eil101 | tsp225 | pcb442 | pr1002 | Wins vs *k* = 2 | Rel. diff. | *p* |
|---|---|---|---|---|---|---|---|---|
| 1 | 48.77 | 138.57 | 428.36 | 797.29 | 1589.69 | 0 | +66.86% | 6·10⁻⁸ |
| **2** | **14.97** | **45.76** | **155.06** | **377.30** | 974.96 | | | |
| 3 | 27.13 | 66.61 | 210.49 | 410.07 | **904.34** | 5 | +8.90% | 0.0006 |
| 5 | 36.54 | 85.09 | 259.23 | 500.48 | 1028.86 | 0 | +22.88% | 6·10⁻⁸ |
| 7 | 43.22 | 106.45 | 293.03 | 549.99 | 1075.96 | 0 | +32.15% | 6·10⁻⁸ |

*k* = 2, the weakest real tournament, is significantly better than every other size. Only pr1002 prefers *k* = 3. Random parent selection (*k* = 1) is by far the worst, so parent selection is clearly needed. More pressure than *k* = 2, however, makes the population converge too early. The small population (μ = 50) and the 5 elites already provide strong selection, so extra pressure from the tournament only reduces diversity [15]. This differs from the earlier, rejected design (Section 6, item 5), where *k* = 5 was best with μ = 50 but with a different mutation and 2 elites. Again, the best value of one parameter depends on the others.

**Decision:** keep **tournament size 2**.

**Step 5b: crossover and mutation probabilities.** A 3 × 3 grid around the current values, *p_c* ∈ {0.6, 0.9, 1.0} × *p_m* ∈ {0.05, 0.1, 0.2}, with *k* = 2.

**Table 7.** Step 5b, mean deviation % over tuning seeds 101–105. Comparisons are against the current setting (*p_c* = 0.9, *p_m* = 0.1). Generations depend on the probabilities, because unchanged offspring are not evaluated.

| *p_c* | *p_m* | berlin52 | eil101 | tsp225 | pcb442 | pr1002 | Rel. diff. | *p* | Generations |
|---|---|---|---|---|---|---|---|---|---|
| 0.9 | 0.1 (current) | 14.97 | 45.76 | 155.06 | 377.30 | 974.96 | | | 1245 |
| 0.6 | 0.05 | 21.89 | 47.85 | 195.11 | 394.08 | **893.95** | +3.50% | 0.096 | 1828 |
| 0.6 | 0.1 | 19.75 | 44.45 | 182.66 | **371.63** | 911.31 | +1.24% | 0.34 | 1767 |
| 0.6 | 0.2 | 16.80 | 46.30 | 175.75 | 383.43 | 934.96 | +1.38% | 0.58 | 1659 |
| 0.9 | 0.05 | 18.05 | 41.65 | 158.98 | 371.99 | 943.28 | −0.57% | 0.75 | 1253 |
| 0.9 | 0.2 | 11.76 | 39.59 | 159.05 | 401.79 | 975.38 | −0.16% | 0.79 | 1228 |
| 1.0 | 0.05 | 17.94 | 43.02 | **152.22** | 382.56 | 987.31 | +0.36% | 0.75 | 1134 |
| 1.0 | 0.1 | 14.73 | 38.86 | 152.91 | 400.61 | 1003.24 | +0.29% | 0.57 | 1133 |
| 1.0 | 0.2 | **10.89** | **35.71** | 170.38 | 414.22 | 1023.20 | +1.40% | 0.29 | 1131 |

**Findings.**

1. **No setting is significantly better or worse overall** (all *p* ≥ 0.096; all pooled differences within ±3.5%).
2. **The reason is that the instances pull in opposite directions.** The small instances prefer *more* variation: with *p_c* = 1.0, *p_m* = 0.2, tours are 3.5% shorter on berlin52 and 6.9% shorter on eil101. pr1002 prefers *less*: with *p_c* = 0.6, *p_m* = 0.05, tours are 7.6% shorter on all 5 seeds, because more offspring stay unchanged and cost nothing, so the run completes 47% more generations. tsp225 strongly dislikes a low *p_c* (+15.6% longer tours at *p_c* = 0.6, *p_m* = 0.05). Pooled, these effects cancel out.
3. Each probability setting is therefore a compromise. The current setting is never far from the best for any instance (within about 8%).

**Decision:** keep ***p_c* = 0.9, *p_m* = 0.1**. No alternative is significantly better, and the opposite preferences of the instances are the motivation for parameter control (Section 5.6).

**Design after Step 5 (tuned):** random initialization; OX (*p_c* = 0.9); swap mutation (*p_m* = 0.1, `indpb` = 1/*n*); tournament size 2; μ = 50 with 5 elites; 50,000 evaluations. Relative to the baseline, only the population size and the elite size changed, but the steps established *why* each other component is kept.

## 5.6 Step 6: parameter control — does the best setting depend on the instance?

**One shared design or one per instance?** All design steps choose **one configuration for all five instances**. The No Free Lunch (NFL) theorem [20] could seem to argue against this, but it states that *averaged over all possible problems* no algorithm outperforms another. It does not imply that every instance needs its own parameters. Our five instances belong to one problem class (symmetric 2D Euclidean TSPs), and within a class some designs can be better on all instances; edge-preserving operators for the TSP are an example [2], [11]. The practical lesson of NFL is to match the algorithm to the problem class.

Tuning each instance separately was rejected for three reasons:

1. It multiplies the experimental effort by five.
2. With five tuning seeds per instance, it would mostly fit noise.
3. The assignment asks for one EA for the five TSPs.

Instead, the instance dependence is **measured**: every step also reports the best variant per instance (`experiments.py` output), and Section 5.8 quantifies how much a per-instance choice would have gained over the shared design on the reporting seeds.

The steps do show instance dependence. In Step 2 the mutation ranking flipped between the small and the large instances, and in Step 3 the best variant was ERX on the three smaller instances but no crossover on the two largest. This motivates **parameter control**: a mechanism that adapts the design *during each run*, so that each instance can settle on its own settings without separate tuning.

Choosing operators by experiment before the run, as in Steps 2 to 5, is *parameter tuning*. Changing operators or their probabilities *during* a run is *parameter control*, which can be deterministic, adaptive or self-adaptive [21], [18].

**What to control.** Step 5b showed *which* parameters the instances disagree on: the small instances prefer more variation (*p_c* = 1.0, *p_m* = 0.2), while pr1002 prefers less (*p_c* = 0.6, *p_m* = 0.05). We therefore control the crossover and mutation probabilities during the run. We test one controller from each of the two feedback-based classes of Eiben et al. [21]:

- **Adaptive control: adaptive pursuit** [22]. There are three "arms", the low (0.6, 0.05), tuned (0.9, 0.1) and high (1.0, 0.2) settings from Step 5b. Each generation, one arm is drawn with probability *P_i* and used for all offspring. The reward is the mean relative improvement of the evaluated offspring over their better parent, $\frac{1}{E}\sum \max(0, (f_{parent} - f_{child}) / f_{parent})$, where *E* is the number of evaluations spent. The quality estimate of the chosen arm moves towards the reward with learning rate α. The best arm's probability is then pushed towards *P_max* = 1 − 2·*P_min* and the others towards *P_min* = 0.1, with learning rate β. [CHECK the update rule and the parameter values against [22].]
- **Self-adaptive control of *p_m*** [23], [24]. Each individual carries its own mutation probability *p_m*. A child inherits it (the mean of both parents after crossover) and perturbs it with the log-normal rule of Bäck and Schütz [24], $p' = \left(1 + \frac{1-p}{p}e^{-\gamma N(0,1)}\right)^{-1}$ with γ = 0.22, before using it. *p_c* stays at 0.9, and every individual starts with *p_m* = 0.1.

Both controllers are a generational loop in `elitism.py` with the same elitism and evaluation budget as before, and they use only DEAP's operators. As a correctness check, adaptive pursuit with a single arm (0.9, 0.1) reproduces the fixed-parameter EA exactly (identical tours and logbooks on berlin52 and pcb442).

**Table 8.** Step 6, relative difference in tour length against the tuned design (geometric mean over the 5 seeds per instance; negative = shorter tours), with wins out of 5. Last two columns: pooled over the 25 paired runs.

| Variant | berlin52 | eil101 | tsp225 | pcb442 | pr1002 | Pooled | *p* |
|---|---|---|---|---|---|---|---|
| Self-adaptive *p_m* | **−2.66% (5)** | **−6.68% (5)** | +3.78% (1) | +9.91% (0) | +4.44% (0) | +1.59% | 0.28 |
| Adaptive pursuit, α = β = 0.8 | −1.00% (4) | −3.36% (3) | +2.71% (1) | −2.18% (5) | −1.07% (3) | −1.00% | 0.30 |
| Adaptive pursuit, α = β = 0.3 | −1.51% (4) | −5.36% (5) | +1.68% (2) | +0.12% (2) | −3.26% (5) | −1.69% | 0.032 |
| Adaptive pursuit, α = β = 0.1 | −0.99% (3) | −5.90% (3) | −2.26% (3) | −1.16% (2) | −3.41% (5) | **−2.76%** | **0.012** |
| Adaptive pursuit, α = β = 0.03 | +2.28% (2) | −4.84% (4) | +2.25% (1) | +0.51% (2) | −3.19% (4) | −0.64% | 0.65 |
| Uniform mix (β = 0, no learning) | +3.65% (2) | −5.23% (4) | −1.37% (3) | −1.25% (3) | −2.66% (5) | −1.41% | 0.039 |

**Findings.**

1. **Self-adaptation helps the small instances and hurts the large ones.** On berlin52 and eil101 it wins on all 5 seeds (up to 6.7% shorter tours). On pcb442 and pr1002 it loses on all 5 (up to 9.9% longer). The mean *p_m* rose from 0.1 to 0.19–0.31 by the end of the run on *every* instance [TODO Figure: `results/step6/control_self-adaptive-pm.png`], including pr1002, which in Step 5b preferred *less* mutation. This confirms the expected weakness. Self-adaptation selects rates through the fitness of the individuals that carry them. It cannot see that unchanged offspring cost no evaluations, so that a low rate buys extra generations, which is what pr1002 benefits from.
2. **Adaptive pursuit with the initial learning rates (α = β = 0.8) gave no significant gain** (−1.0%, *p* = 0.30), and its arm probabilities jumped from generation to generation. We suspected it was chasing noise in the per-generation reward. A follow-up (Step 6b) varied the learning rate and added a control variant, **uniform mix**, which draws one of the three settings at random each generation and never learns (β = 0).
3. **The gain comes from mixing settings, not from learning.** Slower learning helped: with α = β = 0.1 the tours are 2.76% shorter (*p* = 0.012), with gains on all five instances. But uniform random mixing, with no learning at all, also gives 1.41% shorter tours (*p* = 0.039), and **no pursuit variant is significantly better than uniform mixing** (best: α = 0.1, −1.37%, *p* = 0.28). Even at α = 0.1 the arm probabilities fluctuate around 1/3 without settling on any setting, on every instance [TODO Figure: `results/step6b/control_ap-0.1.png`]. The reward does not consistently favour one setting, so there is little to learn. What helps is alternating between low and high variation.
4. **Our reward did not measure what we intended.** We expected a reward per evaluation to favour the cheap low-variation arm on pr1002. It did not: the reward measures how much each evaluated child improves on its parent, not how much the *run* gains per evaluation (e.g. through extra generations). On pr1002 the α = 0.8 pursuit even ended with the high-variation arm most likely (0.65). A reward based on the improvement of the best tour per evaluation would capture this better, but it is sparse and noisy. We leave it as future work.
5. **Caution about selection.** Five controller variants were compared with the tuned design on the same tuning seeds. With a Bonferroni correction for five comparisons [17], the best *p*-value (0.012 × 5 = 0.06) is no longer below 0.05. The gains of 1–3% are therefore suggestive, not conclusive. They must be confirmed on the separate reporting seeds (Section 5.8).

**Relation to No Free Lunch and per-instance tuning.** Self-adaptation reproduced the instance split of Step 5b: it worked where more mutation helps. Adaptive pursuit did not learn instance-specific settings at all. The best per-instance fixed settings from Step 5b were 1–8% better than the tuned design on single instances, but they were selected on the same seeds and are therefore optimistic (compared on the reporting seeds in Section 5.8). Within this assignment, a simple **randomised mix of settings** was as good as any learning controller.

**Candidates for the next design (Sections 5.8–5.9).** (a) Tuned fixed probabilities, the simplest; (b) uniform mix, simple with no extra parameters; (c) adaptive pursuit with α = β = 0.1, the best mean on the tuning seeds. The final choice is made on the reporting seeds (Section 5.8).

## 5.7 Synthesis: exploration versus exploitation

The results of Steps 1–6 are explained by one trade-off. **Exploration** means searching new regions: diverse populations, disruptive variation and weak selection. **Exploitation** means refining the best tours found so far: strong selection, many generations and conservative variation [25], [26]. As a simple measure of where a design sits, Table 9 shows how far the **final population's average tour is above its best tour**. A large gap means the population is still diverse and exploring; a gap near 0% means it has converged and is only exploiting.

**Table 9.** Final population average above its best tour (%), mean over the tuning seeds, with the generations completed within 50,000 evaluations.

| Variant | berlin52 | eil101 | tsp225 | pcb442 | pr1002 | Generations |
|---|---|---|---|---|---|---|
| Tournament *k* = 1 (random parents) | 52.2 | 36.7 | 18.1 | 7.9 | 4.0 | 1245 |
| Baseline (OX, μ = 300) | 6.0 | 6.5 | 5.4 | 3.0 | 1.3 | 203 |
| *p_c* = 1.0, *p_m* = 0.2 | 9.1 | 5.2 | 2.5 | 1.3 | 0.7 | 1131 |
| Self-adaptive *p_m* | 6.1 | 4.7 | 2.0 | 1.3 | 0.6 | 1218 |
| **Tuned (OX, μ = 50, *k* = 2)** | 2.3 | 1.6 | 1.2 | 0.8 | 0.4 | 1245 |
| *p_c* = 0.6, *p_m* = 0.05 | 1.3 | 0.5 | 0.4 | 0.1 | 0.1 | 1828 |
| Tournament *k* = 7 | 0.6 | 0.2 | 0.1 | 0.1 | 0.1 | 1245 |
| ERX, μ = 50 | 0.6 | 0.4 | 0.2 | 0.3 | 0.1 | 1246 |
| No crossover, μ = 50 | 1.3 | 0.5 | 0.2 | 0.1 | 0.03 | 11103 |

1. **The baseline explored too much for its budget.** After 50,000 evaluations its tours were hundreds of percent above the optimum, yet its population was still diverse (gap 3–6.5% on four instances) after only 203 generations. The largest single improvement, reducing μ from 300 to 50 (Step 4), shifted the balance towards exploitation: six times more generations of selection on a smaller population.
2. **Too much exploitation fails as well.** Every variant that made the population collapse (gap ≤ 0.7%) was worse than the tuned design: tournament sizes 3–7 (Step 5a), μ ≤ 20 (Step 4b), and ERX in a small population (Step 4). These are cases of premature convergence: once all tours are nearly identical, crossover produces copies and the search stalls. The opposite extreme, random parent selection (*k* = 1, gap up to 52%), keeps the population diverse but does not concentrate the search on good tours, and was by far the worst. The tuned design lies between the two extremes (gaps of 0.4–2.3%).
3. **The instance dependence is the same trade-off.** The small instances converge within the budget, so they benefit from *more exploration*: higher *p_c* and *p_m* (Step 5b), and self-adaptation, which raised *p_m* (Step 6). pr1002 is far from converged when the budget runs out, so it benefited from *every* more exploitative option: no crossover (Steps 3–4), μ = 30 (Step 4b), *k* = 3 (Step 5a), and low *p_c* and *p_m* (Step 5b). A larger instance needs more generations to exploit, and 50,000 evaluations provide too few. This explains why pr1002 was the outlier in almost every step, and why one fixed setting cannot suit all five instances.
4. **Parameter control in these terms.** Self-adaptation moved every instance towards exploration (rising *p_m*), which suits the small instances but not the large ones. Uniform mixing alternates generations of conservative variation (exploitation) and disruptive variation (exploration). We *hypothesise* that this alternation is why it beats any single fixed setting, but our data does not test this directly.

5. **Initialization (Step 9) in these terms.** Nearest-neighbour seeding uses problem knowledge to start the search in a good region, so that the budget is spent on exploitation from the start. Starting every tour at a different city keeps enough diversity for crossover to work. On the large instances, however, the population converges (1.08 and 1.05 distinct edges per city at the end on pcb442 and pr1002, final design, tuning seeds) before the operators find many improvements.

**Limitation.** The gap measures diversity in *fitness* only. Two populations with the same gap can differ in how many distinct tours, or distinct edges, they contain. A genotypic measure, such as the number of distinct edges in the population, would test points 2 and 4 more directly. From Step 7 onwards we therefore also log the number of distinct edges in the population per city (1.0 = all tours identical). It confirms points 2 and 3: the variants that collapse the population in fitness also collapse it in edges (Table 10).

## 5.8 Verification step and the design with random initialization

**Step 7: verification.** At the design after Step 5, tournament size, population size and probabilities had already been varied around exactly this design (Steps 4b, 5a, 5b). The mutation (Step 2) and crossover (Step 3) had been chosen at μ = 300, *before* the population size changed, so they were re-tested at μ = 50 (tuning seeds).

**Table 10.** Step 7, alternatives at the Step 5 design (OX, swap, *k* = 2, μ = 50, *p_c* = 0.9, *p_m* = 0.1): mean deviation %, relative difference against the Step 5 design, and the final edge diversity (distinct edges per city; 1.0 = all tours identical).

| Variant | berlin52 | eil101 | tsp225 | pcb442 | pr1002 | Rel. diff. | *p* | Edges/city |
|---|---|---|---|---|---|---|---|---|
| Step 5 design (swap) | 14.97 | 45.76 | 155.06 | **377.30** | 974.96 | | | 1.15–1.53 |
| **Inversion** | **5.97** | **23.18** | **144.47** | 396.91 | 977.07 | **−4.88%** | **0.011** | 1.20–1.83 |
| PMX | 51.49 | 118.86 | 357.13 | 670.74 | 1399.81 | +51.51% | 6·10⁻⁸ | 1.03–1.38 |
| UPMX, `indpb` 0.1 | 53.07 | 117.62 | 370.67 | 673.75 | 1414.16 | +52.91% | 6·10⁻⁸ | 1.03–1.37 |
| ERX | 22.67 | 66.49 | 233.69 | 474.91 | 1102.26 | +16.46% | 6·10⁻⁸ | 1.04–1.15 |
| No crossover | 36.50 | 62.42 | 202.08 | 408.12 | **889.85** | +8.93% | 0.0006 | 1.02–1.56 |

**The verification found an order effect.** At μ = 300 (Step 2), swap was significantly better than inversion (*p* = 0.026). At μ = 50, inversion is significantly better (−4.9%, *p* = 0.011), and it halves the deviation on berlin52 and eil101. The best mutation therefore **depended on the population size**, which was only fixed later in the order. This is the risk of one-component-at-a-time design discussed in Section 4.2. The crossover decision was confirmed: OX remains significantly better than all alternatives.

The edge diversity is consistent with Section 5.7. Inversion keeps the most distinct edges (1.20–1.83 per city), while ERX, PMX and UPMX make the population collapse (down to 1.03–1.05 on the large instances).

**Step 7b: re-checking what had been tuned with swap.** With inversion:

- *p_m* = 0.1 remained best. 0.05 was not significantly different (*p* = 0.35); 0.2 and 0.5 were significantly worse (*p* = 0.005 and 2·10⁻⁷).
- Uniform mixing of the three (*p_c*, *p_m*) settings was again significantly better than fixed probabilities (−1.85%, *p* = 0.022, better on 4 of 5 instances). This replicates Step 6b with a different mutation.
- Adaptive pursuit (α = β = 0.1) was again not better than uniform mixing (*p* = 0.67).

**Decision rule, fixed before the reporting runs.** The chosen design is the simplest variant that is (a) not significantly worse than the best variant and (b) significantly better than the fixed design. On the tuning seeds this is **inversion with uniform mixing**. Fixed *p_m* = 0.05 meets (a) (*p* = 0.16 against uniform mixing) but not (b) (*p* = 0.35 against *p_m* = 0.1).

**Step 8: the random-start design on the reporting seeds (not used for any decision).** The design chosen by this rule (inversion with uniform mixing, random initialization) was run together with four comparison designs:

- the same design with fixed probabilities;
- the Step 5 design (swap, fixed probabilities);
- the original template;
- per-instance tuned *p_m*: the best fixed *p_m* per instance from Step 7b.

**Table 11.** Step 8 design (random initialization), reporting seeds 1–5, 50,000 evaluations per run.

| Instance | Optimum | Best | Average | Worst | Std | **Deviation %** |
|---|---|---|---|---|---|---|
| berlin52 | 7542 | 7804 | 8205.6 | 8514 | 297.9 | **8.80** |
| eil101 | 629 | 733 | 755.2 | 777 | 19.2 | **20.06** |
| tsp225 | 3916 | 8975 | 9373.6 | 9787 | 309.1 | **139.37** |
| pcb442 | 50778 | 233258 | 244881.0 | 258044 | 10459.2 | **382.26** |
| pr1002 | 259045 | 2648409 | 2716950.6 | 2755659 | 42174.6 | **948.83** |

**Table 12.** Mean deviation % on the reporting seeds, and pooled comparisons with the Step 8 design (relative difference in tour length; negative = the Step 8 design's tours are shorter).

| Design | berlin52 | eil101 | tsp225 | pcb442 | pr1002 | Step 8 vs this | Wins (Step 8) | *p* |
|---|---|---|---|---|---|---|---|---|
| **Step 8 design** (inversion, uniform mix) | 8.80 | **20.06** | 139.37 | 382.26 | 948.83 | | | |
| Fixed probabilities (inversion) | **7.53** | 23.69 | 144.93 | 404.13 | 965.22 | −1.97% | 18/25 | 0.045 |
| Step 5 design (swap, fixed) | 17.43 | 41.05 | 152.98 | **369.94** | 963.43 | −5.45% | 20/25 | 0.0015 |
| Original template | 19.28 | 88.24 | 364.59 | 724.68 | 1543.74 | −35.48% | 25/25 | 6·10⁻⁸ |
| Per-instance tuned *p_m* | 7.53 | 27.95 | **137.49** | 376.60 | **945.83** | −0.61% | 13/25 | 0.85 |

**Discussion.**

1. **The decisions made on the tuning seeds hold on fresh seeds.** Inversion beats swap (−3.6%, *p* = 0.048), uniform mixing beats fixed probabilities (−2.0%, *p* = 0.045), and the Step 8 design beats the original template on all 25 runs, with tours 35.5% shorter. It lowers the deviation on every instance (berlin52 19.3% → 8.8%, eil101 88.2% → 20.1%, tsp225 365% → 139%, pcb442 725% → 382%, pr1002 1544% → 949%). The gains from mixing are small (about 2%) and only just significant, so they should be read as modest.
2. **Tuning every instance separately does not pay off.** The per-instance tuned *p_m* is not better than the Step 8 design (*p* = 0.85). It also did not beat the fixed design significantly on the fresh seeds (−1.4%, *p* = 0.079), although on the tuning seeds the same choices looked 1–8% better on single instances. This is the optimism of choosing on the same seeds (Section 5.6). One shared design with simple parameter control is as good as five tuned designs.
3. **The Step 8 design is not best on every instance.** Fixed probabilities are slightly better on berlin52 (7.5% vs 8.8%), and the swap design is better on pcb442. As the NFL discussion in Section 5.6 predicts, no single design is best everywhere.
4. **The large instances remain far from optimal.** Deviations of 140–950% for tsp225–pr1002 are large in absolute terms. With 50,000 evaluations, pr1002 gets about 50 evaluations per city, starting from random tours whose length is more than 20 times the optimum (Section 5.1). An EA that only uses general permutation operators cannot close this gap within the budget. This pointed to the one design component not yet tested: the **initialization** (Step 9).

## 5.9 Step 9: heuristic initialization and final results

**Question.** The assignment lists initialization as a design component, but Steps 1–8 kept the template's random start. Does heuristic initialization [27], [1] help, and how much of the initial population should be seeded?

At the Step 8 design, *s* of the 50 initial individuals were built by the nearest-neighbour rule from *s* distinct random start cities (Section 3.4), and the rest stayed random: *s* ∈ {0, 1, 5, 15, 50}. Two variants with fixed probabilities (*p_c* = 0.9, *p_m* = 0.1) checked whether uniform mixing still helps with a seeded start.

**Table 13.** Step 9, mean deviation % over the tuning seeds 101–105 (decisions), and pooled comparisons.

| Initial population | berlin52 | eil101 | tsp225 | pcb442 | pr1002 | Comparison | Rel. diff. | *p* |
|---|---|---|---|---|---|---|---|---|
| Random (Step 8 design) | 6.37 | 21.84 | 136.15 | 377.31 | 964.84 | | | |
| 1 NN tour | 4.12 | 8.11 | 20.63 | 21.00 | 26.68 | vs random | −57.8% | 6·10⁻⁷ |
| 5 NN tours | 4.40 | 6.49 | 17.19 | 17.32 | 21.94 | vs random | −58.7% | 6·10⁻⁷ |
| 15 NN tours | 2.43 | 5.50 | 15.27 | 16.07 | 22.23 | vs 5 NN | −1.05% | 0.063 |
| 50 NN tours | **0.59** | **5.31** | **13.66** | 15.62 | **20.95** | vs 5 NN | −2.00% | 0.0004 |
| 5 NN, fixed probabilities | 4.60 | 6.77 | 16.01 | 17.42 | 22.06 | vs 5 NN mixed | −0.07% | 0.96 |
| 50 NN, fixed probabilities | 1.66 | 5.72 | 14.51 | **15.62** | 21.16 | vs 50 NN mixed | +0.47% | 0.26 |

**Findings.**

1. **Initialization has by far the largest effect in the whole study.** A single nearest-neighbour tour in the initial population shortens the final tours by 58% (23 of 25 runs). On pr1002 the deviation falls from 965% to 27%. Under a budget of 50,000 evaluations, where the EA starts matters more than any operator or parameter choice.
2. **Seeding the whole population is best.** 50 NN tours are significantly better than 5 (−2.0%, *p* = 0.0004). The worry that seeding destroys diversity does not apply here: each tour starts from a different city, so the seeded tours differ, and the population keeps about 1.8 distinct edges per city on berlin52 at the end of the run.
3. **Parameter control no longer helps.** With a seeded start, uniform mixing is not better than fixed probabilities (*p* = 0.96 with 5 NN tours, *p* = 0.26 with 50). Its benefit in Steps 6b–8 came from searching *from random tours*. In terms of Section 5.7, a seeded population already starts in a good region and needs little exploration.

**Decision (same rule as Step 8, fixed before the reporting runs):** the simplest variant not significantly worse than the best. This is **50 NN tours with fixed probabilities** (+0.47% vs the best, *p* = 0.26). This is the final design (Table 1b).

**Final results (reporting seeds 1–5, not used for any decision).** All Step 9 variants were run on the reporting seeds. The findings hold:

- Seeding vs random: tours 59.7% shorter, *p* = 1.2·10⁻⁷.
- 50 vs 5 NN tours: 2.4% shorter, *p* = 0.0002.
- Fixed vs mixed probabilities: not significantly different (+0.57%, *p* = 0.063), although mixing won 16 of 25 runs.

**Table 14. Final results.** Final design (Table 1b), reporting seeds 1–5, 50,000 evaluations per run. Deviation % = (Average − Optimum) / Optimum × 100.

| Instance | Optimum | Best | Average | Worst | Std | **Deviation %** |
|---|---|---|---|---|---|---|
| berlin52 | 7542 | **7542** | 7676.8 | 7807 | 115.4 | **1.79** |
| eil101 | 629 | 654 | 666.0 | 681 | 9.9 | **5.88** |
| tsp225 | 3916 | 4369 | 4442.8 | 4520 | 61.3 | **13.45** |
| pcb442 | 50778 | 58421 | 59095.2 | 59620 | 483.5 | **16.38** |
| pr1002 | 259045 | 311547 | 312796.0 | 313703 | 895.0 | **20.75** |

On berlin52, 1 of the 5 runs finds the optimal tour (7542) and stops after 37,254 evaluations. The same design with mixed probabilities found it in 3 of 5 runs, after 19,533 evaluations on average. Relative to the original template on the same seeds (Table 12), the deviation fell from 19.3% to 1.8% on berlin52, from 88.2% to 5.9% on eil101, from 365% to 13.4% on tsp225, from 725% to 16.4% on pcb442, and from 1544% to 20.8% on pr1002.

[TODO Figures: `results/step9-reporting/best_tour_nn-50-fixed_<tsp>.png` (best tour per instance) and `results/step9-reporting/convergence_<tsp>.png`.]

**How much of this is the EA, and how much the heuristic?** A seeded EA must be compared with the heuristic it starts from. Table 15 compares, per run, the best of the 50 nearest-neighbour tours in the initial population with the final result.

**Table 15.** Mean deviation % of the best initial tour and of the final tour (reporting seeds), and the EA's improvement.

| Instance | Best initial NN tour | Final (EA) | Improvement by the EA |
|---|---|---|---|
| berlin52 | 8.47% | 1.79% | −6.16% |
| eil101 | 19.17% | 5.88% | −11.15% |
| tsp225 | 17.70% | 13.45% | −3.61% |
| pcb442 | 17.43% | 16.38% | −0.90% |
| pr1002 | 21.25% | 20.75% | −0.41% |

On the two small instances the EA does most of the work: it turns tours 8–19% above the optimum into tours 2–6% above it. On pcb442 and pr1002 it improves the starting tours by less than 1%, so **on the large instances the result is mostly due to the heuristic**. The reason lies in the operators. Inversion reverses a *random* segment, and on an already good tour of 442 or 1002 cities almost every such move makes the tour longer. Improving a good large tour requires moves between *nearby* cities, which is local search. The natural next step is therefore a memetic EA that applies a neighbourhood-restricted 2-opt move to offspring [13], with every evaluation counted against the budget. An earlier draft of our design (Section 6, item 5) used such a neighbour-guided move and reached about 12% on pr1002, which supports this explanation.

# 6. Problems and pitfalls

1. **Rounding (Section 2.4).** Three mistakes are easy to make: Python's `round()` (halves to even; 3861 instead of 3916 on tsp225), rounding the total instead of each edge (wrong on all five instances), and using unrounded distances as the fitness while comparing with integer optima. We use `(int)(d + 0.5)` per edge everywhere and check that the five optimal tours reproduce TSP-BEST exactly.
2. **Budget by generations vs. evaluations.** The template stops after 200 generations. Its number of evaluations (≈ 49,440 here) depends on the population size and the operator probabilities, so variants with different settings would get different budgets. Counting evaluations explicitly fixes this. DEAP's `varAnd` does not re-evaluate unchanged offspring, so they must not be counted.
3. **Stale cache.** The distance matrix is cached with `pickle` and loaded without checking. If the distance formula changes, old values are used silently. The rounded matrix is therefore derived from the cache on every load rather than stored in it.
4. **DEAP operator details.**
   - `mutInversion` can choose the same two positions and change nothing, but the offspring is still marked as mutated and re-evaluated, wasting an evaluation. Its slice also leaves out the end position, although the docstring says `[low, high]`.
   - `selRoulette` assumes maximisation and positive fitness.
   - DEAP's crossovers modify the parents in place; `varAnd` clones them first.
5. **Custom vs. established operators.** An earlier draft used a custom "nearest-neighbour-guided inversion" that performed well, but it was not an operator from the literature and was hard to justify. We replaced it with DEAP operators and the published ERX, so every design choice can be traced to a citable source. Our first ERX followed the original paper [11]; it did not give priority to edges shared by both parents, so we switched to the enhanced version [12] and re-ran every step that uses it.
6. **Reproducibility.**
   - A random choice among elements of a Python `set` depends on its iteration order, so ERX sorts the candidates before `random.choice`.
   - Each parallel run seeds itself, so results are identical for any number of worker processes.
   - Tuning and reporting seeds are kept separate.
7. **A plausible hypothesis that the data rejected (step 2).** We expected inversion to beat swap mutation, based on locality and the literature. It did not, at least at the baseline's weak selection pressure and short run length. Running the comparison in several settings, rather than only one, also revealed that the mutation ranking changes with the crossover and the instance.
8. **Statistics with few runs.** With 5 seeds, a per-instance exact Wilcoxon test can never give *p* < 0.05 (two-sided minimum 0.0625). We therefore pool the paired runs over all instances and also report wins per instance.
9. **A hidden confound: unchanged offspring are free.** In Step 3 the no-crossover variant appeared to be the best design on the two largest instances. The cause was that its unchanged offspring are not re-evaluated, so it ran 9× more generations for the same 50,000 evaluations. Comparing variants under an evaluation budget is fair, but explaining *why* a variant wins requires also looking at the generations completed. We now log them for every run.
10. **The best setting at the edge of the tested range.** In Step 4, μ = 50 was both the best and the smallest value tested. Stopping there would have left open whether an even smaller population is better. Extending the range (Step 4b) showed the best value lies between 30 and 50.
11. **An operator's merit depends on the rest of the design.** ERX was the best crossover in Step 3 (μ = 300) and significantly worse than OX in Step 4 (μ = 50: +16.5%, *p* = 6·10⁻⁸), because it makes small populations collapse. Had we fixed the crossover in Step 3 and never revisited it, the final design would have been worse.
12. **A biased pooled test (our own mistake).** At first we pooled the Wilcoxon test over the *differences in deviation %*. Deviations on pr1002 are about 100× those on berlin52, so its differences always received the largest ranks, and the pooled test effectively measured pr1002 alone. For example, berlin52 improving from 15.0% to 10.9% deviation is a 3.6% shorter tour, as large in relative terms as pr1002 changing from 975% to 1023% (4.6%). Testing the *log length ratios* gives every instance equal weight. Correcting this changed several *p*-values: inversion vs swap went from 0.063 to 0.026, ERX vs OX at μ = 300 from 0.09 to 0.0004 (with the original ERX, before the switch in item 5), and no crossover vs OX at μ = 50 from 0.2 to 0.0006. None of the decisions changed, but several became better supported. Because all runs were saved, the analysis could be redone without re-running any EA (`--reanalyse`).
13. **Opposite preferences cancel out in pooled results (Step 5b).** A pooled test showing "no significant difference" can hide strong but opposite effects on individual instances. Per-instance wins and relative differences must be read alongside the pooled result.
14. **A controller needs a control experiment.** Adaptive pursuit looked like a success (−2.76%, *p* = 0.012). Only a variant that mixes the same settings *without learning* showed that the learning itself contributed nothing measurable. Without that control we would have credited the wrong mechanism.
15. **The controller's own parameters need tuning too.** Adaptive pursuit adds α, β, *P_min*, the arm set and the reward definition. Its result depended strongly on the learning rate (from −0.6% to −2.8%). Parameter control does not remove tuning; it moves it to a different level.
16. **Our own prediction about the reward was wrong.** We expected a per-evaluation reward to capture the "unchanged offspring are free" effect. It did not, because it measures improvement per child, not progress of the run per evaluation.
17. **The order of the steps mattered, and the verification step caught it.** Swap mutation was chosen at μ = 300 (Step 2) but was the wrong choice at the final μ = 50 (Step 7). Without the verification step we would have kept swap, although on the reporting seeds inversion gives 3.6% shorter tours (both with fixed probabilities).
18. **Separate reporting seeds expose optimistic choices.** Per-instance tuned probabilities looked better on the tuning seeds but gave no significant gain on the reporting seeds. Without separate seeds, the per-instance design would have looked better than it is.
19. **Fixing the decision rule before the final runs.** We wrote down how the final design would be chosen before running the reporting seeds. This prevents picking whichever variant happens to look best on the seeds used for reporting.
20. **The untested default mattered most.** Initialization was the one design component we had not tested, because the template's random start was kept by default. It turned out to have by far the largest effect (−59.7%). Defaults deserve testing as much as the parameters that look interesting.
21. **Crediting the right component.** With a seeded start, the final tours on pcb442 and pr1002 are less than 1% shorter than the best starting tour. Reporting "20.8% on pr1002" without that comparison would credit the EA for what the heuristic did.

**Knowledge and experience gained.** [TODO 1 paragraph per member.]

# References

[CHECK every entry before submission.]

[1] A. E. Eiben and J. E. Smith, *Introduction to Evolutionary Computing*, 2nd ed. Berlin, Germany: Springer, 2015.

[2] P. Larrañaga, C. M. H. Kuijpers, R. H. Murga, I. Inza, and S. Dizdarevic, "Genetic algorithms for the travelling salesman problem: A review of representations and operators," *Artif. Intell. Rev.*, vol. 13, no. 2, pp. 129–170, 1999.

[3] F.-A. Fortin, F.-M. De Rainville, M.-A. Gardner, M. Parizeau, and C. Gagné, "DEAP: Evolutionary algorithms made easy," *J. Mach. Learn. Res.*, vol. 13, pp. 2171–2175, 2012.

[4] G. Reinelt, "TSPLIB—A traveling salesman problem library," *ORSA J. Comput.*, vol. 3, no. 4, pp. 376–384, 1991.

[5] G. Reinelt, "TSPLIB 95," Universität Heidelberg, Heidelberg, Germany, 1995. [Online]. Available: https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/tsp95.pdf. Best known solutions: https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html

[6] Python Software Foundation, "Built-in functions: round()," *Python 3.12 Documentation*. [Online]. Available: https://docs.python.org/3.12/library/functions.html#round

[7] "std::round, std::lround, std::llround," *cppreference.com*. [Online]. Available: https://en.cppreference.com/w/cpp/numeric/math/round

[8] L. Davis, "Applying adaptive algorithms to epistatic domains," in *Proc. 9th Int. Joint Conf. Artif. Intell. (IJCAI)*, 1985, pp. 162–164.

[9] D. E. Goldberg and R. Lingle, "Alleles, loci, and the traveling salesman problem," in *Proc. 1st Int. Conf. Genetic Algorithms and Their Applications*, 1985, pp. 154–159.

[10] V. A. Cicirello and S. F. Smith, "Modeling GA performance for control parameter optimization," in *Proc. Genetic and Evolutionary Computation Conf. (GECCO)*, 2000, pp. 235–242.

[11] D. Whitley, T. Starkweather, and D. Fuquay, "Scheduling problems and traveling salesmen: The genetic edge recombination operator," in *Proc. 3rd Int. Conf. Genetic Algorithms*, 1989, pp. 133–140.

[12] T. Starkweather, S. McDaniel, K. Mathias, D. Whitley, and C. Whitley, "A comparison of genetic sequencing operators," in *Proc. 4th Int. Conf. Genetic Algorithms*, 1991, pp. 69–76.

[13] G. A. Croes, "A method for solving traveling-salesman problems," *Oper. Res.*, vol. 6, no. 6, pp. 791–812, 1958.

[14] J. H. Holland, *Adaptation in Natural and Artificial Systems*. Ann Arbor, MI, USA: Univ. of Michigan Press, 1975.

[15] D. E. Goldberg and K. Deb, "A comparative analysis of selection schemes used in genetic algorithms," in *Foundations of Genetic Algorithms*, G. J. E. Rawlins, Ed. San Mateo, CA, USA: Morgan Kaufmann, 1991, pp. 69–93.

[16] J. Derrac, S. García, D. Molina, and F. Herrera, "A practical tutorial on the use of nonparametric statistical tests as a methodology for comparing evolutionary and swarm intelligence algorithms," *Swarm Evol. Comput.*, vol. 1, no. 1, pp. 3–18, 2011.

[17] D. C. Montgomery, *Design and Analysis of Experiments*, 9th ed. Hoboken, NJ, USA: Wiley, 2017.

[18] A. E. Eiben and S. K. Smit, "Parameter tuning for configuring and analyzing evolutionary algorithms," *Swarm Evol. Comput.*, vol. 1, no. 1, pp. 19–31, 2011.

[19] F. Rothlauf, *Representations for Genetic and Evolutionary Algorithms*, 2nd ed. Berlin, Germany: Springer, 2006.

[20] D. H. Wolpert and W. G. Macready, "No free lunch theorems for optimization," *IEEE Trans. Evol. Comput.*, vol. 1, no. 1, pp. 67–82, 1997.

[21] A. E. Eiben, R. Hinterding, and Z. Michalewicz, "Parameter control in evolutionary algorithms," *IEEE Trans. Evol. Comput.*, vol. 3, no. 2, pp. 124–141, 1999.

[22] D. Thierens, "An adaptive pursuit strategy for allocating operator probabilities," in *Proc. Genetic and Evolutionary Computation Conf. (GECCO)*, 2005, pp. 1539–1546.

[23] J. E. Smith and T. C. Fogarty, "Self adaptation of mutation rates in a steady state genetic algorithm," in *Proc. IEEE Int. Conf. Evolutionary Computation (ICEC)*, 1996, pp. 318–323.

[24] T. Bäck and M. Schütz, "Intelligent mutation rate control in canonical genetic algorithms," in *Foundations of Intelligent Systems (ISMIS 1996)*, Lecture Notes in Computer Science, vol. 1079. Berlin, Germany: Springer, 1996, pp. 158–167.

[25] A. E. Eiben and C. A. Schippers, "On evolutionary exploration and exploitation," *Fundamenta Informaticae*, vol. 35, no. 1–4, pp. 35–50, 1998.

[26] M. Črepinšek, S.-H. Liu, and M. Mernik, "Exploration and exploitation in evolutionary algorithms: A survey," *ACM Comput. Surv.*, vol. 45, no. 3, art. 35, pp. 1–33, 2013.

[27] J. J. Grefenstette, "Incorporating problem specific knowledge into genetic algorithms," in *Genetic Algorithms and Simulated Annealing*, L. Davis, Ed. London, U.K.: Pitman, 1987, pp. 42–60.
