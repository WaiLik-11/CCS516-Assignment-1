<!--
CONDENSED REPORT (target: max 15 pages, Times New Roman 12 pt, single spacing, incl. figures and references).
The long draft CCS516_A01_Report.md is the full reference; every number here is taken from it / from results/.
Markers: [TODO] = to fill in, [CHECK] = verify before submitting. Citations: IEEE style.
-->

# Cover page

**CCS516 Computational Intelligence — Assignment 01**
**Solving the Traveling Salesman Problem Using Distributed Evolutionary Algorithms in Python (DEAP)**

School of Computer Sciences, Universiti Sains Malaysia — Semester 1, 2026/2027

Group: [TODO GroupXX]

| Name | Matric No. | Contribution |
|---|---|---|
| [TODO] | [TODO] | [TODO] |
| [TODO] | [TODO] | [TODO] |

Supporting documents (in the submission package): source code (`DEAP-source-codes/`), all experiment results (`results/`, one folder per design step), and the step log (`README.md`).

---

# 1. Introduction

The Traveling Salesman Problem (TSP) asks for the shortest closed tour that visits every city exactly once. It is NP-hard, so evolutionary algorithms (EAs) are a common way to find good tours within a limited budget [1], [2]. We design an EA with DEAP [3] for five TSPLIB instances [4]: berlin52, eil101, tsp225, pcb442 and pr1002. Each run is limited to **50,000 fitness evaluations**, results are averaged over **5 runs**, and quality is measured as

$$\text{Deviation (\%)} = \frac{\text{Average} - \text{Optimum}}{\text{Optimum}} \times 100 \quad (0\% \text{ is best}).$$

Section 2 covers the optimal tours and their lengths (Tasks 1 and 2). Section 3 presents the EA. Section 4 describes how the design was chosen by experiment. Section 5 reports the results, and Section 6 the problems and lessons.

# 2. Optimal tours and tour lengths (Tasks 1 and 2)

**Reading the data.** In each `.tsp` file, `tsp.py` skips the header and reads one `index x y` line per city from `NODE_COORD_SECTION` until `EOF` (pr1002 has no `EOF` line; the end of the file also stops it). The `.opt.tour` files list the optimal tour as 1-based city IDs ending in `-1`. `loadTour()` converts them to 0-based indices and checks that the tour is a permutation of all cities. `plotData()` draws the cities as red dots and connects consecutive cities, including the return to the start, with blue lines (Figure 1).

[TODO Figure 1: the five optimal tours, `plots/<tsp>_opt.png`, in a grid as in Figure 1 of the assignment.]

**Euclidean distance as defined by TSPLIB.** All five instances have type `EUC_2D`, for which TSPLIB defines [5]

```
xd = x[i] - x[j];  yd = y[i] - y[j];
dij = nint( sqrt( xd*xd + yd*yd ) );
```

where `nint` rounds to the nearest integer, "implemented as `(int)(x + 0.5)`" in C. **Each edge is rounded before summing**, and halves are rounded up. `tsp.py` precomputes the integer matrix once as `(distances + 0.5).astype(int)`; truncation is correct here because distances are never negative.

**The rounding trap in Python.** Python's `round()` does not behave like C/C++. Since Python 3 it rounds halves **to even** (`round(0.5) = 0`, `round(2.5) = 2`, `round(28.5) = 28`) [6], while C++ `std::round` rounds halves **away from zero** (`std::round(2.5) = 3`) [7]. NumPy's `np.round` also rounds halves to even, and binary floating point adds surprises such as `round(2.675, 2) = 2.67`. Table 1 shows the effect.

**Table 1.** Length of the optimal tour under three rounding methods (output of `tsp.py`).

| Instance | TSP-BEST [5] | `nint` per edge (TSPLIB) | Python `round()` per edge | `round(total)` | Edges ending in .5 |
|---|---|---|---|---|---|
| berlin52 | 7542 | **7542** | 7542 | 7544 | 0 |
| eil101 | 629 | **629** | 629 | 642 | 0 |
| tsp225 | 3916 | **3916** | 3861 | 3859 | 124 |
| pcb442 | 50778 | **50778** | 50778 | 50784 | 0 |
| pr1002 | 259045 | **259045** | 259045 | 259067 | 0 |

**Answer to Task 2: 7542, 629, 3916, 50778 and 259045**, identical to TSP-BEST. Only `nint` per edge reproduces all five optima. Python's `round()` fails only on tsp225. For integer coordinates, √(xd² + yd²) is either an integer or irrational, so it never ends in exactly .5. tsp225 is a rectilinear drawing of the letters "TSP" with half-unit offsets (e.g. 183.92 − 155.42 = 28.5), so 124 of its 225 optimal edges end in .5. Rounding only the total is wrong for every instance.

# 3. The evolutionary algorithm (Task 3)

**Framework.** The given `tsp.py`, `solve_tsp.py` and `elitism.py` were extended. As given, an individual is a permutation of the city indices (`array('i')`), so every individual is a valid tour. The **fitness is the TSPLIB integer tour length**, the same measure as the optima. The template used the unrounded length, which on tsp225 differs from the TSPLIB length by 1.5% for the optimal tour (3859 vs 3916, Table 1).

**Evolutionary cycle and budget.** `elitism.py` extends DEAP's `eaSimple` with **elitism**: the best tours (`tools.HallOfFame`) are copied unchanged into every generation. It also enforces a **budget of 50,000 evaluations**, counting the initial population. Offspring that are neither crossed over nor mutated keep their fitness and cost nothing. If the last generation would exceed the budget, only the offspring that still fit are evaluated. The template instead stopped after 200 generations, which happens to cost about 49,440 evaluations with its settings, but changes with any population size or probability. Counting evaluations gives every variant the same budget.

**Termination.** A run stops on whichever comes first: (1) 50,000 evaluations, or (2) its best tour reaching the known optimum. Rule (2) is for benchmarking only, since it needs the optimum to be known. It cannot change the result: elitism keeps the best tour and nothing beats the optimum, so it only saves evaluations and measures how fast the optimum is found. Stopping on stagnation was rejected: with a fixed budget, stopping early can only make the result worse, and the curves of the large instances are still falling at 50,000 evaluations. Restarting on stagnation is left as future work.

**Operators considered.** TSP fitness depends on which cities are **adjacent** (the edges), not on their positions, so good operators pass on edges and change few at a time [1], [2]. All operators are from DEAP, except edge recombination and nearest-neighbour initialization. These two are not in DEAP, so we implemented them from the literature in DEAP's operator style (`operators.py`) and registered them in the toolbox like the others.

| Role | Operator | Idea | Ref. |
|---|---|---|---|
| Crossover | OX `tools.cxOrdered` | Keep a segment of parent 1; fill the rest in parent 2's order (keeps the segment's edges) | [8] |
| | PMX `tools.cxPartialyMatched`, UPMX `tools.cxUniformPartialyMatched` | Exchange positions and repair through the position mapping (preserve *positions*) | [9], [10] |
| | Enhanced ERX `operators.cxEdgeRecombination` | Build the child from the parents' edge map; follow an edge shared by both parents if possible, else move to the neighbour with the fewest remaining edges | [11], [12] |
| Mutation | Swap `tools.mutShuffleIndexes` | Exchange cities (up to 4 edges change) | [2] |
| | Inversion `tools.mutInversion` | Reverse a segment (2 edges change: a 2-opt move) | [13], [14] |
| Parent selection | Tournament `tools.selTournament` | Best of *k* random individuals; *k* sets the selection pressure | [15] |
| Initialization | Random (`tools.initIterate`); nearest neighbour (`operators.nearestNeighbourTour`) | Greedy tour: always go to the closest unvisited city, from a random start city | [16] |
| Parameter control | Adaptive pursuit; self-adaptive *p_m*; uniform mix | See Section 5.3 | [17]–[19] |

We use the enhanced ERX [12], which gives priority to edges present in both parents; the original [11] merges the parents' edges and can break an edge both parents share. Only about 3% of child edges are in neither parent. These are the known "edge failures" of the operator [12]. ERX makes one child per pair, so the second child starts from the other parent's first city, to fit DEAP's two-children `mate` interface.

**Table 2. Final design** and the evidence for each choice (details in Section 5).

| Component | Final choice | Evidence |
|---|---|---|
| Initialization | 50 nearest-neighbour tours from 50 distinct random start cities | Step 9: tours 59.5% shorter than with random initialization (reporting seeds, 25 of 25 runs, *p* = 6·10⁻⁸) |
| Parent selection | Tournament, *k* = 2 | Step 5a: better than *k* = 1, 3, 5, 7 (all *p* < 0.001) |
| Crossover | OX, *p_c* = 0.9 per pair | Steps 3, 4, 7: PMX, UPMX, ERX and no crossover significantly worse at μ = 50 |
| Mutation | Inversion, *p_m* = 0.1 per offspring | Step 7: better than swap at μ = 50 (−4.9%, *p* = 0.011); confirmed on the reporting seeds |
| Population model | Generational, μ = 50, of which 5 elites | Steps 4, 4b: better than μ = 300, 100, 20, 10 |
| Survivor selection | Offspring replace the parents; the 5 best-ever tours survive | Elitism keeps the best tour |
| Termination | 50,000 evaluations, or earlier when the known optimum is reached | Assignment budget; the optimum rule cannot change the result |
| Random seeds | Tuning 101–105; reporting 1–5; each run seeds `random` and NumPy | Section 4 |

# 4. How the design was chosen

**Protocol.** Every design decision was made by experiment, in nine steps (Steps 1–9). Steps 4, 6 and 7 have a follow-up (4b, 6b, 7b) and Step 5 has two parts (5a, 5b), so there are 13 experiments, all defined in `experiments.py`. With the final runs of Step 9 on the reporting seeds, this makes 2,050 runs.

- **Budget:** each run uses 50,000 evaluations, or fewer if it reaches the optimum first.
- **Seeds:** all decisions use **tuning seeds 101–105**. The reported results use **separate reporting seeds 1–5**, so they are not tuned on their own seeds.
- **Decision rule:** each step changes **one** component and adopts the winner. When variants tie, the choice is "the simplest variant not significantly worse than the best", fixed before the reporting runs.
- **Statistics:** variants are compared on the same seeds, so each pair of runs starts from the same initial population. We report wins per instance and a two-sided **Wilcoxon signed-rank test** [21] on the 25 paired **log length ratios** (5 instances × 5 seeds). Relative differences give every instance equal weight. A per-instance test is not meaningful with 5 seeds, because the exact two-sided *p*-value cannot go below 0.0625.
- **Reproducibility:** every step writes its parameters, seeds and command to `metadata.json`. Runs are deterministic and independent of how many execute in parallel.

**The order of the steps.** Changing one component at a time ignores interactions [22], [23], so the order matters. We studied mutation first, because the baseline analysis pointed to it. A **verification step** (Step 7) then re-tested the early choices at the final design. It found exactly such an interaction (Section 5.2).

# 5. Results

## 5.1 Baseline

The baseline keeps the template's operators and parameters: OX with *p_c* = 0.9, swap mutation with *p_m* = 0.1, *k* = 2, μ = 300 with 30 elites, and random initialization. Only the generation limit is replaced by the evaluation budget. On the tuning seeds it ends 14.8%, 83.7%, 360%, 745% and 1507% above the optimum. For comparison, the best of 300 random tours is 223%, 380%, 874%, 1333% and 2288% above it. The EA improves on random tours, but less and less as the instance grows. The convergence curves are still falling steeply at 50,000 evaluations: the search is far from converged.

## 5.2 The design path

Table 3 summarises the decisions. Each row is a design step on the tuning seeds, except Step 8; the full tables are in `results/<step>/` and the README step log.

**Table 3.** Design steps. "Rel. diff." is the pooled relative difference in tour length (negative = shorter tours), with the Wilcoxon *p*-value.

| Step | Question | Key result | Decision |
|---|---|---|---|
| 2 | Mutation (μ = 300) | Inversion vs swap: +2.95%, *p* = 0.026; *p_m* 0.5 worse than 0.1 on all 10 comparisons | Keep swap |
| 3 | Crossover (μ = 300) | PMX +19.3%, UPMX +13.9% (*p* < 10⁻⁶); ERX −19.9% (*p* = 2·10⁻⁷); no crossover −10.2%, but with 9× more generations | Drop PMX and UPMX; decide OX vs ERX with μ |
| 4, 4b | Population size μ × crossover | OX, μ = 50 vs 300: −30.1% (*p* = 2·10⁻⁶); at μ = 50, ERX +16.5% and no crossover +8.9% vs OX; μ = 30 tied, μ ≤ 20 worse | OX, μ = 50, 5 elites |
| 5a | Tournament size | *k* = 1: +66.9%, *k* = 3: +8.9%, *k* = 5: +22.9% vs *k* = 2 (all *p* < 0.001) | *k* = 2 |
| 5b | *p_c* × *p_m* grid | No setting significantly different (all *p* ≥ 0.096); small instances prefer more variation, pr1002 less | Keep 0.9 / 0.1 |
| 6, 6b | Parameter control | See Section 5.3 | Uniform mix |
| 7, 7b | Verification at μ = 50 | **Inversion vs swap: −4.88%, *p* = 0.011** (reversed from Step 2); OX confirmed | Inversion |
| 8 | Check on the reporting seeds (random initialization; not the final design) | vs fixed probabilities −2.0% (*p* = 0.045); inversion vs swap −3.6% (*p* = 0.048); per-instance tuning no better (*p* = 0.85) | Confirms Steps 6b–7b |
| 9 | Initialization | Nearest-neighbour seeding −58.7% (*p* < 10⁻⁶); 50 vs 5 seeded tours −2.0% (*p* = 0.0004); mixing no longer helps (*p* = 0.26) | 50 NN tours, fixed probabilities |

The main findings, in order of importance:

1. **Initialization matters most (Step 9).** Under this budget, where the EA starts matters more than any operator. Even a single nearest-neighbour tour in the initial population reduces pr1002's deviation from 965% to 27%. Seeding all 50 tours from different start cities is best.
2. **The number of generations is the next largest factor (Steps 3–4).** Reducing μ from 300 to 50 gives six times more generations within the budget and 30% shorter tours. It also explains a confound in Step 3: "no crossover" seemed best only because its unchanged offspring cost no evaluations, so it completed 1842 instead of 203 generations.
3. **Edge-preserving operators win, but their merit depends on the rest of the design.** Operators that preserve positions (PMX, UPMX) are clearly worse than OX, which preserves adjacency, as the literature predicts [2], [12]. ERX, designed for the TSP, was the best crossover at μ = 300 (better than OX on all five instances) but significantly the worst at μ = 50. In a small population it makes all tours nearly identical (final diversity ≈ 1.04–1.15 distinct edges per city, where 1.0 means all tours are the same), so crossover only produces copies.
4. **The order of the steps mattered (Step 7).** Swap beat inversion at μ = 300 (Step 2), but inversion beat swap at μ = 50 (Step 7). Without the verification step the final design would have kept the wrong mutation, which gives 3.6% longer tours on the reporting seeds.

## 5.3 Parameter control and instance dependence

Step 5b showed that the instances prefer **opposite** probabilities. With *p_c* = 1.0, *p_m* = 0.2, tours on eil101 are 6.9% shorter. With *p_c* = 0.6, *p_m* = 0.05, tours on pr1002 are 7.6% shorter, on all 5 seeds. No single setting suits all instances. Choosing parameters before the run is *tuning*; changing them during the run is *control* [20]. We implemented one controller of each feedback-based class [20], using only DEAP operators:

- **Adaptive pursuit** [17]. Each generation one of three (*p_c*, *p_m*) settings, (0.6, 0.05), (0.9, 0.1) or (1.0, 0.2), is drawn with probabilities that move towards the setting whose children improved most on their parents. [CHECK the update rule against [17].]
- **Self-adaptive *p_m*** [18], [19]. Each individual carries its own *p_m*, which is inherited and perturbed by a log-normal step before use.

**Table 4.** Parameter control vs fixed probabilities (random initialization, tuning seeds; pooled relative difference in tour length).

| Variant | Small instances | Large instances | Pooled | *p* |
|---|---|---|---|---|
| Self-adaptive *p_m* | berlin52 −2.7%, eil101 −6.7% (5/5 each) | pcb442 +9.9%, pr1002 +4.4% (0/5 each) | +1.6% | 0.28 |
| Adaptive pursuit, α = β = 0.8 | −1.0%, −3.4% | −2.2%, −1.1% | −1.0% | 0.30 |
| Adaptive pursuit, α = β = 0.1 | −1.0%, −5.9% | −1.2%, −3.4% | −2.8% | 0.012 |
| **Uniform mix (no learning, control)** | +3.7%, −5.2% | −1.3%, −2.7% | −1.4% | 0.039 |

- **Self-adaptation raised *p_m* on every instance** (from 0.1 to 0.19–0.31). That suits the small instances but not pr1002, which prefers *less* mutation. Selection between individuals cannot see that unchanged offspring cost no evaluations.
- **Adaptive pursuit looked successful at a low learning rate.** But a control variant that **mixes the same three settings at random, without learning**, was just as good: no pursuit variant beat it (best *p* = 0.28), and the arm probabilities fluctuated around 1/3 without settling on any setting. **The gain came from alternating settings, not from learning.** After a Bonferroni correction for the five controller variants, none of these gains is significant.

Uniform mixing was confirmed with inversion (Step 7b: −1.85%, *p* = 0.022) and on the reporting seeds (Step 8: −2.0%, *p* = 0.045). Once the population was seeded with nearest-neighbour tours, however, mixing no longer helped (Step 9: *p* = 0.26), so the final design uses fixed probabilities.

**One design or one per instance?** The No Free Lunch theorem [24] states that no algorithm is best *averaged over all problems*. It does not require a separate design per instance of one problem class. Choosing the best probabilities per instance looked up to 8% better on single instances on the tuning seeds (Step 5b). But a design with *p_m* tuned per instance was not better than the shared design on the reporting seeds (Step 8, *p* = 0.85). With five tuning seeds, the per-instance choices mostly fitted noise.

## 5.4 Exploration and exploitation

One trade-off explains most findings [25], [26]. As a simple measure, we use how far the final population's average tour is above its best tour. A large gap means the population is still diverse and **exploring**; a gap near 0% means it has converged and is only **exploiting**.

**Table 5.** Final population average above its best tour (%), tuning seeds.

| Variant | berlin52 | pcb442 | pr1002 |
|---|---|---|---|
| *k* = 1 (random parents) | 52.2 | 7.9 | 4.0 |
| Baseline (μ = 300) | 6.0 | 3.0 | 1.3 |
| **Design after Step 5 (μ = 50, *k* = 2)** | 2.3 | 0.8 | 0.4 |
| *k* = 7 | 0.6 | 0.1 | 0.1 |
| ERX, μ = 50 | 0.6 | 0.3 | 0.1 |

- **The baseline explored too much** for its budget: it was still diverse after 203 generations. The largest operator-level gain (μ = 50) shifted it towards exploitation.
- **Too much exploitation failed too.** Larger tournaments, μ ≤ 20 and ERX in a small population all collapsed the population and were worse. The chosen design sits between the extremes.
- **The instances differ in the same way.** The small instances converge within the budget and gain from *more exploration* (higher *p_c*, *p_m*; self-adaptation). pr1002 is far from converged and gained from *every* more exploitative option: no crossover, μ = 30, *k* = 3, lower *p_c* and *p_m*.
- **Nearest-neighbour seeding** uses problem knowledge to start in a good region, so the budget goes into exploitation from the start, while 50 different start cities keep the tours diverse.

The gap measures diversity in fitness only. A tour-based measure, the number of distinct edges per city (logged in every step), confirms that the collapsing variants also collapse in edges.

## 5.5 Final results

**Table 6. Final results.** Final design (Table 2) on the reporting seeds 1–5, 50,000 evaluations per run, averaged over 5 runs.

| Instance | Optimum | Best | Average | Worst | Std | **Deviation %** |
|---|---|---|---|---|---|---|
| berlin52 | 7542 | **7542** | 7676.8 | 7807 | 115.4 | **1.79** |
| eil101 | 629 | 654 | 666.0 | 681 | 9.9 | **5.88** |
| tsp225 | 3916 | 4369 | 4442.8 | 4520 | 61.3 | **13.45** |
| pcb442 | 50778 | 58421 | 59095.2 | 59620 | 483.5 | **16.38** |
| pr1002 | 259045 | 311547 | 312796.0 | 313703 | 895.0 | **20.75** |

[TODO Figure 2: best tour per instance, `results/step9-reporting/best_tour_nn-50-fixed_<tsp>.png`. Figure 3: convergence, e.g. `results/step9-reporting/convergence_pr1002.png`.]

On berlin52, 1 of the 5 runs finds the optimal tour, after 37,254 evaluations. On the same seeds, the final design's tours are **73.9% shorter than the original template's** (25 of 25 runs, *p* = 6·10⁻⁸). The deviation falls from 19.3% to 1.8% (berlin52), 88.2% to 5.9% (eil101), 365% to 13.4% (tsp225), 725% to 16.4% (pcb442) and 1544% to 20.8% (pr1002).

**How much is the EA, and how much the heuristic?** Table 7 compares each run's best *initial* nearest-neighbour tour with its final tour.

**Table 7.** Mean deviation % of the best initial tour and of the final tour (reporting seeds).

| | berlin52 | eil101 | tsp225 | pcb442 | pr1002 |
|---|---|---|---|---|---|
| Best initial NN tour | 8.47 | 19.17 | 17.70 | 17.43 | 21.25 |
| Final tour (EA) | 1.79 | 5.88 | 13.45 | 16.38 | 20.75 |
| Improvement by the EA | −6.2% | −11.2% | −3.6% | −0.9% | −0.4% |

On the small instances the EA does most of the work. On pcb442 and pr1002 it improves the starting tours by less than 1%, so **there the result is mostly due to the heuristic**. Inversion reverses a *random* segment, and on a good tour of 442 or 1002 cities almost every such move makes the tour longer. Further gains on large instances need moves between *nearby* cities, i.e. local search: a memetic EA with neighbourhood-restricted 2-opt [13]. An early draft of our design used such a move and reached about 12% on pr1002, which supports this explanation (Section 6).

**Alternative applications.** The same permutation EA applies to other ordering problems:

- **scheduling and sequencing** (job-shop and flow-shop scheduling; ERX was introduced for scheduling problems [11]);
- **vehicle routing**, as several tours with capacity constraints;
- **drilling and manufacturing paths**: pcb442 itself is a circuit-board drilling problem [4].

Three lessons transfer to these problems: choose operators by what the fitness depends on (adjacency vs position); count the budget in evaluations, not generations; and test a parameter-control mechanism against a no-learning control.

# 6. Problems, pitfalls and reflection

1. **Rounding and measurement.** Python's `round()` (halves to even), rounding the total instead of each edge, and an unrounded fitness compared with integer optima all give wrong lengths (Table 1). We check that the five optimal tours reproduce TSP-BEST exactly.
2. **Budget accounting.** The template limited generations, not evaluations. Unchanged offspring are free under DEAP's `varAnd`, which made "no crossover" look best in Step 3 until we looked at the generation counts.
3. **Interactions and order.** ERX was best at μ = 300 and worst at μ = 50; swap was best at μ = 300 and inversion at μ = 50. Testing one component at a time is only safe with a verification step.
4. **The untested default mattered most.** Initialization was kept as given until Step 9, but it had by far the largest effect. And crediting the result correctly required comparing the EA with the heuristic it starts from (Table 7).
5. **Statistics.** With 5 seeds, a per-instance Wilcoxon test can never reach *p* < 0.05. Our first pooled test used differences in deviation %, which let pr1002 dominate. Switching to log length ratios changed several *p*-values, though no decision. Opposite effects on different instances can also cancel out in a pooled test (Step 5b), so we always read the per-instance wins as well.
6. **Testing our own hypotheses.** Two of our predictions were wrong: that inversion would beat swap from the start, and that a per-evaluation reward would make adaptive pursuit learn pr1002's preference. Only a control variant (uniform mix) showed that the controller's learning contributed nothing. We also replaced an early, well-performing but custom "guided inversion" operator with DEAP and literature operators, so that every choice can be justified by a source.

**Knowledge and experience gained.** [TODO one paragraph per member, in your own words: what you learned about EA design, experimentation and DEAP, and what you would do differently.]

# References

[CHECK every entry before submission.]

[1] A. E. Eiben and J. E. Smith, *Introduction to Evolutionary Computing*, 2nd ed. Berlin, Germany: Springer, 2015.

[2] P. Larrañaga, C. M. H. Kuijpers, R. H. Murga, I. Inza, and S. Dizdarevic, "Genetic algorithms for the travelling salesman problem: A review of representations and operators," *Artif. Intell. Rev.*, vol. 13, no. 2, pp. 129–170, 1999.

[3] F.-A. Fortin, F.-M. De Rainville, M.-A. Gardner, M. Parizeau, and C. Gagné, "DEAP: Evolutionary algorithms made easy," *J. Mach. Learn. Res.*, vol. 13, pp. 2171–2175, 2012.

[4] G. Reinelt, "TSPLIB—A traveling salesman problem library," *ORSA J. Comput.*, vol. 3, no. 4, pp. 376–384, 1991.

[5] G. Reinelt, "TSPLIB 95," Universität Heidelberg, 1995. [Online]. Available: https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/tsp95.pdf. Best known solutions: https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html

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

[16] J. J. Grefenstette, "Incorporating problem specific knowledge into genetic algorithms," in *Genetic Algorithms and Simulated Annealing*, L. Davis, Ed. London, U.K.: Pitman, 1987, pp. 42–60.

[17] D. Thierens, "An adaptive pursuit strategy for allocating operator probabilities," in *Proc. Genetic and Evolutionary Computation Conf. (GECCO)*, 2005, pp. 1539–1546.

[18] J. E. Smith and T. C. Fogarty, "Self adaptation of mutation rates in a steady state genetic algorithm," in *Proc. IEEE Int. Conf. Evolutionary Computation (ICEC)*, 1996, pp. 318–323.

[19] T. Bäck and M. Schütz, "Intelligent mutation rate control in canonical genetic algorithms," in *Foundations of Intelligent Systems (ISMIS 1996)*, LNCS, vol. 1079. Berlin, Germany: Springer, 1996, pp. 158–167.

[20] A. E. Eiben, R. Hinterding, and Z. Michalewicz, "Parameter control in evolutionary algorithms," *IEEE Trans. Evol. Comput.*, vol. 3, no. 2, pp. 124–141, 1999.

[21] J. Derrac, S. García, D. Molina, and F. Herrera, "A practical tutorial on the use of nonparametric statistical tests as a methodology for comparing evolutionary and swarm intelligence algorithms," *Swarm Evol. Comput.*, vol. 1, no. 1, pp. 3–18, 2011.

[22] D. C. Montgomery, *Design and Analysis of Experiments*, 9th ed. Hoboken, NJ, USA: Wiley, 2017.

[23] A. E. Eiben and S. K. Smit, "Parameter tuning for configuring and analyzing evolutionary algorithms," *Swarm Evol. Comput.*, vol. 1, no. 1, pp. 19–31, 2011.

[24] D. H. Wolpert and W. G. Macready, "No free lunch theorems for optimization," *IEEE Trans. Evol. Comput.*, vol. 1, no. 1, pp. 67–82, 1997.

[25] A. E. Eiben and C. A. Schippers, "On evolutionary exploration and exploitation," *Fundamenta Informaticae*, vol. 35, no. 1–4, pp. 35–50, 1998.

[26] M. Črepinšek, S.-H. Liu, and M. Mernik, "Exploration and exploitation in evolutionary algorithms: A survey," *ACM Comput. Surv.*, vol. 45, no. 3, art. 35, pp. 1–33, 2013.
