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
│   ├── solve_tsp.py                  # the EA (DEAP toolbox, operators, one run)
│   └── elitism.py                    # modified eaSimple with elitism
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
uv run python DEAP-source-codes/solve_tsp.py berlin52   # run the EA on one problem
uv run python DEAP-source-codes/tsp.py --help
```

No need to activate `.venv` manually; `uv run` uses it automatically.

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
