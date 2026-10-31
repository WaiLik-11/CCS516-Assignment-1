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

## Running scripts

```bash
uv run python DEAP-source-codes/solve_tsp.py
```

No need to activate `.venv` manually; `uv run` uses it automatically.

## Notes

- Dependencies are the same as the DEAP course environment, except `multidict` is 6.0.5 (6.0.4 fails to build on macOS arm64 with Python 3.12).
- To add a package: `uv add <package>`, then commit `pyproject.toml` and `uv.lock`.
