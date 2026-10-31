"""Shared paths and experiment settings, imported by every script."""

from pathlib import Path
import argparse

ROOT_DIR = Path(__file__).resolve().parent.parent
TSP_DATA_DIR = ROOT_DIR / "tsp-data"
PLOTS_DIR = ROOT_DIR / "plots"
RESULTS_DIR = ROOT_DIR / "results"

TSP_NAMES = ["berlin52", "eil101", "tsp225", "pcb442", "pr1002"]

# EA settings
MAX_EVALUATIONS = 50_000
NUM_RUNS = 5
RANDOM_SEEDS = [1, 2, 3, 4, 5]

def parse_tsp_names(description="Run on selected TSP instances"):
    """Return the TSP names chosen on the command line (default: all five)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "tsp",
        nargs="*",
        default=TSP_NAMES,
        choices=TSP_NAMES,
        help="one or more of: " + ", ".join(TSP_NAMES) + " (default: all)",
    )
    return parser.parse_args().tsp