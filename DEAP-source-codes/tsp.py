import pickle
import os

import numpy as np
import matplotlib.pyplot as plt

import config


class TravelingSalesmanProblem:
    """This class encapsulates the Traveling Salesman Problem.
    City coordinates are read from an online file and distance matrix is calculated.
    The data is serialized to disk.
    The total distance can be calculated for a path represented by a list of city indices.
    A plot can be created for a path represented by a list of city indices.

    :param name: The name of the corresponding TSPLIB problem, e.g. 'burma14' or 'bayg29'.
    """

    def __init__(self, name):
        """
        Creates an instance of a TSP

        :param name: name of the TSP problem
        """

        # initialize instance variables:
        self.name = name
        self.locations = []
        self.distances = []
        self.tspSize = 0

        self.data_path = str(config.TSP_DATA_DIR)

        # initialize the data:
        self.__initData()

    def __len__(self):
        """
        returns the length of the underlying TSP
        :return: the length of the underlying TSP (number of cities)
        """
        return self.tspSize

    def __initData(self):
        """Reads the serialized data, and if not available - calls __create_data() to prepare it"""

        # attempt to read serialized data:
        try:
            with open(
                os.path.join(self.data_path, f"{self.name}-loc.pickle"), "rb"
            ) as f:
                self.locations = pickle.load(f)
            with open(
                os.path.join(self.data_path, f"{self.name}-dist.pickle"), "rb"
            ) as f:
                self.distances = pickle.load(f)
        except (OSError, IOError):
            pass

        # serailized data not found - create the data from scratch:
        if not self.locations or not self.distances:
            self.__createData()

        # set the problem 'size':
        self.tspSize = len(self.locations)

        # TSPLIB EUC_2D distances: dij = nint(sqrt(xd*xd + yd*yd)), nint(x) = (int)(x + 0.5) as in C.
        # Python's round() would be wrong here: it rounds halves to even (round(28.5) == 28).
        # astype() truncates like a C cast, which is correct because distances are >= 0.
        self.distancesRounded = (np.asarray(self.distances) + 0.5).astype(np.int64)

    def __createData(self):
        """Reads the desired TSP file from the Internet, extracts the city coordinates, calculates the distances
        between every two cities and uses them to populate a distance matrix (two-dimensional array).
        It then serializes the city locations and the calculated distances to disk using the pickle utility.
        """
        self.locations = []

        # open the whitespace-delimited TSPLIB file and read lines from it:
        with open(os.path.join(self.data_path, f"{self.name}.tsp")) as f:
            # skip lines until one of these lines is found:
            for line in f:
                if line.strip() in ("DISPLAY_DATA_SECTION", "NODE_COORD_SECTION"):
                    break

            # read data lines until 'EOF' (or end of file, e.g. pr1002) is found;
            # split() ignores leading/trailing/repeated spaces:
            for line in f:
                parts = line.split()
                if not parts or parts[0] == "EOF":
                    break

                # drop the index at the beginning of the line, keep x,y:
                self.locations.append(np.asarray(parts[1:3], dtype=np.float64))

            # set the problem 'size':
            self.tspSize = len(self.locations)

            # initialize distance matrix by filling it with 0's:
            self.distances = [[0] * self.tspSize for _ in range(self.tspSize)]

            # populate the distance matrix with calculated distances:
            for i in range(self.tspSize):
                for j in range(i + 1, self.tspSize):
                    # calculate euclidean distance between two ndarrays:
                    distance = np.linalg.norm(self.locations[j] - self.locations[i])
                    self.distances[i][j] = distance
                    self.distances[j][i] = distance

            # serialize locations and distances:
            with open(
                os.path.join(self.data_path, f"{self.name}-loc.pickle"), "wb"
            ) as f:
                pickle.dump(self.locations, f)
            with open(
                os.path.join(self.data_path, f"{self.name}-dist.pickle"), "wb"
            ) as f:
                pickle.dump(self.distances, f)

    def getTotalDistance(self, indices):
        """Calculates the total distance of the path described by the given indices of the cities

        :param indices: A list of ordered city indices describing the given path.
        :return: total distance of the path described by the given indices
        """
        # distance between th elast and first city:
        distance = self.distances[indices[-1]][indices[0]]

        # add the distance between each pair of consequtive cities:
        for i in range(len(indices) - 1):
            distance += self.distances[indices[i]][indices[i + 1]]

        return distance

    def getTotalDistanceRounded(self, indices):
        """TSPLIB EUC_2D length: round each edge to the nearest integer, then sum.

        Used as the EA fitness, so the EA optimises the same quantity as the known optima.
        """
        tour = np.asarray(indices)
        return int(self.distancesRounded[tour, np.roll(tour, -1)].sum())

    def getTotalDistancePythonRounded(self, indices):
        """Sum tour edges after applying Python's round() to each edge."""
        total = 0
        for i, city in enumerate(indices):
            next_city = indices[(i + 1) % len(indices)]
            total += round(self.distances[city][next_city])
        return total

    def plotData(self, indices):
        """plots the path described by the given indices of the cities

        :param indices: A list of ordered city indices describing the given path.
        :return: the resulting plot
        """

        # plot the dots representing the cities:
        plt.scatter(*zip(*self.locations), marker=".", color="red")

        # create a list of the corresponding city locations:
        locs = [self.locations[i] for i in indices]
        locs.append(locs[0])

        # plot a line between each pair of consequtive cities:
        plt.plot(*zip(*locs), linestyle="-", color="blue")

        return plt

    def loadTour(self, tour_file=None):
        path = tour_file or os.path.join(self.data_path, f"{self.name}.opt.tour")
        tour, started = [], False
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line == "TOUR_SECTION":
                    started = True
                elif started:
                    for token in line.split():
                        if token in ("-1", "EOF"):
                            return tour
                        tour.append(int(token) - 1)  # TSPLIB IDs are 1-based
        return tour


# testing the class:
def main():
    config.PLOTS_DIR.mkdir(exist_ok=True)

    # optimal tour length with the TSPLIB rounding, compared with two common mistakes
    header = (
        f"{'TSP':<10}{'cities':>7}{'TSP-BEST':>10}{'TSPLIB nint':>13}{'round()/edge':>14}"
        f"{'round(total)':>14}{'unrounded':>12}{'x.5 edges':>11}"
    )
    table_lines = [header]
    print(header)
    for name in config.parse_tsp_names("Plot optimal tours"):
        tsp = TravelingSalesmanProblem(name)
        tour = tsp.loadTour()
        assert sorted(tour) == list(range(len(tsp))), "tour must be a valid permutation"

        n = len(tour)
        edges = [tsp.distances[tour[i]][tour[(i + 1) % n]] for i in range(n)]
        length = tsp.getTotalDistanceRounded(
            tour
        )  # (int)(d + 0.5) per edge, as in TSPLIB
        assert length == config.KNOWN_OPTIMA[name], "must equal the known optimum"
        row = (
            f"{name:<10}{n:>7}{config.KNOWN_OPTIMA[name]:>10}{length:>13}"
            f"{tsp.getTotalDistancePythonRounded(tour):>14}"  # Python round(): halves to even
            f"{round(sum(edges)):>14}"  # rounding the total instead of each edge
            f"{sum(edges):>12.2f}"
            f"{sum(1 for e in edges if abs(e % 1 - 0.5) < 1e-9):>11}"
        )
        print(row)
        table_lines.append(row)

        plot = tsp.plotData(tour)
        plot.title(f"{name} optimal tour (length {length})")
        plot.savefig(config.PLOTS_DIR / f"{name}_opt.png", dpi=150)
        plot.close()

    output_path = config.RESULTS_DIR / "optimal_length.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(f"\nSaved table to {output_path.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
