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
        """Reads the serialized data, and if not available - calls __create_data() to prepare it
        """

        # attempt to read serialized data:
        try:
            self.locations = pickle.load(open(os.path.join(self.data_path, f"{self.name}-loc.pickle"), "rb"))
            self.distances = pickle.load(open(os.path.join(self.data_path, f"{self.name}-dist.pickle"), "rb"))
        except (OSError, IOError):
            pass

        # serailized data not found - create the data from scratch:
        if not self.locations or not self.distances:
            self.__createData()

        # set the problem 'size':
        self.tspSize = len(self.locations)

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
                if line.strip() in ('DISPLAY_DATA_SECTION', 'NODE_COORD_SECTION'):
                    break

            # read data lines until 'EOF' (or end of file, e.g. pr1002) is found;
            # split() ignores leading/trailing/repeated spaces:
            for line in f:
                parts = line.split()
                if not parts or parts[0] == 'EOF':
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
            pickle.dump(self.locations, open(os.path.join(self.data_path, f"{self.name}-loc.pickle"), "wb"))
            pickle.dump(self.distances, open(os.path.join(self.data_path, f"{self.name}-dist.pickle"), "wb"))

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
        """TSPLIB EUC_2D length: round each edge to the nearest integer, then sum."""
        total = 0
        for i in range(len(indices)):
            a, b = indices[i], indices[(i+1) % len(indices)]
            total += int(self.distances[a][b] + 0.5)
        return total

    def plotData(self, indices):
        """plots the path described by the given indices of the cities

        :param indices: A list of ordered city indices describing the given path.
        :return: the resulting plot
        """

        # plot the dots representing the cities:
        plt.scatter(*zip(*self.locations), marker='.', color='red')

        # create a list of the corresponding city locations:
        locs = [self.locations[i] for i in indices]
        locs.append(locs[0])

        # plot a line between each pair of consequtive cities:
        plt.plot(*zip(*locs), linestyle='-', color='blue')

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
    for name in config.parse_tsp_names("Plot optimal tours"):
        tsp = TravelingSalesmanProblem(name)
        tour = tsp.loadTour()
        assert sorted(tour) == list(range(len(tsp))), "tour must be a valid permutation"
        print(
            f"{name}: {len(tsp)} cities, "
            f"length = {tsp.getTotalDistance(tour):.2f}, "
            f"optimal (TSPLIB, rounded) = {tsp.getTotalDistanceRounded(tour)}"
        )

        plot = tsp.plotData(tour)
        plot.title(f"{name} optimal tour")
        plot.savefig(config.PLOTS_DIR / f"{name}_opt.png", dpi=150)
        plot.close()


if __name__ == "__main__":
    main()
