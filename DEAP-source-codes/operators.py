"""Permutation operators that DEAP does not provide, written in the DEAP operator style
(modify the individuals in place and return them), so they can be registered in a toolbox:

    toolbox.register("mate", operators.cxEdgeRecombination)
"""

import random
import numpy as np

def _edgeMap(ind1, ind2):
    """Adjacency list of every city: its neighbours in either parent (a tour is a cycle), and
    the subset of those neighbours that are adjacent to it in both parents (common edges)."""
    size = len(ind1)
    edges = [set() for _ in range(size)]
    common = [set() for _ in range(size)]
    for tour in (ind1, ind2):
        for i in range(size):
            city = tour[i]
            for neighbour in (tour[i - 1], tour[(i + 1) % size]):
                if neighbour in edges[city]:  # already seen in the first parent
                    common[city].add(neighbour)
                edges[city].add(neighbour)
    return edges, common

def _edgeRecombinationChild(edges, common, start):
    """Build one child from the edge map (enhanced ERX, Starkweather et al., 1991).

    1. Start at `start`.
    2. Remove the current city from all adjacency lists.
    3. If the current city still has a common edge (one in both parents), follow it (ties at random).
    4. Otherwise go to the neighbour of the current city that has the fewest remaining neighbours
       of its own (ties at random), so cities that are about to run out of edges are visited first.
    5. If the current city has no neighbours left, go to a random unvisited city. This and the
       edge closing the tour are the only ways a child gets an edge that is in neither parent.
    """
    size = len(edges)
    edges = [set(e) for e in edges]  # copy: the map is consumed while building the child
    unvisited = list(range(size))
    position = list(range(size))     # position[city] = index of city in `unvisited` (O(1) removal)

    def visit(city):
        i, last = position[city], unvisited[-1]
        unvisited[i], position[last] = last, i
        unvisited.pop()
        for neighbour in edges[city]:
            edges[neighbour].discard(city)

    child = [start]
    visit(start)
    while unvisited:
        candidates = edges[child[-1]]
        shared = candidates & common[child[-1]]  # `common` is never consumed: intersect with what is left
        if shared:
            nxt = random.choice(sorted(shared))
        elif candidates:
            fewest = min(len(edges[c]) for c in candidates)
            nxt = random.choice(sorted(c for c in candidates if len(edges[c]) == fewest))
        else:
            nxt = random.choice(unvisited)
        child.append(nxt)
        visit(nxt)
    return child

def cxEdgeRecombination(ind1, ind2):
    """Enhanced edge recombination crossover (ERX) for permutations (Whitley, Starkweather &
    Fuquay, 1989; enhanced by Starkweather et al., 1991).

    TSP fitness depends on which cities are adjacent (the edges), not on their positions.
    ERX builds a child almost entirely from edges present in at least one parent. The enhanced
    version always follows an edge that both parents share when one is available, so common
    edges are passed on. The operator makes one child per pair; to fit DEAP's mate interface
    (two parents in, two children out, modified in place) the first child starts from ind1's
    first city and the second from ind2's.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    """
    edges, common = _edgeMap(ind1, ind2)
    child1 = _edgeRecombinationChild(edges, common, ind1[0])
    child2 = _edgeRecombinationChild(edges, common, ind2[0])
    for i in range(len(ind1)):
        ind1[i], ind2[i] = child1[i], child2[i]
    return ind1, ind2


def nearestNeighbourTour(distances, start):
    """Greedy nearest-neighbour tour (heuristic initialization, e.g. Grefenstette, 1987): from
    `start`, always move to the closest unvisited city (ties: lowest index).

    `distances` is the (integer) distance matrix as a NumPy array. Building the tour reads the
    distance matrix but evaluates no tour, so it costs no fitness evaluations.
    """
    n = len(distances)
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        row = np.where(visited, np.iinfo(np.int64).max, distances[tour[-1]])
        nxt = int(np.argmin(row))
        tour.append(nxt)
        visited[nxt] = True
    return tour
