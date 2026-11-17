"""Permutation operators that DEAP does not provide, written in the DEAP operator style
(modify the individuals in place and return them), so they can be registered in a toolbox:

    toolbox.register("mate", operators.cxEdgeRecombination)
"""

import random

def _edgeMap(ind1, ind2):
    """Adjacency list of every city: its neighbours in either parent (a tour is a cycle)."""
    size = len(ind1)
    edges = [set() for _ in range(size)]
    for tour in (ind1, ind2):
        for i in range(size):
            city = tour[i]
            edges[city].add(tour[i - 1])
            edges[city].add(tour[(i + 1) % size])
    return edges

def _edgeRecombinationChild(edges, start):
    """Build one child from the edge map (Whitley et al., 1989).

    1. Start at `start`.
    2. Remove the current city from all adjacency lists.
    3. Go to the neighbour of the current city that has the fewest remaining neighbours of its
       own (ties broken at random), so cities that are about to run out of edges are visited first.
    4. If the current city has no neighbours left, go to a random unvisited city (the only way a
       child gets an edge that is in neither parent).
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
        if candidates:
            fewest = min(len(edges[c]) for c in candidates)
            nxt = random.choice(sorted(c for c in candidates if len(edges[c]) == fewest))
        else:
            nxt = random.choice(unvisited)
        child.append(nxt)
        visit(nxt)
    return child

def cxEdgeRecombination(ind1, ind2):
    """Edge recombination crossover (ERX) for permutations (Whitley, Starkweather & Fuquay, 1989).

    TSP fitness depends on which cities are adjacent (the edges), not on their positions.
    ERX builds a child almost entirely from edges present in at least one parent. The original
    operator makes one child per pair; to fit DEAP's mate interface (two parents in, two children
    out, modified in place) the first child starts from ind1's first city and the second from ind2's.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    """
    edges = _edgeMap(ind1, ind2)
    child1 = _edgeRecombinationChild(edges, ind1[0])
    child2 = _edgeRecombinationChild(edges, ind2[0])
    for i in range(len(ind1)):
        ind1[i], ind2[i] = child1[i], child2[i]
    return ind1, ind2


def nearestNeighbourTour(distances, start):
    """Greedy nearest-neighbour tour (heuristic initialization, e.g. Grefenstette, 1987): from
    `start`, always move to the closest unvisited city (ties: lowest index).

    `distances` is the (integer) distance matrix as a NumPy array. Building the tour reads the
    distance matrix but evaluates no tour, so it costs no fitness evaluations.
    """
    import numpy as np
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
