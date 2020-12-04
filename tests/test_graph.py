import itertools
import math
import hypothesis as hyp
import hypothesis.strategies as strats
from utils.graph import *


def _construct_graph_dict(paths, cycles):
    graph = {}
    for edges, val in itertools.chain(paths, cycles):
        for arc in edges:
            if arc in graph:
                graph[arc] += val
            else:
                graph[arc] = val
    return graph


@strats.composite
def graph_strat(draw,
                nsources=strats.integers(min_value=1, max_value=5),
                nsinks=strats.integers(min_value=1, max_value=5),
                nodes=strats.integers(min_value=0, max_value=200),
                edge_weights=strats.integers(min_value=1, max_value=20),
                npaths=strats.integers(min_value=0, max_value=10),
                ncycles=strats.integers(min_value=0, max_value=10),
                ):
    sources = draw(nsources.map(lambda n: [f'O{i}' for i in range(n)]))
    sinks = draw(nsinks.map(lambda n: [f'D{i}' for i in range(n)]))
    npaths = draw(npaths)
    ncycles = draw(ncycles)

    path_strat = strats.lists(nodes, unique=True) \
        .map(lambda p: (draw(strats.sampled_from(sources)), *p, draw(strats.sampled_from(sinks)))) \
        .map(lambda p: (tuple(zip(p, p[1:])), draw(edge_weights)))

    paths = draw(strats.lists(path_strat, unique_by=lambda kv: kv[0], min_size=npaths, max_size=npaths))

    def shifted_cycle(c):
        kmin, _ = min(enumerate(c), key=lambda x: x[::-1])
        return c[kmin:] + c[:kmin]

    cycle_strat = strats.lists(nodes, unique=True, min_size=2) \
        .map(shifted_cycle) \
        .map(lambda c: c + [c[0]]) \
        .map(lambda c: (tuple(zip(c, c[1:])), draw(edge_weights)))

    cycles = draw(strats.lists(cycle_strat, unique_by=lambda kv: kv[0], min_size=ncycles, max_size=ncycles))

    return _construct_graph_dict(paths, cycles)


@hyp.given(graph_strat())
def test(graph):
    sources = {n for n, _ in graph if isinstance(n, str) and n[0] == 'O'}
    sinks = {n for _, n in graph if isinstance(n, str) and n[0] == 'D'}
    original_graph = graph.copy()

    def outgoing_arc(g, n):
        for arc, val in g.items():
            if arc[0] == n:
                return arc, val, arc[1]
        raise Exception

    def subtract_arc(g, arc, val):
        new_val = g[arc] - val
        if math.isclose(new_val, 0, abs_tol=1e-12):
            del g[arc]
        else:
            assert new_val > 0
            g[arc] = new_val

    def find_start(g):
        if len(g) == 0:
            return None
        for start, end in graph:
            if start in sources:
                break
        return start

    paths, cycles = decompose_paths_and_cycles(graph, set(sinks), find_start, outgoing_arc, subtract_arc)
    reconstructed = _construct_graph_dict(paths, cycles)
    assert reconstructed == original_graph


if __name__ == '__main__':
    test()
