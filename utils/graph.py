from typing import TypeVar, Iterable, Callable, Set, Union, Tuple


N = TypeVar('N')
A = TypeVar('A')
G = TypeVar('G')
def decompose_paths_and_cycles(graph: G,
                               sinks: Set[N],
                               find_start: Callable[[G], Union[N, None]],
                               outgoing_arc: Callable[[G, N], Iterable[Tuple[A, float, N]]],
                               subtract_arc: Callable[[G, A, float], None]):
    paths = []
    cycles = []
    print('noodles')
    while True:
        node = find_start(graph)
        if node is None:
            break

        visited_nodes = {node: 0}
        node_order = [node]
        edges = []
        while True:
            arc, val, next_node = outgoing_arc(graph, node)
            edges.append((arc, val))

            if next_node in visited_nodes:
                # cycle
                k = visited_nodes[next_node]
                cycle = edges[k:]
                cycle_val = min(v for _, v in cycle)
                cycle = tuple(e for e, _ in cycle)
                cycles.append((cycle, cycle_val))
                for arc in cycle:
                    subtract_arc(graph, arc, cycle_val)

                if k > 0:
                    # backtrack
                    edges = edges[:k]
                    for n in node_order[k + 1:]:
                        del visited_nodes[n]
                    node_order = node_order[:k + 1]
                    node = node_order[-1]
                    continue  # backtrack
                else:
                    break  # Start from scratch
            elif next_node in sinks:
                # path
                path = tuple(e for e, _ in edges)
                path_val = min(v for _, v in edges)
                paths.append((path, path_val,))
                for arc in path:
                    subtract_arc(graph, arc, path_val)
                break

            visited_nodes[next_node] = len(node_order)
            node_order.append(next_node)
            node = next_node

    return paths, cycles
