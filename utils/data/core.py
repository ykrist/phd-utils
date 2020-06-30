import glob
import tarfile
import itertools
from pathlib import Path
from typing import Tuple
from oru import *
from .parse import *
from ..constants import DATA_DIR, ITSRSP_TIME_INFINITY
from .types import *
from . import modify
from . import utils as _u

def data_directory(subdir) -> Path:
    path = DATA_DIR/subdir
    if not path.exists():
        archive = DATA_DIR/f"compressed/{subdir}.tar.xz"
        tar = tarfile.open(archive, 'r:xz')
        tar.extractall(DATA_DIR)
        tar.close()
        assert path.exists()
    return path

# name resolvers should be by dataset not by datatype
_schrotenboer_filepattern_index = None
def resolve_name_schrotenboer(name : str) -> Tuple[Path]:
    global _schrotenboer_filepattern_index
    if _schrotenboer_filepattern_index is None:
        root_dir = data_directory("MSPRP_schrotenboer")
        index_file = root_dir/'INDEX.txt'
        _schrotenboer_filepattern_index = {}
        with open(index_file) as f:
            for line in f:
                k, v = line.strip().split()
                _schrotenboer_filepattern_index[k] = tuple(sorted(root_dir.glob(v)))
    if name not in _schrotenboer_filepattern_index:
        raise ValueError(f"{name} is not a valid Schrotenboer-instance name")

    return _schrotenboer_filepattern_index[name]

def resolve_name_cordeau_PDPTW(name : str) -> Path:
    datasubdir = data_directory("PDPTW_cordeau")
    path = datasubdir/name
    if not os.path.exists(path):
        raise ValueError(f"{name} is not a valid Cordeau-instance name")
    return path


def resolve_name_cordeau_DARP(name : str) -> Path:
    datasubdir = data_directory("DARP_cordeau")
    path = datasubdir/f"{name}.txt"
    if not os.path.exists(path):
        raise ValueError(f"{name} is not a valid Cordeau-instance name")
    return path

def resolve_name_hemmati(name : str) -> Path:
    datasubdir = data_directory('ITSRSP_hemmati')
    path = datasubdir/f"{name}.txt"
    if path.exists():
       return path
    raise ValueError(f"{name} is not a recognised ITSRSP Hemmati name")

def resolve_name_homsi(name : str) -> Path:
    datasubdir = data_directory('ITSRSP_homsi')
    path = datasubdir/f"{name}.txt"
    if path.exists():
        return path
    raise ValueError(f"{name} is not a recognised ITSRSP Homsi name")

def resolve_name_hemmati_hdf5(name : str) -> Path:
    datasubdir = data_directory('ITSRSP_hdf5_ti')
    path = datasubdir/f'{name}.hdf5'
    if path.exists():
        return path
    raise ValueError(f"{name} is not a recognised ITSRSP HDF5 name")


# def load_cordeau_instance(path, rehandling_cost=None) -> PDPTW_Data:
#     with open(path, 'r') as raw:
#         header = tuple(filter(lambda x: len(x) > 0, raw.readline().strip().split()))
#         num_request = int(header[1])
#         num_nodes = 2 * num_request + 1
#         vehicle_cap = int(header[3])
#
#         travel_times = np.full((num_nodes, num_nodes), np.nan)
#         travel_costs = np.full((num_nodes, num_nodes), np.nan)
#         locations = np.full((num_nodes, 2), np.nan)
#         service_time = np.full(num_nodes, np.nan)
#         earliest_times = np.full(num_nodes, np.nan)
#         latest_times = np.full(num_nodes, np.nan)
#         demand = np.full(num_nodes, np.nan)
#
#         for l in raw.readlines():
#             node_id, x, y, st, d, tw_start, tw_end = tuple(filter(lambda x: len(x) > 0, l.strip().split()))
#             node_id = int(node_id)
#
#             if node_id == num_nodes:
#                 assert locations[0, 0] == float(x) and locations[0, 1] == float(y)
#                 continue
#
#             locations[node_id, 0] = float(x)
#             locations[node_id, 1] = float(y)
#             service_time[node_id] = float(st)
#             demand[node_id] = float(d)
#             earliest_times[node_id] = float(tw_start)
#             latest_times[node_id] = float(tw_end)
#
#         for i in range(num_nodes):
#             for j in range(i, num_nodes):
#                 distance = np.sqrt(np.sum((locations[i] - locations[j]) ** 2))
#                 travel_costs[i, j] = distance
#                 travel_costs[j, i] = distance
#                 travel_times[i, j] = distance + service_time[i]
#                 travel_times[j, i] = distance + service_time[j]
#
#         # Check data
#         for d in [locations, service_time, demand, earliest_times, latest_times, travel_times, travel_costs]:
#             assert not np.any(np.isnan(d))
#
#         # Modify the data to match Ali's paper
#         name = os.path.basename(path)
#         problem_group = name[0]
#
#
#         T = dict()
#         C = dict()
#         E = dict(enumerate(earliest_times))
#         L = dict(enumerate(latest_times))
#         D = dict(enumerate(demand))
#         del E[0]
#         del L[0]
#         del D[0]
#
#         for i in range(num_nodes):
#             for j in range(num_nodes):
#                 T[i, j] = _travel_time_round(travel_times[i, j])
#                 C[i, j] = _travel_time_round(travel_costs[i, j])
#
#         if rehandling_cost is not None:
#             return PDPTWLH_Data(
#                 T=frozendict(T),
#                 C=frozendict(C),
#                 D=frozendict(D),
#                 Q=vehicle_cap,
#                 loc=frozendict(enumerate(map(tuple, locations[:]))),
#                 E=frozendict(E),
#                 L=frozendict(L),
#                 n=num_request,
#                 delivery_loc=frozenset(range(num_request + 1, 2 * num_request + 1)),
#                 pickup_loc=frozenset(range(1, num_request + 1)),
#                 vehicle_cost=10000,
#                 rehandle_cost=rehandling_cost,
#                 id=name,
#             )
#         else:
#             return PDPTW_Data(
#                 T=frozendict(T),
#                 C=frozendict(C),
#                 D=frozendict(D),
#                 Q=vehicle_cap,
#                 loc=frozendict(enumerate(map(tuple, locations[:]))),
#                 E=frozendict(E),
#                 L=frozendict(L),
#                 n=num_request,
#                 pickup_loc=frozenset(range(1, num_request + 1)),
#                 delivery_loc=frozenset(range(num_request + 1, 2 * num_request + 1)),
#                 vehicle_cost=10000,
#                 id=name,
#             )
#
# def restrict_requests(data: PDPTW_Data, num_requests: int):
#     node_map = {i: i for i in range(1, num_requests + 1)}
#     node_map.update({i + data.n: i + num_requests for i in node_map})
#     node_map[0] = 0
#
#     arc_dicts_to_modify = ('C', 'T')
#     node_dicts_to_modify = ('E', 'L', 'D', 'loc')
#     changes = dict()
#     for prop in arc_dicts_to_modify:
#         d = getattr(data, prop)
#         changes[prop] = frozendict({(node_map[i], node_map[j]): v for (i, j), v in d.items()
#                                     if i in node_map and j in node_map})
#     for prop in node_dicts_to_modify:
#         d = getattr(data, prop)
#         changes[prop] = frozendict({node_map[i]: v for i, v in d.items() if i in node_map})
#
#     changes['n'] = num_requests
#     changes['pickup_loc'] = frozenset(range(1, num_requests + 1))
#     changes['delivery_loc'] = frozenset(range(num_requests + 1, 2 * num_requests + 1))
#     new_data = dataclasses.replace(data, **changes)
#     return new_data
#
#
#
#
# def random_veenstra_instance(n, p_class, rehandle_cost, seed=None):
#     if seed is None:
#         seed = np.random.randint(10000, 99999)
#
#     if p_class in _PROBLEM_CLASSES:
#         max_demand, vehicle_cap, tw_width, delivery_delay = _PROBLEM_CLASSES[p_class]
#     else:
#         raise ValueError('p_class must be one of the following: ' + ','.join(_PROBLEM_CLASSES.keys()))
#
#     return _random_instance(
#         num_loc=n,
#         max_demand=max_demand,
#         vehicle_cap=vehicle_cap,
#         tw_width=tw_width,
#         delivery_delay=delivery_delay,
#         rehandling_cost=rehandle_cost,
#         id_str=f'veenstra_{p_class.replace("*", "s"):s}{n:d}_{seed:d}',
#         seed=seed
#     )
#
#
# def random_cherkesly_instance(n, p_class, seed=None):
#     if seed is None:
#         seed = np.random.randint(10000, 99999)
#
#     if p_class in _PROBLEM_CLASSES:
#         max_demand, vehicle_cap, tw_width, delivery_delay = _PROBLEM_CLASSES[p_class]
#     else:
#         raise ValueError('p_class must be one of the following: ' + ','.join(_PROBLEM_CLASSES.keys()))
#
#     return _random_instance(
#         num_loc=n,
#         max_demand=max_demand,
#         vehicle_cap=vehicle_cap,
#         tw_width=tw_width,
#         delivery_delay=delivery_delay,
#         rehandling_cost=None,
#         id_str=f'cherkesly_{p_class.replace("*", "s"):s}{n:d}_{seed:d}',
#         seed=seed
#     )
#
# def random_ropke_instance(n, p_class, seed=None):
#     if seed is None:
#         seed = np.random.randint(10000, 99999)
#
#     if p_class in _PROBLEM_CLASSES and p_class[-1] != '*':
#         max_demand, _, tw_width, _ = _PROBLEM_CLASSES[p_class]
#     else:
#         raise ValueError('p_class must be one of the following: '
#                          + ','.join(filter(lambda x: '*' not in x, _PROBLEM_CLASSES.keys())))
#
#     return _random_instance(
#         num_loc=n,
#         max_demand=max_demand,
#         vehicle_cap=max_demand,
#         tw_width=tw_width,
#         delivery_delay=0,
#         rehandling_cost=None,
#         id_str=f'ropke_{p_class:s}{n:d}_{seed:d}',
#         seed=seed
#     )
#
#
# _PROBLEM_CLASSES = {
#     # p_class : (max_demand, vehicle_cap, tw_width, delivery_delay)
#     'AA': (15, 22, 60, 45),
#     'BB': (20, 30, 60, 45),
#     'CC': (15, 18, 120, 15),
#     'DD': (20, 25, 120, 15),
#     'AA*': (15, 26, 60, 60),
#     'BB*': (20, 35, 60, 60),
# }


#
# def load_schrotenboer_instance(instance_name, service_time_multiplier=1) -> MSPRP_Data:
#     root_dir = DATA_DIR + '/MSPRP_schrotenboer/'
#     index_file = root_dir + 'INDEX.txt'
#     with open(index_file) as f:
#         for line in f:
#             k, v = line.strip().split()
#             if k == instance_name:
#                 depot_file, general_file, jobs_file, max_travel_file = sorted(glob.glob(root_dir + v))
#                 break
#         else:
#             raise ValueError(f'`{instance_name}` is not in `{index_file}`')
#
#     with open(general_file, 'r') as f:
#         first_line, second_line = f.readline().strip().split(), f.readline().strip().split()
#         problem_size_info = dict(zip(first_line, map(int, second_line)))
#         f.readline()  # next line is a comment
#         vehicle_info = dict()
#         for _ in range(problem_size_info['nr_vehicles']):
#             idx, servicemen_max, parts_cap, cost_parameter, travel_parameter = f.readline().strip().split()
#             vehicle_info[int(idx)] = {
#                 'servicemen_capacity': int(servicemen_max),
#                 'parts_capacity': int(parts_cap),
#                 'cost_factor': float(cost_parameter),
#                 'speed': float(travel_parameter),
#             }
#
#     n = problem_size_info['nr_jobs']
#     depot_origin_idx = 0
#     depot_dest_idx = 2*n+1
#     L = range(problem_size_info['nr_reusables'])
#     T = range(problem_size_info['nr_periods'])
#     Nd = range(1,n+1)
#     Np = range(n+1, 2*n+1)
#     K = range(problem_size_info['nr_vehicles'])
#
#     with open(depot_file, 'r') as f:
#         # header is useless, format seems to be Cap(1), ..., Cap(L), Cost(1), ..., Cost(L) where Cap( ) is the maximum
#         # number of servicement of a type per time period and Cost( ) is the unit cost of a type of serviceman.
#         # There is a line for each time period
#         contents = list(map(lambda x: x.strip().split(), f.readlines()[1:]))
#
#         servicemen_max = map(lambda line: tuple(int(x) for x in line[:problem_size_info['nr_reusables']]), contents)
#         servicemen_cost = map(lambda line: tuple(float(x) for x in line[problem_size_info['nr_reusables']:]), contents)
#
#         servicemen_max = {(l, t): v for t, vals in zip(T, servicemen_max) for l, v in zip(L, vals)}
#         servicemen_cost = {(l, t): v for t, vals in zip(T, servicemen_cost) for l, v in zip(L, vals)}
#
#         assert (len(servicemen_max) == problem_size_info['nr_periods'] * problem_size_info['nr_reusables'])
#         assert (len(servicemen_cost) == problem_size_info['nr_periods'] * problem_size_info['nr_reusables'])
#
#     servicemen_cost = {l:servicemen_cost[l,0] for l in L}
#     servicemen_max = {l:servicemen_max[l,0] for l in L}
#
#     with open(max_travel_file, 'r') as f:
#         contents = map(lambda line : line.strip().split(), f.readlines())
#         travel_time_max = { (k, t) : int(v) for t, row in zip(T, contents) for k,v in zip(K, row)}
#
#     with open(jobs_file, 'r') as f:
#         contents = list(map(lambda line : line.strip().split(), f.readlines()[1:]))
#         num_cols = len(contents[0])
#         assert all(map(lambda l : len(l) == num_cols, contents[1:])), 'number of columns is not consistent'
#         columns = [[contents[r][c] for r in range(len(contents))] for c in range(num_cols)]
#         index = list(map(int,columns[0]))
#         loc = {int(idx) + 1 : (float(x),float(y)) for idx, x, y in zip(index, columns[1], columns[2])}
#         servicemen_demand = {(i, l) : int(val) for l in L for i,val in zip(Nd, columns[3+l]) }
#         servicemen_demand.update({(i+len(Nd), l) : -val for (i,l),val in servicemen_demand.items()})
#         col_offset = 3 + len(L)
#         parts_demand = dict(zip(Nd, map(int, columns[col_offset])))
#         col_offset += 1
#         service_time = dict(zip(Nd, map(lambda x : float(x)*service_time_multiplier, columns[col_offset])))
#         col_offset += 1
#         periodic_costs = {(i,t) : float(val) for t in T for i,val in zip(Nd, columns[col_offset+t])}
#
#     periodic_costs.update({(i, t) : 0 for i in itertools.chain(Np,[depot_origin_idx,depot_dest_idx]) for t in T})
#
#     loc.update({i+len(Nd) : v for i,v in loc.items()})
#     loc[depot_origin_idx] = loc[depot_dest_idx] = (0,30)
#
#     distance_matrix = {
#         (i, j) : ((loc[i][0]-loc[j][0])**2 + (loc[i][1]-loc[j][1])**2)**0.5
#         for i in loc for j in loc
#     }
#
#     const_loading_time = 0.25
#
#     travel_times = {
#         (i,j,k) : distance_matrix[i,j]/vehicle_info[k]['speed'] + const_loading_time
#         for i in loc for j in loc for k in K
#     }
#
#     travel_costs = {
#         (i,j,k) : distance_matrix[i,j]*vehicle_info[k]['cost_factor']
#         for i in loc for j in loc for k in K
#     }
#
#     instance_name = schrotenboer_instance_id(instance_name, service_time_multiplier)
#
#     return MSPRP_Data(
#         servicemen_demand = frozendict(servicemen_demand),
#         servicemen_max=frozendict(servicemen_max),
#         servicemen_cost = frozendict(servicemen_cost),
#         servicemen_capacity= frozendict({k : vehicle_info[k]['servicemen_capacity'] for k in K}),
#         service_time = frozendict(_round_dict_values(service_time, 3)),
#         travel_time = frozendict(_round_dict_values(travel_times, 3)),
#         travel_time_max=frozendict(_round_dict_values(travel_time_max, 3)),
#         travel_cost = frozendict(_round_dict_values(travel_costs,3)),
#         parts_demand = frozendict(parts_demand),
#         parts_capacity= frozendict({k : vehicle_info[k]['parts_capacity'] for k in K}),
#         late_penalty=frozendict(_round_dict_values(periodic_costs,3)),
#         nr=n,
#         Nd=Nd,
#         Np=Np,
#         T = T,
#         K = K,
#         L = L,
#         loc = frozendict(loc),
#         id=instance_name
#     )

def build_PDPTWLH_from_cordeau(raw : RawDataCordeau, vehicle_cost : float, rehandling_cost : float, id_str : str) -> PDPTWLH_Data:
    n = raw.num_requests
    loc = frozendict((i,(raw.pos_x, raw.pos_y)) for i in range(2*n+1))
    pickup_loc = range(1,n+1)
    delivery_loc = range(n+1,2*n+1)

    T = {(i,j) : ((raw.pos_y[i]-raw.pos_y[j])**2 + (raw.pos_x[i]-raw.pos_x[j])**2)**0.5 for i in loc for j in range(i+1)}
    T.update({(j,i) : v for (i,j), v in T.items()})
    T = frozendict(T)
    C = T.copy()

    return PDPTWLH_Data(
        T = T,
        C = C,
        n= n,
        E = raw.tw_start,
        L = raw.tw_end,
        D = raw.demand,
        Q = raw.vehicle_cap,
        loc = loc,
        pickup_loc = pickup_loc,
        delivery_loc = delivery_loc,
        vehicle_cost = vehicle_cost,
        rehandle_cost = rehandling_cost,
        id=id_str
    )

def build_MSPRP_from_schrotenboer(raw : RawDataSchrotenboer, id_str:str) -> MSPRP_Data:
    K = range(raw.num_vehicles)
    distance_matrix = _u.euclidean_distance_matrix(raw.pos_x, raw.pos_y)

    travel_times = frozendict(
        ((i,j,k),(distance_matrix[i,j]/raw.vehicle_speed[k] + raw.loading_time))
        for i,j in distance_matrix for k in K
    )

    travel_costs = frozendict(
        ((i,j,k),distance_matrix[i,j]*raw.vehicle_travel_cost[k])
        for i,j in distance_matrix for k in K
    )

    return MSPRP_Data(
        servicemen_demand = raw.servicemen_demand,
        servicemen_max = raw.servicemen_avail,
        servicemen_capacity = raw.vehicle_servicemen_cap,
        servicemen_cost = raw.servicemen_cost,
        service_time = raw.service_time,
        travel_time = travel_times,
        travel_cost = travel_costs,
        parts_demand = raw.parts_demand,
        parts_capacity = raw.vehicle_parts_cap,
        travel_time_max = raw.max_travel_time,
        nr = raw.num_requests,
        Nd = range(1,raw.num_requests+1),
        Np = range(raw.num_requests+1,2*raw.num_requests+1),
        T = range(raw.num_time_periods),
        K = K,
        L = range(raw.num_servicemen_types),
        loc = frozendict((i,(raw.pos_x[i],raw.pos_y[i])) for i in range(2*raw.num_requests+2)),
        late_penalty = raw.periodic_costs,
        id=id_str
    )

def build_DARP_from_cordeau(raw : RawDataCordeau, id_str : str) -> DARP_Data:
    # Note - service time must be stored separately, since a client's service time does not contribute to their ride
    # time.  I handle this by storing adding the service time onto the travel time, adding each client's service times
    # (both at pickup and delivery) to their max ride time.
    distance = _u.euclidean_distance_matrix(raw.pos_x, raw.pos_y)
    P = range(1, raw.num_requests+1)
    max_ride_times = { i : raw.max_ride_time + raw.service_time[i] for i in P }
    max_ride_times[0] = raw.max_route_duration
    return DARP_Data(
        travel_time = frozendict({(i,j) : d + raw.service_time[i] for (i,j), d in distance.items()}),
        service_time = raw.service_time,
        travel_cost = frozendict(distance),
        tw_start = raw.tw_start,
        tw_end = raw.tw_end,
        demand = raw.demand,
        loc = frozendict((i, (raw.pos_x[i], raw.pos_y[i])) for i in raw.pos_x),
        max_ride_time = frozendict(max_ride_times),
        capacity = raw.vehicle_cap,
        n = raw.num_requests,
        P = P,
        D = range(raw.num_requests+1, 2*raw.num_requests+1),
        K = range(raw.num_vehicles),
        N = range(0, 2*raw.num_requests+2),
        id=id_str
    )

def build_ITSRSP_from_hemmati(raw : RawDataHemmati, id_str : str) -> ITSRSP_Data:
    n = raw.num_cargos
    P = range(0, n)
    D = range(n, 2*n)
    V = range(raw.num_vessels)
    o_depots = range(2*n, 2*n+raw.num_vessels)
    d_depot = 2*n+raw.num_vessels

    tw_start = {}
    tw_end = {}

    for p in P:
        tw_start[p] = raw.cargo_origin_tw_start[p]
        tw_start[p+n] = raw.cargo_dest_tw_start[p]
        tw_end[p] = raw.cargo_origin_tw_end[p]
        tw_end[p+n] = raw.cargo_dest_tw_end[p]

    for v in V:
        tw_start[o_depots[v]] = raw.vessel_start_time[v]
        tw_end[o_depots[v]] = 2**32 - 1

    tw_start[d_depot] = 0
    tw_end[d_depot] = 2**32 - 1

    travel_time = {}
    travel_cost = {}
    port_group = defaultdict(list)

    for i in P:
        port_group[raw.cargo_origin[i]].append(i)
        port_group[raw.cargo_dest[i]].append(i + n)


    for v in V:
        travel_time[v] = {}
        travel_cost[v] = {}

        for i in P:
            if i not in raw.vessel_compatible[v]:
                continue
            o_port_i = raw.cargo_origin[i]
            d_port_i = raw.cargo_dest[i]

            for j in P:
                if j not in raw.vessel_compatible[v]:
                    continue

                o_port_j = raw.cargo_origin[j]
                d_port_j = raw.cargo_dest[j]

                if i != j:
                    travel_time[v][i,j] = raw.travel_time[v, o_port_i, o_port_j] + raw.cargo_origin_port_time[v, i]
                    travel_time[v][i+n, j] = raw.travel_time[v, d_port_i, o_port_j] + raw.cargo_dest_port_time[v, i]
                    travel_time[v][i+n, j+n] = raw.travel_time[v, d_port_i, d_port_j] + raw.cargo_dest_port_time[v,i]
                    travel_cost[v][i,j] = raw.travel_cost[v, o_port_i, o_port_j] + raw.cargo_origin_port_cost[v, i]
                    travel_cost[v][i+n, j] = raw.travel_cost[v, d_port_i, o_port_j] + raw.cargo_dest_port_cost[v, i]
                    travel_cost[v][i+n, j+n] = raw.travel_cost[v, d_port_i, d_port_j] + raw.cargo_dest_port_cost[v,i]

                travel_time[v][i,j+n] = raw.travel_time[v, o_port_i, d_port_j] + raw.cargo_origin_port_time[v, i]
                travel_cost[v][i,j+n] = raw.travel_cost[v, o_port_i, d_port_j] + raw.cargo_origin_port_cost[v, i]

            travel_time[v][i+n, d_depot] = raw.cargo_dest_port_time[v, i]
            travel_cost[v][i+n, d_depot] = raw.cargo_dest_port_cost[v, i]

    # These are allowed to be different for vehicles in the same group
    travel_time_vehicle_origin_depot = {v : {} for v in V}
    travel_cost_vehicle_origin_depot = {v: {} for v in V}


    for v in V:
        od_port = raw.vessel_start_port[v]
        for i in P:
            if i not in raw.vessel_compatible[v]:
                continue

            o_port_i = raw.cargo_origin[i]
            travel_time_vehicle_origin_depot[v][i] = raw.travel_time[v, od_port, o_port_i]
            travel_cost_vehicle_origin_depot[v][i] = raw.travel_cost[v, od_port, o_port_i]

    vehicle_groups = defaultdict(set)

    for v in V:
        key = (
            frozendict(travel_time[v]),
            frozendict(travel_cost[v]),
            raw.vessel_capacity[v],
            raw.vessel_compatible[v],
        )
        vehicle_groups[key].add(v)
    vehicle_groups = frozendict((g,frozenset(grp)) for g,grp in enumerate(vehicle_groups.values()))
    vehicle_groups_inv = frozendict((v,vg) for vg,Vg in vehicle_groups.items() for v in Vg)

    # Now, after grouping, we may merge the travel_times and travel_costs to/from depots
    travel_time_vg = {}
    travel_cost_vg = {}
    capacity_vg = {}
    P_compat_vg = {}
    char_v = {}
    for vg, Vg in vehicle_groups.items():
        v = min(Vg)
        travel_time_vg[vg] = travel_time[v]
        travel_cost_vg[vg] = travel_cost[v]
        capacity_vg[vg] = raw.vessel_capacity[v]
        P_compat_vg[vg] = raw.vessel_compatible[v]

        for v in Vg:
            del travel_time[v]
            del travel_cost[v]
            o = o_depots[v]
            travel_time_vg[vg].update({(o, i) : t for i,t in travel_time_vehicle_origin_depot[v].items()})
            travel_cost_vg[vg].update({(o, i) : t for i,t in travel_cost_vehicle_origin_depot[v].items()})

        for p in P_compat_vg[vg]:
            _, v_char = min(
                (max(tw_start[o_depots[v]] + travel_time_vehicle_origin_depot[v][p], tw_start[p]), v) for v in Vg
            )
            char_v[vg,p] = v_char

    # for group in port_group.values():
    #     for i,j in itertools.combinations(group, 2):
    #         assert all(travel_time[v][i,j] == raw.cargo_origin_port_time[v,i] for v in V
    #                    if (i,j) in travel_time[v] and (v,i) in raw.cargo_origin_port_time)
    #         assert all(travel_cost[v][i,j] == raw.cargo_origin_port_cost[v,i] for v in V
    #                    if (i,j) in travel_time[v] and (v,i) in raw.cargo_origin_port_time)

    demand = frozendict(itertools.chain(
        raw.cargo_size.items(),
        ((p+n, -sz) for p,sz in raw.cargo_size.items())
    ))

    return ITSRSP_Data(
        id=id_str,
        n=n,
        P=P,
        D=D,
        V=V,
        o_depots=o_depots,
        d_depot=d_depot,
        demand=demand,
        vehicle_capacity=frozendict(capacity_vg),
        P_compatible=frozendict(P_compat_vg),
        customer_penalty=raw.cargo_penalty,
        tw_start=frozendict(tw_start),
        tw_end=frozendict(tw_end),
        travel_time=frozendict((v,frozendict(tt)) for v,tt in travel_time_vg.items()),
        travel_cost=frozendict((v,frozendict(tc)) for v,tc in travel_cost_vg.items()),
        port_groups = frozendict((i, frozenset(g)) for g in port_group.values() for i in g),
        vehicle_groups=vehicle_groups,
        group_by_vehicle=vehicle_groups_inv,
        char_vehicle=frozendict(char_v)
    )

def get_named_instance_PDPTWLH(name, rehandling_cost=0) -> PDPTWLH_Data:
    problem_group = name[0]
    id_str = name
    if name.endswith('s'):
        name = name[:-1]
        problem_group += '*'

    if problem_group == 'A*':
        vehicle_cap = 26
        delay_amount = 60
    elif problem_group == "A":
        vehicle_cap = 22
        delay_amount =  45
    elif problem_group == 'B*':
        vehicle_cap = 35
        delay_amount = 60
    elif problem_group == 'B':
        vehicle_cap = 30
        delay_amount =  45
    elif problem_group == 'C':
        vehicle_cap = 18
        delay_amount =  15
    elif problem_group == 'D':
        vehicle_cap = 25
        delay_amount =  15
    else:
        raise Exception

    raw = parse_format_cordeau(resolve_name_cordeau_PDPTW(name))
    data = build_PDPTWLH_from_cordeau(raw, 10000, rehandling_cost, name)
    data = modify.delay_delivery_windows(data, delay_amount)
    return dataclasses.replace(data, id = id_str, Q = vehicle_cap)


def get_named_instance_MSPRP(name : str) -> MSPRP_Data:
    rawdata = parse_format_schrotenboer(*resolve_name_schrotenboer(name))
    return build_MSPRP_from_schrotenboer(rawdata, name)

def get_named_instance_DARP(name : str) -> DARP_Data:
    filename = resolve_name_cordeau_DARP(name)
    raw = parse_format_cordeau(filename)
    data = build_DARP_from_cordeau(raw, name)
    return data

def get_named_instance_ITSRSP(name : str) -> ITSRSP_Data:
    # for resolve in (resolve_name_hemmati, resolve_name_homsi):
    #     try:
    #         filename = resolve(name)
    #         break
    #     except ValueError:
    #         continue
    # else:
    #     raise ValueError(f"{name} is not a valid ITSRSP instance name")

    raw = parse_format_hemmati_hdf5(resolve_name_hemmati_hdf5(name))
    data = build_ITSRSP_from_hemmati(raw, name)
    return data


def get_named_instance_skeleton_ITSRSP(name : str) -> ITSRSP_Skeleton_Data:
    raw = parse_format_hemmati_hdf5_to_skeleton(resolve_name_hemmati_hdf5(name))
    return dataclasses.replace(raw, id=name)


def get_index_file(dataset : str, **kwargs) -> Path:
    datasets = {
        'itsrsp' : data_directory("ITSRSP_hdf5_ti")/"INDEX.txt"
    }

    if dataset not in datasets:
        raise ValueError(f"no known index file for `{dataset!s}`")

    indexfile = datasets[dataset]

    return indexfile