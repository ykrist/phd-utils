from oru import frozendict, LazyHashFrozenDataclass
import dataclasses
from typing import Union, Sequence
from . import utils as _u

frozen_dataclass = dataclasses.dataclass(frozen=True)

@frozen_dataclass
class ProblemDataBase(LazyHashFrozenDataclass):
    id : str

@frozen_dataclass
class DARP_Data(ProblemDataBase):
    travel_time : frozendict #+idx:IJ
    # service_time : frozendict #+idx:I
    travel_cost : frozendict #+idx:IJ
    demand : frozendict #+idx:IJ
    loc : frozendict #+idx:I
    tw_start : frozendict #+idx:I
    tw_end : frozendict #+idx:I

    # Note: the max_ride_time includes the service time, but service_time is needed for some modify.* functionality
    service_time : frozendict #+idx:I
    max_ride_time: frozendict #+idx:I
    capacity : int

    n : int
    P : range
    D : range
    K : range
    N : range


@frozen_dataclass
class PDPTW_Data(ProblemDataBase):
    T: frozendict
    C: frozendict
    n: int
    Q: int
    E: frozendict
    L: frozendict
    D: frozendict
    loc: frozendict
    pickup_loc: range
    delivery_loc: range
    vehicle_cost: float

@frozen_dataclass
class PDPTWLH_Data(PDPTW_Data):
    rehandle_cost : float


@frozen_dataclass
class GenMSPRP_Data(ProblemDataBase):
    # indexed by i,l
    servicemen_demand : frozendict
    # indexed by l,t
    servicemen_max : frozendict
    # indexed by k,t
    servicemen_capacity : frozendict
    # indexed by l,t
    servicemen_cost : frozendict
    # indexed by i
    service_time : frozendict
    # indexed by i,j,k,t
    travel_time : frozendict
    # indexed by i,j,k,t
    travel_cost : frozendict
    # indexed by i
    parts_demand : frozendict
    # indexed by k
    parts_capacity : frozendict
    # indexed by k,t
    travel_time_max : frozendict

    nr : int
    Nd : range
    Np : range
    T : range
    L : range
    K : range
    loc : frozendict


@frozen_dataclass
class MSPRP_Data(GenMSPRP_Data):
    # note that travel_cost, travel_time are only indexed by (i,j,k) not (i,j,k,t)
    # likewise servicemen_cost,servicemen_max are indexed only by l, not (l,t)

    # indexed by i,t
    late_penalty : frozendict
    #
    # def convert_to_general_form(self) -> GenMSPRP_Data:
    #     travel_time = {(i,j,k,t) : v for (i,j,k),v in self.travel_time.items() for t in self.T}
    #     travel_cost = {(i,j,k,t) : v1+self.late_penalty.get((i,t),0) for (i,j,k),v1 in self.travel_cost.items() for t in self.T}
    #     servicemen_cost = {(l,t) : v for l,v in self.servicemen_cost.items() for t in self.T}
    #     servicemen_max = {(l,t) : v for l,v in self.servicemen_max.items() for t in self.T}
    #     return GenMSPRP_Data(
    #         servicemen_demand=self.servicemen_demand,
    #         servicemen_max=frozendict(servicemen_max),
    #         servicemen_capacity=self.servicemen_capacity,
    #         servicemen_cost = frozendict(servicemen_cost),
    #         service_time=self.service_time,
    #         travel_time = frozendict(travel_time),
    #         travel_cost = frozendict(travel_cost),
    #         parts_demand=self.parts_demand,
    #         parts_capacity=self.parts_capacity,
    #         travel_time_max=self.travel_time_max,
    #         nr=self.nr,
    #         Nd=self.Nd,
    #         Np=self.Np,
    #         loc=self.loc,
    #         T=self.T,
    #         L=self.L,
    #         K=self.K,
    #         id=self.id
    #     )


    def reduce_size(self, req_keep: Union[int,Sequence]):
        node_map, num_req_keep = _u.get_node_map(req_keep, self.nr)

@frozen_dataclass
class AFVSP_Data(ProblemDataBase):
    T : range
    S : range
    D : range


    num_buses : frozendict

    fuel_capacity : float

    start_time_trip : frozendict
    end_time_trip : frozendict

    cost_trip: frozendict
    fuel_trip: frozendict

    cost_depot_station: frozendict
    cost_depot_trip: frozendict
    cost_station_depot: frozendict
    cost_station_trip: frozendict
    cost_trip_depot: frozendict
    cost_trip_station: frozendict
    cost_trip_trip: frozendict

    fuel_depot_station: frozendict
    fuel_depot_trip: frozendict
    fuel_station_depot: frozendict
    fuel_station_trip: frozendict
    fuel_trip_depot: frozendict
    fuel_trip_station: frozendict
    fuel_trip_trip: frozendict

    time_depot_station: frozendict
    time_depot_trip: frozendict
    time_station_depot: frozendict
    time_station_trip: frozendict
    time_trip_depot: frozendict
    time_trip_station: frozendict
    time_trip_trip: frozendict

@frozen_dataclass
class ITSRSP_Data(ProblemDataBase):
    P : range
    D : range
    o_depots : frozendict
    d_depots : frozendict


    customer_penalty : frozendict
    service_time : frozendict
    travel_time : frozendict
    travel_cost : frozendict
