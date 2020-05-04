"""
Binary serialisation/deserialisation functions for interprocess communation
"""
from .types import DARP_Data
import msgpack

def _msgpack_convert_types(x):
    if isinstance(x, tuple):
        return list(x)
    raise ValueError(f"unsupported type conversion: {type(x)}")

def serialise_DARP_Data(stream, data : DARP_Data):
    return msgpack.dump({
        'id' : data.id,
        'n' : data.n,
        'k' : len(data.K),
        'capacity' : float(data.capacity),
        'travel_time' : { k : float(v) for k,v in data.travel_time.items() },
        'travel_cost' : { k : float(v) for k,v in data.travel_cost.items() },
        'demand' : { k : float(v) for k,v in data.demand.items() },
        'loc' : { k : (float(x),float(y)) for k,(x,y) in data.loc.items() },
        'tw_end' : { k : float(v) for k,v in data.tw_end.items() },
        'tw_start' : { k : float(v) for k,v in data.tw_start.items() },
        'max_ride_time' : { k : float(v) for k,v in data.max_ride_time.items() },
    }, stream, default=_msgpack_convert_types, use_single_float=True, strict_types=True)
