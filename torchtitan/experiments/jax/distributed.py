"""JAX distributed utilities: sharding, mesh setup, parameter initialization."""

import fnmatch
import functools

import jax
import jax.numpy as jnp
from flax import nnx
import torchtitan.tools.logging

logger = torchtitan.tools.logging.logger

P = jax.sharding.PartitionSpec


def sharded_device_put(
    array: jax.Array,
    sharding,
    num_global_devices: int,
    num_local_devices: int,
) -> jax.Array:
    """Place an array on devices with the given sharding."""
    if num_global_devices == num_local_devices:
        return jax.device_put(array, sharding)

    # Multi-host: each host only provides its addressable slices.
    shape = array.shape
    x_split = [
        jax.device_put(array[i], device)
        for device, i in sharding.addressable_devices_indices_map(shape).items()
    ]
    return jax.make_array_from_single_device_arrays(shape, sharding, x_split)


def _match_path(path_str: str, sharding_map: dict) -> tuple | None:
    """Return the partition spec tuple for the first matching pattern."""
    # Exact match first.
    if path_str in sharding_map:
        return sharding_map[path_str]
    # Wildcard / fnmatch patterns.
    for pattern, spec in sharding_map.items():
        if fnmatch.fnmatch(path_str, pattern):
            return spec
    return None


def apply_sharding_to_state(
    state: nnx.State,
    sharding_map: dict,
    mesh: jax.sharding.Mesh,
) -> nnx.State:
    """Apply JAX sharding to every leaf in an NNX state pytree.

    Args:
        state: NNX state (e.g. from nnx.split(model)).
        sharding_map: dict mapping path patterns to partition spec tuples.
        mesh: JAX device mesh.
    Returns:
        New state with all arrays placed on-device with proper sharding.
    """
    def shard_leaf(path, array):
        # Build a '/'-separated path string from the pytree key path.
        # nnx.Variable registers itself as a pytree with a trailing
        # GetAttrKey('value') leaf — strip that so paths match the sharding map.
        parts = []
        for k in path:
            if isinstance(k, jax.tree_util.DictKey):
                parts.append(str(k.key))
            elif isinstance(k, jax.tree_util.GetAttrKey):
                if k.name != 'value':
                    parts.append(k.name)
            elif isinstance(k, jax.tree_util.SequenceKey):
                parts.append(str(k.idx))
            else:
                parts.append(str(k))
        path_str = '/'.join(p for p in parts if p)

        spec = _match_path(path_str, sharding_map)
        if spec is None:
            logger.warning('No sharding spec for %s — replicating.', path_str)
            spec = ()
        sharding = jax.sharding.NamedSharding(mesh, P(*spec))
        logger.info(
            'Sharding %s  shape=%s  spec=%s',
            path_str, array.shape, spec,
        )
        return jax.device_put(array, sharding)

    return jax.tree_util.tree_map_with_path(shard_leaf, state)


def shard_input(
    x: jax.Array,
    mesh: jax.sharding.Mesh,
    num_global_devices: int,
    num_local_devices: int,
) -> jax.Array:
    """Shard a batch input tensor across the FSDP axis (batch dimension)."""
    sharding = jax.sharding.NamedSharding(mesh, P('fsdp'))
    return sharded_device_put(x, sharding, num_global_devices, num_local_devices)
