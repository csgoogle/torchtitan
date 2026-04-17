"""Shared utilities for JAX-based experiments (jax/ and torchax/)."""


def get_accelerator_short_name(device_kind: str) -> str:
    """Parse a JAX device_kind string into a short accelerator tag.

    Returns tags like 'v4', 'v5p', 'v5e', 'v6e', 'v7x', 'h100', 'a100', 'cpu'.
    Used to select peak-FLOPS tables for MFU computation.
    """
    device_lower = device_kind.lower()
    if 'v4' in device_lower:
        return 'v4'
    # v5e is reported as 'TPU v5 lite'
    if 'v5 lite' in device_lower:
        return 'v5e'
    # v5p is reported as 'TPU v5'
    if device_kind == 'TPU v5':
        return 'v5p'
    if 'v6' in device_lower:
        return 'v6e'
    if 'tpu7x' in device_lower:
        return 'v7x'
    if 'h100' in device_lower:
        return 'h100'
    if 'a100' in device_lower:
        return 'a100'
    if 'cpu' in device_lower:
        return 'cpu'
    raise RuntimeError(
        f'Could not determine accelerator type from JAX device_kind: {device_kind}'
    )
