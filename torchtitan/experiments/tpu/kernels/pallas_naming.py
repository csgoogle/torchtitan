"""Names Pallas kernels from libraries that do not expose ``name=``.

XLA:TPU names each Pallas custom call after the last name scope around its
``pl.pallas_call``. MegaBlox ``gmm`` and ``tgmm`` take no ``name`` and open no
scope, so every grouped matmul shows up in profiles under the same generic
name. ``call_named`` opens a ``jax.named_scope`` around a single library call.
"""

from typing import Any, Callable

import jax


def call_named(
    fn: Callable[..., Any], name: str, /, *args: Any, **kwargs: Any
) -> Any:
    """Calls Pallas library function ``fn`` inside ``jax.named_scope(name)``.

    ``fn`` is traced through ``__wrapped__`` when it is a ``jax.jit`` function.
    Through torch_tpu, a scope opened outside the jit does not reach the
    kernels inside it.

    Args:
      fn: A function that calls ``pl.pallas_call``, such as MegaBlox ``gmm``.
      name: Name scope, and therefore XLA:TPU kernel name, for ``fn``'s kernels.
      *args: Positional arguments for ``fn``.
      **kwargs: Keyword arguments for ``fn``.

    Returns:
      The result of ``fn(*args, **kwargs)``.
    """
    with jax.named_scope(name):
        return getattr(fn, "__wrapped__", fn)(*args, **kwargs)


__all__ = ["call_named"]
