"""Which XLA memory-space number does the TPU backend use for SMEM?

Puts one Pallas operand in SMEM and prints the compiled HLO: XLA copies that operand into S(6)
before the kernel. (S(2) = SFLAG comes from the copy-start context element; see src/lib/memory-space.ts.)
Run on a TPU VM: python scripts/probe_memory_spaces.py
"""

import re

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def kernel(s_ref, x_ref, o_ref):
    o_ref[...] = x_ref[...] * s_ref[0]


def scale(s, x):
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
    )(s, x)


text = (
    jax.jit(scale)
    .lower(jnp.array([3.0], jnp.float32), jnp.ones((8, 128), jnp.float32))
    .compile()
    .as_text()
)
for line in text.splitlines():
    if re.search(r"S\(\d+\)|custom-call", line):
        print(re.sub(r", (backend_config|metadata)=.*", "", line.strip()))
# On v6e with JAX 0.11.2:  %copy.s_ref = f32[1]{0:T(128)S(6)} copy(%s_ref)
