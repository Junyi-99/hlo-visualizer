"""Compile a set of JAX programs on TPU and write their HLO text.

For each program: <name>.after.hlo  = compiled (post-optimization, scheduled) module
                  <name>.before.hlo = lowered module before XLA optimizations

Usage (on a TPU VM with JAX installed):  python scripts/dump_hlo.py OUT
Also writes XLA's own dumps to OUT/xla_dump (buffer assignment included), from the same compilation,
for scripts/extract_memory_truth.py. Compilation is not byte-identical across runs, so always take
the examples and the memory ground truth from one run.
The files in examples/tpu-v6e were produced on a v6e-1 with JAX 0.11.2.
"""
import os
import sys
import traceback

out = sys.argv[1] if len(sys.argv) > 1 else "hlo-out"
os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_dump_to={out}/xla_dump --xla_dump_hlo_as_text".strip()

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl

os.makedirs(out, exist_ok=True)
key = jax.random.key(0)
f32, bf16 = jnp.float32, jnp.bfloat16


def arr(shape, dtype=f32, seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape), dtype)


programs = {}


def program(*args, **jit_kwargs):
    def register(fn):
        programs[fn.__name__] = (fn, args, jit_kwargs)
        return fn
    return register


@program(arr((256, 512), bf16), arr((512, 1024), bf16), arr((1024,), bf16), arr((1024, 128), bf16))
def mlp(x, w1, b1, w2):
    h = jax.nn.gelu(x @ w1 + b1)
    return jax.nn.softmax((h @ w2).astype(f32), axis=-1)


@program(arr((8, 32, 32, 16)), arr((3, 3, 16, 32)))
def conv_pool(x, k):
    y = lax.conv_general_dilated(x, k, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    mean = y.mean(axis=(0, 1, 2)); var = y.var(axis=(0, 1, 2))
    y = jax.nn.relu((y - mean) * lax.rsqrt(var + 1e-5))
    mx = lax.reduce_window(y, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
    avg = lax.reduce_window(y, 0.0, lax.add, (1, 2, 2, 1), (1, 2, 2, 1), "VALID") / 4
    return mx + avg


@program(arr((8, 32, 32, 16)), arr((3, 3, 16, 32)))
def conv_grad(x, k):
    def loss(x, k):
        y = lax.conv_general_dilated(x, k, (2, 2), [(1, 1), (1, 1)], rhs_dilation=(1, 1),
                                     dimension_numbers=("NHWC", "HWIO", "NHWC"))
        y = lax.reduce_window(y, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
        return (y ** 2).sum()
    return jax.grad(loss, argnums=(0, 1))(x, k)


@program(arr((2, 4, 128, 64), bf16), arr((2, 4, 128, 64), bf16), arr((2, 4, 128, 64), bf16))
def attention(q, k, v):
    s = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(f32) / 8.0
    mask = jnp.arange(128)[:, None] >= jnp.arange(128)[None, :]
    s = jnp.where(mask, s, -1e9)
    p = jax.nn.softmax(s, axis=-1).astype(bf16)
    return jnp.einsum("bhqk,bhkd->bhqd", p, v)


@program(arr((128, 128), bf16), arr((128, 128), bf16))
def fori_matmul(a, b):
    return lax.fori_loop(0, 10, lambda i, acc: jnp.tanh(acc @ b) + i.astype(bf16), a)


@program(arr((16, 64)), arr((64, 64)))
def scan_rnn(xs, w):
    def step(h, x):
        h = jnp.tanh(h @ w + x)
        return h, h.sum()
    return lax.scan(step, jnp.zeros((64,)), xs)


@program(arr((64,)), jnp.int32(2))
def cond_switch(x, i):
    y = lax.cond(x.sum() > 0, lambda v: jnp.sin(v), lambda v: jnp.cos(v) * 2, x)
    return lax.switch(i, [lambda v: v + 1, lambda v: v * v, lambda v: jnp.exp(v)], y)


@program(arr((1000, 64)), jnp.arange(32, dtype=jnp.int32) * 7, arr((32, 64)))
def gather_scatter(table, idx, upd):
    g = table[idx]
    s = table.at[idx].add(upd)
    t = jnp.take_along_axis(table[:32], (idx[:, None] % 64).astype(jnp.int32), axis=1)
    m = table.at[idx].max(upd)
    return g, s, t, m


@program(arr((256, 256)), jnp.int32(5))
def dynamic_slices(x, i):
    d = lax.dynamic_slice(x, (i, i * 2), (16, 32))
    x = lax.dynamic_update_slice(x, d * 2, (i + 1, 0))
    s = x[3:200:3, 10:50]
    p = jnp.pad(x, ((1, 2), (0, 3)), constant_values=0.5)
    c = jnp.concatenate([x, x[::-1]], axis=0)
    return d, s, p, c, x.T.reshape(128, 512)


@program(arr((8, 1000)))
def sort_topk(x):
    return jnp.sort(x, axis=-1), jnp.argsort(x, axis=-1), lax.top_k(x, 5), jnp.argmax(x, axis=-1), jnp.cumsum(x, axis=-1)


@program()
def random_ops():
    k1, k2 = jax.random.split(key)
    return jax.random.normal(k1, (128, 128)), jax.random.randint(k2, (64,), 0, 10), jax.random.bernoulli(k2, 0.3, (32,))


@program(arr((64, 64)), arr((64, 64)))
def elementwise_zoo(x, y):
    xi = (x * 100).astype(jnp.int32)
    yi = (y * 100).astype(jnp.int32)
    return (
        jnp.exp(x), jnp.log1p(jnp.abs(x)), jnp.expm1(x), lax.erf(x), jax.nn.sigmoid(x), jnp.arctan2(x, y),
        jnp.power(jnp.abs(x), y), jnp.remainder(x, 1.5), jnp.floor(x), jnp.ceil(x), jnp.round(x), jnp.sign(x),
        jnp.clip(x, -0.5, 0.5), jnp.isfinite(x / y), jnp.cbrt(x), lax.reduce_precision(x, 5, 10),
        xi & yi, xi | yi, xi ^ yi, ~xi, xi << 2, xi >> 1, lax.shift_right_logical(xi, 3),
        lax.population_count(xi), lax.clz(xi), lax.bitcast_convert_type(x, jnp.int32),
        x.astype(jnp.int8), x.astype(jnp.float16), (x > y) & (x < 1), jnp.maximum(x, y), jnp.minimum(xi, yi),
    )


@program(arr((256,)), arr((8, 64)))
def fft_complex(x, y):
    f = jnp.fft.fft(x)
    r = jnp.fft.irfft(jnp.fft.rfft(y, axis=-1), axis=-1)
    return jnp.abs(f), jnp.real(f) + jnp.imag(f), r


@program(arr((64, 64)), arr((64, 8)))
def linalg(a, b):
    spd = a @ a.T + 64 * jnp.eye(64)
    l = jnp.linalg.cholesky(spd)
    x = jax.scipy.linalg.solve_triangular(l, b, lower=True)
    q, r = jnp.linalg.qr(a)
    return l, x, q, r


@program(arr((128, 256)), arr((256, 256)))
def remat_grad(x, w):
    @jax.checkpoint
    def layer(x):
        return jnp.tanh(x @ w)
    loss = lambda w: layer(layer(x)).mean()
    return jax.value_and_grad(loss)(w)


@program(arr((8, 128)), arr((8, 128)))
def pallas_add(x, y):
    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...] * 2
    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x, y)


@program(arr((1, 64)))
def collectives(x):
    f = jax.pmap(lambda v: (lax.psum(v, "i"), lax.pmax(v, "i"), lax.all_gather(v, "i"), lax.axis_index("i")), axis_name="i")
    return f(x)


@program(arr((32, 32)), arr((32, 32)))
def barrier_and_callback(x, y):
    x, y = lax.optimization_barrier((x * 2, y + 1))
    jax.debug.print("sum {}", x.sum())
    return x @ y


@program(arr((128, 128)), arr((128,)), donate_argnums=(0,))
def donated_update(state, delta):
    return state + delta[None, :], jnp.linalg.norm(state)


@program(jnp.arange(96, dtype=jnp.int8).reshape(8, 12), arr((8, 12), bf16))
def int8_quant(q, scale):
    w = q.astype(bf16) * scale
    return jnp.einsum("ij,kj->ik", w, w, preferred_element_type=f32), lax.iota(jnp.int32, 12)


def _inner(x):
    return jnp.sin(x) * 2  # innermost frame


def _middle(x):
    return _inner(x) + 1


@program(arr((16,)))
def nested_calls(x):
    return _middle(x)


failed = []
for name, (fn, args, jit_kwargs) in programs.items():
    try:
        lowered = jax.jit(fn, **jit_kwargs).lower(*args)
        with open(f"{out}/{name}.before.hlo", "w") as f:
            f.write(lowered.as_text(dialect="hlo", debug_info=True))  # keep metadata (op_name, source lines)
        with open(f"{out}/{name}.after.hlo", "w") as f:
            f.write(lowered.compile().as_text())
        print("ok", name)
    except Exception:
        failed.append(name)
        print("FAIL", name); traceback.print_exc(limit=3)
print("jax", jax.__version__, jax.devices()[0].device_kind, "failed:", failed)
