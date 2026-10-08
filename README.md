# HLO Visualizer - explore and understand XLA HLO

Interactive viewer for XLA HLO text, such as the HLO that JAX produces on TPU.

**Open it here: https://www.junyi.dev/hlo-visualizer/**

## Use it

1. Pick a module from **Examples**, or click **Open HLO** to paste text or choose a `.hlo` / `.txt` file. Your HLO stays in your browser.
2. The overview shows how computations call each other. Open one to see its instructions as a dependency graph.
3. Click a node. The inspector explains the instruction piece by piece and lists its inputs, consumers and source location.

## What you can see

- Data and control dependencies, with upstream and downstream paths highlighted
- A plain-language explanation of every part of an instruction: types, layouts, operands and attributes
- Where each value lives on TPU (HBM, VMEM, SMEM, …) in compiled HLO, verified against XLA's buffer assignment on TPU v6e
- Source call stacks from HLO metadata
- Search (`/`), N-hop neighborhood view, shortest path between two nodes, and nested computations expanded in place

## Get HLO from JAX

```python
lowered = jax.jit(f).lower(*args)
print(lowered.compile().as_text())                    # compiled HLO, with memory placement
print(lowered.as_text(dialect="hlo", debug_info=True)) # HLO before XLA optimizations
```

Output from XLA's `--xla_dump_to` also works.

## Examples

The bundled examples are 21 small JAX programs (attention, MLP, convolution, while loops, scatter, sort, FFT, Pallas, …) compiled on a TPU v6e, each before and after optimization. Link to one directly, for example [`?example=attention.after`](https://www.junyi.dev/hlo-visualizer/?example=attention.after).
