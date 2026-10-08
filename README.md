# HLO Visualizer - explore and understand XLA HLO

Interactive viewer for XLA HLO text, such as the HLO that JAX produces on TPU.

**Open it here: https://www.junyi.dev/hlo-visualizer/**

<table>
  <tr>
    <th>XLA's built-in HLO graph</th>
    <th>HLO Visualizer</th>
  </tr>
  <tr>
    <td width="50%"><img src="doc/xla-lowered-hlo.png" alt="XLA's HLO graph rendering of a small lowered computation"></td>
    <td width="50%"><img src="doc/screenshot.png" alt="HLO Visualizer showing the entry computation of a JAX attention program compiled for TPU"></td>
  </tr>
</table>

<sub>Left image: from the <a href="https://openxla.org/xla/gpu_architecture">XLA GPU architecture overview</a>, OpenXLA, licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.</sub>

## Use it

1. Pick a module from **Examples**, or click **Open HLO** to paste text or choose a `.hlo` / `.txt` file.
2. Click a node. The inspector explains the instruction piece by piece.

## What you can see

- A plain-language explanation of every part of an instruction
- Where each value lives on TPU (HBM, VMEM, SMEM, …)
- Data and control dependencies between instructions

## Get HLO from JAX

```python
lowered = jax.jit(f).lower(*args)
print(lowered.compile().as_text())                    # compiled HLO, with memory placement
print(lowered.as_text(dialect="hlo", debug_info=True)) # HLO before XLA optimizations
```

Output from XLA's `--xla_dump_to` also works.

## Examples

The bundled examples are 21 small JAX programs (attention, MLP, convolution, while loops, scatter, sort, FFT, Pallas, …) compiled on a TPU v6e, each before and after optimization. Link to one directly, for example [`?example=attention.after`](https://www.junyi.dev/hlo-visualizer/?example=attention.after).
