# HLO Visualizer - explore and understand XLA HLO

Interactive viewer for XLA HLO text, such as the HLO that JAX produces on TPU.

The same app now has an [LLO Visualizer](?view=llo) for TPU backend instruction dumps. It shows scheduled VLIW bundles by hardware unit, or source-order instructions in earlier LLO passes. Click an instruction to inspect its text, register inputs and users, and referenced allocations. You can paste a dump, choose a local file, or drop one onto the page; parsing runs in the browser.

**Open it here: https://www.junyi.dev/hlo-visualizer/**

The site can also be installed as a PWA from your browser's install or Add to Home Screen menu. After the first production visit finishes caching, both HLO and LLO views and their bundled examples work offline. Local files and pasted text are processed in the browser. The development server does not register a service worker; use `npm run build && npm run preview` to try the installed app locally.

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

<table>
  <tr>
    <td width="33%"><img src="doc/decomposition.png" alt="Inspector explaining each part of a while instruction"></td>
    <td width="33%"><img src="doc/detailed-explanation.png" alt="Inspector expanding a scalar's type, tiling and memory space"></td>
    <td width="33%"><img src="doc/array-layout.png" alt="Inspector drawing the T(8,128)(2,1) tiling of a bf16 array in VMEM"></td>
  </tr>
  <tr>
    <td>Every part of an instruction, explained</td>
    <td>Types and layouts, layer by layer</td>
    <td>TPU array tiling, drawn out</td>
  </tr>
</table>

## Get HLO from JAX

```python
lowered = jax.jit(f).lower(*args)
print(lowered.compile().as_text())                    # compiled HLO, with memory placement
print(lowered.as_text(dialect="hlo", debug_info=True)) # HLO before XLA optimizations
```

Output from XLA's `--xla_dump_to` also works.

## Get LLO from TPU compilation

For libtpu versions that support the Jellyfish LLO dumper, set these flags before importing JAX:

```sh
LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/llo --xla_jf_dump_llo_text=true" python your_program.py
```

Open a dumped LLO pass text file or `*-final_bundles.txt` in the [LLO view](?view=llo). This uses a separate dump directory from HLO's `--xla_dump_to`. LLO text is backend-specific and can vary by libtpu version; unknown instructions remain visible with their original text.

### Offline v6e examples

The LLO view ships every verified final-bundle backend program of the 21 JAX examples, compiled for **v6e:1** and **v6e:2x2** with JAX 0.11.2 and libtpu 0.0.48: 129 programs per topology, one per HLO instruction (fusions, copies, custom calls, …). A program is included only when the HLO instruction named in its entry-bundle comment, with the same operands, exists in that run's compiled HLO; TLP glue, `<late-*>` programs and programs of other modules compiled by the same example are left out. Choose an example from **Examples** (its largest program loads first) and switch between the example's programs from the **HLO instruction** picker in the sidebar. Link to a program with [`?view=llo&example=v6e-1/attention/fusion.5`](?view=llo&example=v6e-1/attention/fusion.5); older `v6e-1/mlp` links resolve to that example's default program. The initial screen remains an illustrative excerpt.

HLO and LLO are linked at the instruction level: a node in the HLO graph whose program exists shows a **TPU backend program** link in the inspector, and the LLO sidebar links back to the same node with [`?example=v6e-1/attention.after&node=fusion.5`](?example=v6e-1/attention.after&node=fusion.5). The program text lives in `public/llo/<topology>/<example>/<instruction>.llo`, indexed by `src/llo/llo-index.json`; it is fetched on demand and cached for offline use the first time it is opened. The two `examples/llo-v6e-*-full.tar.gz` archives contain **all** final-bundle programs, the HLO text from the same offline compilation, and a manifest.

No TPU hardware was used. The compiler ran in a Linux x86_64 Docker container with a virtual topology. On Apple Silicon, Docker's default x86_64 translation lacked AVX, so we ran the container's Python through `qemu-x86_64-static -cpu max`. To reproduce inside a Linux x86_64 environment with `jax[tpu]==0.11.2` installed:

```sh
TPU_OFFLINE_TOPOLOGY=v6e:1 LLO_DUMP_DIR=/tmp/llo-v6e-1 \
LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/llo-v6e-1 --xla_jf_dump_llo_text=true --xla_jf_dump_llo_pass_label_regex=final_bundles" \
python scripts/dump_hlo.py /tmp/build-v6e-1
python scripts/select_llo_examples.py /tmp/build-v6e-1 v6e-1
```

Repeat with `v6e:2x2` and separate output paths. The script maps `v6e:1` to libtpu's `v6e:1x1` topology with `chips_per_host_bounds=(1,1,1)`; libtpu 0.0.48 rejects the literal `v6e:1` name. The 2x2 case replicates the existing single-device examples across four virtual devices. `pallas_add` and `collectives` retain their single-device behavior within that slice.

## Examples

The bundled examples are 21 small JAX programs (attention, MLP, convolution, while loops, scatter, sort, FFT, Pallas, …) compiled for TPU **v6e:1** and **v6e:2x2** (`examples/v6e-1`, `examples/v6e-2x2`), each before and after optimization. The 2x2 modules are the same programs replicated over four partitions, so their compiled HLO carries shardings and differs from the single-device version. Link to one directly, for example [`?example=v6e-1/attention.after`](https://www.junyi.dev/hlo-visualizer/?example=v6e-1/attention.after); links without the topology prefix open the v6e-1 module.
