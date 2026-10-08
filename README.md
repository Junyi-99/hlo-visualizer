# HLO Atlas

An interactive dependency viewer for textual HLO. The bundled `sample.hlo` is a compact version of the supplied matmul loop: it keeps every instruction and computation edge while omitting verbose backend configuration. You can paste or load the complete original HLO in the UI; its raw instruction lines remain available in the inspector.

## Run

From this directory:

```sh
python3 -m http.server 8765
```

Open <http://localhost:8765>. No build or package install is required. HLO input is parsed locally in the browser.

## Explore

- Start with **Overview** to see which computation invokes which other computation. These arrows are call/control links, not tensor data edges. Click a computation to open its instruction graph.
- The entry computation, while body and condition, and fusion computations have separate instruction views.
- Click a node to highlight every upstream and downstream data dependency. The inspector shows direct inputs, consumers, source HLO, and links to called computations.
- The inspector color codes HLO syntax and explains each part. For `copy-start` with `T(8,128)(2,1)`, it also shows a small diagram of the array's physical layout and copy direction.
- Tuple results are explained slot by slot. For while body and condition parameters, the inspector links each slot to its initial value, the body result for later iterations, and any tuple extraction in the current computation.
- Drag empty canvas space to pan. Use the zoom buttons or **Fit view** for an overview.
- Press `/` to search nodes across the module. **Open HLO** accepts pasted text or a local file.

Run parser checks with `node --test parser.test.js`.
