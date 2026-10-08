# HLO Atlas

A React and TypeScript visualizer for textual XLA HLO modules. It shows computation calls, instruction dependencies, and an inspector that explains result types, tuple slots, layouts, and copy direction.

## Run

Requires a current Node.js release supported by Vite 8.

```sh
npm install
npm run dev
```

Open the URL printed by Vite. For a production build, run `npm run build`. Run the parser and instruction guide checks with `npm test`.

## Explore

- **Overview** shows computation call relationships. Open a computation to see instruction data dependencies.
- Select a node to highlight upstream and downstream paths. The inspector shows direct inputs and consumers, the complete HLO instruction, and links to called computations.
- Use the canvas **View** control to show only nodes within 1–3 dependency hops. **Find path** lets you pick a second node and displays their shortest connection through data and control dependencies.
- Select a fusion or while node and choose **Expand calls** to inspect its called computation inside the canvas. While bodies and conditions have separate tabs; nested calls have breadcrumbs. **Open full graph** remains available for a larger view.
- Control dependencies appear as dashed edges and have separate predecessor/successor lists in the inspector. Parse notes can be opened from the status bar when an instruction or reference needs attention.
- Result types expand by layer. Tuple results expose individually expandable slots; arrays and scalars expose their element type, logical shape, layout, and memory space.
- `copy-start` nodes display the transfer direction, such as `HBM → VMEM` or `VMEM → HBM`. The inspector also explains the destination, source, and context tuple entries.
- Use `/` to search nodes across the module. **Open HLO** accepts pasted text or a local file; parsing stays in the browser.
- Drag empty graph space to pan. Use zoom controls, trackpad pinch, or **Fit view** to adjust the canvas.

## Code layout

- `src/App.tsx` owns navigation, selection, and module state.
- `src/components/` contains the sidebar, toolbar, graph canvas, inspector, type tree, search overlay, and import dialog.
- `src/lib/parser.ts` parses textual HLO, including multiline instruction attributes and control predecessors, and provides graph relationships.
- `src/lib/graph-layout.ts` computes positions for both graph views.
- `src/lib/instruction-guide.ts` explains HLO syntax and generates the array layout diagram.
- `src/styles.css` contains the graph and inspector styles. Tailwind CSS 4 is integrated through its Vite plugin and used for shared utility styling.
- `sample.hlo` is the bundled example and test fixture. It keeps the instructions and computation edges from the provided matmul loop while omitting verbose backend configuration.

The input parser targets `HloModule` text emitted by XLA. The [StableHLO specification](https://openxla.org/stablehlo/spec) is useful for operation and type terminology, but its MLIR `stablehlo.*` syntax is a different input format and is not parsed here.
