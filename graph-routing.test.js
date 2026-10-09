import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import ELK from 'elkjs/lib/elk.bundled.js';
import { parseHlo } from './src/lib/parser.ts';
import { instructionGraph, overviewGraph, routeGraph, edgePath, curvePoints, movedEdgePoints } from './src/lib/graph-routing.ts';

const engine = new ELK();
const sample = parseHlo(readFileSync(new URL('./sample.hlo', import.meta.url), 'utf8'));
const epsilon = 0.01;
function checkGeometry(layout, nodes, edges, direction = 'horizontal') {
  assert.equal(layout.positions.size, nodes.length);
  assert.equal(layout.edges.length, edges.filter(e => nodes.some(n => n.name === e.source) && nodes.some(n => n.name === e.target)).length);
  const cards = nodes.map(node => ({ ...node, ...layout.positions.get(node.name) }));
  for (const card of cards) {
    assert(card.x >= 0 && card.y >= 0);
    assert(card.x + card.width <= layout.width && card.y + card.height <= layout.height);
  }
  for (const [i, a] of cards.entries())
    for (const b of cards.slice(i + 1)) {
      assert(
        a.x + a.width <= b.x || b.x + b.width <= a.x || a.y + a.height <= b.y || b.y + b.height <= a.y,
        a.name + ' overlaps ' + b.name
      );
    }
  for (const edge of layout.edges) {
    assert.equal((edge.points.length - 1) % 3, 0, 'invalid cubic control points');
    assert(edgePath(edge.points).includes('C'));
    const points = curvePoints(edge.points, direction);
    // Test points on the actual Bézier curves, not their control polygons.
    for (let i = 1; i < points.length; i += 3) {
      const p0 = points[i - 1],
        p1 = points[i],
        p2 = points[i + 1],
        p3 = points[i + 2];
      for (let step = 1; step < 80; step++) {
        const t = step / 80,
          u = 1 - t;
        const x = u * u * u * p0.x + 3 * u * u * t * p1.x + 3 * u * t * t * p2.x + t * t * t * p3.x;
        const y = u * u * u * p0.y + 3 * u * u * t * p1.y + 3 * u * t * t * p2.y + t * t * t * p3.y;
        for (const c of cards) {
          assert(
            !(x > c.x + epsilon && x < c.x + c.width - epsilon && y > c.y + epsilon && y < c.y + c.height - epsilon),
            edge.source + ' → ' + edge.target + ' crosses ' + c.name
          );
        }
      }
    }
  }
}

for (const direction of ['horizontal', 'vertical']) {
  test(direction + ': sample routes avoid cards, including variable heights and cross-layer edges', async () => {
    for (const computation of sample.computations) {
      const heights = new Map(computation.nodes.map((n, i) => [n.name, 91 + (i % 4) * 40]));
      const graph = instructionGraph(computation, heights);
      checkGeometry(await routeGraph(engine, graph.nodes, graph.edges, direction), graph.nodes, graph.edges, direction);
    }
    const graph = overviewGraph(sample);
    const layout = await routeGraph(engine, graph.nodes, graph.edges, direction);
    checkGeometry(layout, graph.nodes, graph.edges, direction);
    assert(layout.edges.every(e => e.label && Number.isFinite(e.label.x)));
  });
  test(direction + ': duplicate inputs, control dependencies, cycles and missing references', async () => {
    const nodes = ['a', 'b', 'c', 'isolated'].map(name => ({ name, width: 218, height: 91 }));
    const edges = [
      { source: 'a', target: 'b' },
      { source: 'a', target: 'b' },
      { source: 'b', target: 'c' },
      { source: 'c', target: 'a', control: true },
      { source: 'missing', target: 'c' }
    ];
    const layout = await routeGraph(engine, nodes, edges, direction);
    checkGeometry(layout, nodes, edges, direction);
    assert.equal(layout.edges.filter(e => e.control).length, 1);
    const again = await routeGraph(engine, nodes, edges, direction);
    assert.deepEqual(layout, again);
  });
}
test('empty and filtered graphs remain finite; unknown dependencies are omitted', async () => {
  const empty = await routeGraph(engine, [], [], 'horizontal');
  assert.equal(empty.edges.length, 0);
  assert(Number.isFinite(empty.width));
  const computation = sample.computations[0];
  const filtered = { ...computation, nodes: computation.nodes.slice(-2) };
  const graph = instructionGraph(filtered, new Map());
  checkGeometry(await routeGraph(engine, graph.nodes, graph.edges, 'vertical'), graph.nodes, graph.edges, 'vertical');
});
test('cubic curves leave room for arrows and stay attached during dragging', () => {
  const edge = {
    source: 'a',
    target: 'b',
    points: [
      { x: 218, y: 40 },
      { x: 260, y: 40 },
      { x: 280, y: 140 },
      { x: 320, y: 140 }
    ]
  };
  const path = edgePath(edge.points);
  assert(path.includes('C'));
  assert(!/[LQ]/.test(path));
  assert(path.endsWith('315 140'));
  const moved = movedEdgePoints(
    edge,
    new Map([
      ['a', { x: 20, y: 30 }],
      ['b', { x: -10, y: 15 }]
    ])
  );
  assert.deepEqual(moved[0], { x: 238, y: 70 });
  assert.deepEqual(moved[1], { x: 280, y: 70 });
  assert.deepEqual(moved.at(-1), { x: 310, y: 155 });
  assert.deepEqual(moved.at(-2), { x: 270, y: 155 });
  assert.equal(moved.length, edge.points.length);
  assert(edgePath(moved).includes('C'));
  assert.deepEqual(edge.points[0], { x: 218, y: 40 });
});

for (const direction of ['horizontal', 'vertical']) {
  test(direction + ': MLP entry has distinct, separated ports for every dependency', async () => {
    const module = parseHlo(readFileSync(new URL('./examples/tpu-v6e/mlp.after.hlo', import.meta.url), 'utf8'));
    const computation = module.computations.find(c => c.entry);
    const graph = instructionGraph(computation, new Map());
    const layout = await routeGraph(engine, graph.nodes, graph.edges, direction);
    checkGeometry(layout, graph.nodes, graph.edges, direction);
    for (const node of graph.nodes) {
      for (const side of ['input', 'output']) {
        const ports = layout.edges
          .filter(e => (side === 'input' ? e.target === node.name : e.source === node.name))
          .map(e => (side === 'input' ? e.points.at(-1) : e.points[0]));
        assert.equal(new Set(ports.map(p => p.x + ':' + p.y)).size, ports.length, node.name + ' merged ' + side + ' ports');
        const offsets = ports.map(p => (direction === 'horizontal' ? p.y : p.x)).sort((a, b) => a - b);
        for (let i = 1; i < offsets.length; i++) assert(offsets[i] - offsets[i - 1] >= 8, node.name + ' ports too close');
      }
    }
  });
}

test('grouped copy pairs retain all external data and control routes in both directions', async () => {
  const { groupCopyPairs } = await import('./src/lib/copy-grouping.ts');
  const original = sample.computations.find(c => c.entry);
  const grouped = groupCopyPairs(original, true).computation;
  const graph = instructionGraph(grouped, new Map());
  for (const direction of ['horizontal', 'vertical']) {
    const layout = await routeGraph(engine, graph.nodes, graph.edges, direction);
    checkGeometry(layout, graph.nodes, graph.edges, direction);
    assert(!layout.edges.some(e => e.source.startsWith('copy-start') || e.target.startsWith('copy-start')));
  }
});

test('angled spline controls enter input ports squarely with a small, fixed gap', () => {
  for (const direction of ['horizontal', 'vertical']) {
    const raw = [
      { x: 100, y: 100 },
      { x: 100, y: 100 },
      { x: 150, y: 300 },
      { x: 200, y: 400 }
    ];
    const controls = curvePoints(raw, direction);
    const end = controls.at(-1),
      tangent = controls.at(-2);
    if (direction === 'vertical') {
      assert.equal(end.x, raw.at(-1).x);
      assert.equal(end.x, tangent.x);
      assert(tangent.y < end.y);
      assert.equal(end.y, raw.at(-1).y - 5);
    } else {
      assert.equal(end.y, raw.at(-1).y);
      assert.equal(end.y, tangent.y);
      assert(tangent.x < end.x);
      assert.equal(end.x, raw.at(-1).x - 5);
    }
    assert.deepEqual(raw.at(-1), { x: 200, y: 400 });
    assert(!/[LQ]/.test(edgePath(raw, direction)));
  }
});

test('routed edges keep their input index, and straight or non-spline routes still draw', async () => {
  const nodes = ['a', 'b'].map(name => ({ name, width: 218, height: 91 }));
  const edges = [
    { source: 'missing', target: 'b' },
    { source: 'a', target: 'b', label: 'BODY' }
  ];
  const layout = await routeGraph(engine, nodes, edges, 'horizontal');
  assert.deepEqual(
    layout.edges.map(e => e.index),
    [1]
  );
  assert(
    edgePath([
      { x: 0, y: 0 },
      { x: 100, y: 0 }
    ]).startsWith('M0 0 C')
  );
  assert.equal(
    edgePath([
      { x: 0, y: 0 },
      { x: 50, y: 0 },
      { x: 100, y: 0 }
    ]),
    'M0 0 L50 0 L100 0'
  );
});
