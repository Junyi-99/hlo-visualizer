import { useEffect, useMemo, useState } from 'react';
import ELK from 'elkjs/lib/elk-api';
import workerUrl from 'elkjs/lib/elk-worker.min.js?url';
import { instructionGraph, overviewGraph, routeGraph, fallbackRoutes, type RoutedLayout } from '../lib/graph-routing';
import { layoutInstructions, layoutOverview, NODE_HEIGHT } from '../lib/graph-layout';
import type { LayoutDirection } from '../lib/graph-layout';
import type { Computation, HloModule } from '../lib/types';

// One real worker per mounted viewer. Cleanup also drops queued work when an
// imported module replaces the current graph.
export function useGraphLayouts(module: HloModule, computation: Computation | null | undefined, heights: Record<string, number>) {
  const [revision, setRevision] = useState(0);
  const input = useMemo(() => {
    const measured = new Map((computation?.nodes ?? []).map(node => [node.name, heights[node.id] ?? NODE_HEIGHT]));
    const graph = computation ? instructionGraph(computation, measured) : overviewGraph(module);
    const place = (direction: LayoutDirection) => computation ? layoutInstructions(computation, measured, direction) : layoutOverview(module, direction);
    // Simple routes keep dependencies visible while the worker runs, and after it fails.
    const fallback: Record<LayoutDirection, RoutedLayout> = {
      horizontal: fallbackRoutes(place('horizontal'), graph.nodes, graph.edges, 'horizontal'),
      vertical: fallbackRoutes(place('vertical'), graph.nodes, graph.edges, 'vertical')
    };
    return { graph, fallback };
  }, [module, computation, heights, revision]);
  const [result, setResult] = useState<{ input: typeof input; layouts: Record<LayoutDirection, RoutedLayout> } | null>(null);
  const [failure, setFailure] = useState<typeof input | null>(null);
  useEffect(() => {
    let cancelled = false;
    const createEngine = () => new ELK({ workerFactory: () => {
      const worker = new Worker(workerUrl);
      worker.addEventListener('error', () => { if (!cancelled) setFailure(input); });
      return worker;
    } });
    let engine: ReturnType<typeof createEngine>;
    try { engine = createEngine(); } catch (error) { console.error('Graph worker failed', error); setFailure(input); return; }
    Promise.all([
      routeGraph(engine, input.graph.nodes, input.graph.edges, 'horizontal'),
      routeGraph(engine, input.graph.nodes, input.graph.edges, 'vertical')
    ]).then(([horizontal, vertical]) => {
      if (!cancelled) { setResult({ input, layouts: { horizontal, vertical } }); setFailure(null); }
    }).catch(error => {
      if (!cancelled) { console.error('Graph layout failed', error); setFailure(input); }
    });
    return () => { cancelled = true; engine.terminateWorker(); };
  }, [input]);
  const ready = result?.input === input;
  return {
    layouts: ready ? result.layouts : input.fallback,
    pending: !ready && failure !== input,
    failed: failure === input,
    retry: () => setRevision(value => value + 1)
  };
}
