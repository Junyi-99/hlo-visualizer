import { useEffect, useMemo, useState } from 'react';
import ELK from 'elkjs/lib/elk-api';
import workerUrl from 'elkjs/lib/elk-worker.min.js?url';
import { instructionGraph, overviewGraph, routeGraph, fallbackRoutes, type RoutedLayout } from '../lib/graph-routing';
import { layoutInstructions, layoutOverview, NODE_HEIGHT, type LayoutDirection } from '../lib/graph-layout';
import type { Computation, HloModule } from '../lib/types';

// One ELK worker per layout attempt; cleanup terminates it, dropping queued work when the graph changes.
export function useGraphLayouts(module: HloModule, computation: Computation | null | undefined, heights: Record<string, number>) {
  const [revision, setRevision] = useState(0);

  const input = useMemo(() => {
    const measured = new Map((computation?.nodes ?? []).map(node => [node.name, heights[node.id] ?? NODE_HEIGHT]));
    const graph = computation ? instructionGraph(computation, measured) : overviewGraph(module);
    const place = (direction: LayoutDirection) =>
      computation ? layoutInstructions(computation, measured, direction) : layoutOverview(module, direction);
    // Simple routes keep dependencies visible while the worker runs, and after it fails.
    const fallback: Record<LayoutDirection, RoutedLayout> = {
      horizontal: fallbackRoutes(place('horizontal'), graph.nodes, graph.edges, 'horizontal'),
      vertical: fallbackRoutes(place('vertical'), graph.nodes, graph.edges, 'vertical')
    };
    return { graph, fallback };
  }, [module, computation, heights]);

  // A retry is a new attempt on the same input, so its result and failure are tracked separately.
  const attempt = useMemo(() => ({ input, revision }), [input, revision]);
  const [result, setResult] = useState<{ attempt: typeof attempt; layouts: Record<LayoutDirection, RoutedLayout> } | null>(null);
  const [failure, setFailure] = useState<typeof attempt | null>(null);

  useEffect(() => {
    let cancelled = false;
    let engine: InstanceType<typeof ELK> | undefined;
    // Constructing the worker can throw; the promise turns that into an asynchronous failure.
    new Promise<[RoutedLayout, RoutedLayout]>(resolve => {
      const created = new ELK({
        workerFactory: () => {
          const worker = new Worker(workerUrl);
          worker.addEventListener('error', () => {
            if (!cancelled) setFailure(attempt);
          });
          return worker;
        }
      });
      engine = created;

      const { nodes, edges } = attempt.input.graph;
      resolve(Promise.all([routeGraph(created, nodes, edges, 'horizontal'), routeGraph(created, nodes, edges, 'vertical')]));
    })
      .then(([horizontal, vertical]) => {
        if (!cancelled) {
          setResult({ attempt, layouts: { horizontal, vertical } });
          setFailure(null);
        }
      })
      .catch(error => {
        if (!cancelled) {
          console.error('Graph layout failed', error);
          setFailure(attempt);
        }
      });
    return () => {
      cancelled = true;
      engine?.terminateWorker();
    };
  }, [attempt]);

  const ready = result?.attempt === attempt;
  return {
    layouts: ready ? result.layouts : input.fallback,
    pending: !ready && failure !== attempt,
    failed: failure === attempt,
    retry: () => setRevision(value => value + 1)
  };
}
