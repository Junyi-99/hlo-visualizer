import { useEffect, useMemo, useRef, useState } from 'react';
import { layoutInstructions, NODE_HEIGHT, NODE_WIDTH } from '../lib/graph-layout';
import { nodeCategory, nodeSummary } from '../lib/parser';
import type { HloModule, HloNode } from '../lib/types';

interface Step { name: string; role: string }

interface ComputationExplorerProps {
  module: HloModule;
  rootNode: HloNode;
  onClose: () => void;
  onOpenFull: (name: string) => void;
}

export function ComputationExplorer({ module, rootNode, onClose, onOpenFull }: ComputationExplorerProps) {
  const entries = Object.entries(rootNode.calls).filter(([, name]) => module.byName.has(name)).sort(([left], [right]) => Number(right === 'body') - Number(left === 'body'));
  const [steps, setSteps] = useState<Step[]>(() => entries.length ? [{ role: entries[0][0], name: entries[0][1] }] : []);
  const [selected, setSelected] = useState<string | null>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const active = steps.at(-1);
  const computation = active ? module.byName.get(active.name) : null;
  const layout = useMemo(() => computation ? layoutInstructions(computation) : null, [computation]);
  useEffect(() => { scrollerRef.current?.scrollTo(0, 0); }, [computation]);
  const selectedNode = selected && computation?.byName.get(selected);
  const scale = 0.72;

  const chooseRoot = (role: string, name: string) => { setSteps([{ role, name }]); setSelected(null); };
  const drill = (role: string, name: string) => {
    if (steps.some(step => step.name === name)) return;
    setSteps(prior => [...prior, { role, name }]);
    setSelected(null);
  };

  return <aside id="computation-explorer" className="computation-explorer" aria-label="Expanded computation">
    <header className="explorer-header"><div><span className="eyebrow">EXPANDED IN CANVAS</span><strong>%{rootNode.name}</strong></div><button type="button" className="icon-button" aria-label="Close expanded computation" onClick={onClose}>×</button></header>
    <div className="explorer-breadcrumb"><button type="button" onClick={() => { setSteps([]); setSelected(null); }}>%{rootNode.name}</button>{steps.map((step, index) =>
      <span key={`${index}/${step.name}`}><span>›</span><button type="button" aria-current={index === steps.length - 1 ? 'page' : undefined} onClick={() => { setSteps(prior => prior.slice(0, index + 1)); setSelected(null); }}>%{step.name}</button></span>)}</div>
    <div className="explorer-tabs">{entries.map(([role, name]) => <button type="button" key={role} className={steps[0]?.role === role ? 'active' : ''} onClick={() => chooseRoot(role, name)}>{role.toUpperCase()}</button>)}</div>
    {computation && layout ? <>
      <div ref={scrollerRef} className="nested-scroller"><div className="nested-content" style={{ width: layout.width * scale, height: layout.height * scale }}><div className="nested-stage" style={{ width: layout.width, height: layout.height, transform: `scale(${scale})` }}>
        <svg width={layout.width} height={layout.height} aria-hidden="true">{computation.nodes.flatMap(node => node.operands.concat(node.controlPredecessors).map((source, index) => {
          const from = layout.positions.get(source), to = layout.positions.get(node.name);
          if (!from || !to) return null;
          return <path key={`${source}/${node.name}/${index}`} d={`M${from.x + NODE_WIDTH} ${from.y + NODE_HEIGHT / 2} C${from.x + NODE_WIDTH + 42} ${from.y + NODE_HEIGHT / 2},${to.x - 42} ${to.y + NODE_HEIGHT / 2},${to.x} ${to.y + NODE_HEIGHT / 2}`} className={`nested-edge${node.controlPredecessors.includes(source) ? ' control-edge' : ''}`} />;
        }))}</svg>
        {computation.nodes.map(node => {
          const point = layout.positions.get(node.name)!;
          return <button type="button" key={node.name} className={`nested-node ${nodeCategory(node)}${selected === node.name ? ' selected' : ''}`} style={{ left: point.x, top: point.y }} onClick={() => setSelected(node.name)}>
            <small>{node.op}{node.root ? ' · ROOT' : ''}</small><strong title={`%${node.name}`}>%{node.name}</strong><span>{nodeSummary(node)}</span>
          </button>;
        })}
      </div></div></div>
      {selectedNode && <div className="explorer-detail"><strong>%{selectedNode.name}</strong><pre>{selectedNode.raw}</pre>{Object.entries(selectedNode.calls).filter(([, name]) => module.byName.has(name)).map(([role, name]) =>
        <button type="button" key={role} onClick={() => drill(role, name)}>Expand {role} → %{name}</button>)}</div>}
      <button type="button" className="explorer-full" onClick={() => onOpenFull(computation.name)}>Open %{computation.name} as full graph ↗</button>
    </> : <p className="explorer-empty">Choose BODY or CONDITION to see its dependency graph.</p>}
  </aside>;
}
