import { useEffect, useMemo, useRef, useState } from 'react';
import { linkLabel } from '../lib/graph-layout';
import { calledComputations, nodeCategory } from '../lib/parser';
import type { HloModule, HloNode } from '../lib/types';
import { copyGroupCaption, groupCopyPairs } from '../lib/copy-grouping';
import { useGraphLayouts } from '../hooks/useGraphLayouts';
import { edgePath } from '../lib/graph-routing';
import { Icon } from './Icon';

const NESTED_SCALE = 0.72;
const emptyHeights: Record<string, number> = {};

interface Step {
  name: string;
  role: string;
}

interface ComputationExplorerProps {
  autoGroup: boolean;
  module: HloModule;
  rootNode: HloNode;
  onClose: () => void;
  onOpenFull: (name: string) => void;
}

export function ComputationExplorer({ autoGroup, module, rootNode, onClose, onOpenFull }: ComputationExplorerProps) {
  const entries = calledComputations(module, rootNode).sort(([left], [right]) => Number(right === 'body') - Number(left === 'body'));

  const [steps, setSteps] = useState<Step[]>(() => (entries.length ? [{ role: entries[0][0], name: entries[0][1] }] : []));
  const [selected, setSelected] = useState<string | null>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);

  const active = steps.at(-1);
  const original = active ? module.byName.get(active.name) : null;
  const grouping = useMemo(() => (original ? groupCopyPairs(original, autoGroup) : null), [original, autoGroup]);
  const computation = grouping?.computation;
  const layout = useGraphLayouts(module, computation, emptyHeights).layouts.horizontal;
  const selectedGroup = selected ? grouping?.groups.get(selected) : null;
  const selectedNode = selected && computation?.byName.get(selected);

  useEffect(() => {
    if (computation) scrollerRef.current?.scrollTo(0, 0);
  }, [computation]);

  const showSteps = (next: Step[]) => {
    setSteps(next);
    setSelected(null);
  };

  const drill = (role: string, name: string) => {
    if (!steps.some(step => step.name === name)) showSteps([...steps, { role, name }]);
  };

  return (
    <aside id="computation-explorer" className="computation-explorer" aria-label="Expanded computation">
      <header className="explorer-header">
        <div>
          <span className="eyebrow">EXPANDED IN CANVAS</span>
          <strong>%{rootNode.name}</strong>
        </div>
        <button type="button" className="icon-button" aria-label="Close expanded computation" onClick={onClose}>
          <Icon name="close" />
        </button>
      </header>

      <div className="explorer-breadcrumb">
        <button type="button" onClick={() => showSteps([])}>
          %{rootNode.name}
        </button>
        {steps.map((step, index) => (
          <span key={`${index}/${step.name}`}>
            <span>›</span>
            <button
              type="button"
              aria-current={index === steps.length - 1 ? 'page' : undefined}
              onClick={() => showSteps(steps.slice(0, index + 1))}
            >
              %{step.name}
            </button>
          </span>
        ))}
      </div>

      <div className="explorer-tabs">
        {entries.map(([role, name]) => (
          <button type="button" key={role} className={steps[0]?.role === role ? 'active' : ''} onClick={() => showSteps([{ role, name }])}>
            {linkLabel(role)}
          </button>
        ))}
      </div>

      {computation ? (
        <>
          <div ref={scrollerRef} className="nested-scroller">
            <div className="nested-content" style={{ width: layout.width * NESTED_SCALE, height: layout.height * NESTED_SCALE }}>
              <div className="nested-stage" style={{ width: layout.width, height: layout.height, transform: `scale(${NESTED_SCALE})` }}>
                <svg width={layout.width} height={layout.height} aria-hidden="true">
                  {layout.edges.map(edge => (
                    <path
                      key={edge.id}
                      d={edgePath(edge.points, 'horizontal', 0)}
                      className={edge.control ? 'nested-edge control-edge' : 'nested-edge'}
                    />
                  ))}
                </svg>
                {computation.nodes.map(node => {
                  const group = grouping?.groups.get(node.name);
                  const point = layout.positions.get(node.name)!;
                  const { label, detail } = copyGroupCaption(node, group);
                  const className = ['nested-node', nodeCategory(node), group && 'copy-group', selected === node.name && 'selected']
                    .filter(Boolean)
                    .join(' ');

                  return (
                    <button
                      type="button"
                      key={node.name}
                      className={className}
                      style={{ left: point.x, top: point.y }}
                      onClick={() => setSelected(node.name)}
                    >
                      <small>
                        {node.op}
                        {node.root ? ' · ROOT' : ''}
                      </small>
                      <strong title={`%${node.name}`}>{label}</strong>
                      <span>{detail}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {selectedNode && (
            <div className="explorer-detail">
              <strong>%{selectedNode.name}</strong>
              <pre>{selectedGroup ? `${selectedGroup.start.raw}\n${selectedGroup.done.raw}` : selectedNode.raw}</pre>
              {calledComputations(module, selectedNode).map(([role, name]) => (
                <button type="button" key={role} onClick={() => drill(role, name)}>
                  Expand {role} → %{name}
                </button>
              ))}
            </div>
          )}

          <button type="button" className="explorer-full" onClick={() => onOpenFull(computation.name)}>
            Open %{computation.name} as full graph ↗
          </button>
        </>
      ) : (
        <p className="explorer-empty">Choose a called computation above to see its dependency graph.</p>
      )}
    </aside>
  );
}
