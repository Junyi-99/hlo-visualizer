import { useEffect, useMemo, useRef, useState } from 'react';
import { instructionGuide, layoutDiagram } from '../lib/instruction-guide';
import { extractHloMetadata } from '../lib/metadata.ts';
import { nodeCategory, nodeSummary } from '../lib/parser';
import type { Computation, HloModule, HloNode } from '../lib/types';
import { TypeTree } from './TypeTree';

interface InspectorProps {
  module: HloModule;
  computation: Computation | null;
  node: HloNode | null;
  upstreamCount: number;
  downstreamCount: number;
  onClose: () => void;
  onNode: (name: string) => void;
  onComputation: (name: string) => void;
}

function ReferenceList({ names, onNode }: { names: string[]; onNode: (name: string) => void }) {
  return names.length ? <>{names.map(name => <button type="button" className="reference" key={name} onClick={() => onNode(name)}><span>%{name}</span><span>↗</span></button>)}</> : <p className="none">None in this computation</p>;
}

export function Inspector({ module, computation, node, upstreamCount, downstreamCount, onClose, onNode, onComputation }: InspectorProps) {
  return <aside className={`inspector${node ? ' inspector-visible' : ''}`}>
    <div className="inspector-header"><div><div className="eyebrow">INSPECTOR</div><h2>Node details</h2></div><button className="icon-button" type="button" aria-label="Clear selection" onClick={onClose}>×</button></div>
    <div className="inspector-content overscroll-contain">{node && computation ? <InstructionDetails key={node.id} {...{ module, computation, node, upstreamCount, downstreamCount, onNode, onComputation }} /> :
      <div className="empty-inspector"><div className="empty-icon">◇</div><strong>Explore the graph</strong><p>Select any node to see its inputs, consumers, raw HLO, and linked computations.</p></div>}
    </div>
  </aside>;
}

function InstructionDetails({ module, computation, node, upstreamCount, downstreamCount, onNode, onComputation }: Omit<InspectorProps, 'onClose'> & { node: HloNode; computation: Computation }) {
  const guide = useMemo(() => instructionGuide(node, { computation, module }), [node, computation, module]);
  const [activePart, setActivePart] = useState<string | null>(null);
  const [activeSlot, setActiveSlot] = useState<string | null>(null);
  const [showFullInstruction, setShowFullInstruction] = useState(false);
  const detailsRef = useRef<HTMLDivElement>(null);
  const codeRef = useRef<HTMLPreElement>(null);
  const group = guide.resultGroup;
  const metadata = extractHloMetadata(node.raw);
  const typeKeys = new Set(['tuple', 'shape', 'order', 'tile', 'space', 'dest', 'source', 'context', ...(group?.kind === 'tuple' ? group.slots.map(slot => slot.key) : [])]);
  const explanations = guide.parts.filter(part => !typeKeys.has(part.key) && !(metadata && part.key === 'metadata'));
  const diagram = layoutDiagram(node);
  const semanticsUrl = node.op === 'copy-start' ? 'https://openxla.org/xla/operation_semantics#copy' : node.type.startsWith('(') ? 'https://openxla.org/xla/operation_semantics#tuple' : 'https://openxla.org/xla/operation_semantics';
  const semanticsLabel = node.op === 'copy-start' ? 'Copy' : node.type.startsWith('(') ? 'Tuple' : 'Operations';
  const displayType = node.type.startsWith('(') ? nodeSummary(node) : node.type;
  const displayedHtml = showFullInstruction ? guide.html : guide.compactHtml || guide.html;

  useEffect(() => {
    const elements = [...(codeRef.current?.querySelectorAll<HTMLElement>('[data-part]') || [])];
    const hasMatchingPart = elements.some(element => element.dataset.part === activePart && (!activeSlot || element.closest<HTMLElement>('.hlo-slot')?.dataset.part === activeSlot));
    elements.forEach(element => {
      const key = element.dataset.part;
      const slot = element.closest<HTMLElement>('.hlo-slot')?.dataset.part;
      const matchesPart = key === activePart && (!activeSlot || slot === activeSlot || key === activeSlot);
      const matchesSlot = element.classList.contains('hlo-slot') && key === activeSlot;
      const fallbackType = !hasMatchingPart && !activeSlot && activePart === 'space' && element.classList.contains('hlo-type');
      element.classList.toggle('active-part', !!(matchesPart || matchesSlot || fallbackType));
    });
  }, [activePart, activeSlot, displayedHtml]);

  const activate = (part: string | null, slot: string | null = null) => {
    setActivePart(part);
    setActiveSlot(slot);
  };

  const codeTarget = (target: EventTarget) => {
    const part = target instanceof Element ? target.closest<HTMLElement>('[data-part]') : null;
    return { key: part?.dataset.part || null, slot: part?.closest<HTMLElement>('.hlo-slot')?.dataset.part || null };
  };
  const activateFromCode = (event: React.MouseEvent<HTMLPreElement>) => {
    const { key, slot } = codeTarget(event.target);
    if (!key) return;
    activate(key, slot);
    const container = detailsRef.current;
    const slotItem = slot ? container?.querySelector<HTMLDetailsElement>(`.type-child[data-part="${slot}"]`) : null;
    const item = (slotItem || container?.querySelector('.type-tree'))?.querySelector<HTMLElement>(`.type-detail[data-part="${key}"]`) ||
      slotItem || container?.querySelector<HTMLElement>(`.type-child[data-part="${key}"]`) ||
      (typeKeys.has(key) ? container?.querySelector<HTMLElement>('.type-tree') : null) ||
      container?.querySelector<HTMLElement>(`.guide-row[data-part="${key}"]`) ||
      container?.querySelector<HTMLElement>(`.metadata-section[data-part="${key}"]`);
    if (!item) return;
    const tree = item.closest<HTMLDetailsElement>('.type-tree');
    if (tree) tree.open = true;
    if (slotItem) slotItem.open = true;
    if (item instanceof HTMLDetailsElement) item.open = true;
    item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  };

  return <div ref={detailsRef}>
    <div className="inspector-title"><span className={`type-badge ${nodeCategory(node)}`}>{node.op}</span>{node.root && <span className="root-pill">ROOT</span>}<h3>%{node.name}</h3><div className="muted">{displayType}</div></div>
    <div className="metric-row"><div><strong>{upstreamCount}</strong><span>Upstream</span></div><div><strong>{downstreamCount}</strong><span>Downstream</span></div><div><strong>{node.line}</strong><span>Source line</span></div></div>
    <div className="inspector-section instruction-section"><h4>HLO instruction</h4>
      <pre ref={codeRef} className="hlo-code" onClick={activateFromCode} onMouseOver={event => { const { key, slot } = codeTarget(event.target); activate(key, slot); }} onMouseLeave={() => activate(null)} dangerouslySetInnerHTML={{ __html: displayedHtml }} />
      {guide.compactHtml && <button className="instruction-toggle" type="button" onClick={() => setShowFullInstruction(value => !value)}>{showFullInstruction ? '收起编码配置' : '展开完整 HLO 指令'}</button>}
      {node.op === 'copy-start' && diagram && <div dangerouslySetInnerHTML={{ __html: diagram }} />}
      <div className="guide-heading">逐段解读 <span>悬停双向高亮 · 点按指令定位说明</span></div>
      <div className="guide-list">
        {group && <TypeTree group={group} parts={guide.parts} activePart={activePart} activeSlot={activeSlot} onActivate={activate} />}
        {explanations.map(part => <div key={part.key} className={`guide-row part-${part.key}${activePart === part.key ? ' active-part' : ''}`} data-part={part.key} tabIndex={0} onMouseEnter={() => activate(part.key)} onMouseLeave={() => activate(null)} onFocus={() => activate(part.key)} onBlur={() => activate(null)}>
          <span className="guide-swatch" /><div><strong>{part.label}</strong><p>{part.text}</p></div>
        </div>)}
      </div>
      {node.op !== 'copy-start' && diagram && <div dangerouslySetInnerHTML={{ __html: diagram }} />}
      <div className="guide-sources">参考：<a href="https://jax-ml.github.io/scaling-book/profiling/#how-to-read-an-xla-op" target="_blank" rel="noreferrer">How to read an XLA op</a> · <a href="https://openxla.org/xla/shapes" target="_blank" rel="noreferrer">Shapes and layout</a> · <a href={semanticsUrl} target="_blank" rel="noreferrer">{semanticsLabel}</a></div>
    </div>
    {metadata && <div className={`inspector-section metadata-section part-metadata${activePart === 'metadata' ? ' active-part' : ''}`} data-part="metadata" tabIndex={0}
      onMouseEnter={() => activate('metadata')} onMouseLeave={() => activate(null)} onFocus={() => activate('metadata')} onBlur={() => activate(null)}>
      <h4>HLO metadata</h4>
      <p className="metadata-intro">记录源操作和源码位置。</p>
      <div className="metadata-fields">{metadata.fields.map(field => <div className="metadata-field" key={field.name}>
        <span>{field.name}</span><code>{field.value}</code>
      </div>)}</div>
      <details className="metadata-original"><summary>查看原始 metadata</summary><pre>{metadata.raw}</pre></details>
    </div>}
    <div className="inspector-section"><h4>Direct inputs <span>{node.operands.length}</span></h4><ReferenceList names={node.operands} onNode={onNode} /></div>
    <div className="inspector-section"><h4>Direct consumers <span>{node.users.length}</span></h4><ReferenceList names={node.users} onNode={onNode} /></div>
    {!!node.controlPredecessors.length && <div className="inspector-section"><h4>Control predecessors <span>{node.controlPredecessors.length}</span></h4><ReferenceList names={node.controlPredecessors} onNode={onNode} /></div>}
    {!!node.controlSuccessors.length && <div className="inspector-section"><h4>Control successors <span>{node.controlSuccessors.length}</span></h4><ReferenceList names={node.controlSuccessors} onNode={onNode} /></div>}
    {!!Object.keys(node.calls).length && <div className="inspector-section"><h4>Open computation</h4>{Object.entries(node.calls).map(([role, name]) => <button type="button" className="jump-link" key={role} onClick={() => onComputation(name)}><span><small>{role.toUpperCase()}</small><strong>%{name}</strong></span><span>↗</span></button>)}</div>}
  </div>;
}
