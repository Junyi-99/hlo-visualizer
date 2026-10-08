import { computationLinks } from '../lib/parser';
import { computationRole } from '../lib/graph-layout';
import type { HloModule } from '../lib/types';
import { Icon } from './Icon';

interface SidebarProps {
  module: HloModule | null;
  current: string | null;
  onOverview: () => void;
  onComputation: (name: string) => void;
}

const legend = ['Input', 'Compute', 'Transfer', 'Control', 'Fusion', 'Tuple'];

export function Sidebar({ module, current, onOverview, onComputation }: SidebarProps) {
  const links = module ? computationLinks(module) : [];
  const ordered = module ? [...module.computations].sort((a, b) => Number(b.entry) - Number(a.entry)) : [];
  const count = module?.computations.reduce((total, c) => total + c.nodes.length, 0) || 0;

  return <aside className="sidebar">
    <div className="brand"><div className="brand-mark"><Icon name="graph" size={18} /></div><div><strong>HLO Visualizer</strong><span>Explore and understand XLA HLO</span></div></div>
    <div className="sidebar-section module-section">
      <div className="eyebrow">MODULE</div>
      <h1>{module?.name || 'Loading…'}</h1>
      <div className="muted">{module && `${module.computations.length} ${module.computations.length === 1 ? 'computation' : 'computations'} · ${count} instructions`}</div>
    </div>
    <div className="sidebar-section">
      <div className="section-title"><span>Computations</span><span className="counter">{module?.computations.length || 0}</span></div>
      <nav aria-label="Computations" id="computation-list">
        <button type="button" className={`computation-item overview-item${current === null ? ' active' : ''}`} onClick={onOverview}>
          <span className="comp-icon"><Icon name="overview" /></span><span className="comp-copy"><strong>Overview</strong><small>COMPUTATION LINKS</small></span><span className="comp-arrow"><Icon name="chevron" size={14} /></span>
        </button>
        {ordered.map(c => {
          const role = computationRole(c, links);
          const icon = c.entry ? 'entry' : role === 'FUSION' ? 'fusion' : role.startsWith('WHILE') ? 'loop' : 'computation';
          return <button type="button" key={c.name} className={`computation-item${current === c.name ? ' active' : ''}`} onClick={() => onComputation(c.name)}>
            <span className="comp-icon"><Icon name={icon} /></span><span className="comp-copy"><strong title={`%${c.name}`}>%{c.name}</strong><small>{role} · {c.nodes.length} nodes</small></span><span className="comp-arrow"><Icon name="chevron" size={14} /></span>
          </button>;
        })}
      </nav>
    </div>
    <div className="sidebar-footer"><div className="legend-title">NODE TYPES</div><div className="legend">
      {legend.map(label => <span key={label}><i className={`dot ${label.toLowerCase()}`} />{label}</span>)}
    </div><p>Click a node to trace its dependencies. Double-click empty space to clear.</p></div>
  </aside>;
}
