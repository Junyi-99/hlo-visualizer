import type { HloModule } from '../lib/types';
import { Icon, type IconName } from './Icon';
import { useComputationRoles } from '../hooks/useComputationRoles';

interface SidebarProps {
  module: HloModule | null;
  current: string | null;
  onOverview: () => void;
  onComputation: (name: string) => void;
}

const legend = ['Input', 'Compute', 'Transfer', 'Control', 'Fusion', 'Tuple'];

function roleIcon(role: string): IconName {
  if (role === 'ENTRY') return 'entry';
  if (role === 'FUSION') return 'fusion';
  if (role.startsWith('WHILE')) return 'loop';
  return 'computation';
}

interface ComputationItemProps {
  active: boolean;
  overview?: boolean;
  icon: IconName;
  title: string;
  detail: string;
  onClick: () => void;
}

function ComputationItem({ active, overview = false, icon, title, detail, onClick }: ComputationItemProps) {
  const className = ['computation-item', overview && 'overview-item', active && 'active'].filter(Boolean).join(' ');

  return (
    <button type="button" className={className} onClick={onClick}>
      <span className="comp-icon">
        <Icon name={icon} />
      </span>
      <span className="comp-copy">
        <strong title={overview ? undefined : title}>{title}</strong>
        <small>{detail}</small>
      </span>
      <span className="comp-arrow">
        <Icon name="chevron" size={14} />
      </span>
    </button>
  );
}

export function Sidebar({ module, current, onOverview, onComputation }: SidebarProps) {
  const roles = useComputationRoles(module);

  const computations = module?.computations ?? [];
  const entryFirst = [...computations].sort((a, b) => Number(b.entry) - Number(a.entry));
  const instructionCount = computations.reduce((total, c) => total + c.nodes.length, 0);
  const summary =
    module && `${computations.length} ${computations.length === 1 ? 'computation' : 'computations'} · ${instructionCount} instructions`;

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">
          <Icon name="graph" size={18} />
        </div>
        <div>
          <strong>HLO Visualizer</strong>
          <span>Explore and understand XLA HLO</span>
        </div>
      </div>

      <div className="sidebar-section module-section">
        <div className="eyebrow">MODULE</div>
        <h1>{module?.name || 'Loading…'}</h1>
        <div className="muted">{summary}</div>
      </div>

      <div className="sidebar-section">
        <div className="section-title">
          <span>Computations</span>
          <span className="counter">{computations.length}</span>
        </div>
        <nav aria-label="Computations" id="computation-list">
          <ComputationItem
            active={current === null}
            overview
            icon="overview"
            title="Overview"
            detail="COMPUTATION LINKS"
            onClick={onOverview}
          />

          {entryFirst.map(c => {
            const role = roles.get(c.name) ?? '';
            return (
              <ComputationItem
                key={c.name}
                active={current === c.name}
                icon={roleIcon(role)}
                title={`%${c.name}`}
                detail={`${role} · ${c.nodes.length} nodes`}
                onClick={() => onComputation(c.name)}
              />
            );
          })}
        </nav>
      </div>

      <div className="sidebar-footer">
        <div className="legend-title">NODE TYPES</div>
        <div className="legend">
          {legend.map(label => (
            <span key={label}>
              <i className={`dot ${label.toLowerCase()}`} />
              {label}
            </span>
          ))}
        </div>
        <p>Click a node to trace its dependencies. Double-click empty space to clear.</p>
      </div>
    </aside>
  );
}
