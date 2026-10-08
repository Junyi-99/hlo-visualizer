import type { GuidePart, ResultGroup, TypeDetail } from '../lib/types';

interface TypeTreeProps {
  group: ResultGroup;
  parts: GuidePart[];
  activePart: string | null;
  activeSlot: string | null;
  onActivate: (part: string | null, slot: string | null) => void;
}

function DetailRows({ details, activePart, highlighted, slot, onActivate }: { details: TypeDetail[]; activePart: string | null; highlighted: boolean; slot: string | null; onActivate: TypeTreeProps['onActivate'] }) {
  return <>{details.map(detail => <div className={`type-detail${detail.key ? ` part-${detail.key}` : ''}${highlighted && activePart === (detail.key || slot) ? ' active-part' : ''}`} data-part={detail.key} key={detail.key || detail.label}
    onMouseEnter={() => onActivate(detail.key || slot, slot)} onMouseLeave={() => onActivate(null, null)}>
    <strong>{detail.label}</strong><p>{detail.text}</p>
  </div>)}</>;
}

export function TypeTree({ group, parts, activePart, activeSlot, onActivate }: TypeTreeProps) {
  const tuple = group.kind === 'tuple';
  const key = tuple ? 'tuple' : 'shape';
  const kind = tuple ? `${group.slots.length}-element tuple` : /\[\]/.test(group.rawType) ? 'scalar' : 'array';
  return <details className={`type-tree part-${key}${activePart === key ? ' active-part' : ''}`} data-part={key} open>
    <summary onMouseEnter={() => onActivate(key, null)} onMouseLeave={() => onActivate(null, null)}><span className="type-tree-title">Result type <b>{kind}</b></span><code>{group.rawType}</code></summary>
    <div className="type-tree-children">
      {group.kind === 'tuple' ? group.slots.map(slot => {
        const explanation = parts.find(part => part.key === slot.key);
        return <details key={slot.key} className={`type-child part-${slot.key}${activePart === slot.key || activeSlot === slot.key ? ' active-part' : ''}`} data-part={slot.key}>
          <summary onMouseEnter={() => onActivate(slot.key, slot.key)} onMouseLeave={() => onActivate(null, null)}><span>{slot.label}</span><code>{slot.raw}</code></summary>
          <div className="type-child-body">{explanation && <p className="type-child-intro" onMouseEnter={() => onActivate(slot.key, slot.key)} onMouseLeave={() => onActivate(null, null)}>{explanation.text}</p>}<DetailRows details={slot.details} activePart={activePart} highlighted={activeSlot === slot.key} slot={slot.key} onActivate={onActivate} /></div>
        </details>;
      }) : <DetailRows details={group.details} activePart={activePart} highlighted slot={null} onActivate={onActivate} />}
    </div>
  </details>;
}
