import type { GuidePart, ResultGroup, TypeDetail, TypeSlot } from '../lib/types';

type Activate = (part: string | null, slot: string | null) => void;

interface TypeTreeProps {
  group: ResultGroup;
  parts: GuidePart[];
  activePart: string | null;
  activeSlot: string | null;
  onActivate: Activate;
}

// Class list for an element tied to an HLO part: `part-<key>` sets its colour, `active-part` highlights it.
export function partClass(base: string, key: string | null | undefined, active: boolean) {
  const keyClass = key ? ` part-${key}` : '';
  const activeClass = active ? ' active-part' : '';
  return `${base}${keyClass}${activeClass}`;
}

function DetailRows({
  details,
  activePart,
  highlighted,
  slot,
  onActivate
}: {
  details: TypeDetail[];
  activePart: string | null;
  highlighted: boolean;
  slot: string | null;
  onActivate: Activate;
}) {
  return (
    <>
      {details.map(detail => {
        const part = detail.key || slot;
        return (
          <div
            className={partClass('type-detail', detail.key, highlighted && activePart === part)}
            data-part={detail.key}
            key={detail.key || detail.label}
            onMouseEnter={() => onActivate(part, slot)}
            onMouseLeave={() => onActivate(null, null)}
          >
            <strong>{detail.label}</strong>
            <p>{detail.text}</p>
          </div>
        );
      })}
    </>
  );
}

function SlotItem({
  slot,
  explanation,
  activePart,
  activeSlot,
  onActivate
}: Omit<TypeTreeProps, 'group' | 'parts'> & { slot: TypeSlot; explanation: GuidePart | undefined }) {
  const active = activePart === slot.key || activeSlot === slot.key;
  const activateSlot = () => onActivate(slot.key, slot.key);
  const clear = () => onActivate(null, null);

  return (
    <details className={partClass('type-child', slot.key, active)} data-part={slot.key}>
      <summary onMouseEnter={activateSlot} onMouseLeave={clear}>
        <span>{slot.label}</span>
        <code>{slot.raw}</code>
      </summary>
      <div className="type-child-body">
        {explanation && (
          <p className="type-child-intro" onMouseEnter={activateSlot} onMouseLeave={clear}>
            {explanation.text}
          </p>
        )}
        <DetailRows
          details={slot.details}
          activePart={activePart}
          highlighted={activeSlot === slot.key}
          slot={slot.key}
          onActivate={onActivate}
        />
      </div>
    </details>
  );
}

export function TypeTree({ group, parts, activePart, activeSlot, onActivate }: TypeTreeProps) {
  const key = group.kind === 'tuple' ? 'tuple' : 'shape';
  const scalarOrArray = /\[\]/.test(group.rawType) ? 'scalar' : 'array';
  const kind = group.kind === 'tuple' ? `${group.slots.length}-element tuple` : scalarOrArray;

  return (
    <details className={partClass('type-tree', key, activePart === key)} data-part={key} open>
      <summary onMouseEnter={() => onActivate(key, null)} onMouseLeave={() => onActivate(null, null)}>
        <span className="type-tree-title">
          Result type <b>{kind}</b>
        </span>
        <code>{group.rawType}</code>
      </summary>
      <div className="type-tree-children">
        {group.kind === 'tuple' ? (
          group.slots.map(slot => (
            <SlotItem
              key={slot.key}
              slot={slot}
              explanation={parts.find(part => part.key === slot.key)}
              activePart={activePart}
              activeSlot={activeSlot}
              onActivate={onActivate}
            />
          ))
        ) : (
          <DetailRows details={group.details} activePart={activePart} highlighted slot={null} onActivate={onActivate} />
        )}
      </div>
    </details>
  );
}
