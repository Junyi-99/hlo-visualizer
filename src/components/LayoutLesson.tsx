import { useMemo, useState } from 'react';
import type { HloNode } from '../lib/types';
import {
  layoutCoordinates,
  layoutGeometry,
  layoutSlots,
  logicalAtOffset,
  parseLayout,
  physicalOffset,
  type LayoutSpec
} from '../lib/layout-lesson';
import './LayoutLesson.css';

const vector = (values: number[]) => `[${values.join(', ')}]`;
const count = (value: number) => value.toLocaleString('en-US');

export function LayoutLesson({ node }: { node: HloNode }) {
  return <LayoutLessonContent key={`${node.id}:${node.type}`} node={node} />;
}

function LayoutLessonContent({ node }: { node: HloNode }) {
  const slots = useMemo(() => layoutSlots(node.type), [node.type]);
  const [selected, setSelected] = useState('');
  const slot = slots.find(item => item.path === selected) ?? slots[0];
  if (!slot) return null;
  const parsed = parseLayout(slot.raw);
  return (
    <details className="layout-lesson">
      <summary>Explore array layout</summary>
      <div className="layout-lesson-body">
        {slots.length > 1 && (
          <label className="layout-lesson-slot">
            Array result
            <select value={slot.path} onChange={event => setSelected(event.target.value)}>
              {slots.map(item => (
                <option key={item.path} value={item.path}>
                  Slot {item.path}: {item.raw}
                </option>
              ))}
            </select>
          </label>
        )}
        <code className="layout-lesson-type">{slot.raw}</code>
        {parsed.spec ? <LayoutExplorer key={`${node.id}:${slot.path}:${slot.raw}`} spec={parsed.spec} /> : <p>{parsed.reason}</p>}
        <a href="https://openxla.org/xla/tiled_layout" target="_blank" rel="noreferrer">
          OpenXLA layout reference ↗
        </a>
      </div>
    </details>
  );
}

function LayoutExplorer({ spec }: { spec: LayoutSpec }) {
  const [position, setPosition] = useState(spec.dimensions.map(() => 0));
  const geometry = layoutGeometry(spec);
  const selected = position.map((c, i) => Math.min(c, spec.dimensions[i] - 1));
  if (!geometry) return <p>This layout exceeds the explorer’s exact integer range.</p>;
  const offset = physicalOffset(spec, selected, geometry);
  const coordinateStages = layoutCoordinates(spec, selected);
  const columnsAxis = spec.dimensions.length - 1;
  const rowsAxis = columnsAxis - 1;
  const rowStart = rowsAxis < 0 ? 0 : Math.floor(selected[rowsAxis] / 4) * 4;
  const columnStart = Math.floor(selected[columnsAxis] / 8) * 8;
  const rowCount = rowsAxis < 0 ? 1 : Math.min(4, spec.dimensions[rowsAxis] - rowStart);
  const columnCount = Math.min(8, spec.dimensions[columnsAxis] - columnStart);
  const logicalCells = Array.from({ length: rowCount * columnCount }, (_, i) => {
    const coord = [...selected];
    coord[columnsAxis] = columnStart + (i % columnCount);
    if (rowsAxis >= 0) coord[rowsAxis] = rowStart + Math.floor(i / columnCount);
    return { coord, offset: physicalOffset(spec, coord, geometry) };
  });
  const physicalStart = Math.floor(offset / 32) * 32;
  const physicalCells = Array.from({ length: Math.min(32, geometry.storageElements - physicalStart) }, (_, i) => {
    const cellOffset = physicalStart + i;
    return { offset: cellOffset, coord: logicalAtOffset(spec, cellOffset, geometry) };
  });
  const packed = spec.elementType === 'bf16' && spec.tiles.at(-1)?.join(',') === '2,1';
  const changePosition = (index: number, value: number) => {
    if (!Number.isFinite(value)) return;
    setPosition(selected.map((c, i) => (i === index ? Math.max(0, Math.min(spec.dimensions[i] - 1, Math.floor(value))) : c)));
  };

  return (
    <>
      <p>
        Dimension {spec.minorToMajor[0]} is most minor: it changes fastest before tiling. Numbers below are element offsets, not byte
        addresses.
      </p>
      <div className="layout-lesson-stats">
        <span>
          <strong>{count(geometry.logicalElements)}</strong> logical elements
        </span>
        <span>
          <strong>{count(geometry.storageElements)}</strong> layout slots
        </span>
        <span>
          <strong>{count(geometry.storageElements - geometry.logicalElements)}</strong> padding slots
        </span>
        <span>
          <strong>{((100 * geometry.logicalElements) / geometry.storageElements).toFixed(1)}%</strong> layout fill
        </span>
      </div>
      <p className="layout-lesson-note">Derived from the layout.</p>
      <div className="layout-lesson-inputs">
        {selected.map((value, i) => (
          <label key={i}>
            Index d{i}
            <input
              aria-label={`Logical index dimension ${i}`}
              type="number"
              min={0}
              max={spec.dimensions[i] - 1}
              value={value}
              onChange={event => changePosition(i, Number(event.target.value))}
            />
          </label>
        ))}
      </div>
      <div className="layout-lesson-selection" aria-live="polite">
        Logical {vector(selected)} → offset <strong>{count(offset)}</strong>
      </div>

      <div className="layout-lesson-view">
        <h4>1 · Logical array</h4>
        <p>
          {rowsAxis < 0 ? 'One-dimensional slice.' : `Rows d${rowsAxis}, columns d${columnsAxis}.`}{' '}
          {spec.dimensions.length > 2 ? 'Other indices stay at the values above.' : ''} Hover, focus or tap a cell to trace it.
        </p>
        <div className="layout-lesson-grid" style={{ gridTemplateColumns: `repeat(${columnCount}, minmax(0, 1fr))` }}>
          {logicalCells.map(cell => (
            <button
              type="button"
              key={cell.coord.join(',')}
              className={cell.offset === offset ? 'is-selected' : ''}
              aria-label={`Logical ${vector(cell.coord)}, element offset ${cell.offset}`}
              aria-pressed={cell.offset === offset}
              title={`${vector(cell.coord)} → offset ${cell.offset}`}
              onPointerEnter={event => {
                if (event.pointerType === 'mouse') setPosition(cell.coord);
              }}
              onFocus={() => setPosition(cell.coord)}
              onClick={() => setPosition(cell.coord)}
            >
              {cell.offset}
            </button>
          ))}
        </div>
        <p className="layout-lesson-note">
          Showing {rowsAxis < 0 ? '' : `rows ${rowStart}–${rowStart + rowCount - 1}, `}columns {columnStart}–{columnStart + columnCount - 1}{' '}
          of shape {vector(spec.dimensions)}.
        </p>
      </div>

      <div className="layout-lesson-view">
        <h4>2 · Inside the tiles</h4>
        {spec.tiles.length ? (
          spec.tiles.map((tile, level) => {
            const previous = coordinateStages[level];
            const start = previous.length - tile.length;
            const tileIndex = tile.map((size, i) => Math.floor(previous[start + i] / size));
            const within = tile.map((size, i) => previous[start + i] % size);
            return (
              <div className="layout-lesson-tile" key={level}>
                <strong>
                  Level {level + 1} · {tile.join(' × ')}
                </strong>
                <span>
                  Tile index <code>{vector(tileIndex)}</code>
                </span>
                <span>
                  Position inside <code className="layout-lesson-coordinate">{vector(within)}</code>
                </span>
              </div>
            );
          })
        ) : (
          <p>No tiles are recorded. Dimension order alone determines the linear offsets.</p>
        )}
        {spec.tiles.length > 0 && (
          <p className="layout-lesson-note">
            Tile coordinates follow physical major-to-minor order. Each level acts on the trailing axes produced by the preceding level.
          </p>
        )}
        {packed && (
          <p className="layout-lesson-packing">
            BF16 packing: the final (2,1) tile puts two 16-bit elements together. Selected element: 32-bit word{' '}
            {count(Math.floor(offset / 2))}, half {offset % 2} (in storage order).
          </p>
        )}
      </div>

      <div className="layout-lesson-view">
        <h4>3 · Linear memory order</h4>
        <div className="layout-lesson-grid layout-lesson-memory">
          {physicalCells.map(cell => (
            <button
              type="button"
              key={cell.offset}
              className={`${cell.offset === offset ? 'is-selected' : ''} ${cell.coord === null ? 'is-padding' : ''} ${packed && Math.floor(cell.offset / 2) === Math.floor(offset / 2) ? 'is-pair' : ''}`}
              aria-label={`Offset ${cell.offset}: ${cell.coord ? `logical ${vector(cell.coord)}` : 'padding'}`}
              aria-pressed={cell.offset === offset}
              title={`Offset ${cell.offset}: ${cell.coord ? vector(cell.coord) : 'padding'}`}
              disabled={!cell.coord}
              onPointerEnter={event => {
                if (event.pointerType === 'mouse' && cell.coord) setPosition(cell.coord);
              }}
              onFocus={() => {
                if (cell.coord) setPosition(cell.coord);
              }}
              onClick={() => {
                if (cell.coord) setPosition(cell.coord);
              }}
            >
              {cell.coord ? cell.offset : 'pad'}
            </button>
          ))}
        </div>
        <p className="layout-lesson-note">
          Offsets {physicalStart}–{physicalStart + physicalCells.length - 1}; up to 32 slots around the selection. Shading marks padding
          {packed ? '; an outline groups the selected BF16 pair' : ''}.
        </p>
      </div>
      <p className="layout-lesson-note">
        A memory tile describes storage. Vector register shape and MXU matrix dimensions describe hardware.
      </p>
    </>
  );
}
