import { FUSION_OUTPUT_PREVIEW_LIMIT, fusionLessonData } from '../lib/fusion-lesson';
import type { Computation, HloModule, HloNode } from '../lib/types';
import './FusionLesson.css';

interface FusionLessonProps {
  module: HloModule;
  computation: Computation;
  node: HloNode;
  onNode: (name: string) => void;
  onNavigate: (computation: string, node: string) => void;
}

export function FusionLesson({ module, computation, node, onNode, onNavigate }: FusionLessonProps) {
  const lesson = fusionLessonData(module, computation, node);
  if (!lesson) return null;
  const body = lesson.body;
  const internalButton = (item: HloNode) => (
    <button type="button" onClick={() => body && onNavigate(body.name, item.name)} title={`Open %${item.name} in %${body?.name}`}>
      %{item.name}
    </button>
  );

  return (
    <details className="inspector-section fusion-lesson">
      <summary>Inside this fusion</summary>
      {!body ? (
        <p>
          The called computation {lesson.name ? `%${lesson.name} ` : ''}is not present in this file. Its internal structure cannot be shown.
        </p>
      ) : (
        <>
          <p>
            <strong>From this HLO:</strong> {body.nodes.length} instructions in <code>%{body.name}</code>, with {lesson.inputs.length} input
            parameters. The operation counts below describe this computation only; nested calls have their own bodies.
          </p>
          <div className="fusion-lesson-ops" aria-label="Operations in this fusion">
            {lesson.operations.map(([op, count]) => (
              <span key={op}>
                {op} <b>×{count}</b>
              </span>
            ))}
          </div>
          <h5>Inputs cross the boundary</h5>
          <p>Operand position maps to parameter(n), even when parameters appear in a different order in the file.</p>
          <ul className="fusion-lesson-mappings">
            {lesson.inputs.map(input => (
              <li key={input.parameter.name}>
                <span className="fusion-lesson-position">{input.index === null ? 'Unknown position' : `Operand ${input.index}`}</span>
                <div>
                  {input.operand ? (
                    <button type="button" onClick={() => onNode(input.operand!.name)} title="Select the external operand">
                      %{input.operand.name}
                    </button>
                  ) : (
                    <span>{input.operandName ? `%${input.operandName} (missing)` : 'No matching operand'}</span>
                  )}
                  <span aria-hidden="true"> → </span>
                  {internalButton(input.parameter)}
                </div>
              </li>
            ))}
          </ul>
          {!lesson.inputs.length && <p>No parameter instructions are present.</p>}
          <h5>ROOT returns to the caller</h5>
          {lesson.root ? (
            <>
              <p>
                {internalButton(lesson.root)} defines the fusion result
                {lesson.root.op === 'tuple' ? '; a preview of its returned tuple elements is shown below.' : '.'}
              </p>
              {lesson.root.op === 'tuple' && (
                <ul className="fusion-lesson-mappings">
                  {lesson.outputs.map(output => (
                    <li key={output.path.join('.')}>
                      <span className="fusion-lesson-position">Result [{output.path.join(', ')}]</span>
                      {output.node ? internalButton(output.node) : <span>%{output.name} (missing)</span>}
                      {output.unexpanded && <span> (tuple expansion stopped)</span>}
                    </li>
                  ))}
                </ul>
              )}
              {lesson.outputsTruncated && (
                <p className="fusion-lesson-note">
                  Output preview truncated: at most {FUSION_OUTPUT_PREVIEW_LIMIT} paths are shown. Deep, repeated or cyclic tuple branches
                  have limited expansion. Internal-value classification still includes every reachable tuple element.
                </p>
              )}
            </>
          ) : (
            <p>No ROOT instruction was found, so the returned values cannot be identified.</p>
          )}
          {lesson.root && (
            <details className="fusion-lesson-internals">
              <summary>{lesson.internal.length} internal values are not directly returned</summary>
              <p>
                These values belong to the fusion body and do not form separate outputs at its HLO boundary. Lowering determines their
                storage; local memory and temporary buffers can still be needed.
              </p>
              <div className="fusion-lesson-node-list">
                {lesson.internal.map(item => (
                  <span key={item.name}>{internalButton(item)}</span>
                ))}
              </div>
            </details>
          )}
        </>
      )}
      <details className="fusion-lesson-background">
        <summary>Why fuse operations?</summary>
        <p>
          Fusion can reduce memory traffic by keeping intermediate results within a compiled region. One fusion can contain a branching
          graph and lower to many machine instructions; it is not a single hardware instruction or bundle.
        </p>
        <p>
          Common constraints include local memory capacity, supported operation and layout combinations, duplicated work, and dependency
          legality. This file shows the resulting structure, not why another pair was left unfused. That decision needs compiler logs.
        </p>
        <div className="fusion-lesson-links">
          <a href="https://docs.cloud.google.com/tpu/docs/performance-guide" target="_blank" rel="noreferrer">
            TPU performance guide ↗
          </a>
          <a href="https://gh.evko.io/crucible-notes/libtpu/compiler/fusion-patterns.html" target="_blank" rel="noreferrer">
            Fusion research notes ↗
          </a>
        </div>
        <p className="fusion-lesson-note">Research notes describe libtpu 0.0.40 internals; compiler versions and hardware can differ.</p>
      </details>
    </details>
  );
}
