export interface HloNode {
  id: string;
  name: string;
  op: string;
  type: string;
  operands: string[];
  controlPredecessors: string[];
  controlSuccessors: string[];
  calls: Record<string, string>;
  index: number | null;
  kind: string | null;
  direction: string | null;
  root: boolean;
  raw: string;
  line: number;
  users: string[];
}

export interface Computation {
  name: string;
  entry: boolean;
  nodes: HloNode[];
  byName: Map<string, HloNode>;
  header: string;
}

export interface HloModule {
  name: string;
  computations: Computation[];
  byName: Map<string, Computation>;
  warnings: string[];
}

export interface ComputationLink {
  from: string;
  to: string;
  role: string;
  via: string;
  op: string;
}

export interface GuidePart {
  key: string;
  label: string;
  text: string;
  raw?: string;
  details?: TypeDetail[];
}

export interface TypeDetail {
  key?: string;
  label: string;
  text: string;
}

export interface TypeSlot {
  key: string;
  label: string;
  raw: string;
  details: TypeDetail[];
}

export type ResultGroup =
  | { kind: 'tuple'; rawType: string; slots: TypeSlot[] }
  | { kind: 'array'; rawType: string; details: TypeDetail[] };

export interface InstructionGuide {
  html: string;
  compactHtml?: string;
  parts: GuidePart[];
  resultGroup: ResultGroup | null;
}
