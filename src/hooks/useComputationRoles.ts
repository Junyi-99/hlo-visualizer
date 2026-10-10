import { useMemo } from 'react';
import { computationLinks } from '../lib/parser';
import { computationRole } from '../lib/graph-layout';
import type { HloModule } from '../lib/types';

// Role label ('ENTRY', 'FUSION', 'WHILE BODY', …) for each computation, keyed by name.
export function useComputationRoles(module: HloModule) {
  return useMemo(() => {
    const roles = new Map<string, string>();
    const links = computationLinks(module);
    for (const computation of module.computations) roles.set(computation.name, computationRole(computation, links));
    return roles;
  }, [module]);
}
