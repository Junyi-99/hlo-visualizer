import { useEffect, useState } from 'react';

// State mirrored to localStorage. `write` returning null removes the key; blocked storage is ignored.
export function useStoredState<T>(key: string, read: (saved: string | null) => T, write: (value: T) => string | null = String) {
  const [value, setValue] = useState(() => {
    try {
      return read(localStorage.getItem(key));
    } catch {
      return read(null);
    }
  });

  useEffect(() => {
    try {
      const saved = write(value);
      if (saved === null) localStorage.removeItem(key);
      else localStorage.setItem(key, saved);
    } catch {
      /* storage blocked */
    }
  }, [key, value, write]);

  return [value, setValue] as const;
}
