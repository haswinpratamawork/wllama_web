import { useSyncExternalStore } from 'react';

export type PerfEvent = {
  id: number;
  name: string;
  durationMs: number;
  startedAt: number;
  endedAt: number;
  wallClock: number;
  meta?: Record<string, unknown>;
};

const MAX_EVENTS = 50;

type PerfState = {
  events: PerfEvent[];
  latestByName: Record<string, PerfEvent>;
};

let perfState: PerfState = {
  events: [],
  latestByName: {},
};

const listeners = new Set<() => void>();
let eventCounter = 0;

const notify = () => {
  for (const listener of listeners) {
    listener();
  }
};

export const recordPerfEvent = (
  name: string,
  durationMs: number,
  meta?: Record<string, unknown>
) => {
  const now = performance.now();
  const event: PerfEvent = {
    id: ++eventCounter,
    name,
    durationMs,
    startedAt: now - durationMs,
    endedAt: now,
    wallClock: Date.now(),
    meta,
  };
  const events = [...perfState.events, event];
  if (events.length > MAX_EVENTS) {
    events.splice(0, events.length - MAX_EVENTS);
  }
  perfState = {
    events,
    latestByName: {
      ...perfState.latestByName,
      [name]: event,
    },
  };
  notify();
};

export const withPerfTimer = async <T>(
  name: string,
  fn: () => Promise<T> | T,
  meta?:
    | Record<string, unknown>
    | (() => Record<string, unknown> | undefined)
): Promise<T> => {
  const startedAt = performance.now();
  try {
    const result = await fn();
    const durationMs = performance.now() - startedAt;
    recordPerfEvent(name, durationMs, typeof meta === 'function' ? meta() : meta);
    return result;
  } catch (err) {
    const durationMs = performance.now() - startedAt;
    recordPerfEvent(name, durationMs, {
      ...(typeof meta === 'function' ? meta() : meta),
      error: err instanceof Error ? err.message : String(err),
    });
    throw err;
  }
};

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

const getSnapshot = () => perfState;

export const usePerfMonitor = () =>
  useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

export const formatDuration = (value: number) =>
  `${value.toFixed(1)} ms`;

export const PERF_LABELS: Record<string, string> = {
  'embedding:generate': 'Generate embedding',
  'rag:search': 'RAG search',
  'hnsw:search': 'HNSW search',
  'hnsw:rebuild': 'HNSW rebuild',
  'hnsw:addMany': 'HNSW addMany',
  'hnsw:createIndex': 'HNSW create index',
  'answer:generate': 'Generate answer',
  'idb:getAllKnowledge': 'IndexedDB load all knowledge',
  'idb:putKnowledge': 'IndexedDB put knowledge',
  'idb:deleteKnowledge': 'IndexedDB delete knowledge',
  'idb:open': 'IndexedDB open database',
};

export const getPerfLabel = (name: string) => PERF_LABELS[name] ?? name;
