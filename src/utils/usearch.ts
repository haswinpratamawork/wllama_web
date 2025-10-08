import type { HierarchicalNSW, HnswlibModule } from 'hnswlib-wasm';
import { loadHnswlib } from 'hnswlib-wasm';

type UFloatVector = Float32Array | Float64Array | number[];

export interface USearchMatch {
  id: number;
  score: number;
  distance: number;
}

const SPACE = 'cosine' as const;
const DEFAULT_CAPACITY = 64;
const DEFAULT_M = 16;
const DEFAULT_EF_CONSTRUCTION = 200;
const DEFAULT_RANDOM_SEED = 100;
const DEFAULT_EF_SEARCH = 64;
const AUTO_SAVE_FILENAME = 'rag-hnsw-index.bin';

const toFloat32 = (vector: UFloatVector, expected: number): Float32Array => {
  const arr =
    vector instanceof Float32Array
      ? vector
      : vector instanceof Float64Array
      ? Float32Array.from(vector)
      : Float32Array.from(vector);
  if (arr.length !== expected) {
    throw new Error(
      `Vector dimension mismatch: expected ${expected}, got ${arr.length}`
    );
  }
  return arr;
};

const cosineSimilarity = (a: Float32Array, b: Float32Array): number => {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    const va = a[i];
    const vb = b[i];
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
};

const loadLibrary = async (): Promise<HnswlibModule | null> => {
  try {
    return await loadHnswlib('IDBFS');
  } catch (err) {
    console.error('[HNSW] Gagal memuat WASM hnswlib, fallback ke cosine search', err);
    return null;
  }
};

export class USearchIndex {
  readonly dimensions: number;

  private readonly lib: HnswlibModule | null;
  private index: HierarchicalNSW | null;
  private capacity: number;
  private readonly vectorStore = new Map<number, Float32Array>(); // external id -> vector
  private readonly externalToInternal = new Map<number, number>();
  private readonly internalToExternal = new Map<number, number>();
  private nextLabel = 0;
  private freeLabels: number[] = [];

  private constructor(lib: HnswlibModule | null, dimensions: number) {
    if (!Number.isFinite(dimensions) || dimensions <= 0) {
      throw new Error('USearchIndex requires a positive dimension');
    }
    this.dimensions = Math.floor(dimensions);
    this.lib = lib;
    this.capacity = DEFAULT_CAPACITY;
    this.index = this.lib ? this.createNativeIndex(DEFAULT_CAPACITY) : null;
  }

  static async create(dimensions: number) {
    const lib = await loadLibrary();
    return new USearchIndex(lib, dimensions);
  }

  private createNativeIndex(initialCapacity: number): HierarchicalNSW {
    if (!this.lib) {
      throw new Error('Native HNSW lib is unavailable');
    }
    const idx = new this.lib.HierarchicalNSW(
      SPACE,
      this.dimensions,
      AUTO_SAVE_FILENAME
    );
    idx.initIndex(initialCapacity, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, DEFAULT_RANDOM_SEED);
    idx.setEfSearch(DEFAULT_EF_SEARCH);
    this.capacity = initialCapacity;
    return idx;
  }

  private ensureIndex(): HierarchicalNSW {
    if (!this.index) {
    this.index = this.createNativeIndex(Math.max(DEFAULT_CAPACITY, this.vectorStore.size || 1));
    for (const [externalId, vector] of this.vectorStore.entries()) {
      const label = this.externalToInternal.get(externalId);
      if (label === undefined) continue;
      this.addNativePoint(vector, label);
    }
    }
    return this.index;
  }

  clear() {
    this.vectorStore.clear();
    this.externalToInternal.clear();
    this.internalToExternal.clear();
    this.freeLabels = [];
    this.nextLabel = 0;
    if (this.lib) {
      this.index = this.createNativeIndex(DEFAULT_CAPACITY);
    } else {
      this.index = null;
    }
  }

  has(id: number): boolean {
    return this.externalToInternal.has(id);
  }

  remove(id: number) {
    const label = this.externalToInternal.get(id);
    if (label === undefined) return;
    this.vectorStore.delete(id);
    this.externalToInternal.delete(id);
    this.internalToExternal.delete(label);
    this.freeLabels.push(label);
    if (this.index) {
      try {
        this.index.markDelete(label);
      } catch (err) {
        console.warn('[HNSW] markDelete gagal', err);
      }
    }
  }

  size() {
    return this.vectorStore.size;
  }

  private ensureCapacity(required: number) {
    if (!this.index) return;
    if (required <= this.capacity) return;
    const newCapacity = Math.max(required, Math.ceil(this.capacity * 1.5));
    try {
      this.index.resizeIndex(newCapacity);
      this.capacity = newCapacity;
    } catch (err) {
      console.warn('[HNSW] resizeIndex gagal, akan membuat ulang indeks', err);
      this.rebuildNativeIndex();
    }
  }

  private rebuildNativeIndex() {
    if (!this.lib) return;
    const count = this.vectorStore.size || 1;
    this.index = this.createNativeIndex(Math.max(DEFAULT_CAPACITY, count));
    for (const [externalId, vector] of this.vectorStore.entries()) {
      const label = this.externalToInternal.get(externalId);
      if (label === undefined) continue;
      this.addNativePoint(vector, label);
    }
  }

  private allocateLabel(): number {
    if (this.freeLabels.length) return this.freeLabels.pop()!;
    const label = this.nextLabel;
    this.nextLabel += 1;
    return label;
  }

  private bindLabel(externalId: number): number {
    const existing = this.externalToInternal.get(externalId);
    if (existing !== undefined) return existing;
    const label = this.allocateLabel();
    this.externalToInternal.set(externalId, label);
    this.internalToExternal.set(label, externalId);
    return label;
  }

  private addNativePoint(vector: Float32Array, label: number) {
    const idx = this.ensureIndex();
    this.ensureCapacity(this.vectorStore.size);
    try {
      idx.addPoint(vector, label, true);
    } catch (err) {
      console.warn('[HNSW] addPoint gagal, akan rebuild', err);
      this.rebuildNativeIndex();
      this.ensureIndex().addPoint(vector, label, true);
    }
  }

  add(id: number, vector: UFloatVector) {
    const arr = toFloat32(vector, this.dimensions);
    const label = this.bindLabel(id);
    this.vectorStore.set(id, Float32Array.from(arr));
    if (this.index) {
      try {
        this.index.markDelete(label);
      } catch {
        // ignore
      }
    }
    if (this.lib) {
      this.addNativePoint(arr, label);
    }
  }

  addMany(entries: { id: number; vector: UFloatVector }[]) {
    if (!entries.length) return;
    const converted: { label: number; arr: Float32Array }[] = [];
    for (const { id, vector } of entries) {
      const arr = toFloat32(vector, this.dimensions);
      const label = this.bindLabel(id);
      this.vectorStore.set(id, Float32Array.from(arr));
      if (this.index) {
        try {
          this.index.markDelete(label);
        } catch {
          // ignore
        }
      }
      converted.push({ label, arr });
    }

    if (!this.lib) return;
    const idx = this.ensureIndex();
    this.ensureCapacity(this.vectorStore.size);
    const vectors = converted.map(({ arr }) => arr);
    const labels = converted.map(({ label }) => label);
    try {
      idx.addPoints(vectors, labels, true);
    } catch (err) {
      console.warn('[HNSW] addPoints gagal, rebuild seluruh indeks', err);
      this.rebuildNativeIndex();
    }
  }

  search(query: UFloatVector, k: number): USearchMatch[] {
    const total = this.vectorStore.size;
    if (!total) return [];
    const arr = toFloat32(query, this.dimensions);
    const topK = Math.max(1, Math.min(Math.floor(k) || 1, total));

    if (this.index) {
      try {
        const result = this.index.searchKnn(arr, topK, undefined);
        const matches: USearchMatch[] = [];
        const { neighbors, distances } = result;
        for (let i = 0; i < neighbors.length; i += 1) {
          const label = neighbors[i];
          if (label === -1) continue;
          const externalId = this.internalToExternal.get(label);
          if (externalId === undefined) continue;
          const distance = distances[i];
          const score = 1 - distance;
          matches.push({ id: externalId, score, distance });
        }
        return matches;
      } catch (err) {
        console.warn('[HNSW] searchKnn gagal, fallback ke cosine search', err);
        this.index = null;
      }
    }

    const results: USearchMatch[] = [];
    for (const [externalId, vec] of this.vectorStore.entries()) {
      const score = cosineSimilarity(arr, vec);
      const distance = 1 - score;
      results.push({ id: externalId, score, distance });
    }
    results.sort((a, b) => a.distance - b.distance);
    return results.slice(0, topK);
  }
}
