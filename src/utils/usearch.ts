type UFloatVector = Float32Array | Float64Array | number[];

export interface USearchMatch {
  id: number;
  score: number;
  distance: number;
}

type USearchModule = typeof import('usearch');
type MetricKindValue = USearchModule['MetricKind'][keyof USearchModule['MetricKind']];

const resolveUSearchModule = (mod: any): USearchModule => {
  const candidate = mod?.default?.Index ? mod.default : mod;
  if (!candidate?.Index) {
    throw new Error('USearch module tidak valid: export Index tidak ditemukan');
  }
  return candidate as USearchModule;
};

let usearchModulePromise: Promise<USearchModule> | null = null;

const isNodeRuntime = () => {
  const maybeProcess = (globalThis as any)?.process;
  return (
    typeof maybeProcess === 'object' &&
    typeof maybeProcess?.versions === 'object' &&
    typeof maybeProcess?.versions?.node === 'string'
  );
};

const loadUSearchModule = async (): Promise<USearchModule> => {
  if (!usearchModulePromise) {
    usearchModulePromise = (async () => {
      if (!isNodeRuntime()) {
        throw new Error('USearch native bindings hanya tersedia di lingkungan Node');
      }
      const dynamicImport = new Function(
        'specifier',
        'return import(specifier);'
      ) as (specifier: string) => Promise<unknown>;
      const mod = await dynamicImport('usearch');
      return resolveUSearchModule(mod);
    })().catch((err) => {
      usearchModulePromise = null;
      throw err;
    });
  }
  return usearchModulePromise;
};

const toBigUint64 = (id: number) => {
  if (!Number.isFinite(id) || id < 0) {
    throw new Error('USearchIndex key must be bilangan bulat positif');
  }
  if (!Number.isInteger(id)) {
    throw new Error('USearchIndex key harus bilangan bulat');
  }
  return BigInt(id);
};

const toFlatVector = (vector: UFloatVector, dim: number): Float32Array => {
  const arr =
    vector instanceof Float32Array
      ? vector
      : vector instanceof Float64Array
      ? Float32Array.from(vector)
      : Float32Array.from(vector);
  if (arr.length !== dim) {
    throw new Error(
      `Vector dimension mismatch: expected ${dim}, got ${arr.length}`
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
  if (normA === 0 || normB === 0) {
    return 0;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
};

export class USearchIndex {
  readonly dimensions: number;
  private module: USearchModule | null;
  private index: InstanceType<USearchModule['Index']> | null;
  private readonly metric: MetricKindValue | 'cos';
  private readonly fallbackVectors: Map<number, Float32Array> | null;

  private constructor(
    module: USearchModule | null,
    dimensions: number,
    metric: MetricKindValue | 'cos'
  ) {
    if (!Number.isFinite(dimensions) || dimensions <= 0) {
      throw new Error('USearchIndex requires a positive dimension');
    }
    this.module = module;
    this.metric = metric;
    this.dimensions = Math.floor(dimensions);

    if (module) {
      this.index = this.createNativeIndex();
      this.fallbackVectors = null;
    } else {
      console.warn(
        '[USearch] Modul native tidak tersedia, fallback ke cosine search'
      );
      this.index = null;
      this.fallbackVectors = new Map();
    }
  }

  static async create(
    dimensions: number,
    options?: { metric?: MetricKindValue }
  ) {
    let module: USearchModule | null = null;
    try {
      module = await loadUSearchModule();
    } catch (err) {
      console.error('[USearch] Gagal memuat modul native', err);
    }
    const metric = module
      ? options?.metric ?? module.MetricKind.Cos
      : 'cos';
    return new USearchIndex(module, dimensions, metric);
  }

  private createNativeIndex() {
    if (!this.module) {
      throw new Error('USearch native module tidak tersedia');
    }
    return new this.module.Index(this.dimensions, this.metric as MetricKindValue);
  }

  clear() {
    if (this.index) {
      this.index = this.createNativeIndex();
    } else {
      this.fallbackVectors?.clear();
    }
  }

  has(id: number): boolean {
    if (this.index) {
      return Boolean(this.index.contains(toBigUint64(id)));
    }
    return this.fallbackVectors?.has(id) ?? false;
  }

  remove(id: number) {
    if (this.index) {
      this.index.remove(toBigUint64(id));
    } else {
      this.fallbackVectors?.delete(id);
    }
  }

  size() {
    if (this.index) {
      return Number(this.index.size());
    }
    return this.fallbackVectors?.size ?? 0;
  }

  add(id: number, vector: UFloatVector) {
    const vec = toFlatVector(vector, this.dimensions);
    if (this.index) {
      const key = toBigUint64(id);
      this.index.add(key, vec, 0);
    } else {
      this.fallbackVectors?.set(Number(id), vec);
    }
  }

  addMany(entries: { id: number; vector: UFloatVector }[]) {
    if (!entries.length) return;
    if (this.index) {
      const keys = new BigUint64Array(entries.length);
      const vectors = new Float32Array(entries.length * this.dimensions);
      for (let i = 0; i < entries.length; i += 1) {
        const { id, vector } = entries[i];
        keys[i] = toBigUint64(id);
        const vec = toFlatVector(vector, this.dimensions);
        vectors.set(vec, i * this.dimensions);
      }
      this.index.add(keys, vectors, 0);
    } else {
      for (const { id, vector } of entries) {
        const vec = toFlatVector(vector, this.dimensions);
        this.fallbackVectors?.set(Number(id), vec);
      }
    }
  }

  search(query: UFloatVector, k: number): USearchMatch[] {
    const total = this.size();
    if (total === 0) return [];
    const q = toFlatVector(query, this.dimensions);
    const normalizedK = Math.max(1, Math.min(Math.floor(k) || 1, total));

    if (this.index) {
      const matches = this.index.search(q, normalizedK, 0);
      const { keys, distances } = matches;
      const results: USearchMatch[] = [];
      for (let i = 0; i < keys.length; i += 1) {
        const distance = distances[i];
        const id = Number(keys[i]);
        const score = 1 - distance;
        results.push({ id, score, distance });
      }
      return results;
    }

    const results: USearchMatch[] = [];
    if (!this.fallbackVectors) return results;
    for (const [id, vec] of this.fallbackVectors.entries()) {
      const score = cosineSimilarity(q, vec);
      const distance = 1 - score;
      results.push({ id, score, distance });
    }
    results.sort((a, b) => a.distance - b.distance);
    return results.slice(0, normalizedK);
  }
}
