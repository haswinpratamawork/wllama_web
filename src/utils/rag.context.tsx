import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { openDB, DBSchema, IDBPDatabase } from 'idb';
import {
  createEmbeddingVector,
  getCurrentEmbeddingModel,
  onEmbeddingModelChange,
  type EmbeddingModelState,
} from './embedding-runtime';
import { USearchIndex, type USearchMatch } from './usearch';
import { withPerfTimer } from './perf-monitor';

type KnowledgeInput = {
  text: string;
};

type StoredKnowledge = {
  id: number;
  text: string;
  embedding: number[];
  embeddingDim: number;
  modelUrl?: string;
  createdAt: number;
  updatedAt: number;
  // legacy fields kept for migration from previous schema
  title?: string;
  content?: string;
  url?: string;
  tags?: string[];
};

export type KnowledgeItem = {
  id: number;
  text: string;
  embedding: Float32Array;
  embeddingDim: number;
  modelUrl?: string;
  createdAt: number;
  updatedAt: number;
};

export type RagSearchHit = {
  item: KnowledgeItem;
  score: number;
  distance: number;
};

export type KnowledgeImportRecord = {
  id?: number | string;
  text: string;
  embedding?: number[] | Float32Array;
};

export type KnowledgeImportResult = {
  imported: number;
  updated: number;
  skipped: number;
  errors: string[];
};

interface RagDB extends DBSchema {
  knowledge: {
    key: number;
    value: StoredKnowledge;
    indexes: { 'by-updated': number };
  };
}

interface RagContextValue {
  items: KnowledgeItem[];
  isLoading: boolean;
  ragEnabled: boolean;
  toggleRag(enabled: boolean): void;
  addKnowledge(input: KnowledgeInput): Promise<KnowledgeItem>;
  updateKnowledge(id: number, patch: KnowledgeInput): Promise<KnowledgeItem>;
  deleteKnowledge(id: number): Promise<void>;
  reembedKnowledge(id: number): Promise<KnowledgeItem>;
  importKnowledgeFromJson(
    records: KnowledgeImportRecord[]
  ): Promise<KnowledgeImportResult>;
  searchByText(
    text: string,
    options?: { topK?: number }
  ): Promise<RagSearchHit[]>;
  searchByEmbedding(
    embedding: Float32Array,
    options?: { topK?: number }
  ): Promise<RagSearchHit[]>;
  latestHits: RagSearchHit[] | null;
  embeddingModel?: EmbeddingModelState | null;
  knowledgeCount: number;
}

const RagContext = createContext<RagContextValue>({} as RagContextValue);

const DB_NAME = 'rag-knowledge-db';
const STORE_NAME = 'knowledge';
const RAG_PREF_KEY = 'rag_enabled';

const toKnowledgeItem = (stored: StoredKnowledge): KnowledgeItem => ({
  id: stored.id,
  text:
    stored.text ??
    [stored.title, stored.content].filter(Boolean).join('\n\n') ??
    '',
  embedding: Float32Array.from(stored.embedding),
  embeddingDim: stored.embeddingDim,
  modelUrl: stored.modelUrl,
  createdAt: stored.createdAt,
  updatedAt: stored.updatedAt,
});

const toStoredKnowledge = (item: KnowledgeItem): StoredKnowledge => ({
  id: item.id,
  text: item.text,
  title: undefined,
  content: undefined,
  url: undefined,
  tags: undefined,
  embedding: Array.from(item.embedding),
  embeddingDim: item.embeddingDim,
  modelUrl: item.modelUrl,
  createdAt: item.createdAt,
  updatedAt: item.updatedAt,
});

const openKnowledgeDb = () =>
  withPerfTimer('idb:open', () =>
    openDB<RagDB>(DB_NAME, 1, {
      upgrade(db) {
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, {
            keyPath: 'id',
            autoIncrement: false,
          });
          store.createIndex('by-updated', 'updatedAt');
        }
      },
    })
  );

const loadPreference = () => {
  try {
    const stored = localStorage.getItem(RAG_PREF_KEY);
    return stored ? stored === 'true' : true;
  } catch {
    return true;
  }
};

const savePreference = (value: boolean) => {
  try {
    localStorage.setItem(RAG_PREF_KEY, String(value));
  } catch {
    // ignore
  }
};

export const RagProvider = ({ children }: { children: React.ReactNode }) => {
  const [items, setItems] = useState<KnowledgeItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [ragEnabled, setRagEnabled] = useState<boolean>(loadPreference);
  const [latestHits, setLatestHits] = useState<RagSearchHit[] | null>(null);
  const [embeddingModel, setEmbeddingModel] = useState<
    EmbeddingModelState | null | undefined
  >(getCurrentEmbeddingModel());

  const dbRef = useRef<IDBPDatabase<RagDB> | null>(null);
  const indexRef = useRef<USearchIndex | null>(null);
  const indexReadyRef = useRef<Promise<void>>(Promise.resolve());
  const itemMapRef = useRef<Map<number, KnowledgeItem>>(new Map());

  const ensureIndex = async (dim: number) => {
    if (dim <= 0 || !Number.isFinite(dim)) {
      throw new Error('Dimensi embedding tidak valid untuk indeks HNSW');
    }
    const existing = indexRef.current;
    if (!existing || existing.dimensions !== dim) {
      console.info('[RagContext] Membuat indeks HNSW baru', { dim });
      indexRef.current = await withPerfTimer(
        'hnsw:createIndex',
        () => USearchIndex.create(dim),
        () => ({ dim })
      );
    } else {
      console.info('[RagContext] Membersihkan indeks HNSW lama', { dim });
      existing.clear();
    }
    return indexRef.current!;
  };

  const rebuildIndex = (records: KnowledgeItem[]) => {
    const previous = indexReadyRef.current.catch(() => {});
    const promise = previous
      .then(() =>
        withPerfTimer(
          'hnsw:rebuild',
          async () => {
            console.info('[RagContext] Rebuild indeks HNSW dimulai', {
              total: records.length,
            });
            if (!records.length) {
              indexRef.current = null;
              itemMapRef.current.clear();
              console.info(
                '[RagContext] Rebuild selesai, tidak ada item yang diindeks'
              );
              return;
            }
            const index = await ensureIndex(records[0].embeddingDim);
            index.clear();
            itemMapRef.current.clear();
            const validEntries: { id: number; vector: Float32Array }[] = [];
            for (const item of records) {
              if (item.embedding.length !== index.dimensions) {
                console.warn(
                  `[RagContext] Skipping item ${item.id} due to dimension mismatch`
                );
                continue;
              }
              validEntries.push({ id: item.id, vector: item.embedding });
              itemMapRef.current.set(item.id, item);
            }
            if (validEntries.length) {
              await withPerfTimer(
                'hnsw:addMany',
                async () => {
                  index.addMany(validEntries);
                },
                () => ({ count: validEntries.length })
              );
            }
            console.info('[RagContext] Rebuild indeks HNSW selesai', {
              indexed: validEntries.length,
            });
          },
          () => ({ total: records.length })
        )
      )
      .catch((err) => {
        console.error('[RagContext] Failed to rebuild HNSW index', err);
        throw err;
      });
    indexReadyRef.current = promise;
    return promise;
  };

  useEffect(() => {
    let cancelled = false;
    const init = async () => {
      try {
        const db = await openKnowledgeDb();
        if (cancelled) return;
        dbRef.current = db;
        let entryCount = 0;
        const stored = await withPerfTimer(
          'idb:getAllKnowledge',
          async () => {
            const result = await db.getAll(STORE_NAME);
            entryCount = result.length;
            return result;
          },
          () => ({
            store: STORE_NAME,
            count: entryCount,
          })
        );
        if (cancelled) return;
        const list = stored.map(toKnowledgeItem);
        setItems(list);
        await rebuildIndex(list);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };
    init();
    const unsubscribe = onEmbeddingModelChange(setEmbeddingModel);
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);

  const persistItem = async (item: KnowledgeItem) => {
    const db = dbRef.current;
    if (!db) throw new Error('Database belum siap');
    await withPerfTimer(
      'idb:putKnowledge',
      () => db.put(STORE_NAME, toStoredKnowledge(item)),
      () => ({
        id: item.id,
      })
    );
  };

  const removeItemFromDb = async (id: number) => {
    const db = dbRef.current;
    if (!db) throw new Error('Database belum siap');
    await withPerfTimer(
      'idb:deleteKnowledge',
      () => db.delete(STORE_NAME, id),
      () => ({ id })
    );
  };

  const addKnowledge = async (input: KnowledgeInput) => {
    const model = getCurrentEmbeddingModel();
    if (!model) {
      throw new Error(
        'Load embedding model terlebih dahulu sebelum menambah knowledge'
      );
    }
    const text = input.text.trim();
    if (!text) {
      throw new Error('Teks knowledge tidak boleh kosong');
    }
    const embedding = await createEmbeddingVector(text);
    if (indexRef.current && embedding.length !== indexRef.current.dimensions) {
      throw new Error(
        'Dimensi embedding baru tidak cocok dengan indeks yang sudah ada. Kosongkan knowledge atau gunakan model yang sama.'
      );
    }
    const id = Date.now();
    const now = Date.now();
    const item: KnowledgeItem = {
      id,
      text,
      embedding,
      embeddingDim: embedding.length,
      modelUrl: model.url,
      createdAt: now,
      updatedAt: now,
    };
    await persistItem(item);
    setItems((prev) => {
      const next = [item, ...prev];
      void rebuildIndex(next);
      return next;
    });
    itemMapRef.current.set(item.id, item);
    return item;
  };

  const updateKnowledge = async (id: number, patch: KnowledgeInput) => {
    const existing = itemMapRef.current.get(id);
    if (!existing) {
      throw new Error('Knowledge tidak ditemukan');
    }
    const text = patch.text.trim();
    if (!text) {
      throw new Error('Teks knowledge tidak boleh kosong');
    }

    let embedding = existing.embedding;
    let embeddingDim = existing.embeddingDim;
    let modelUrl = existing.modelUrl;

    if (text !== existing.text) {
      const model = getCurrentEmbeddingModel();
      if (!model) {
        throw new Error(
          'Load embedding model terlebih dahulu sebelum memperbarui knowledge'
        );
      }
      embedding = await createEmbeddingVector(text);
      if (
        indexRef.current &&
        embedding.length !== indexRef.current.dimensions
      ) {
        throw new Error(
          'Dimensi embedding baru tidak cocok dengan indeks yang sudah ada.'
        );
      }
      embeddingDim = embedding.length;
      modelUrl = model.url;
    }

    const updated: KnowledgeItem = {
      ...existing,
      text,
      embedding,
      embeddingDim,
      modelUrl,
      updatedAt: Date.now(),
    };
    await persistItem(updated);
    setItems((prev) => {
      const next = prev.map((item) => (item.id === id ? updated : item));
      void rebuildIndex(next);
      return next;
    });
    itemMapRef.current.set(updated.id, updated);
    return updated;
  };

  const deleteKnowledge = async (id: number) => {
    await removeItemFromDb(id);
    setItems((prev) => {
      const next = prev.filter((item) => item.id !== id);
      void rebuildIndex(next);
      return next;
    });
    itemMapRef.current.delete(id);
  };

  const reembedKnowledge = async (id: number) => {
    const existing = itemMapRef.current.get(id);
    if (!existing) {
      throw new Error('Knowledge tidak ditemukan');
    }
    const model = getCurrentEmbeddingModel();
    if (!model) {
      throw new Error('Load embedding model terlebih dahulu');
    }
    const embedding = await createEmbeddingVector(existing.text);
    if (indexRef.current && embedding.length !== indexRef.current.dimensions) {
      throw new Error(
        'Dimensi embedding baru tidak cocok dengan indeks yang sudah ada.'
      );
    }
    const updated: KnowledgeItem = {
      ...existing,
      embedding,
      embeddingDim: embedding.length,
      modelUrl: model.url,
      updatedAt: Date.now(),
    };
    await persistItem(updated);
    setItems((prev) => {
      const next = prev.map((item) => (item.id === id ? updated : item));
      void rebuildIndex(next);
      return next;
    });
    itemMapRef.current.set(updated.id, updated);
    return updated;
  };

  const importKnowledgeFromJson = async (
    records: KnowledgeImportRecord[]
  ): Promise<KnowledgeImportResult> => {
    if (!records.length) {
      return { imported: 0, updated: 0, skipped: 0, errors: [] };
    }

    const needEmbeddingGeneration = records.some(
      (rec) => !rec.embedding || rec.embedding.length === 0
    );
    const model = needEmbeddingGeneration ? getCurrentEmbeddingModel() : null;
    if (needEmbeddingGeneration && !model) {
      throw new Error(
        'Load embedding model terlebih dahulu sebelum mengimpor knowledge tanpa embedding'
      );
    }

    const currentDim = (() => {
      const iterator = itemMapRef.current.values().next();
      if (!iterator.done) return iterator.value.embeddingDim;
      return indexRef.current?.dimensions ?? null;
    })();

    const results: KnowledgeImportResult = {
      imported: 0,
      updated: 0,
      skipped: 0,
      errors: [],
    };

    const prepared = new Map<number, KnowledgeItem>();
    const existedBefore = new Map<number, boolean>();
    let autoIdSeed = Date.now();

    for (const record of records) {
      try {
        if (!record || typeof record !== 'object') {
          throw new Error('Record bukan objek');
        }
        if (typeof record.text !== 'string') {
          throw new Error('Field "text" wajib berupa string');
        }
        const trimmed = record.text.trim();
        if (!trimmed) {
          throw new Error('Field "text" wajib diisi');
        }

        let numericId: number;
        if (
          record.id !== undefined &&
          record.id !== null &&
          record.id !== ''
        ) {
          const maybeNumber =
            typeof record.id === 'number'
              ? record.id
              : Number.parseInt(String(record.id), 10);
          if (Number.isFinite(maybeNumber)) {
            numericId = maybeNumber;
          } else {
            throw new Error('Field "id" harus numerik');
          }
        } else {
          numericId = autoIdSeed;
          autoIdSeed += 1;
        }

        while (prepared.has(numericId)) {
          numericId += 1;
        }

        const existingItem = itemMapRef.current.get(numericId);

        let vector: Float32Array;
        let modelUrl: string | undefined;
        if (record.embedding && record.embedding.length) {
          vector = Float32Array.from(record.embedding);
          modelUrl = 'json-import';
        } else {
          vector = await createEmbeddingVector(trimmed);
          modelUrl = model?.url;
        }

        if (!vector.length) {
          throw new Error('Embedding kosong');
        }
        if (currentDim !== null && vector.length !== currentDim) {
          throw new Error(
            `Dimensi embedding ${vector.length} tidak cocok dengan existing ${currentDim}`
          );
        }
        const firstPreparedValue = prepared.values().next().value;
        if (
          prepared.size > 0 &&
          firstPreparedValue &&
          vector.length !== firstPreparedValue.embeddingDim
        ) {
          throw new Error('Dimensi embedding antar record tidak konsisten');
        }

        const timestamp = Date.now();
        const text = trimmed;
        const item: KnowledgeItem = {
          id: numericId,
          text,
          embedding: vector,
          embeddingDim: vector.length,
          modelUrl,
          createdAt: existingItem?.createdAt ?? timestamp,
          updatedAt: timestamp,
        };

        prepared.set(numericId, item);
        existedBefore.set(numericId, existingItem !== undefined);
      } catch (err: any) {
        results.skipped += 1;
        results.errors.push(err?.message ?? String(err));
      }
    }

    if (prepared.size === 0) {
      return results;
    }

    await Promise.all(
      Array.from(prepared.values()).map((item) => persistItem(item))
    );

    setItems((prev) => {
      const map = new Map<number, KnowledgeItem>();
      for (const item of prev) {
        map.set(item.id, item);
      }
      for (const item of prepared.values()) {
        map.set(item.id, item);
      }
      const next = Array.from(map.values());
      void rebuildIndex(next);
      return next;
    });

    for (const item of prepared.values()) {
      itemMapRef.current.set(item.id, item);
      if (existedBefore.get(item.id)) {
        results.updated += 1;
      } else {
        results.imported += 1;
      }
    }

    return results;
  };

  const waitForIndexReady = async () => {
    try {
      await indexReadyRef.current;
    } catch (err) {
      console.error('[RagContext] Gagal menyiapkan indeks HNSW', err);
      throw err;
    }
  };

  const searchByEmbedding = async (
    embedding: Float32Array,
    options?: { topK?: number }
  ): Promise<RagSearchHit[]> => {
    await waitForIndexReady();
    const index = indexRef.current;
    if (!index) {
      console.warn('[RagContext] Pencarian dibatalkan: indeks HNSW belum siap');
      setLatestHits([]);
      return [];
    }
    if (embedding.length !== index.dimensions) {
      throw new Error(
        `Dimensi embedding tidak cocok dengan indeks (${embedding.length} vs ${index.dimensions})`
      );
    }
    console.info('[RagContext] Menjalankan pencarian embeddings', {
      topK: options?.topK ?? 5,
      dim: embedding.length,
    });
    const hits: USearchMatch[] = await withPerfTimer(
      'hnsw:search',
      () => index.search(embedding, options?.topK ?? 5),
      () => ({
        topK: options?.topK ?? 5,
        dim: embedding.length,
      })
    );
    const mapped: RagSearchHit[] = [];
    for (const hit of hits) {
      const item = itemMapRef.current.get(hit.id);
      if (item) {
        mapped.push({
          item,
          score: hit.score,
          distance: hit.distance,
        });
      }
    }
    setLatestHits(mapped);
    console.info('[RagContext] Hasil pencarian HNSW', {
      hits: mapped.length,
    });
    return mapped;
  };

  const searchByText = async (
    text: string,
    options?: { topK?: number }
  ) => {
    if (!text.trim()) return [];
    return withPerfTimer(
      'rag:search',
      async () => {
        const model = getCurrentEmbeddingModel();
        if (!model) {
          throw new Error(
            'Load embedding model terlebih dahulu sebelum melakukan pencarian'
          );
        }
        const embedding = await createEmbeddingVector(text);
        console.info('[RagContext] Pencarian teks menghasilkan embedding', {
          dim: embedding.length,
        });
        return searchByEmbedding(embedding, options);
      },
      () => ({
        queryLength: text.length,
        topK: options?.topK ?? 5,
      })
    );
  };

  const toggleRag = (enabled: boolean) => {
    setRagEnabled(enabled);
    savePreference(enabled);
  };

  const knowledgeCount = useMemo(() => items.length, [items]);

  const value = useMemo<RagContextValue>(
    () => ({
      items,
      isLoading,
      ragEnabled,
      toggleRag,
      addKnowledge,
      updateKnowledge,
      deleteKnowledge,
      reembedKnowledge,
      importKnowledgeFromJson,
      searchByText,
      searchByEmbedding,
      latestHits,
      embeddingModel,
      knowledgeCount,
    }),
    [
      items,
      isLoading,
      ragEnabled,
      toggleRag,
      addKnowledge,
      updateKnowledge,
      deleteKnowledge,
      reembedKnowledge,
      importKnowledgeFromJson,
      searchByText,
      searchByEmbedding,
      latestHits,
      embeddingModel,
      knowledgeCount,
    ]
  );

  return <RagContext.Provider value={value}>{children}</RagContext.Provider>;
};

export const useRag = () => useContext(RagContext);
