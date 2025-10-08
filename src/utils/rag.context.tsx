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

type KnowledgeInput = {
  title: string;
  content: string;
  url?: string;
  tags?: string[];
};

type StoredKnowledge = KnowledgeInput & {
  id: number;
  embedding: number[];
  embeddingDim: number;
  modelUrl?: string;
  createdAt: number;
  updatedAt: number;
};

export type KnowledgeItem = KnowledgeInput & {
  id: number;
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
  title: stored.title,
  content: stored.content,
  url: stored.url,
  tags: stored.tags ?? [],
  embedding: Float32Array.from(stored.embedding),
  embeddingDim: stored.embeddingDim,
  modelUrl: stored.modelUrl,
  createdAt: stored.createdAt,
  updatedAt: stored.updatedAt,
});

const toStoredKnowledge = (item: KnowledgeItem): StoredKnowledge => ({
  id: item.id,
  title: item.title,
  content: item.content,
  url: item.url,
  tags: item.tags ?? [],
  embedding: Array.from(item.embedding),
  embeddingDim: item.embeddingDim,
  modelUrl: item.modelUrl,
  createdAt: item.createdAt,
  updatedAt: item.updatedAt,
});

const openKnowledgeDb = () =>
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
  });

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
      indexRef.current = await USearchIndex.create(dim);
    } else {
      console.info('[RagContext] Membersihkan indeks HNSW lama', { dim });
      existing.clear();
    }
    return indexRef.current!;
  };

  const rebuildIndex = (records: KnowledgeItem[]) => {
    const previous = indexReadyRef.current.catch(() => {});
    const promise = previous.then(async () => {
      console.info('[RagContext] Rebuild indeks HNSW dimulai', {
        total: records.length,
      });
      if (!records.length) {
        indexRef.current = null;
        itemMapRef.current.clear();
        console.info('[RagContext] Rebuild selesai, tidak ada item yang diindeks');
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
      index.addMany(validEntries);
      console.info('[RagContext] Rebuild indeks HNSW selesai', {
        indexed: validEntries.length,
      });
    }).catch((err) => {
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
        const stored = await db.getAll(STORE_NAME);
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
    await db.put(STORE_NAME, toStoredKnowledge(item));
  };

  const removeItemFromDb = async (id: number) => {
    const db = dbRef.current;
    if (!db) throw new Error('Database belum siap');
    await db.delete(STORE_NAME, id);
  };

  const addKnowledge = async (input: KnowledgeInput) => {
    const model = getCurrentEmbeddingModel();
    if (!model) {
      throw new Error('Load embedding model terlebih dahulu sebelum menambah knowledge');
    }
    const text = `${input.title}\n\n${input.content}`;
    const embedding = await createEmbeddingVector(text);
    if (indexRef.current && embedding.length !== indexRef.current.dimensions) {
      throw new Error(
        'Dimensi embedding baru tidak cocok dengan indeks yang sudah ada. Kosongkan knowledge atau gunakan model yang sama.'
      );
    }
    const id = Date.now();
    const now = Date.now();
    const item: KnowledgeItem = {
      ...input,
      id,
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
    const shouldReembed =
      patch.content !== existing.content || patch.title !== existing.title;
    let embedding = existing.embedding;
    let embeddingDim = existing.embeddingDim;
    let modelUrl = existing.modelUrl;

    if (shouldReembed) {
      const model = getCurrentEmbeddingModel();
      if (!model) {
        throw new Error(
          'Load embedding model terlebih dahulu sebelum memperbarui knowledge'
        );
      }
      const text = `${patch.title}\n\n${patch.content}`;
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
      ...patch,
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
    const text = `${existing.title}\n\n${existing.content}`;
    const embedding = await createEmbeddingVector(text);
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
    const hits: USearchMatch[] = index.search(embedding, options?.topK ?? 5);
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
    const model = getCurrentEmbeddingModel();
    if (!model) {
      throw new Error('Load embedding model terlebih dahulu sebelum melakukan pencarian');
    }
    const embedding = await createEmbeddingVector(text);
    console.info('[RagContext] Pencarian teks menghasilkan embedding', {
      dim: embedding.length,
    });
    return searchByEmbedding(embedding, options);
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
