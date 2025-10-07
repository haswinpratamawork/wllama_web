import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type ChangeEvent,
} from 'react';
import { Link } from 'react-router-dom';
import {
  createEmbeddingVector,
  DEFAULT_EMBEDDING_MODEL,
  getSavedEmbeddingPreference,
  loadEmbeddingModel,
  loadEmbeddingModelFromFiles,
  resetEmbeddingRuntime,
} from '../../utils/embedding-runtime';
import {
  useRag,
  type KnowledgeItem,
  type RagSearchHit,
} from '../../utils/rag.context';

const formatDate = (value: number) =>
  new Date(value).toLocaleString(undefined, {
    hour12: false,
  });

const clampPreview = (vector: Float32Array | null, take: number) => {
  if (!vector) return null;
  const slice = vector.slice(0, take);
  return Array.from(slice);
};

export default function EmbeddingPage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const autoLoadAttempted = useRef(false);

  const [status, setStatus] = useState<string>('idle');
  const [progressPercent, setProgressPercent] = useState<number | null>(null);
  const [modelUrl, setModelUrl] = useState<string>(DEFAULT_EMBEDDING_MODEL);
  const [text, setText] = useState(
    'Saya sedang menguji embedding Gemma untuk demo WASM di browser.'
  );
  const [embeddingPreview, setEmbeddingPreview] =
    useState<Float32Array | null>(null);
  const [embeddingBusy, setEmbeddingBusy] = useState(false);

  const [form, setForm] = useState<{ title: string; content: string; url: string }>(
    { title: '', content: '', url: '' }
  );
  const [editing, setEditing] = useState<KnowledgeItem | null>(null);
  const [savingKnowledge, setSavingKnowledge] = useState(false);

  const [searchText, setSearchText] = useState('');
  const [searchResults, setSearchResults] = useState<RagSearchHit[]>([]);
  const [searching, setSearching] = useState(false);

  const {
    ragEnabled,
    toggleRag,
    addKnowledge,
    updateKnowledge,
    deleteKnowledge,
    reembedKnowledge,
    searchByText,
    items,
    embeddingModel,
    knowledgeCount,
  } = useRag();

  const modelLoaded = Boolean(embeddingModel);

  const sortedItems = useMemo(() => {
    return [...items].sort((a, b) => b.updatedAt - a.updatedAt);
  }, [items]);

  const embeddingPreviewNumbers = useMemo(
    () => clampPreview(embeddingPreview, 64),
    [embeddingPreview]
  );

  useEffect(() => {
    if (embeddingModel) {
      setModelUrl(embeddingModel.url);
      setStatus(`model ready — ${embeddingModel.label}`);
    }
  }, [embeddingModel]);

  useEffect(() => {
    if (embeddingModel || autoLoadAttempted.current) {
      return;
    }
    autoLoadAttempted.current = true;
    const saved = getSavedEmbeddingPreference();
    if (!saved) {
      return;
    }
    setStatus('loading saved model');
    setProgressPercent(0);
    loadEmbeddingModel({
      url: saved.url,
      source: saved.source,
      label: saved.name,
      silent: true,
      onProgress: (pct) =>
        setProgressPercent(pct === null ? null : Math.round(pct * 100)),
    })
      .then(() => {
        setStatus(
          `model ready — ${
            saved.name ?? saved.url.split('/').pop() ?? saved.url
          }`
        );
      })
      .catch((err: any) => {
        console.error(err);
        setStatus(
          'failed to load saved model: ' + (err?.message ?? String(err))
        );
      })
      .finally(() => setProgressPercent(null));
  }, [embeddingModel]);

  const handleLoadModel = async () => {
    const targetUrl = modelUrl.trim();
    if (!targetUrl) {
      setStatus('model URL tidak valid');
      return;
    }
    const source = targetUrl.startsWith('local://') ? 'local' : 'remote';
    setStatus(
      source === 'local' ? 'loading local model' : 'loading model from url'
    );
    setProgressPercent(0);
    try {
      await loadEmbeddingModel({
        url: targetUrl,
        source,
        label: targetUrl.split('/').pop() ?? targetUrl,
        onProgress: (pct) =>
          setProgressPercent(pct === null ? null : Math.round(pct * 100)),
      });
    } catch (err: any) {
      console.error(err);
      setStatus('failed to load model: ' + (err?.message ?? String(err)));
    } finally {
      setProgressPercent(null);
    }
  };

  const handleUploadLocalModel = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setStatus('saving local model');
    setProgressPercent(0);
    try {
      await loadEmbeddingModelFromFiles({
        files,
        onProgress: (pct) =>
          setProgressPercent(pct === null ? null : Math.round(pct * 100)),
      });
    } catch (err: any) {
      console.error(err);
      setStatus('failed to load local model: ' + (err?.message ?? String(err)));
    } finally {
      setProgressPercent(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleResetRuntime = async () => {
    await resetEmbeddingRuntime();
    setEmbeddingPreview(null);
    setStatus('runtime reset — model unloaded');
  };

  const handleCreateEmbedding = async () => {
    if (!modelLoaded) {
      setStatus('load a model first');
      return;
    }
    if (!text.trim()) {
      setStatus('type some text to embed');
      return;
    }
    setEmbeddingBusy(true);
    try {
      setStatus('creating embedding');
      const vec = await createEmbeddingVector(text);
      setEmbeddingPreview(vec);
      setStatus(
        `embedding created (length ${vec.length.toLocaleString()})`
      );
    } catch (err: any) {
      console.error(err);
      setStatus('embedding failed: ' + (err?.message ?? String(err)));
    } finally {
      setEmbeddingBusy(false);
    }
  };

  const resetForm = () => {
    setForm({ title: '', content: '', url: '' });
    setEditing(null);
  };

  const handleKnowledgeSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const title = form.title.trim();
    const content = form.content.trim();
    const url = form.url.trim();
    if (!title || !content) {
      setStatus('isi judul dan konten terlebih dahulu');
      return;
    }
    setSavingKnowledge(true);
    try {
      if (editing) {
        await updateKnowledge(editing.id, {
          title,
          content,
          url: url || undefined,
        });
        setStatus('knowledge updated');
      } else {
        await addKnowledge({
          title,
          content,
          url: url || undefined,
        });
        setStatus('knowledge added');
      }
      resetForm();
    } catch (err: any) {
      console.error(err);
      setStatus('gagal menyimpan knowledge: ' + (err?.message ?? String(err)));
    } finally {
      setSavingKnowledge(false);
    }
  };

  const handleEditKnowledge = (item: KnowledgeItem) => {
    setEditing(item);
    setForm({
      title: item.title,
      content: item.content,
      url: item.url ?? '',
    });
  };

  const handleDeleteKnowledge = async (item: KnowledgeItem) => {
    const confirmed = window.confirm(`Hapus "${item.title}" dari knowledge?`);
    if (!confirmed) return;
    try {
      await deleteKnowledge(item.id);
      setStatus('knowledge deleted');
      if (editing?.id === item.id) {
        resetForm();
      }
    } catch (err: any) {
      console.error(err);
      setStatus('gagal menghapus knowledge: ' + (err?.message ?? String(err)));
    }
  };

  const handleReembedKnowledge = async (item: KnowledgeItem) => {
    try {
      setStatus(`re-embedding "${item.title}"`);
      await reembedKnowledge(item.id);
      setStatus('knowledge re-embedded');
    } catch (err: any) {
      console.error(err);
      setStatus(
        'gagal membuat ulang embedding: ' + (err?.message ?? String(err))
      );
    }
  };

  const handleSearch = async (event: FormEvent) => {
    event.preventDefault();
    const query = searchText.trim();
    if (!query) {
      setSearchResults([]);
      return;
    }
    setSearching(true);
    try {
      const hits = await searchByText(query, { topK: 5 });
      setSearchResults(hits);
      setStatus(
        hits.length
          ? `menemukan ${hits.length} knowledge relevan`
          : 'tidak ada knowledge yang cocok'
      );
    } catch (err: any) {
      console.error(err);
      setStatus('gagal mencari knowledge: ' + (err?.message ?? String(err)));
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="min-h-screen bg-[oklch(0.15_0.03_260)] text-[oklch(0.9_0.02_260)] transition-colors duration-200 p-6 md:p-10">
      <div className="max-w-5xl mx-auto space-y-6 md:space-y-10">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[oklch(0.3_0.12_270)] text-[oklch(0.98_0.02_260)] text-sm font-medium hover:opacity-95">
          <Link to="/">Kembali ke Chat</Link>
        </div>

        <div className="bg-[oklch(0.18_0.04_260)] shadow-lg rounded-2xl border border-[oklch(0.25_0.04_260)] overflow-hidden transition-colors">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between p-6 md:p-8">
            <div>
              <h1 className="text-2xl md:text-3xl font-semibold text-[oklch(0.95_0.02_260)]">
                Embedding & Knowledge Base
              </h1>
              <p className="text-sm text-[oklch(0.7_0.02_260)] mt-1">
                Kelola model embedding Gemma dan knowledge untuk RAG lokal.
              </p>
            </div>
            <div className="flex flex-col items-end gap-2">
              <div
                className={`px-3 py-1 rounded-full text-sm font-medium ${
                  modelLoaded
                    ? 'bg-[oklch(0.4_0.2_150)] text-[oklch(0.95_0.02_150)]'
                    : 'bg-[oklch(0.4_0.1_80)] text-[oklch(0.95_0.02_80)]'
                }`}
              >
                {modelLoaded ? 'Model ready' : 'Model not loaded'}
              </div>
              <div className="text-sm text-[oklch(0.7_0.02_260)] text-right">
                {status}
                {progressPercent !== null ? ` • ${progressPercent}%` : ''}
              </div>
            </div>
          </div>

          <div className="p-6 md:p-8 border-t border-[oklch(0.25_0.04_260)] space-y-8">
            <section className="space-y-4">
              <div className="flex items-center justify-between flex-wrap gap-3">
                <h2 className="text-xl font-semibold text-[oklch(0.95_0.02_260)]">
                  Model Embedding
                </h2>
                <label className="inline-flex items-center cursor-pointer gap-2 text-sm text-[oklch(0.75_0.03_260)]">
                  <span>Gunakan RAG di chat</span>
                  <input
                    type="checkbox"
                    className="toggle toggle-sm"
                    checked={ragEnabled}
                    onChange={(e) => toggleRag(e.target.checked)}
                  />
                </label>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
                <div className="md:col-span-2 space-y-3">
                  <textarea
                    className="w-full rounded-lg border border-[oklch(0.25_0.04_260)] shadow-sm p-3 text-sm focus:outline-none focus:ring-2 focus:ring-[oklch(0.6_0.1_260)] resize-none bg-[oklch(0.18_0.04_260)] text-[oklch(0.95_0.02_260)]"
                    value={modelUrl}
                    onChange={(e) => setModelUrl(e.target.value)}
                    placeholder="https://.../model.gguf atau local://..."
                    rows={2}
                  />
                  <div className="flex flex-wrap gap-3">
                    <button
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[oklch(0.3_0.12_270)] text-[oklch(0.98_0.02_260)] text-sm font-medium hover:opacity-95"
                      onClick={handleLoadModel}
                    >
                      Load model
                    </button>
                    <label className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-[oklch(0.25_0.04_260)] text-sm cursor-pointer bg-[oklch(0.2_0.04_260)] shadow-sm">
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".gguf"
                        multiple
                        className="hidden"
                        onChange={(e: ChangeEvent<HTMLInputElement>) =>
                          handleUploadLocalModel(e.target.files)
                        }
                      />
                      Upload GGUF
                    </label>
                    <button
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-[oklch(0.25_0.04_260)] text-sm bg-[oklch(0.18_0.04_260)] hover:bg-[oklch(0.22_0.05_260)]"
                      onClick={handleResetRuntime}
                    >
                      Reset runtime
                    </button>
                  </div>

                  {embeddingModel && (
                    <div className="text-xs text-[oklch(0.7_0.02_260)] space-y-1">
                      <div>
                        • Dimensi embedding:{' '}
                        <span className="font-semibold text-[oklch(0.95_0.02_260)]">
                          {embeddingModel.dim.toLocaleString()}
                        </span>
                      </div>
                      <div>• URL sumber: {embeddingModel.url}</div>
                    </div>
                  )}
                </div>

                <div className="space-y-3 text-sm text-[oklch(0.78_0.03_260)] bg-[oklch(0.2_0.05_260)] px-4 py-3 rounded-lg border border-[oklch(0.25_0.04_260)]">
                  <div className="font-semibold text-[oklch(0.95_0.02_260)]">
                    Tips
                  </div>
                  <ul className="list-disc list-inside space-y-1">
                    <li>Gunakan model embedding berbasis GGUF.</li>
                    <li>Upload semua shard jika model terpecah.</li>
                    <li>Model yang pernah dimuat disimpan otomatis.</li>
                  </ul>
                </div>
              </div>
            </section>

            <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h2 className="text-xl font-semibold text-[oklch(0.95_0.02_260)]">
                  Coba buat embedding
                </h2>
                <textarea
                  className="w-full rounded-lg border border-[oklch(0.25_0.04_260)] shadow-sm p-3 text-sm focus:outline-none focus:ring-2 focus:ring-[oklch(0.6_0.1_260)] resize-none bg-[oklch(0.18_0.04_260)] text-[oklch(0.95_0.02_260)]"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  rows={5}
                />
                <div className="flex flex-wrap gap-3">
                  <button
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[oklch(0.3_0.12_270)] text-[oklch(0.98_0.02_260)] text-sm font-medium hover:opacity-95 disabled:opacity-60"
                    onClick={handleCreateEmbedding}
                    disabled={embeddingBusy}
                  >
                    {embeddingBusy ? 'Creating...' : 'Create embedding'}
                  </button>
                  <button
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-[oklch(0.25_0.04_260)] text-sm bg-[oklch(0.18_0.04_260)] hover:bg-[oklch(0.22_0.05_260)]"
                    onClick={() => {
                      setEmbeddingPreview(null);
                      setStatus('idle');
                    }}
                  >
                    Clear
                  </button>
                </div>
              </div>

              <div className="space-y-3">
                <div className="bg-[oklch(0.2_0.05_260)] border border-[oklch(0.25_0.04_260)] rounded-lg p-4">
                  <div className="font-semibold text-[oklch(0.95_0.02_260)] mb-2">
                    Embedding preview
                  </div>
                  <div className="text-xs text-[oklch(0.7_0.02_260)]">
                    Panjang vector:{' '}
                    <span className="font-medium text-[oklch(0.95_0.02_260)]">
                      {embeddingPreview
                        ? embeddingPreview.length.toLocaleString()
                        : '-'}
                    </span>
                  </div>
                  <pre className="mt-2 text-xs whitespace-pre-wrap break-all text-[oklch(0.75_0.03_260)] bg-[oklch(0.18_0.04_260)] rounded-lg p-3 border border-[oklch(0.25_0.04_260)] max-h-60 overflow-auto">
                    {embeddingPreviewNumbers
                      ? JSON.stringify(embeddingPreviewNumbers, null, 2)
                      : '—'}
                  </pre>
                </div>
              </div>
            </section>

            <section className="space-y-5">
              <div className="flex items-center justify-between flex-wrap gap-3">
                <h2 className="text-xl font-semibold text-[oklch(0.95_0.02_260)]">
                  Knowledge Management
                </h2>
                <div className="text-sm text-[oklch(0.7_0.02_260)]">
                  Total knowledge: {knowledgeCount}
                </div>
              </div>

              <form
                onSubmit={handleKnowledgeSubmit}
                className="grid grid-cols-1 gap-4 bg-[oklch(0.2_0.05_260)] border border-[oklch(0.25_0.04_260)] rounded-lg p-4"
              >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <label className="space-y-2">
                    <span className="block text-sm font-medium text-[oklch(0.8_0.02_260)]">
                      Judul
                    </span>
                    <input
                      type="text"
                      className="input input-sm w-full bg-[oklch(0.18_0.04_260)]"
                      value={form.title}
                      onChange={(e) =>
                        setForm((prev) => ({
                          ...prev,
                          title: e.target.value,
                        }))
                      }
                      placeholder="Judul knowledge"
                    />
                  </label>
                  <label className="space-y-2">
                    <span className="block text-sm font-medium text-[oklch(0.8_0.02_260)]">
                      URL (opsional)
                    </span>
                    <input
                      type="url"
                      className="input input-sm w-full bg-[oklch(0.18_0.04_260)]"
                      value={form.url}
                      onChange={(e) =>
                        setForm((prev) => ({
                          ...prev,
                          url: e.target.value,
                        }))
                      }
                      placeholder="https://..."
                    />
                  </label>
                </div>
                <label className="space-y-2">
                  <span className="block text-sm font-medium text-[oklch(0.8_0.02_260)]">
                    Konten
                  </span>
                  <textarea
                    className="textarea textarea-sm w-full min-h-[150px] bg-[oklch(0.18_0.04_260)]"
                    value={form.content}
                    onChange={(e) =>
                      setForm((prev) => ({
                        ...prev,
                        content: e.target.value,
                      }))
                    }
                    placeholder="Masukkan konten yang ingin ditambahkan ke knowledge base"
                  />
                </label>
                <div className="flex flex-wrap gap-3 justify-end">
                  {editing && (
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={resetForm}
                    >
                      Batalkan edit
                    </button>
                  )}
                  <button
                    type="submit"
                    className="btn btn-sm btn-primary"
                    disabled={savingKnowledge}
                  >
                    {savingKnowledge
                      ? 'Menyimpan...'
                      : editing
                        ? 'Update knowledge'
                        : 'Tambah knowledge'}
                  </button>
                </div>
              </form>

              <div className="space-y-4">
                <form
                  className="flex flex-col md:flex-row gap-3 items-stretch md:items-center"
                  onSubmit={handleSearch}
                >
                  <input
                    type="text"
                    className="input input-sm flex-1 bg-[oklch(0.18_0.04_260)]"
                    placeholder="Cari knowledge (menggunakan embedding)..."
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                  />
                  <button
                    type="submit"
                    className="btn btn-sm btn-secondary"
                    disabled={searching}
                  >
                    {searching ? 'Searching...' : 'Search'}
                  </button>
                </form>

                {searchResults.length > 0 && (
                  <div className="bg-[oklch(0.2_0.05_260)] border border-[oklch(0.25_0.04_260)] rounded-lg p-4 space-y-3">
                    <div className="text-sm font-medium text-[oklch(0.95_0.02_260)]">
                      Hasil RAG preview
                    </div>
                    <div className="space-y-3">
                      {searchResults.map((hit) => (
                        <div
                          key={hit.item.id}
                          className="border border-[oklch(0.25_0.04_260)] rounded-lg p-3 bg-[oklch(0.18_0.04_260)]"
                        >
                          <div className="flex justify-between gap-3 text-sm">
                            <div className="font-semibold text-[oklch(0.95_0.02_260)]">
                              {hit.item.title}
                            </div>
                            <div className="text-[oklch(0.6_0.02_260)]">
                              score {hit.score.toFixed(3)}
                            </div>
                          </div>
                          <p className="text-xs text-[oklch(0.7_0.02_260)] mt-1 line-clamp-2">
                            {hit.item.content}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="space-y-3">
                {sortedItems.length === 0 ? (
                  <div className="text-sm text-[oklch(0.7_0.02_260)] border border-dashed border-[oklch(0.25_0.04_260)] rounded-lg p-6 text-center">
                    Belum ada knowledge yang ditambahkan.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {sortedItems.map((item) => (
                      <div
                        key={item.id}
                        className="border border-[oklch(0.25_0.04_260)] rounded-lg p-4 bg-[oklch(0.2_0.05_260)] space-y-3"
                      >
                        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
                          <div>
                            <div className="text-lg font-semibold text-[oklch(0.95_0.02_260)]">
                              {item.title}
                            </div>
                            {item.url && (
                              <a
                                href={item.url}
                                target="_blank"
                                rel="noreferrer"
                                className="text-xs text-[oklch(0.7_0.02_260)] underline"
                              >
                                {item.url}
                              </a>
                            )}
                          </div>
                          <div className="text-xs text-[oklch(0.6_0.02_260)] text-right space-y-1">
                            <div>Updated {formatDate(item.updatedAt)}</div>
                            <div>
                              Dim: {item.embeddingDim} • Model:{' '}
                              {item.modelUrl?.split('/').pop() ?? 'unknown'}
                            </div>
                          </div>
                        </div>
                        <p className="text-sm text-[oklch(0.75_0.03_260)] whitespace-pre-wrap">
                          {item.content}
                        </p>
                        <div className="flex flex-wrap gap-2 justify-end">
                          <button
                            className="btn btn-xs"
                            onClick={() => handleEditKnowledge(item)}
                          >
                            Edit
                          </button>
                          <button
                            className="btn btn-xs"
                            onClick={() => handleReembedKnowledge(item)}
                          >
                            Re-embed
                          </button>
                          <button
                            className="btn btn-xs btn-error"
                            onClick={() => handleDeleteKnowledge(item)}
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}

