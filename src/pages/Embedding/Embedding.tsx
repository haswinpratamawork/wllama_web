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
  DEFAULT_EMBEDDING_MODEL,
  getSavedEmbeddingPreference,
  loadEmbeddingModel,
  loadEmbeddingModelFromFiles,
  resetEmbeddingRuntime,
} from '../../utils/embedding-runtime';
import {
  useRag,
  type KnowledgeItem,
  type KnowledgeImportRecord,
  type RagSearchHit,
} from '../../utils/rag.context';
import {
  formatDuration as formatDurationMs,
  getPerfLabel,
  usePerfMonitor,
  type PerfEvent,
} from '../../utils/perf-monitor';

const formatDate = (value: number) =>
  new Date(value).toLocaleString(undefined, {
    hour12: false,
  });

export default function EmbeddingPage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const knowledgeJsonInputRef = useRef<HTMLInputElement | null>(null);
  const autoLoadAttempted = useRef(false);

  const [status, setStatus] = useState<string>('idle');
  const [progressPercent, setProgressPercent] = useState<number | null>(null);
  const [modelUrl, setModelUrl] = useState<string>(DEFAULT_EMBEDDING_MODEL);
  const [formText, setFormText] = useState<string>('');
  const [editing, setEditing] = useState<KnowledgeItem | null>(null);
  const [savingKnowledge, setSavingKnowledge] = useState(false);
  const [importingKnowledge, setImportingKnowledge] = useState(false);

  const [searchText, setSearchText] = useState('');
  const [searchResults, setSearchResults] = useState<RagSearchHit[]>([]);
  const [searching, setSearching] = useState(false);
  const perfState = usePerfMonitor();

  const summaryItems = useMemo(() => {
    const keys = [
      'embedding:generate',
      'rag:search',
      'hnsw:search',
      'hnsw:rebuild',
      'answer:generate',
      'idb:getAllKnowledge',
    ];
    return keys
      .map((key) => {
        const event = perfState.latestByName[key];
        if (!event) return null;
        return { key, event };
      })
      .filter(Boolean) as Array<{ key: string; event: PerfEvent }>;
  }, [perfState.latestByName]);

  const recentEvents = useMemo(
    () => [...perfState.events].reverse().slice(0, 10),
    [perfState.events]
  );

  const {
    ragEnabled,
    toggleRag,
    addKnowledge,
    updateKnowledge,
    deleteKnowledge,
    reembedKnowledge,
    importKnowledgeFromJson,
    searchByText,
    items,
    embeddingModel,
    knowledgeCount,
  } = useRag();

  const modelLoaded = Boolean(embeddingModel);

  const sortedItems = useMemo(() => {
    return [...items].sort((a, b) => b.updatedAt - a.updatedAt);
  }, [items]);

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
          `model ready — ${saved.name ?? saved.url.split('/').pop() ?? saved.url
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

  const handleUploadKnowledgeJson = async (files: FileList | null) => {
    if (!files || files.length === 0 || importingKnowledge) return;
    const file = files[0];
    setImportingKnowledge(true);
    try {
      setStatus(`memuat knowledge dari ${file.name}`);
      const rawText = await file.text();
      const parsed = JSON.parse(rawText);
      if (!Array.isArray(parsed)) {
        throw new Error('File JSON harus berupa array knowledge');
      }
      const normalized = parsed.filter(
        (item: any): item is KnowledgeImportRecord =>
          item &&
          typeof item === 'object' &&
          typeof item.text === 'string'
      );
      if (!normalized.length) {
        throw new Error('Tidak ada record valid pada JSON');
      }
      const result = await importKnowledgeFromJson(normalized);
      const summaryParts: string[] = [];
      if (result.imported) summaryParts.push(`${result.imported} baru`);
      if (result.updated) summaryParts.push(`${result.updated} diperbarui`);
      if (result.skipped) summaryParts.push(`${result.skipped} dilewati`);
      setStatus(
        summaryParts.length
          ? `Impor knowledge selesai (${summaryParts.join(', ')})`
          : 'Impor knowledge selesai'
      );
      if (result.errors.length) {
        console.warn(
          '[EmbeddingPage] Beberapa record gagal diimpor',
          result.errors
        );
      }
    } catch (err: any) {
      console.error('[EmbeddingPage] Gagal mengimpor knowledge', err);
      setStatus('gagal mengimpor knowledge: ' + (err?.message ?? String(err)));
    } finally {
      setImportingKnowledge(false);
      if (knowledgeJsonInputRef.current) {
        knowledgeJsonInputRef.current.value = '';
      }
    }
  };

  const handleResetRuntime = async () => {
    await resetEmbeddingRuntime();
    setStatus('runtime reset — model unloaded');
  };

  const resetForm = () => {
    setFormText('');
    setEditing(null);
  };

  const handleKnowledgeSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const text = formText.trim();
    if (!text) {
      setStatus('isi teks knowledge terlebih dahulu');
      return;
    }
    setSavingKnowledge(true);
    try {
      if (editing) {
        await updateKnowledge(editing.id, {
          text,
        });
        setStatus('knowledge updated');
      } else {
        await addKnowledge({
          text,
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
    setFormText(item.text);
  };

  const handleDeleteKnowledge = async (item: KnowledgeItem) => {
    const preview = item.text.slice(0, 80);
    const confirmed = window.confirm(
      `Hapus knowledge berikut?\n\n${preview}${item.text.length > 80 ? '...' : ''}`
    );
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
      setStatus(`re-embedding knowledge #${item.id}`);
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
      const hits = await searchByText(query, { topK: 50 });
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
                className={`px-3 py-1 rounded-full text-sm font-medium ${modelLoaded
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
                  Performance Monitor
                </h2>
                <div className="text-xs text-[oklch(0.7_0.02_260)]">
                  Pantau durasi proses embedding, RAG, HNSW, dan IndexedDB.
                </div>
              </div>

              {summaryItems.length ? (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {summaryItems.map(({ key, event }) => {
                    const metaEntries = event.meta
                      ? Object.entries(event.meta).map(([metaKey, metaValue]) => {
                        if (metaValue === undefined || metaValue === null) {
                          return `${metaKey}: -`;
                        }
                        if (typeof metaValue === 'object') {
                          return `${metaKey}: ${JSON.stringify(metaValue)}`;
                        }
                        return `${metaKey}: ${metaValue}`;
                      })
                      : [];
                    return (
                      <div
                        key={key}
                        className="p-4 bg-[oklch(0.2_0.05_260)] border border-[oklch(0.25_0.04_260)] rounded-lg space-y-2"
                      >
                        <div className="text-xs uppercase tracking-wide text-[oklch(0.7_0.02_260)]">
                          {getPerfLabel(key)}
                        </div>
                        <div className="text-2xl font-semibold text-[oklch(0.95_0.02_260)]">
                          {formatDurationMs(event.durationMs)}
                        </div>
                        <div className="text-[0.7rem] text-[oklch(0.65_0.02_260)]">
                          Update {formatDate(event.wallClock)}
                        </div>
                        {metaEntries.length > 0 && (
                          <div className="text-[0.65rem] text-[oklch(0.7_0.02_260)] space-y-1">
                            {metaEntries.map((entry) => (
                              <div key={entry}>{entry}</div>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-sm text-[oklch(0.7_0.02_260)] border border-dashed border-[oklch(0.25_0.04_260)] rounded-lg p-4">
                  Belum ada data performa yang terekam. Lakukan operasi seperti menambah knowledge atau menjalankan RAG untuk mulai mengumpulkan data.
                </div>
              )}

              {recentEvents.length > 0 && (
                <div className="space-y-2">
                  <div className="text-sm font-medium text-[oklch(0.9_0.02_260)]">
                    Riwayat terbaru
                  </div>
                  <div className="overflow-auto border border-[oklch(0.25_0.04_260)] rounded-lg">
                    <table className="table table-xs text-[0.7rem]">
                      <thead>
                        <tr className="bg-[oklch(0.2_0.05_260)] text-[oklch(0.75_0.02_260)]">
                          <th>Waktu</th>
                          <th>Event</th>
                          <th>Durasi</th>
                          <th>Detail</th>
                        </tr>
                      </thead>
                      <tbody>
                        {recentEvents.map((event) => {
                          const metaEntries = event.meta
                            ? Object.entries(event.meta)
                              .map(
                                ([metaKey, metaValue]) =>
                                  `${metaKey}: ${
                                    typeof metaValue === 'object'
                                      ? JSON.stringify(metaValue)
                                      : metaValue ?? '-'
                                  }`
                              )
                              .join(' • ')
                            : '';
                          return (
                            <tr key={event.id} className="hover:bg-[oklch(0.18_0.04_260)]">
                              <td>{formatDate(event.wallClock)}</td>
                              <td>{getPerfLabel(event.name)}</td>
                              <td>{formatDurationMs(event.durationMs)}</td>
                              <td className="max-w-xs whitespace-pre-wrap">
                                {metaEntries || '-'}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </section>

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

            <section className="space-y-5">
              <div className="flex items-center justify-between flex-wrap gap-3">
                <h2 className="text-xl font-semibold text-[oklch(0.95_0.02_260)]">
                  Knowledge Management
                </h2>
                <div className="text-sm text-[oklch(0.7_0.02_260)]">
                  Total knowledge: {knowledgeCount}
                </div>
              </div>

              <div className="bg-[oklch(0.2_0.05_260)] border border-[oklch(0.25_0.04_260)] rounded-lg p-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                <div>
                  <div className="text-sm font-medium text-[oklch(0.95_0.02_260)]">
                    Upload JSON knowledge
                  </div>
                  <p className="text-xs text-[oklch(0.7_0.02_260)]">
                    Gunakan schema {`{ id, text, embedding }`} untuk mengimpor beberapa knowledge sekaligus.
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  <input
                    ref={knowledgeJsonInputRef}
                    type="file"
                    accept="application/json,.json"
                    className="hidden"
                    onChange={(e) => handleUploadKnowledgeJson(e.target.files)}
                  />
                  <button
                    type="button"
                    className="btn btn-sm btn-accent"
                    onClick={() => knowledgeJsonInputRef.current?.click()}
                    disabled={importingKnowledge}
                  >
                    {importingKnowledge ? 'Mengimpor...' : 'Upload JSON'}
                  </button>
                </div>
              </div>

              <form
                onSubmit={handleKnowledgeSubmit}
                className="space-y-4 bg-[oklch(0.2_0.05_260)] border border-[oklch(0.25_0.04_260)] rounded-lg p-4"
              >
                <label className="space-y-2 block">
                  <span className="block text-sm font-medium text-[oklch(0.8_0.02_260)]">
                    Teks knowledge
                  </span>
                  <textarea
                    className="textarea textarea-sm w-full min-h-[160px] bg-[oklch(0.18_0.04_260)]"
                    value={formText}
                    onChange={(e) => setFormText(e.target.value)}
                    placeholder="Tuliskan pengetahuan yang ingin kamu simpan untuk RAG"
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
                            <p className="font-semibold text-[oklch(0.95_0.02_260)] line-clamp-3">
                              {hit.item.text}
                            </p>
                            <div className="text-[oklch(0.6_0.02_260)]">
                              score {hit.score.toFixed(3)}
                            </div>
                          </div>
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
                        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-2">
                          <p className="text-sm text-[oklch(0.95_0.02_260)] whitespace-pre-wrap flex-1">
                            {item.text}
                          </p>
                          <div className="text-xs text-[oklch(0.6_0.02_260)] text-right space-y-1">
                            <div>Updated {formatDate(item.updatedAt)}</div>
                            <div>
                              Dim: {item.embeddingDim} • Model:{' '}
                              {item.modelUrl?.split('/').pop() ?? 'unknown'}
                            </div>
                          </div>
                        </div>
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
