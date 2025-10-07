import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type JSX,
} from 'react';
import { ModelManager, Wllama } from '@wllama/wllama';
import { Link } from 'react-router-dom';
import { DebugLogger, WllamaStorage } from '../../utils/utils';
import { storeLocalModelFiles } from '../../utils/local-model-cache';

type SavedEmbeddingModel = {
  url: string;
  source: 'remote' | 'local';
  name?: string;
};

const CONFIG_PATHS = {
  'single-thread/wllama.wasm': '/wllama/esm/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm': '/wllama/esm/multi-thread/wllama.wasm',
};

const DEFAULT_MODEL =
  'https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf';

const embeddingModelManager = new ModelManager({ logger: DebugLogger });
let sharedEmbeddingInstance: Wllama | null = null;
let sharedEmbeddingLoaded = false;
let sharedEmbeddingUrl: string | null = null;
let sharedEmbeddingLabel: string | null = null;

export default function EmbeddingPage(): JSX.Element {
  const savedModelRef = useRef<SavedEmbeddingModel | null>(
    WllamaStorage.load<SavedEmbeddingModel | null>('embedding_model', null)
  );

  const [wllama, setWllama] = useState<Wllama | null>(sharedEmbeddingInstance);
  const [modelLoaded, setModelLoaded] = useState<boolean>(
    sharedEmbeddingLoaded
  );
  const [status, setStatus] = useState<string>(
    sharedEmbeddingLoaded && sharedEmbeddingLabel
      ? `model ready — ${sharedEmbeddingLabel}`
      : 'idle'
  );
  const [progress, setProgress] = useState<number | null>(null);
  const [modelUrl, setModelUrl] = useState<string>(
    savedModelRef.current?.url ?? DEFAULT_MODEL
  );
  const [text, setText] = useState(
    'Saya sedang menguji embedding Gemma untuk demo WASM di browser.'
  );
  const [embedding, setEmbedding] = useState<number[] | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const importWllama = useCallback(async () => {
    if (sharedEmbeddingInstance) {
      setWllama(sharedEmbeddingInstance);
      return sharedEmbeddingInstance;
    }
    setStatus('initializing runtime');
    const inst = new Wllama(CONFIG_PATHS);
    sharedEmbeddingInstance = inst;
    setWllama(inst);
    setStatus('runtime ready');
    return inst;
  }, []);

  const finalizeLoad = useCallback(
    (
      label: string,
      url: string,
      source: 'remote' | 'local',
      tookMs: number,
      dim: number | string,
      silent?: boolean
    ) => {
      setModelLoaded(true);
      setModelUrl(url);
      const dimLabel =
        dim === 'unknown' ? String(dim) : Number(dim).toLocaleString();
      if (silent) {
        setStatus(`model ready — ${label}`);
      } else {
        const prefix =
          source === 'remote' ? 'model loaded' : 'local model loaded';
        setStatus(`${prefix} (took ${tookMs.toLocaleString()} ms). dim=${dimLabel}`);
      }
      sharedEmbeddingLoaded = true;
      sharedEmbeddingUrl = url;
      sharedEmbeddingLabel = label;
      savedModelRef.current = { url, source, name: label };
      WllamaStorage.save('embedding_model', savedModelRef.current);
    },
    []
  );

  const loadModelFromCache = useCallback(
    async (
      targetUrl: string,
      meta: { label?: string; source: 'remote' | 'local'; silent?: boolean }
    ) => {
      if (!targetUrl) {
        setStatus('model URL tidak valid');
        return;
      }
      const silent = meta.silent ?? false;
      const label =
        meta.label ?? targetUrl.split('/').pop() ?? targetUrl;
      try {
        if (!silent) {
          setStatus(
            meta.source === 'remote'
              ? 'loading model from url'
              : 'loading local model'
          );
          setProgress(0);
        }
        const inst = await importWllama();
        const start = Date.now();
        let model;
        if (meta.source === 'local' || targetUrl.startsWith('local://')) {
          const available = await embeddingModelManager.getModels({
            includeInvalid: true,
          });
          model = available.find((m) => m.url === targetUrl);
          if (!model) {
            throw new Error('Local model tidak ditemukan di cache');
          }
        } else {
          model = await embeddingModelManager.getModelOrDownload(targetUrl, {
            progressCallback: ({ loaded, total }) => {
              if (!silent && total) {
                const pct = Math.round((loaded / total) * 100);
                setProgress(pct);
              }
            },
          });
        }
        const blobs = await model.open();
        await inst.loadModel(blobs, {
          embeddings: true,
          n_ctx: 2048,
          pooling_type: 'LLAMA_POOLING_TYPE_MEAN',
        });
        const took = Date.now() - start;
        const dimGetter = (inst as any)?.getEmbeddingSize;
        const dim = typeof dimGetter === 'function' ? dimGetter.call(inst) : 'unknown';
        finalizeLoad(label, targetUrl, meta.source, took, dim, silent);
      } catch (err: any) {
        console.error(err);
        if (!silent) {
          setStatus(
            `failed to load ${
              meta.source === 'remote' ? 'model' : 'local model'
            }: ${err?.message ?? String(err)}`
          );
        }
        setModelLoaded(false);
      } finally {
        setProgress(null);
      }
    },
    [finalizeLoad, importWllama]
  );

  const loadModelFromUrl = useCallback(
    async (url: string, opts: { silent?: boolean } = {}) => {
      const trimmed = url.trim();
      if (!trimmed) {
        setStatus('model URL tidak valid');
        return;
      }
      await loadModelFromCache(trimmed, {
        source: 'remote',
        label: trimmed.split('/').pop() ?? trimmed,
        silent: opts.silent,
      });
    },
    [loadModelFromCache]
  );

  const loadModelFromFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const fileArr = Array.from(files);
      for (const file of fileArr) {
        if (!file.name.toLowerCase().endsWith('.gguf')) {
          setStatus('Hanya file .gguf yang didukung');
          return;
        }
        if (file.size === 0) {
          setStatus('File model tidak boleh kosong');
          return;
        }
      }
      try {
        setStatus('saving local model');
        setProgress(0);
        const { modelUrl: storedUrl, baseName } = await storeLocalModelFiles(
          fileArr,
          embeddingModelManager,
          {
            namespace: `embedding-${Date.now()
              .toString(36)
              .slice(2)}-${Math.random().toString(36).slice(2, 6)}`,
          }
        );
        await loadModelFromCache(storedUrl, {
          source: 'local',
          label: baseName,
        });
      } catch (err: any) {
        console.error(err);
        setStatus(
          'failed to load local model: ' + (err?.message ?? String(err))
        );
        setModelLoaded(false);
      } finally {
        setProgress(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    },
    [loadModelFromCache]
  );

  const createEmbedding = useCallback(
    async (inputText: string) => {
      const inst = wllama ?? sharedEmbeddingInstance;
      if (!inst || !modelLoaded) {
        setStatus('load a model first');
        return;
      }
      if (!inputText) {
        setStatus('type some text to embed');
        return;
      }
      try {
        setStatus('creating embedding');
        const t0 = Date.now();
        const vec: number[] = await (inst as any).createEmbedding(inputText, {
          skipBOS: true,
          skipEOS: true,
        });
        const took = Date.now() - t0;
        setEmbedding(vec);
        setStatus(
          `embedding created (took ${took.toLocaleString()} ms). length=${vec.length}`
        );
      } catch (err: any) {
        console.error(err);
        setStatus('embedding failed: ' + (err?.message ?? String(err)));
      }
    },
    [wllama, modelLoaded]
  );

  const resetRuntime = useCallback(() => {
    if (sharedEmbeddingInstance) {
      try {
        (sharedEmbeddingInstance as any)?.release?.();
      } catch (e) {
        console.warn('failed to release embedding runtime', e);
      }
    }
    sharedEmbeddingInstance = null;
    sharedEmbeddingLoaded = false;
    sharedEmbeddingUrl = null;
    sharedEmbeddingLabel = null;
    savedModelRef.current = null;
    WllamaStorage.remove('embedding_model');
    setWllama(null);
    setModelLoaded(false);
    setStatus('runtime reset');
    setEmbedding(null);
    setProgress(null);
  }, []);

  useEffect(() => {
    if (sharedEmbeddingInstance) {
      setWllama(sharedEmbeddingInstance);
    }
    if (sharedEmbeddingLoaded) {
      setModelLoaded(true);
      if (sharedEmbeddingUrl) {
        setModelUrl(sharedEmbeddingUrl);
      }
      if (sharedEmbeddingLabel) {
        setStatus(`model ready — ${sharedEmbeddingLabel}`);
      }
      return;
    }
    if (savedModelRef.current?.url) {
      loadModelFromCache(savedModelRef.current.url, {
        source: savedModelRef.current.source,
        label: savedModelRef.current.name,
      });
    }
  }, [loadModelFromCache]);

  const first128 = useMemo(
    () => (embedding ? embedding.slice(0, 128) : null),
    [embedding]
  );

  return (
    <div className="min-h-screen bg-[oklch(0.15_0.03_260)] text-[oklch(0.9_0.02_260)] transition-colors duration-200 p-6 md:p-10">
      <div className="max-w-4xl mx-auto">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[oklch(0.3_0.12_270)] text-[oklch(0.98_0.02_260)] text-sm font-medium hover:opacity-95">
          <Link to="/"> Kembali ke Chat</Link>
        </div>

        <div className="bg-[oklch(0.18_0.04_260)] shadow-lg rounded-2xl border border-[oklch(0.25_0.04_260)] overflow-hidden transition-colors">
          <div className="flex items-center justify-between gap-4 p-6 md:p-8">
            <div>
              <h1 className="text-2xl md:text-3xl font-semibold text-[oklch(0.95_0.02_260)]">
                Embedding Llama
              </h1>
              <p className="text-sm text-[oklch(0.7_0.02_260)] mt-1">
                Tool demo — Mohamad Arsya Kaukabi
              </p>
            </div>

            <div className="flex items-center gap-3">
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
                {progress !== null ? ` • ${progress}%` : ''}
              </div>
            </div>
          </div>

          <div className="p-6 md:p-8 border-t border-[oklch(0.25_0.04_260)] space-y-6">
            <section className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-[oklch(0.8_0.02_260)] mb-2">
                  Model URL (GGUF)
                </label>
                <textarea
                  className="w-full rounded-lg border border-[oklch(0.25_0.04_260)] shadow-sm p-3 text-sm focus:outline-none focus:ring-2 focus:ring-[oklch(0.6_0.1_260)] resize-none bg-[oklch(0.18_0.04_260)] text-[oklch(0.95_0.02_260)]"
                  value={modelUrl}
                  onChange={(e) => setModelUrl(e.target.value)}
                  placeholder="https://.../model.gguf or leave empty to upload"
                  rows={2}
                />

                <div className="mt-3 flex flex-wrap gap-3">
                  <button
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[oklch(0.3_0.12_270)] text-[oklch(0.98_0.02_260)] text-sm font-medium hover:opacity-95"
                    onClick={() => loadModelFromUrl(modelUrl)}
                  >
                    Load from URL
                  </button>

                  <label className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-[oklch(0.25_0.04_260)] text-sm cursor-pointer bg-[oklch(0.2_0.04_260)] shadow-sm">
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".gguf"
                      multiple
                      className="hidden"
                      onChange={(e) => loadModelFromFiles(e.target.files)}
                    />
                    Upload GGUF
                  </label>

                  <button
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-[oklch(0.25_0.04_260)] text-sm bg-[oklch(0.18_0.04_260)] hover:bg-[oklch(0.22_0.05_260)]"
                    onClick={resetRuntime}
                  >
                    Reset runtime
                  </button>
                </div>
              </div>

              <div className="space-y-3 text-sm text-[oklch(0.78_0.03_260)] bg-[oklch(0.2_0.05_260)] px-4 py-3 rounded-lg border border-[oklch(0.25_0.04_260)]">
                <div className="font-semibold text-[oklch(0.95_0.02_260)]">
                  Tips
                </div>
                <ul className="list-disc list-inside space-y-1">
                  <li>Gunakan model embedding berbasis GGUF.</li>
                  <li>Upload beberapa shard sekaligus jika model terpecah.</li>
                  <li>
                    Model yang pernah dimuat akan disimpan sehingga bisa dipakai
                    kembali.
                  </li>
                </ul>
              </div>
            </section>

            <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-[oklch(0.8_0.02_260)] mb-2">
                  Text to embed
                </label>
                <textarea
                  className="w-full rounded-lg border border-[oklch(0.25_0.04_260)] shadow-sm p-3 text-sm focus:outline-none focus:ring-2 focus:ring-[oklch(0.6_0.1_260)] resize-none bg-[oklch(0.18_0.04_260)] text-[oklch(0.95_0.02_260)]"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  rows={5}
                />

                <div className="mt-3 flex flex-wrap gap-3">
                  <button
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[oklch(0.3_0.12_270)] text-[oklch(0.98_0.02_260)] text-sm font-medium hover:opacity-95"
                    onClick={() => createEmbedding(text)}
                  >
                    Create embedding
                  </button>
                  <button
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-[oklch(0.25_0.04_260)] text-sm bg-[oklch(0.18_0.04_260)] hover:bg-[oklch(0.22_0.05_260)]"
                    onClick={() => {
                      setEmbedding(null);
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
                    Vector length:{' '}
                    <span className="font-medium text-[oklch(0.95_0.02_260)]">
                      {embedding ? embedding.length : '-'}
                    </span>
                  </div>
                  <pre className="mt-2 text-xs whitespace-pre-wrap break-all text-[oklch(0.75_0.03_260)] bg-[oklch(0.18_0.04_260)] rounded-lg p-3 border border-[oklch(0.25_0.04_260)] max-h-60 overflow-auto">
                    {first128
                      ? JSON.stringify(first128, null, 2) +
                        (embedding && embedding.length > 128
                          ? '\n\n... (first 128 elements)'
                          : '')
                      : '—'}
                  </pre>
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
