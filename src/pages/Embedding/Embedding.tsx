import { useCallback, useEffect, useMemo, useRef, useState, type JSX } from 'react';
import { Wllama } from '@wllama/wllama';
import { Link } from 'react-router-dom';

type Progress = { loaded?: number; total?: number };

const CONFIG_PATHS = {
  'single-thread/wllama.wasm': '/wllama/esm/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm': '/wllama/esm/multi-thread/wllama.wasm',
};

const DEFAULT_MODEL =
  'https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf';

export default function EmbeddingPage(): JSX.Element {
  const [wllama, setWllama] = useState<any | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState<number | null>(null);
  const [modelUrl, setModelUrl] = useState(DEFAULT_MODEL);
  const [text, setText] = useState('Saya sedang menguji embedding Gemma untuk demo WASM di browser.');
  const [embedding, setEmbedding] = useState<number[] | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);


  const importWllama = useCallback(async () => {
    if (wllama) return wllama;
    setStatus('initializing runtime');
    const inst = new Wllama(CONFIG_PATHS);
    setWllama(inst);
    setStatus('runtime ready');
    return inst;
  }, [wllama]);

  const onProgress = useCallback((p: Progress) => {
    if (!p || !p.total) return;
    const pct = Math.round(((p.loaded ?? 0) / (p.total ?? 1)) * 100);
    setProgress(pct);
  }, []);

  const loadModelFromUrl = useCallback(
    async (url: string) => {
      try {
        setStatus('loading model from url');
        setProgress(0);
        const inst = await importWllama();
        const start = Date.now();
        await inst.loadModelFromUrl(url, {
          embeddings: true,
          n_ctx: 2048,
          pooling_type: 'LLAMA_POOLING_TYPE_MEAN',
          progressCallback: onProgress,
        });
        const took = Date.now() - start;
        setStatus(
          `model loaded (took ${took.toLocaleString()} ms). dim=${inst.getEmbeddingSize ? inst.getEmbeddingSize() : 'unknown'}`
        );
        setModelLoaded(true);
      } catch (err: any) {
        console.error(err);
        setStatus('failed to load model: ' + (err?.message ?? String(err)));
        setModelLoaded(false);
      } finally {
        setProgress(null);
      }
    },
    [importWllama, onProgress]
  );

  const loadModelFromFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) return;
      try {
        setStatus('loading model from files');
        setProgress(0);
        const inst = await importWllama();
        const start = Date.now();
        const blobs = Array.from(files);
        await inst.loadModel(blobs, {
          embeddings: true,
          n_ctx: 2048,
          pooling_type: 'LLAMA_POOLING_TYPE_MEAN',
          progressCallback: onProgress,
        });
        const took = Date.now() - start;
        setStatus(`model loaded from files (took ${took.toLocaleString()} ms)`);
        setModelLoaded(true);
      } catch (err: any) {
        console.error(err);
        setStatus('failed to load local model: ' + (err?.message ?? String(err)));
        setModelLoaded(false);
      } finally {
        setProgress(null);
      }
    },
    [importWllama, onProgress]
  );

  const createEmbedding = useCallback(
    async (inputText: string) => {
      if (!wllama || !modelLoaded) {
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
        const vec: number[] = await wllama.createEmbedding(inputText, { skipBOS: true, skipEOS: true });
        const took = Date.now() - t0;
        setEmbedding(vec);
        setStatus(`embedding created (took ${took.toLocaleString()} ms). length=${vec.length}`);
      } catch (err: any) {
        console.error(err);
        setStatus('embedding failed: ' + (err?.message ?? String(err)));
      }
    },
    [wllama, modelLoaded]
  );

  const resetRuntime = useCallback(() => {
    if (wllama) {
      try {
        wllama.release?.();
      } catch (e) { }
    }
    setWllama(null);
    setModelLoaded(false);
    setStatus('runtime reset');
    setEmbedding(null);
    setProgress(null);
  }, [wllama]);

  useEffect(() => {
    return () => {
      try {
        wllama?.release?.();
      } catch (e) { }
    };
  }, [wllama]);

  const first128 = useMemo(() => (embedding ? embedding.slice(0, 128) : null), [embedding]);

  return (
    <div className="min-h-screen bg-[oklch(0.15_0.03_260)] text-[oklch(0.9_0.02_260)] transition-colors duration-200 p-6 md:p-10">
      <div className="max-w-4xl mx-auto">
        <div
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[oklch(0.3_0.12_270)] text-[oklch(0.98_0.02_260)] text-sm font-medium hover:opacity-95"
        >
          <Link to="/"> Kembali ke Chat</Link>
        </div>

        <div className="bg-[oklch(0.18_0.04_260)] shadow-lg rounded-2xl border border-[oklch(0.25_0.04_260)] overflow-hidden transition-colors">
          <div className="flex items-center justify-between gap-4 p-6 md:p-8">

            <div>
              <h1 className="text-2xl md:text-3xl font-semibold text-[oklch(0.95_0.02_260)]">Embedding Llama</h1>
              <p className="text-sm text-[oklch(0.7_0.02_260)] mt-1">Tool demo — Mohamad Arsya Kaukabi</p>
            </div>

            <div className="flex items-center gap-3">


              <div className={`px-3 py-1 rounded-full text-sm font-medium ${modelLoaded ? 'bg-[oklch(0.4_0.2_150)] text-[oklch(0.95_0.02_150)]' : 'bg-[oklch(0.4_0.1_80)] text-[oklch(0.95_0.02_80)]'}`}>
                {modelLoaded ? 'Model ready' : 'Model not loaded'}
              </div>
              <div className="text-sm text-[oklch(0.7_0.02_260)] text-right">{status}{progress !== null ? ` • ${progress}%` : ''}</div>
            </div>
          </div>

          <div className="p-6 md:p-8 border-t border-[oklch(0.25_0.04_260)] space-y-6">
            <section className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-[oklch(0.8_0.02_260)] mb-2">Model URL (GGUF)</label>
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
                    Upload .gguf
                  </label>

                  <button
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[oklch(0.25_0.03_260)] text-sm text-[oklch(0.95_0.02_260)]"
                    onClick={resetRuntime}
                  >
                    Reset runtime
                  </button>
                </div>

                {progress !== null && (
                  <div className="mt-4">
                    <div className="text-xs text-[oklch(0.7_0.02_260)] mb-1">Loading progress</div>
                    <div className="w-full bg-[oklch(0.22_0.03_260)] rounded-md h-3 overflow-hidden">
                      <div className="h-3 bg-[oklch(0.65_0.2_270)] transition-all" style={{ width: `${progress}%` }} />
                    </div>
                  </div>
                )}
              </div>

              <div className="bg-[oklch(0.2_0.04_260)] border border-[oklch(0.25_0.04_260)] rounded-lg p-4">
                <div className="text-xs text-[oklch(0.7_0.02_260)] mb-2">Model info</div>
                <div className="text-sm text-[oklch(0.95_0.02_260)]">{modelLoaded ? 'Embedding model siap digunakan' : 'Belum ada model'}</div>
                <div className="mt-3 text-xs text-[oklch(0.6_0.02_260)]">Tip: gunakan model GGUF yang sudah compatible untuk WASM.</div>
              </div>
            </section>

            <section>
              <label className="block text-sm font-medium text-[oklch(0.8_0.02_260)] mb-2">Text to embed</label>
              <textarea
                className="w-full rounded-lg border border-[oklch(0.25_0.04_260)] shadow-sm p-3 text-sm focus:outline-none focus:ring-2 focus:ring-[oklch(0.6_0.1_260)] bg-[oklch(0.18_0.04_260)] text-[oklch(0.95_0.02_260)]"
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={4}
                placeholder="Masukkan teks untuk di-embed..."
              />

              <div className="mt-3 flex flex-wrap gap-3 items-center">
                <button
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[oklch(0.65_0.2_270)] text-white text-sm font-medium hover:opacity-95"
                  onClick={() => createEmbedding(text)}
                >
                  Create embedding
                </button>
                <button
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[oklch(0.2_0.04_260)] border border-[oklch(0.25_0.04_260)] text-sm text-[oklch(0.95_0.02_260)]"
                  onClick={() => {
                    setEmbedding(null);
                    setStatus('cleared');
                  }}
                >
                  Clear
                </button>

                <div className="ml-auto text-sm text-[oklch(0.7_0.02_260)]">
                  Vector length: <span className="font-medium text-[oklch(0.95_0.02_260)]">{embedding ? embedding.length : '-'}</span>
                </div>
              </div>
            </section>

            <section className="bg-[oklch(0.2_0.04_260)] border border-[oklch(0.25_0.04_260)] rounded-lg p-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-[oklch(0.95_0.02_260)]">Result</h3>
                <div className="text-xs text-[oklch(0.7_0.02_260)]">{embedding ? `Created — ${embedding.length} dims` : 'No embedding yet.'}</div>
              </div>

              <pre className="mt-3 overflow-auto text-xs text-[oklch(0.95_0.02_260)] bg-[oklch(0.18_0.04_260)] p-3 rounded-md border border-[oklch(0.25_0.04_260)]" style={{ maxHeight: 300 }}>
                {first128 ? JSON.stringify(first128, null, 2) + (embedding && embedding.length > 128 ? '\n\n... (first 128 elements)' : '') : '—'}
              </pre>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}