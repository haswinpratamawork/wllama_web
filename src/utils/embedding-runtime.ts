import { ModelManager, Wllama } from '@wllama/wllama';
import { DebugLogger, WllamaStorage } from './utils';
import { withPerfTimer } from './perf-monitor';
import { storeLocalModelFiles } from './local-model-cache';

export type EmbeddingModelSource = 'remote' | 'local';

export interface SavedEmbeddingModel {
  url: string;
  source: EmbeddingModelSource;
  name?: string;
}

export interface EmbeddingModelState {
  url: string;
  label: string;
  source: EmbeddingModelSource;
  dim: number;
  loadedAt: number;
}

export interface LoadEmbeddingOptions {
  url: string;
  source: EmbeddingModelSource;
  label?: string;
  onProgress?: (progress: number | null) => void;
  silent?: boolean;
}

export interface LoadLocalFilesOptions {
  files: FileList | File[];
  namespace?: string;
  onProgress?: (progress: number | null) => void;
}

const CONFIG_PATHS = {
  'single-thread/wllama.wasm': '/wllama/esm/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm': '/wllama/esm/multi-thread/wllama.wasm',
};

export const DEFAULT_EMBEDDING_MODEL =
  'https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf';

const embeddingModelManager = new ModelManager({ logger: DebugLogger });

let embeddingInstance: Wllama | null = null;
let currentModel: EmbeddingModelState | null = null;
let loadingPromise: Promise<EmbeddingModelState> | null = null;

type Listener = (model: EmbeddingModelState | null) => void;
const listeners = new Set<Listener>();

const notify = () => {
  listeners.forEach((listener) => listener(currentModel));
};

export const onEmbeddingModelChange = (listener: Listener) => {
  listeners.add(listener);
  listener(currentModel);
  return () => listeners.delete(listener);
};

const createRuntime = () => {
  embeddingInstance = new Wllama(CONFIG_PATHS, { logger: DebugLogger });
  return embeddingInstance;
};

export const getEmbeddingRuntime = async (): Promise<Wllama> => {
  if (embeddingInstance) return embeddingInstance;
  return createRuntime();
};

export const getEmbeddingModelManager = () => embeddingModelManager;

const finalizeLoad = (
  state: EmbeddingModelState,
  persist = true,
  silent = false
) => {
  currentModel = state;
  if (persist) {
    const saved: SavedEmbeddingModel = {
      url: state.url,
      source: state.source,
      name: state.label,
    };
    WllamaStorage.save('embedding_model', saved);
  }
  if (!silent) {
    console.info(
      `[EmbeddingRuntime] Model ready (${state.label}) dim=${state.dim}`
    );
  }
  notify();
};

export const getCurrentEmbeddingModel = () => currentModel;

const loadModelFromManager = async (
  inst: Wllama,
  opts: LoadEmbeddingOptions
): Promise<EmbeddingModelState> => {
  const { url, source, label, onProgress, silent } = opts;
  const trimmedUrl = url.trim();
  if (!trimmedUrl) {
    throw new Error('Model URL tidak valid');
  }

  const now = performance.now();
  let targetModel;
  if (source === 'local' || trimmedUrl.startsWith('local://')) {
    const available = await embeddingModelManager.getModels({
      includeInvalid: true,
    });
    targetModel = available.find((m) => m.url === trimmedUrl);
    if (!targetModel) {
      throw new Error('Local model tidak ditemukan di cache');
    }
  } else {
    targetModel = await embeddingModelManager.getModelOrDownload(trimmedUrl, {
      progressCallback: ({ loaded, total }) => {
        if (onProgress && total) {
          onProgress(total === 0 ? null : loaded / total);
        }
      },
    });
  }

  const blobs = await targetModel.open();
  await inst.loadModel(blobs, {
    embeddings: true,
    n_ctx: 2048,
    pooling_type: 'LLAMA_POOLING_TYPE_MEAN',
  });
  const dim = 768;

  const state: EmbeddingModelState = {
    url: trimmedUrl,
    label: label ?? trimmedUrl.split('/').pop() ?? trimmedUrl,
    source,
    dim,
    loadedAt: Date.now(),
  };
  if (onProgress) onProgress(null);
  console.info(
    `[EmbeddingRuntime] Loaded ${source} model "${state.label}" in ${Math.round(
      performance.now() - now
    )} ms (${dim} dims)`
  );
  finalizeLoad(state, true, silent ?? false);
  return state;
};

export const loadEmbeddingModel = async (
  opts: LoadEmbeddingOptions
): Promise<EmbeddingModelState> => {
  if (loadingPromise) return loadingPromise;
  const performLoad = async () => {
    try {
      const inst = await getEmbeddingRuntime();
      return await loadModelFromManager(inst, opts);
    } catch (err) {
      throw err;
    } finally {
      loadingPromise = null;
    }
  };
  loadingPromise = performLoad();
  return loadingPromise;
};

export const loadEmbeddingModelFromFiles = async (
  options: LoadLocalFilesOptions
): Promise<EmbeddingModelState> => {
  const files = Array.isArray(options.files)
    ? [...options.files]
    : Array.from(options.files);
  if (!files.length) {
    throw new Error('Tidak ada file yang dipilih');
  }

  const { modelUrl, baseName } = await storeLocalModelFiles(
    files,
    embeddingModelManager,
    { namespace: options.namespace }
  );

  return loadEmbeddingModel({
    url: modelUrl,
    label: baseName,
    source: 'local',
    onProgress: options.onProgress,
  });
};

export const resetEmbeddingRuntime = async () => {
  if (embeddingInstance) {
    try {
      await embeddingInstance.exit();
    } catch (err) {
      console.warn('[EmbeddingRuntime] Failed to exit instance', err);
    }
  }
  embeddingInstance = null;
  currentModel = null;
  loadingPromise = null;
  notify();
};

export const createEmbeddingVector = async (
  text: string,
  options: { skipBos?: boolean; skipEos?: boolean } = {}
): Promise<Float32Array> =>
  withPerfTimer(
    'embedding:generate',
    async () => {
      if (!currentModel) {
        throw new Error('Embedding model belum dimuat');
      }
      const inst = await getEmbeddingRuntime();
      const vec: number[] = await (inst as any).createEmbedding(text, {
        skipBOS: options.skipBos ?? true,
        skipEOS: options.skipEos ?? true,
      });
      return Float32Array.from(vec);
    },
    () => ({
      textLength: text.length,
    })
  );

export const getSavedEmbeddingPreference = (): SavedEmbeddingModel | null => {
  return WllamaStorage.load<SavedEmbeddingModel | null>(
    'embedding_model',
    null
  );
};

export const clearSavedEmbeddingPreference = () => {
  WllamaStorage.remove('embedding_model');
};
