'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { EmbeddingModelManager } from '@/lib/embeddingModelManager';

interface Wllama {
  loadModel: (blobs: Blob[], config: any) => Promise<void>;
  createEmbedding: (text: string, options?: { skipBOS?: boolean; skipEOS?: boolean }) => Promise<number[]>;
  setOptions?: (options: { embeddings: boolean }) => Promise<void>;
  getModelMetadata: () => any;
  getLoadedContextInfo: () => any;
}

interface UseEmbeddingModelReturn {
  embeddingModel: Wllama | null;
  isLoaded: boolean;
  isLoading: boolean;
  loadProgress: number;
  error: string;
  status: string;
  modelCapabilities: any;
  loadEmbeddingModel: (url: string) => Promise<void>;
  loadEmbeddingModelFromFiles: (files: FileList | File[]) => Promise<void>;
  unloadEmbeddingModel: () => void;
}

const CONFIG_PATHS = {
  'single-thread/wllama.wasm': './wllama/esm/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm': './wllama/esm/multi-thread/wllama.wasm',
};

export function useEmbeddingModel(): UseEmbeddingModelReturn {
  const [embeddingModel, setEmbeddingModel] = useState<Wllama | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [loadProgress, setLoadProgress] = useState(0);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const [modelCapabilities, setModelCapabilities] = useState<any>(null);
  
  const wllamaRef = useRef<Wllama | null>(null);
  const modelManagerRef = useRef<any>(null);
  const currentUrlRef = useRef<string | null>(null);
  const isLoadingRef = useRef<boolean>(false);

  const finalizeLoad = useCallback((
    wllama: Wllama,
    {
      sourceType,
      identifier = null,
      modelName,
      silent = false,
    }: {
      sourceType: 'remote' | 'local';
      identifier?: string | null;
      modelName: string;
      silent?: boolean;
    }
  ) => {
    const metadata = wllama.getModelMetadata?.() ?? {};
    const contextInfo = wllama.getLoadedContextInfo?.() ?? {};
    const hparams = (metadata as any).hparams ?? {};
    const meta = (metadata as any).meta ?? {};

    const capabilities = {
      n_ctx_train: hparams.nCtxTrain ?? contextInfo.n_ctx ?? 0,
      n_embd: hparams.nEmbd ?? 0,
      n_vocab: hparams.nVocab ?? 0,
      model_type: meta['general.architecture'] || 'unknown',
    };

    currentUrlRef.current = sourceType === 'remote' ? (identifier ?? null) : null;
    wllamaRef.current = wllama;
    setEmbeddingModel(wllama);
    setIsLoaded(true);
    setModelCapabilities(capabilities);
    setLoadProgress(100);

    EmbeddingModelManager.saveModelState({
      isLoaded: true,
      sourceType,
      modelUrl: sourceType === 'remote' ? (identifier ?? null) : null,
      modelName,
      capabilities,
      loadedAt: Date.now(),
    });

    if (!silent) {
      const contextTokens = contextInfo.n_ctx ?? capabilities.n_ctx_train ?? '?';
      setStatus(
        `Embedding model loaded${sourceType === 'local' ? ' from files' : ''}: ${modelName} ` +
        `(Context ${contextTokens} tokens, Embedding ${capabilities.n_embd}D)`
      );
    }
  }, []);

  const loadEmbeddingModel = useCallback(async (url: string, silent: boolean = false) => {
    if (!url.trim()) {
      setError('Please enter a model URL');
      return;
    }

    // Prevent loading if already loading
    if (isLoadingRef.current) {
      console.log('Already loading a model, skipping...');
      return;
    }

    // Check if already loaded with same URL
    if (currentUrlRef.current === url && wllamaRef.current) {
      if (!silent) setStatus('Embedding model already loaded');
      return;
    }

    isLoadingRef.current = true;
    setIsLoading(true);
    setError('');
    setLoadProgress(0);
    if (!silent) setStatus('Initializing embedding model...');

    try {
      const WllamaModule = await import('@wllama/wllama/esm/index.js');
      const { Wllama, ModelManager } = WllamaModule;
      const wllama = new Wllama(CONFIG_PATHS);
      
      if (!modelManagerRef.current) {
        modelManagerRef.current = new ModelManager();
      }

      if (!silent) setStatus('Downloading/loading embedding model...');

      const model = await modelManagerRef.current.getModelOrDownload(url, {
        progressCallback: ({ loaded, total }: { loaded: number; total: number }) => {
          if (total && !silent) {
            const progressPercentage = Math.round((loaded / total) * 100);
            setLoadProgress(progressPercentage);
            setStatus(`Downloading embedding model... ${progressPercentage}% (${(loaded / 1024 / 1024).toFixed(1)}MB / ${(total / 1024 / 1024).toFixed(1)}MB)`);
          }
        },
      });

      if (!silent) setStatus('Opening model blobs...');
      const blobs = await model.open();

      const modelName = url.split('/').pop() || 'model';
      if (!silent) setStatus('Loading embedding model into runtime...');
      
      await wllama.loadModel(blobs, {
        embeddings: true,
        n_ctx: 2048,
        pooling_type: 'LLAMA_POOLING_TYPE_MEAN',
      });

      finalizeLoad(wllama, {
        sourceType: 'remote',
        identifier: url,
        modelName,
        silent,
      });
    } catch (err: any) {
      const errorMsg = err?.message || String(err);
      
      if (errorMsg.includes('Invalid typed array length') || errorMsg.includes('Array buffer allocation failed')) {
        setError(
          '⚠️ Model too large for browser memory. Try:\n' +
          '1. Close other tabs to free memory\n' +
          '2. Use a smaller quantization (Q4_K_M, Q3_K_M)\n' +
          '3. Restart your browser'
        );
      } else {
        setError('Failed to load embedding model: ' + errorMsg);
      }
      setStatus('');
      console.error(err);
    } finally {
      setIsLoading(false);
      isLoadingRef.current = false;
    }
  }, [finalizeLoad]);

  const loadEmbeddingModelFromFiles = useCallback(async (files: FileList | File[]) => {
    const fileArray = Array.from(files as any).filter(Boolean) as File[];
    if (fileArray.length === 0) {
      setError('Please select embedding model file(s) first');
      return;
    }

    if (isLoadingRef.current) {
      console.log('Already loading a model, skipping...');
      return;
    }

    isLoadingRef.current = true;
    setIsLoading(true);
    setError('');
    setLoadProgress(0);
    setStatus('Initializing embedding model from files...');

    try {
      const WllamaModule = await import('@wllama/wllama/esm/index.js');
      const { Wllama } = WllamaModule;
      const wllama = new Wllama(CONFIG_PATHS);

      setStatus('Loading embedding model from local files...');

      await wllama.loadModel(fileArray, {
        embeddings: true,
        n_ctx: 2048,
        pooling_type: 'LLAMA_POOLING_TYPE_MEAN',
      });

      const modelName = fileArray.length === 1
        ? fileArray[0].name
        : `${fileArray[0].name} (+${fileArray.length - 1} parts)`;

      finalizeLoad(wllama, {
        sourceType: 'local',
        identifier: null,
        modelName,
        silent: false,
      });
    } catch (err: any) {
      const errorMsg = err?.message || String(err);
      setError('Failed to load embedding model from files: ' + errorMsg);
      setStatus('');
      console.error(err);
    } finally {
      setIsLoading(false);
      isLoadingRef.current = false;
    }
  }, [finalizeLoad]);

  const unloadEmbeddingModel = useCallback(() => {
    wllamaRef.current = null;
    currentUrlRef.current = null;
    setEmbeddingModel(null);
    setIsLoaded(false);
    setModelCapabilities(null);
    EmbeddingModelManager.clearModelState();
    setStatus('Embedding model unloaded');
  }, []);

  // Initialize only once
  useEffect(() => {
    let mounted = true;

    const init = async () => {
      // Only load if not already loaded
      if (wllamaRef.current || isLoadingRef.current) {
        return;
      }

      const state = EmbeddingModelManager.getModelState();
      if (state?.isLoaded && state.sourceType === 'remote' && state.modelUrl && mounted) {
        await loadEmbeddingModel(state.modelUrl, true);
      } else if (state?.isLoaded && state.sourceType === 'local' && mounted) {
        setStatus('Embedding model was loaded from local files in a previous session. Reload from files to use again.');
      }
    };

    init();

    const handleModelLoaded = (e: Event) => {
      if (!mounted) return;
      
      const customEvent = e as CustomEvent;
      const state = customEvent.detail;
      
      // Only load if different URL and not already loading
      if (state?.isLoaded && state.sourceType === 'remote' && state.modelUrl && 
          state.modelUrl !== currentUrlRef.current && 
          !isLoadingRef.current) {
        loadEmbeddingModel(state.modelUrl, true);
      } else if (state?.isLoaded && state.sourceType === 'local') {
        setStatus('Embedding model loaded from local files in another tab. Reload here if needed.');
      }
    };

    const handleModelUnloaded = () => {
      if (!mounted) return;
      
      wllamaRef.current = null;
      currentUrlRef.current = null;
      setEmbeddingModel(null);
      setIsLoaded(false);
    };

    window.addEventListener('embedding-model-loaded', handleModelLoaded);
    window.addEventListener('embedding-model-unloaded', handleModelUnloaded);

    return () => {
      mounted = false;
      window.removeEventListener('embedding-model-loaded', handleModelLoaded);
      window.removeEventListener('embedding-model-unloaded', handleModelUnloaded);
    };
  }, [loadEmbeddingModel]);

  return {
    embeddingModel,
    isLoaded,
    isLoading,
    loadProgress,
    error,
    status,
    modelCapabilities,
    loadEmbeddingModel,
    loadEmbeddingModelFromFiles,
    unloadEmbeddingModel,
  };
}
