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
  unloadEmbeddingModel: () => void;
}

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

      const CONFIG_PATHS = {
        'single-thread/wllama.wasm': './wllama/esm/single-thread/wllama.wasm',
        'multi-thread/wllama.wasm': './wllama/esm/multi-thread/wllama.wasm',
      };

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

      if (!silent) setStatus('Loading embedding model into runtime...');
      
      await wllama.loadModel(blobs, {
        embeddings: true,
        n_ctx: 2048,
        pooling_type: 'LLAMA_POOLING_TYPE_MEAN',
      });

      if (!silent) setStatus('Reading model capabilities...');
      const metadata = wllama.getModelMetadata();
      const contextInfo = wllama.getLoadedContextInfo();
      
      const capabilities = {
        n_ctx_train: metadata.hparams.nCtxTrain,
        n_embd: metadata.hparams.nEmbd,
        n_vocab: metadata.hparams.nVocab,
        model_type: metadata.meta['general.architecture'] || 'unknown',
      };

      setModelCapabilities(capabilities);
      
      const modelName = url.split('/').pop() || 'model';
      
      EmbeddingModelManager.saveModelState({
        isLoaded: true,
        modelUrl: url,
        modelName,
        capabilities,
        loadedAt: Date.now(),
      });

      currentUrlRef.current = url;
      wllamaRef.current = wllama;
      setEmbeddingModel(wllama);
      setIsLoaded(true);
      setLoadProgress(100);
      
      if (!silent) {
        setStatus(
          `✓ Embedding model loaded! ${modelName} - Context: ${contextInfo.n_ctx} tokens, ` +
          `Embedding: ${metadata.hparams.nEmbd}D`
        );
      }
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
  }, []);

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
      if (state?.isLoaded && state.modelUrl && mounted) {
        await loadEmbeddingModel(state.modelUrl, true);
      }
    };

    init();

    const handleModelLoaded = (e: Event) => {
      if (!mounted) return;
      
      const customEvent = e as CustomEvent;
      const state = customEvent.detail;
      
      // Only load if different URL and not already loading
      if (state?.isLoaded && state.modelUrl && 
          state.modelUrl !== currentUrlRef.current && 
          !isLoadingRef.current) {
        loadEmbeddingModel(state.modelUrl, true);
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
    unloadEmbeddingModel,
  };
}