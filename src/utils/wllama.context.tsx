import { createContext, useContext, useMemo, useState } from 'react';
import {
  DebugLogger,
  getDefaultScreen,
  useDidMount,
  WllamaStorage,
} from './utils';
import { Model, ModelManager, Wllama } from '@wllama/wllama';
import { DEFAULT_INFERENCE_PARAMS, WLLAMA_CONFIG_PATHS } from '../config';
import { InferenceParams, RuntimeInfo, ModelState, Screen } from './types';
import { verifyCustomModel } from './custom-models';
import {
  DisplayedModel,
  getDisplayedModels,
  getUserAddedModels,
  updateUserAddedModels,
} from './displayed-model';

interface WllamaContextValue {
  // functions for managing models
  models: DisplayedModel[];
  downloadModel(model: DisplayedModel): Promise<void>;
  removeCachedModel(model: DisplayedModel): Promise<void>;
  removeAllCachedModels(): Promise<void>;
  isDownloading: boolean;
  isLoadingModel: boolean;
  currParams: InferenceParams;
  setParams(params: InferenceParams): void;

  // function to load/unload model
  loadedModel?: DisplayedModel;
  currRuntimeInfo?: RuntimeInfo;
  loadModel(model: DisplayedModel): Promise<void>;
  unloadModel(): Promise<void>;

  // function for managing custom user model
  addCustomModel(url: string): Promise<void>;
  removeCustomModel(model: DisplayedModel): Promise<void>;

  // functions for chat completion
  getWllamaInstance(): Wllama;
  createCompletion(
    input: string,
    callback: (piece: string) => void
  ): Promise<void>;
  stopCompletion(): void;
  isGenerating: boolean;
  currentConvId: number;

  // nagivation
  navigateTo(screen: Screen, conversationId?: number): void;
  currScreen: Screen;

  // add from localModel
  handleAddLocalModel(files: FileList | File[]): Promise<void>;


}

const WllamaContext = createContext<WllamaContextValue>({} as any);

const modelManager = new ModelManager();
let wllamaInstance = new Wllama(WLLAMA_CONFIG_PATHS, { logger: DebugLogger });
let stopSignal = false;
const resetWllamaInstance = () => {
  wllamaInstance = new Wllama(WLLAMA_CONFIG_PATHS, { logger: DebugLogger });
};

export const WllamaProvider = ({ children }: any) => {
  const [isGenerating, setGenerating] = useState(false);
  const [currentConvId, setCurrentConvId] = useState(-1);
  const [currScreen, setScreen] = useState<Screen>(getDefaultScreen());
  const [cachedModels, setCachedModels] = useState<Model[]>([]);
  const [isBusy, setBusy] = useState(false);
  const [currRuntimeInfo, setCurrRuntimeInfo] = useState<RuntimeInfo>();
  const [currParams, setCurrParams] = useState<InferenceParams>(
    WllamaStorage.load('params', DEFAULT_INFERENCE_PARAMS)
  );
  const [downloadingProgress, setDownloadingProgress] = useState<
    Record<DisplayedModel['url'], number>
  >({});
  const [loadedModel, setLoadedModel] = useState<DisplayedModel>();

  const refreshCachedModels = async () => {
    setCachedModels(await modelManager.getModels());
  };
  useDidMount(refreshCachedModels);

  // console.log('loadedModel', loadedModel)
  // computed variables
  // console.log("cachedModels", cachedModels)
  const models = useMemo(() => {
    const list = getDisplayedModels(cachedModels);
    // console.log('list', list)
    for (const model of list) {
      model.downloadPercent = downloadingProgress[model.url] ?? -1;
      if (model.downloadPercent >= 0) {
        model.state = ModelState.DOWNLOADING;
      }
      if (loadedModel?.url === model.url) {
        model.state = loadedModel.state;
      }
    }
    return list;
  }, [cachedModels, downloadingProgress, loadedModel]);

  // console.log("models", models)

  const isDownloading = useMemo(
    () => models.some((m) => m.state === ModelState.DOWNLOADING),
    [models]
  );
  const isLoadingModel = useMemo(
    () => isBusy || loadedModel?.state === ModelState.LOADING,
    [loadedModel, isBusy]
  );



  // utils
  const updateModelDownloadState = (
    url: string,
    downloadPercent: number = -1
  ) => {
    if (downloadPercent < 0) {
      setDownloadingProgress((p) => {
        const newProgress = { ...p };
        delete newProgress[url];
        return newProgress;
      });
    } else {
      setDownloadingProgress((p) => ({ ...p, [url]: downloadPercent }));
    }
  };

  const downloadModel = async (model: DisplayedModel) => {
    if (isDownloading || loadedModel || isLoadingModel) return;
    updateModelDownloadState(model.url, 0);
    try {
      await modelManager.downloadModel(model.url, {
        progressCallback(opts) {
          updateModelDownloadState(model.url, opts.loaded / opts.total);
        },
      });
      updateModelDownloadState(model.url, -1);
      await refreshCachedModels();
    } catch (e) {
      alert((e as any)?.message || 'unknown error while downloading model');
    }
  };

  const removeCachedModel = async (model: DisplayedModel) => {
    if (isDownloading || loadedModel || isLoadingModel) return;
    if (model.cachedModel) {
      await model.cachedModel.remove();
      await refreshCachedModels();
    }
  };

  const removeAllCachedModels = async () => {
    if (isDownloading || loadedModel || isLoadingModel) return;
    await modelManager.clear();
    await refreshCachedModels();
  };

  const loadModel = async (model: DisplayedModel) => {
    if (isDownloading || loadedModel || isLoadingModel) return;
    // make sure the model is cached
    if (!model.cachedModel) {
      throw new Error('Model is not in cache');
    }
    setLoadedModel(model.clone({ state: ModelState.LOADING }));
    try {
      await wllamaInstance.loadModel(model.cachedModel, {
        n_threads: currParams.nThreads > 0 ? currParams.nThreads : undefined,
        n_ctx: currParams.nContext,
        n_batch: currParams.nBatch,
      });
      setLoadedModel(model.clone({ state: ModelState.LOADED }));
      setCurrRuntimeInfo({
        isMultithread: wllamaInstance.isMultithread(),
        hasChatTemplate: !!wllamaInstance.getChatTemplate(),
      });
    } catch (e) {
      resetWllamaInstance();
      alert(`Failed to load model: ${(e as any).message ?? 'Unknown error'}`);
      setLoadedModel(undefined);
    }
  };

  const handleAddLocalModel = async (files: FileList | File[]): Promise<void> => {
    setBusy(true);
    try {
      const inst = wllamaInstance;
      const fileArr = Array.isArray(files) ? files : Array.from(files);
      if (fileArr.length === 0) throw new Error('Tidak ada file yang dipilih');

      // Validasi dasar
      for (const f of fileArr) {
        if (!f.name.toLowerCase().endsWith('.gguf')) {
          throw new Error('Hanya file .gguf yang didukung');
        }
        if (f.size === 0) throw new Error('File model tidak boleh kosong');
      }

      // 1) Load model ke memory agar langsung bisa dipakai untuk inference
      const blobs: Blob[] = fileArr as Blob[];
      console.info('Loading model into wllama memory (in-memory only)...');
      await inst.loadModel(blobs, {
        n_ctx: currParams.nContext,
        n_threads: currParams.nThreads,
        n_batch: currParams.nBatch,
        embeddings: false,
      });
      console.info('wllama.loadModel finished (model loaded in-memory).');

      // 2) Buat URL lokal unik untuk UI referensi
      const baseName = (fileArr[0] as File).name;
      const url = `local://${Date.now()}/${baseName}`;

      // 3) Coba simpan ke cache jika cacheManager punya API (best-effort)
      const mmAny: any = modelManager as any;
      const cacheAny: any = mmAny?.cacheManager;
      console.debug('CacheManager methods:', cacheAny ? Object.keys(cacheAny) : 'no-cache-manager');

      let savedFiles: any[] | undefined = undefined;
      let modelInstance: any = undefined;
      let savedToCache = false;

      if (cacheAny && Object.keys(cacheAny).length > 0) {
        try {
          // Coba beberapa method yang mungkin ada (save / put / addFile / add)
          if (typeof cacheAny.save === 'function') {
            try {
              savedFiles = await cacheAny.save(url, blobs);
              savedToCache = true;
            } catch (e) {
              // fallback signature: array of {name, blob}
              const entries = blobs.map((b: Blob, i: number) => ({
                name: i === 0 ? baseName : `model-${i}.gguf`,
                blob: b,
              }));
              savedFiles = await cacheAny.save(url, entries);
              savedToCache = true;
            }
          }

          if (!savedToCache && typeof cacheAny.put === 'function') {
            try {
              savedFiles = await cacheAny.put(url, blobs);
              savedToCache = true;
            } catch (_) {
              // try per-file put(url, name, blob)
              const arr: any[] = [];
              for (let i = 0; i < blobs.length; ++i) {
                const fname = i === 0 ? baseName : `model-${i}.gguf`;
                const r = await cacheAny.put(url, fname, blobs[i]);
                arr.push(r);
              }
              savedFiles = arr;
              savedToCache = true;
            }
          }

          if (!savedToCache && typeof cacheAny.addFile === 'function') {
            savedFiles = [];
            for (let i = 0; i < blobs.length; ++i) {
              const fname = i === 0 ? baseName : `model-${i}.gguf`;
              const r = await cacheAny.addFile(url, fname, blobs[i]);
              savedFiles.push(r);
            }
            savedToCache = true;
          }

          if (!savedToCache && typeof cacheAny.add === 'function') {
            const entries = blobs.map((b: Blob, i: number) => ({ name: i === 0 ? baseName : `model-${i}.gguf`, blob: b }));
            savedFiles = await cacheAny.add(url, entries);
            savedToCache = true;
          }
        } catch (cacheErr) {
          console.warn('Gagal menyimpan ke cacheManager (ignored):', cacheErr);
          savedToCache = false;
          savedFiles = undefined;
        }
      } else {
        console.info('Tidak ada cacheManager / cache methods kosong — akan memakai in-memory only (no persist).');
      }

      // 4) HANYA bila kita berhasil menyimpan ke cache, coba dapatkan Model dari modelManager
      if (savedToCache) {
        try {
          if (typeof mmAny.getModelOrDownload === 'function') {
            // Jangan memanggil getModelOrDownload untuk schema local:// jika implementasi
            // library melakukan fetch(url). Namun karena kita baru saja menyimpan ke cache,
            // getModelOrDownload mungkin mengembalikan Model langsung.
            modelInstance = await mmAny.getModelOrDownload(url, { useCache: true });
          } else {
            // Sebagai fallback coba buat Model dengan constructor jika tersedia
            const maybeModelCtor = mmAny.Model ?? (await import('@wllama/wllama')).Model;
            if (maybeModelCtor) {
              modelInstance = new maybeModelCtor(modelManager, url, savedFiles);
            }
          }
          console.info('Model instance obtained from modelManager/cache.');
        } catch (e) {
          console.warn('Gagal membuat Model instance meski savedFiles ada — fallback ke in-memory only:', e);
          modelInstance = undefined;
        }
      } else {
        // Important: JANGAN panggil getModelOrDownload/new Model(...) karena itu akan mencoba fetch local:// dan gagal.
        console.info('Skip creating Model instance because model was NOT saved to cache — using in-memory only.');
      }

      // 5) refresh cachedModels jika tersedia
      try {
        if (typeof modelManager.getModels === 'function') {
          setCachedModels(await modelManager.getModels());
        }
      } catch (e) {
        console.warn('refreshCachedModels gagal:', e);
      }

      // 6) Set loaded model (DisplayedModel) sehingga UI dan runtime tahu model siap (in-memory)
      const totalSize = (fileArr as File[]).reduce((s, f) => s + f.size, 0);


      const displayed = new DisplayedModel(url, totalSize, true, modelInstance);
      displayed.state = ModelState.LOADED;
      setLoadedModel(displayed);


      const userAddedModels = getUserAddedModels(cachedModels);

      // baru: daftarkan metadata model ke daftar models yang ditambahkan user
      try {
        updateUserAddedModels([
          ...userAddedModels,
          new DisplayedModel(url, totalSize, true, modelInstance)
        ]);
      } catch (err) {
        console.warn('updateUserAddedModels failed:', err);
      }
      // 7) Update runtime info
      setCurrRuntimeInfo({
        isMultithread: inst.isMultithread(),
        hasChatTemplate: !!inst.getChatTemplate(),
      });

      console.info(`Model loaded in-memory and set as loaded. persisted=${savedToCache}, url=${url}`);
    } catch (e: any) {
      alert('Gagal load model lokal: ' + (e?.message ?? String(e)));
      setLoadedModel(undefined);
    } finally {
      setBusy(false);
    }
  };



  const unloadModel = async () => {
    if (!loadedModel) return;
    await wllamaInstance.exit();
    resetWllamaInstance();
    setLoadedModel(undefined);
    setCurrRuntimeInfo(undefined);
  };

  const createCompletion = async (
    input: string,
    callback: (currentText: string) => void
  ) => {
    if (isDownloading || !loadedModel || isLoadingModel) return;
    setGenerating(true);
    stopSignal = false;
    const result = await wllamaInstance.createCompletion(input, {
      nPredict: currParams.nPredict,
      useCache: true,
      sampling: {
        temp: currParams.temperature,
      },
      // @ts-ignore unused variable
      onNewToken(token, piece, currentText, optionals) {
        callback(currentText);
        if (stopSignal) optionals.abortSignal();
      },
    });
    callback(result);
    stopSignal = false;
    setGenerating(false);
  };

  const stopCompletion = () => {
    stopSignal = true;
  };

  const navigateTo = (screen: Screen, conversationId?: number) => {
    setScreen(screen);
    setCurrentConvId(conversationId ?? -1);
    if (screen === Screen.MODEL) {
      WllamaStorage.save('welcome', false);
    }
  };

  // proxy function for saving to localStorage
  const setParams = (val: InferenceParams) => {
    WllamaStorage.save('params', val);
    setCurrParams(val);
  };

  // function for managing custom user model
  const addCustomModel = async (url: string) => {
    setBusy(true);
    try {
      const custom = await verifyCustomModel(url);
      if (models.some((m) => m.url === custom.url)) {
        throw new Error('Model with the same URL already exist');
      }
      const userAddedModels = getUserAddedModels(cachedModels);
      updateUserAddedModels([
        ...userAddedModels,
        new DisplayedModel(custom.url, custom.size, true, undefined),
      ]);
      await refreshCachedModels();
    } catch (e) {
      setBusy(false);
      throw e; // re-throw
    }
    setBusy(false);
  };

  const removeCustomModel = async (model: DisplayedModel) => {
    setBusy(true);
    if (model.isUserAdded) {
      const userAddedModels = getUserAddedModels(cachedModels);
      const newList = userAddedModels.filter((m) => m.url !== model.url);
      updateUserAddedModels(newList);
      await refreshCachedModels();
    } else {
      throw new Error('Cannot remove non-user-added model');
    }
    setBusy(false);
  };

  return (
    <WllamaContext.Provider
      value={{
        models,
        isDownloading,
        isLoadingModel,
        downloadModel,
        removeCachedModel,
        removeAllCachedModels,
        loadedModel,
        loadModel,
        unloadModel,
        handleAddLocalModel,
        currParams,
        setParams,
        createCompletion,
        stopCompletion,
        isGenerating,
        currentConvId,
        navigateTo,
        currScreen,
        getWllamaInstance: () => wllamaInstance,
        addCustomModel,
        removeCustomModel,
        currRuntimeInfo,
      }}
    >
      {children}
    </WllamaContext.Provider>
  );
};

export const useWllama = () => useContext(WllamaContext);
