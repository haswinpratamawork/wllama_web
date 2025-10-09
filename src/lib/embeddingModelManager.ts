type EmbeddingModelSource = 'remote' | 'local';

interface EmbeddingModelState {
  isLoaded: boolean;
  sourceType: EmbeddingModelSource;
  modelUrl: string | null;
  modelName: string;
  capabilities: {
    n_ctx_train: number;
    n_embd: number;
    n_vocab: number;
    model_type: string;
  } | null;
  loadedAt: number;
}

const STORAGE_KEY = 'wllama-embedding-model-state';

export class EmbeddingModelManager {
  static saveModelState(state: EmbeddingModelState): void {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    window.dispatchEvent(new CustomEvent('embedding-model-loaded', { detail: state }));
  }

  static getModelState(): EmbeddingModelState | null {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return null;
    try {
      const parsed = JSON.parse(stored);
      return {
        sourceType: parsed.sourceType ?? (parsed.modelUrl ? 'remote' : 'local'),
        modelUrl: parsed.modelUrl ?? null,
        isLoaded: parsed.isLoaded ?? false,
        modelName: parsed.modelName ?? 'unknown',
        capabilities: parsed.capabilities ?? null,
        loadedAt: parsed.loadedAt ?? Date.now(),
      } as EmbeddingModelState;
    } catch {
      return null;
    }
  }

  static clearModelState(): void {
    localStorage.removeItem(STORAGE_KEY);
    window.dispatchEvent(new CustomEvent('embedding-model-unloaded'));
  }

  static isModelLoaded(): boolean {
    const state = this.getModelState();
    return state?.isLoaded ?? false;
  }
}
