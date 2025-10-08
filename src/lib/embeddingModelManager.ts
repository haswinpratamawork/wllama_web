interface EmbeddingModelState {
  isLoaded: boolean;
  modelUrl: string;
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
      return JSON.parse(stored);
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