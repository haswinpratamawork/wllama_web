export interface ProgressCallback {
  loaded: number;
  total: number;
}

export interface WllamaConfig {
  n_ctx: number;
  n_batch: number;
  n_threads: number;
  n_gpu_layers: number;
  use_mlock: boolean;
  use_mmap: boolean;
  progressCallback: (progress: ProgressCallback) => void;
}

export interface Wllama {
  loadModelFromUrl: (url: string, config: WllamaConfig) => Promise<void>;
  loadModel: (blobs: File[], config: WllamaConfig) => Promise<void>;
  createCompletion: (prompt: string, options: any) => Promise<string>;
  createEmbedding?: (text: string, options?: { skipBOS?: boolean; skipEOS?: boolean }) => Promise<number[]>;
  setOptions?: (options: { embeddings: boolean }) => Promise<void>;
  getModelMetadata?: () => any;
  getLoadedContextInfo?: () => any;
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  ragContext?: string[];
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

export interface CachedModel {
  url: string;
  size: number;
  name: string;
}

export interface EmbeddingDocument {
  id: string;
  text: string;
  embedding: number[];
  metadata?: {
    source?: string;
    timestamp: number;
  };
}