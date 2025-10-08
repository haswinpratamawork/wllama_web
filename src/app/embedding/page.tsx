'use client';

import { useState, useEffect, useRef } from 'react';
import { Database, Upload, Search, Trash2, Download, Loader2, FileText, Zap, Home, AlertCircle, Play, Globe } from 'lucide-react';

interface EmbeddingDocument {
  id: string;
  text: string;
  embedding: number[];
  metadata?: {
    source?: string;
    timestamp: number;
  };
}

interface SearchResult {
  document: EmbeddingDocument;
  similarity: number;
}

interface Wllama {
  loadModel: (blobs: Blob[], config: any) => Promise<void>;
  createEmbedding: (text: string, options?: { skipBOS?: boolean; skipEOS?: boolean }) => Promise<number[]>;
  setOptions?: (options: { embeddings: boolean }) => Promise<void>;
  getModelMetadata: () => ModelMetadata;
  getLoadedContextInfo: () => LoadedContextInfo;
}

interface ModelMetadata {
  hparams: {
    nVocab: number;
    nCtxTrain: number;
    nEmbd: number;
    nLayer: number;
  };
  meta: Record<string, string>;
}

interface LoadedContextInfo {
  n_ctx: number;
  n_batch: number;
  n_ubatch: number;
  n_embd: number;
}

interface CachedModel {
  url: string;
  size: number;
  name: string;
}

export default function EmbeddingPage() {
  const [documents, setDocuments] = useState<EmbeddingDocument[]>([]);
  const [inputText, setInputText] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [loadingModel, setLoadingModel] = useState(false);
  const [loadProgress, setLoadProgress] = useState(0);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const [dbReady, setDbReady] = useState(false);
  const [loadMethod, setLoadMethod] = useState<'url' | 'file' | 'cached'>('url');
  const [modelUrl, setModelUrl] = useState('https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf');
  const [modelFiles, setModelFiles] = useState<FileList | null>(null);
  const [cachedModels, setCachedModels] = useState<CachedModel[]>([]);
  const [selectedCachedModel, setSelectedCachedModel] = useState('');
  const [modelCapabilities, setModelCapabilities] = useState<{
    n_ctx_train: number;
    n_embd: number;
    n_vocab: number;
    model_type: string;
  } | null>(null);
  
  const wllamaRef = useRef<Wllama | null>(null);
  const modelManagerRef = useRef<any>(null);
  const dbRef = useRef<IDBDatabase | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- Knowledge JSON upload states/refs ---
const knowledgeFileRef = useRef<HTMLInputElement | null>(null);
const [knowledgeBusy, setKnowledgeBusy] = useState(false);
const [knowledgeResult, setKnowledgeResult] = useState<{ inserted: number; failed: number; errors: string[] } | null>(null);
const [knowledgeError, setKnowledgeError] = useState<string | null>(null);

// Save many knowledge docs into 'knowledgeDocs'
async function putManyKnowledge(docs: any[]): Promise<{ inserted: number; failed: number; errors: string[] }> {
  if (!dbRef.current) throw new Error('DB not ready');
  const db = dbRef.current;

  return new Promise((resolve) => {
    const tx = db.transaction('embeddings', 'readwrite');
    const store = tx.objectStore('embeddings');

    let inserted = 0;
    let failed = 0;
    const errors: string[] = [];

    docs.forEach((doc, idx) => {
      try {
        if (!doc || typeof doc !== 'object') {
          failed++; errors.push(`Row ${idx}: not an object`);
          return;
        }
        if (!doc.id) {
          failed++; errors.push(`Row ${idx}: missing "id"`);
          return;
        }
        // optional: backfill modified_date if missing
        if (!doc.modified_date) {
          doc.modified_date = new Date().toISOString().slice(0,19).replace('T',' ');
        }
        store.put(doc);
        inserted++;
      } catch (e: any) {
        failed++;
        errors.push(`Row ${idx}: ${e?.message || String(e)}`);
      }
    });

    tx.oncomplete = () => resolve({ inserted, failed, errors });
    tx.onerror = () => resolve({ inserted, failed: failed + (docs.length - inserted - failed), errors: [...errors, String(tx.error)] });
  });
}

const handleKnowledgePick = () => knowledgeFileRef.current?.click();

const handleKnowledgeFile = async (f: File) => {
  setKnowledgeBusy(true);
  setKnowledgeResult(null);
  setKnowledgeError(null);
  try {
    const text = await f.text();
    const parsed = JSON.parse(text);
    const docs = Array.isArray(parsed) ? parsed : [parsed];

    // shape check
    const normalized = docs.map((d, i) => {
      if (!d || typeof d !== 'object') throw new Error(`Item ${i} is not an object`);
      return d as Record<string, any>;
    });

    const res = await putManyKnowledge(normalized);
    setKnowledgeResult(res);
  } catch (e: any) {
    setKnowledgeError(e?.message || 'Failed to parse or store JSON');
  } finally {
    setKnowledgeBusy(false);
    if (knowledgeFileRef.current) knowledgeFileRef.current.value = '';
  }
};
  // --- End Knowledge JSON upload states/refs ---

  // Initialize IndexedDB
  useEffect(() => {
    const initDB = () => {
      const request = indexedDB.open('VectorDB', 2);

      request.onerror = () => {
        setError('Failed to open IndexedDB');
      };

      request.onsuccess = (event) => {
        dbRef.current = (event.target as IDBOpenDBRequest).result;
        setDbReady(true);
        loadDocumentsFromDB();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains('embeddings')) {
          const objectStore = db.createObjectStore('embeddings', { keyPath: 'id' });
          objectStore.createIndex('timestamp', 'metadata.timestamp', { unique: false });
        }
            // ⬇️ NEW: knowledgeDocs store for your JSON knowledge items
        if (!db.objectStoreNames.contains('embeddings')) {
          db.createObjectStore('embeddings', { keyPath: 'id' });
        }
      };
    };

    initDB();
    loadCachedModels();

    return () => {
      if (dbRef.current) {
        dbRef.current.close();
      }
    };
  }, []);

  const loadDocumentsFromDB = () => {
    if (!dbRef.current) return;

    const transaction = dbRef.current.transaction(['embeddings'], 'readonly');
    const objectStore = transaction.objectStore('embeddings');
    const request = objectStore.getAll();

    request.onsuccess = () => {
      setDocuments(request.result);
    };

    request.onerror = () => {
      setError('Failed to load documents from database');
    };
  };

  const saveDocumentToDB = (doc: EmbeddingDocument) => {
    if (!dbRef.current) return Promise.reject('DB not ready');

    return new Promise<void>((resolve, reject) => {
      const transaction = dbRef.current!.transaction(['embeddings'], 'readwrite');
      const objectStore = transaction.objectStore('embeddings');
      const request = objectStore.add(doc);

      request.onsuccess = () => resolve();
      request.onerror = () => reject('Failed to save document');
    });
  };

  const deleteDocumentFromDB = (id: string) => {
    if (!dbRef.current) return Promise.reject('DB not ready');

    return new Promise<void>((resolve, reject) => {
      const transaction = dbRef.current!.transaction(['embeddings'], 'readwrite');
      const objectStore = transaction.objectStore('embeddings');
      const request = objectStore.delete(id);

      request.onsuccess = () => resolve();
      request.onerror = () => reject('Failed to delete document');
    });
  };

  // Load cached models using ModelManager
  const loadCachedModels = async () => {
    try {
      const WllamaModule = await import('@wllama/wllama/esm/index.js');
      const { ModelManager } = WllamaModule;
      if (!modelManagerRef.current) {
        modelManagerRef.current = new ModelManager();
      }
      const models = await modelManagerRef.current.getModels();
      setCachedModels(models.map((m: any) => ({
        url: m.url,
        size: m.size,
        name: m.url.split('/').pop()?.replace('.gguf', '') || 'Unknown'
      })));
    } catch (err) {
      console.error('Failed to load cached models:', err);
    }
  };

  // Delete cached model
  const deleteCachedModel = async (url: string) => {
    try {
      const models = await modelManagerRef.current.getModels();
      const model = models.find((m: any) => m.url === url);
      if (model) {
        await model.remove();
        await loadCachedModels();
        setStatus('Model deleted from cache');
        setTimeout(() => setStatus(''), 3000);
      }
    } catch (err: any) {
      setError('Failed to delete model: ' + (err?.message || String(err)));
    }
  };

  // Load embedding model using optimized approach
  const loadModelFromUrl = async () => {
    if (!modelUrl.trim()) {
      setError('Please enter a model URL');
      return;
    }

    setLoadingModel(true);
    setError('');
    setLoadProgress(0);
    setStatus('Initializing Wllama...');

    try {
      const WllamaModule = await import('@wllama/wllama/esm/index.js');
      const { Wllama, ModelManager } = WllamaModule;

      const CONFIG_PATHS = {
        'single-thread/wllama.wasm': './wllama/esm/single-thread/wllama.wasm',
        'multi-thread/wllama.wasm': './wllama/esm/multi-thread/wllama.wasm',
      };

      wllamaRef.current = new Wllama(CONFIG_PATHS);
      
      if (!modelManagerRef.current) {
        modelManagerRef.current = new ModelManager();
      }

      setStatus('Downloading/loading model...');

      // Use ModelManager to handle download and caching
      const model = await modelManagerRef.current.getModelOrDownload(modelUrl, {
        progressCallback: ({ loaded, total }: { loaded: number; total: number }) => {
          if (total) {
            const progressPercentage = Math.round((loaded / total) * 100);
            setLoadProgress(progressPercentage);
            setStatus(`Downloading model... ${progressPercentage}% (${(loaded / 1024 / 1024).toFixed(1)}MB / ${(total / 1024 / 1024).toFixed(1)}MB)`);
          }
        },
      });

      setStatus('Opening model blobs...');
      const blobs = await model.open();

      setStatus('Loading model into runtime...');
      
      // Optimized config - simpler is better for large models
      await wllamaRef.current.loadModel(blobs, {
        embeddings: true,
        n_ctx: 2048,
        pooling_type: 'LLAMA_POOLING_TYPE_MEAN',
      });

      setStatus('Reading model capabilities...');
      const metadata = wllamaRef.current.getModelMetadata();
      const contextInfo = wllamaRef.current.getLoadedContextInfo();
      
      setModelCapabilities({
        n_ctx_train: metadata.hparams.nCtxTrain,
        n_embd: metadata.hparams.nEmbd,
        n_vocab: metadata.hparams.nVocab,
        model_type: metadata.meta['general.architecture'] || 'unknown',
      });
      
      const modelName = modelUrl.split('/').pop() || 'model';
      setStatus(
        `✓ Model loaded! ${modelName} - Context: ${contextInfo.n_ctx} tokens, ` +
        `Embedding: ${metadata.hparams.nEmbd}D`
      );
      
      setLoadProgress(100);
      setModelLoaded(true);
      await loadCachedModels();
    } catch (err: any) {
      const errorMsg = err?.message || String(err);
      
      if (errorMsg.includes('Invalid typed array length') || errorMsg.includes('Array buffer allocation failed')) {
        setError(
          '⚠️ Model too large for browser memory. Try:\n' +
          '1. Close other tabs to free memory\n' +
          '2. Use a smaller quantization (Q4_K_M, Q3_K_M)\n' +
          '3. Restart your browser\n' +
          '4. Try a different browser (Chrome/Edge recommended)'
        );
      } else if (errorMsg.includes('unknown model architecture')) {
        const arch = errorMsg.match(/unknown model architecture: '([^']+)'/)?.[1];
        setError(
          `⚠️ Architecture '${arch}' not supported. Use BERT, LLAMA, or Gemma-based embedding models.`
        );
      } else if (errorMsg.includes('out of memory') || errorMsg.includes('OOM')) {
        setError(
          '⚠️ Out of memory. Close other applications and tabs, then try again.'
        );
      } else {
        setError('Failed to load model: ' + errorMsg);
      }
      setStatus('');
      console.error(err);
    } finally {
      setLoadingModel(false);
    }
  };

  // Load from cached model
  const loadCachedModel = async () => {
    if (!selectedCachedModel) {
      setError('Please select a cached model');
      return;
    }
    setModelUrl(selectedCachedModel);
    await loadModelFromUrl();
  };

  // Load embedding model from files
  const loadModelFromFiles = async () => {
    if (!modelFiles || modelFiles.length === 0) {
      setError('Please select model file(s)');
      return;
    }

    setLoadingModel(true);
    setError('');
    setLoadProgress(0);
    setStatus('Initializing Wllama...');

    try {
      const WllamaModule = await import('@wllama/wllama/esm/index.js');
      const Wllama = WllamaModule.Wllama;

      const CONFIG_PATHS = {
        'single-thread/wllama.wasm': './wllama/esm/single-thread/wllama.wasm',
        'multi-thread/wllama.wasm': './wllama/esm/multi-thread/wllama.wasm',
      };

      wllamaRef.current = new Wllama(CONFIG_PATHS);

      setStatus('Loading model from files...');

      const blobs = Array.from(modelFiles);
      
      // Optimized config
      await wllamaRef.current.loadModel(blobs, {
        embeddings: true,
        n_ctx: 2048,
        pooling_type: 'LLAMA_POOLING_TYPE_MEAN',
      });

      setStatus('Reading model capabilities...');
      const metadata = wllamaRef.current.getModelMetadata();
      const contextInfo = wllamaRef.current.getLoadedContextInfo();
      
      setModelCapabilities({
        n_ctx_train: metadata.hparams.nCtxTrain,
        n_embd: metadata.hparams.nEmbd,
        n_vocab: metadata.hparams.nVocab,
        model_type: metadata.meta['general.architecture'] || 'unknown',
      });
      
      setStatus(
        `✓ Local model loaded! Context: ${contextInfo.n_ctx} tokens, ` +
        `Embedding: ${metadata.hparams.nEmbd}D`
      );
      
      setLoadProgress(100);
      setModelLoaded(true);
    } catch (err: any) {
      setError('Failed to load model: ' + (err?.message || String(err)));
      setStatus('');
      console.error(err);
    } finally {
      setLoadingModel(false);
    }
  };

  const generateEmbedding = async (text: string): Promise<number[]> => {
    if (!wllamaRef.current) {
      throw new Error('Model not loaded');
    }
    
    // Enable embeddings mode before creating embeddings
    if (wllamaRef.current.setOptions) {
      await wllamaRef.current.setOptions({ embeddings: true });
    }
    
    const embedding = await wllamaRef.current.createEmbedding(text, {
      skipBOS: true,
      skipEOS: true,
    });
    
    return embedding;
  };

  const cosineSimilarity = (a: number[], b: number[]): number => {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  };

  const handleAddDocument = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text');
      return;
    }

    if (!modelLoaded) {
      setError('Please load the model first');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      setStatus('Creating embedding...');
      const embedding = await generateEmbedding(inputText);
      
      const doc: EmbeddingDocument = {
        id: `${Date.now()}`,
        text: inputText,
        embedding: embedding,
        metadata: {
          timestamp: Date.now(),
        },
      };

      await saveDocumentToDB(doc);
      setDocuments(prev => [...prev, doc]);
      setInputText('');
      setStatus('Document added successfully');
      setTimeout(() => setStatus(''), 3000);
    } catch (err: any) {
      setError('Failed to embed text: ' + (err?.message || String(err)));
      setStatus('');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setError('Please enter a search query');
      return;
    }

    if (!modelLoaded) {
      setError('Please load the model first');
      return;
    }

    if (documents.length === 0) {
      setError('No documents to search');
      return;
    }

    setIsSearching(true);
    setError('');

    try {
      const queryEmbedding = await generateEmbedding(searchQuery);
      
      const results: SearchResult[] = documents.map(doc => ({
        document: doc,
        similarity: cosineSimilarity(queryEmbedding, doc.embedding),
      }));

      results.sort((a, b) => b.similarity - a.similarity);
      setSearchResults(results.slice(0, 10));
    } catch (err: any) {
      setError('Search failed: ' + (err?.message || String(err)));
    } finally {
      setIsSearching(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteDocumentFromDB(id);
      setDocuments(prev => prev.filter(doc => doc.id !== id));
      setSearchResults(prev => prev.filter(result => result.document.id !== id));
    } catch (err: any) {
      setError('Failed to delete: ' + (err?.message || String(err)));
    }
  };

  const handleExport = () => {
    const dataStr = JSON.stringify(documents, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `embeddings-${new Date().toISOString()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const imported = JSON.parse(event.target?.result as string) as EmbeddingDocument[];
        
        for (const doc of imported) {
          await saveDocumentToDB(doc);
        }
        
        loadDocumentsFromDB();
      } catch (err: any) {
        setError('Failed to import: ' + (err?.message || String(err)));
      }
    };
    reader.readAsText(file);
  };

  const handleClearAll = async () => {
    if (!confirm('Delete all documents?')) return;

    try {
      for (const doc of documents) {
        await deleteDocumentFromDB(doc.id);
      }
      setDocuments([]);
      setSearchResults([]);
    } catch (err: any) {
      setError('Failed to clear: ' + (err?.message || String(err)));
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2 flex items-center gap-3">
              <Database className="w-10 h-10" />
              Vector Embeddings
            </h1>
            <p className="text-blue-200">Wllama-powered text embedding with semantic search</p>
          </div>
          <a href="/">
            <button className="bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center gap-2">
              <Home className="w-4 h-4" />
              Back to Chat
            </button>
          </a>
        </div>

        {/* Model Loader */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 p-6 mb-6">
          <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5" />
            Load Embedding Model
          </h2>

          <div className="mb-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
            <p className="text-blue-200 text-sm mb-2">
              <strong>✅ Now supports larger models (up to 500MB+)!</strong>
            </p>
            <ul className="text-blue-200 text-xs space-y-1 list-disc list-inside mb-2">
              <li><strong>Small models (under 150MB):</strong> bge-base-en-v1.5-q4_k_m.gguf (85MB)</li>
              <li><strong>Medium models (150-300MB):</strong> nomic-embed-text-v1.5-Q8_0.gguf</li>
              <li><strong>Large models (300MB+):</strong> embeddinggemma-300M-Q8_0.gguf (default)</li>
            </ul>
            <p className="text-green-200 text-xs mt-2">
              💡 <strong>Optimized loading:</strong> Uses ModelManager with blob loading for better memory efficiency
            </p>
          </div>

          {/* Load Method Tabs */}
          <div className="flex gap-2 mb-4">
            <button
              onClick={() => setLoadMethod('url')}
              className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                loadMethod === 'url'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white/10 text-blue-200 hover:bg-white/20'
              }`}
            >
              <Globe className="w-4 h-4" />
              From URL
            </button>
            <button
              onClick={() => setLoadMethod('file')}
              className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                loadMethod === 'file'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white/10 text-blue-200 hover:bg-white/20'
              }`}
            >
              <Upload className="w-4 h-4" />
              From File
            </button>
            <button
              onClick={() => setLoadMethod('cached')}
              className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                loadMethod === 'cached'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white/10 text-blue-200 hover:bg-white/20'
              }`}
            >
              <Database className="w-4 h-4" />
              Cached ({cachedModels.length})
            </button>
          </div>

          {/* Load from URL */}
          {loadMethod === 'url' && (
            <div className="space-y-3">
              <input
                type="text"
                value={modelUrl}
                onChange={(e) => setModelUrl(e.target.value)}
                placeholder="https://huggingface.co/.../model.gguf"
                className="w-full bg-white/10 border border-white/20 text-white placeholder-gray-400 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                onClick={loadModelFromUrl}
                disabled={loadingModel || modelLoaded || !modelUrl}
                className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                {loadingModel ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Loading... {loadProgress}%
                  </>
                ) : modelLoaded ? (
                  '✓ Model Loaded'
                ) : (
                  <>
                    <Download className="w-5 h-5" />
                    Download & Load Model
                  </>
                )}
              </button>
            </div>
          )}

          {/* Load from File */}
          {loadMethod === 'file' && (
            <div className="space-y-3">
              <input
                ref={fileInputRef}
                type="file"
                accept=".gguf"
                multiple
                onChange={(e) => setModelFiles(e.target.files)}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                <Upload className="w-5 h-5" />
                {modelFiles ? `Selected: ${modelFiles.length} file(s)` : 'Choose Model File(s)'}
              </button>
              <button
                onClick={loadModelFromFiles}
                disabled={!modelFiles || loadingModel || modelLoaded}
                className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                {loadingModel ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Loading... {loadProgress}%
                  </>
                ) : modelLoaded ? (
                  '✓ Model Loaded'
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Load Model
                  </>
                )}
              </button>
            </div>
          )}

          {/* Load from Cached */}
          {loadMethod === 'cached' && (
            <div className="space-y-3">
              {cachedModels.length === 0 ? (
                <p className="text-blue-200 text-sm text-center py-4">
                  No cached models. Download one first using URL or File method.
                </p>
              ) : (
                <>
                  <select
                    value={selectedCachedModel}
                    onChange={(e) => setSelectedCachedModel(e.target.value)}
                    className="w-full bg-white/10 border border-white/20 text-white rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Select a cached model...</option>
                    {cachedModels.map((model) => (
                      <option key={model.url} value={model.url}>
                        {model.name} ({(model.size / 1024 / 1024).toFixed(1)} MB)
                      </option>
                    ))}
                  </select>
                  <div className="flex gap-2">
                    <button
                      onClick={loadCachedModel}
                      disabled={!selectedCachedModel || loadingModel || modelLoaded}
                      className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
                    >
                      {loadingModel ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin" />
                          Loading... {loadProgress}%
                        </>
                      ) : modelLoaded ? (
                        '✓ Model Loaded'
                      ) : (
                        <>
                          <Play className="w-5 h-5" />
                          Load Cached Model
                        </>
                      )}
                    </button>
                    <button
                      onClick={() => selectedCachedModel && deleteCachedModel(selectedCachedModel)}
                      disabled={!selectedCachedModel || loadingModel}
                      className="bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white font-semibold py-3 px-4 rounded-lg transition-colors flex items-center justify-center"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                </>
              )}
            </div>
          )}

          {status && (
            <p className="text-blue-300 text-sm mt-2">{status}</p>
          )}

          {!dbReady && (
            <p className="text-yellow-300 text-sm mt-2 flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              Initializing database...
            </p>
          )}

          {/* Model Info Display */}
          {modelLoaded && modelCapabilities && (
            <div className="mt-4 p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
              <h3 className="text-green-300 font-semibold mb-2">Model Information</h3>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="text-blue-200">
                  <span className="text-white font-medium">Type:</span> {modelCapabilities.model_type}
                </div>
                <div className="text-blue-200">
                  <span className="text-white font-medium">Embedding Dim:</span> {modelCapabilities.n_embd}D
                </div>
                <div className="text-blue-200">
                  <span className="text-white font-medium">Vocabulary:</span> {modelCapabilities.n_vocab.toLocaleString()} tokens
                </div>
                <div className="text-blue-200">
                  <span className="text-white font-medium">Trained Context:</span> {modelCapabilities.n_ctx_train} tokens
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-500/20 border border-red-400/50 text-red-200 rounded-lg p-4 mb-6 flex items-start gap-2">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <div className="whitespace-pre-wrap">{error}</div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-6">
            {/* Add Document */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 p-6">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Add Document
              </h2>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter text to embed..."
                className="w-full bg-white/10 border border-white/20 text-white placeholder-gray-400 rounded-lg p-3 min-h-[120px] focus:outline-none focus:ring-2 focus:ring-blue-500 mb-3"
                disabled={!modelLoaded}
              />
              <button
                onClick={handleAddDocument}
                disabled={!modelLoaded || isLoading}
                className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Embedding...
                  </>
                ) : (
                  <>
                    <Upload className="w-5 h-5" />
                    Add to Database
                  </>
                )}
              </button>
            </div>

            {/* Upload Knowledge JSON -> stores to IndexedDB (knowledgeDocs) */}
<div className="bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 p-6">
  <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
    <Upload className="w-5 h-5" />
    Upload Knowledge JSON
  </h2>

  <p className="text-blue-200 text-sm mb-3">
    Accepts an array or single object. Uses store <code className="text-white/90">Embeddings</code> keyed by <code className="text-white/90">id</code>.
  </p>

  <input
    ref={knowledgeFileRef}
    type="file"
    accept="application/json,.json"
    className="hidden"
    onChange={(e) => {
      const f = e.target.files?.[0];
      if (f) void handleKnowledgeFile(f);
    }}
  />

  <button
    onClick={handleKnowledgePick}
    disabled={!dbReady || knowledgeBusy}
    className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
  >
    {knowledgeBusy ? (
      <>
        <Loader2 className="w-5 h-5 animate-spin" />
        Saving…
      </>
    ) : (
      <>
        <Upload className="w-5 h-5" />
        Select .json & Save to DB
      </>
    )}
  </button>

  {/* Inline feedback */}
  {knowledgeResult && (
    <div className="mt-3 text-sm text-green-200">
      Saved <b>{knowledgeResult.inserted}</b>
      {knowledgeResult.failed ? <> • Failed <b>{knowledgeResult.failed}</b></> : null}
      {knowledgeResult.errors?.length ? (
        <details className="mt-2">
          <summary className="cursor-pointer text-red-200">Errors</summary>
          <ul className="list-disc list-inside text-red-200">
            {knowledgeResult.errors.slice(0,10).map((e, i) => <li key={i}>{e}</li>)}
          </ul>
          {knowledgeResult.errors.length > 10 && (
            <p className="text-red-300 mt-1">…and {knowledgeResult.errors.length - 10} more</p>
          )}
        </details>
      ) : null}
    </div>
  )}

  {knowledgeError && (
    <div className="mt-3 text-sm text-red-200 flex items-start gap-2">
      <AlertCircle className="w-4 h-4 mt-0.5" />
      <span className="whitespace-pre-wrap">{knowledgeError}</span>
    </div>
  )}
</div>


            {/* Search */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 p-6">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <Search className="w-5 h-5" />
                Semantic Search
              </h2>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Search documents..."
                className="w-full bg-white/10 border border-white/20 text-white placeholder-gray-400 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-blue-500 mb-3"
                disabled={!modelLoaded}
              />
              <button
                onClick={handleSearch}
                disabled={!modelLoaded || isSearching || documents.length === 0}
                className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                {isSearching ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search className="w-5 h-5" />
                    Search ({documents.length} docs)
                  </>
                )}
              </button>
            </div>

            {/* Database Actions */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 p-6">
              <h2 className="text-xl font-bold text-white mb-4">Database Actions</h2>
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={handleExport}
                  disabled={documents.length === 0}
                  className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Export
                </button>
                <label className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2 cursor-pointer">
                  <Upload className="w-4 h-4" />
                  Import
                  <input
                    type="file"
                    accept=".json"
                    onChange={handleImport}
                    className="hidden"
                  />
                </label>
                <button
                  onClick={handleClearAll}
                  disabled={documents.length === 0}
                  className="col-span-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  <Trash2 className="w-4 h-4" />
                  Clear All
                </button>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div>
            {/* Search Results */}
            {searchResults.length > 0 && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 p-6 mb-6">
                <h2 className="text-xl font-bold text-white mb-4">
                  Search Results ({searchResults.length})
                </h2>
                <div className="space-y-3 max-h-[500px] overflow-y-auto">
                  {searchResults.map((result, idx) => (
                    <div
                      key={result.document.id}
                      className="bg-white/5 border border-white/10 rounded-lg p-4"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <span className="text-green-300 font-semibold text-sm">
                          #{idx + 1} • {(result.similarity * 100).toFixed(1)}% match
                        </span>
                        <button
                          onClick={() => handleDelete(result.document.id)}
                          className="text-red-400 hover:text-red-300 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                      <p className="text-white text-sm">{result.document.text}</p>
                      {result.document.metadata?.source && (
                        <p className="text-blue-300 text-xs mt-2">
                          {result.document.metadata.source}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* All Documents */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 p-6">
              <h2 className="text-xl font-bold text-white mb-4">
                All Documents ({documents.length})
              </h2>
              <div className="space-y-3 max-h-[600px] overflow-y-auto">
                {documents.length === 0 ? (
                  <p className="text-blue-200 text-sm">No documents yet. Add some text above!</p>
                ) : (
                  documents.map((doc) => (
                    <div
                      key={doc.id}
                      className="bg-white/5 border border-white/10 rounded-lg p-4"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <span className="text-blue-300 text-xs">
                          {new Date(doc.metadata?.timestamp || 0).toLocaleString()}
                        </span>
                        <button
                          onClick={() => handleDelete(doc.id)}
                          className="text-red-400 hover:text-red-300 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                      <p className="text-white text-sm">{doc.text}</p>
                      <div className="flex items-center justify-between mt-2">
                        <p className="text-blue-300 text-xs">
                          Vector: {doc.embedding.length}D
                        </p>
                        {doc.metadata?.source && (
                          <p className="text-purple-300 text-xs">
                            {doc.metadata.source}
                          </p>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}