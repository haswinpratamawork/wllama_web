'use client';

import { useState, useEffect, useRef } from 'react';
import { Database, Upload, Search, Trash2, Download, Loader2, FileText, Zap, Home, AlertCircle, Play, Globe } from 'lucide-react';
import Link from 'next/link';

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
  loadModelFromUrl: (url: string, config: any) => Promise<void>;
  loadModel: (blobs: File[], config: any) => Promise<void>;
  createEmbedding: (text: string, options?: { skipBOS?: boolean; skipEOS?: boolean }) => Promise<number[]>;
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
  const [loadMethod, setLoadMethod] = useState<'url' | 'file'>('url');
  const [modelUrl, setModelUrl] = useState('');
  const [modelFiles, setModelFiles] = useState<FileList | null>(null);
  
  const wllamaRef = useRef<Wllama | null>(null);
  const dbRef = useRef<IDBDatabase | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initialize IndexedDB
  useEffect(() => {
    const initDB = () => {
      const request = indexedDB.open('VectorDB', 1);

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
      };
    };

    initDB();

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

  // Load embedding model from URL
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
      const Wllama = WllamaModule.Wllama;

      const CONFIG_PATHS = {
        'single-thread/wllama.wasm': './wllama/esm/single-thread/wllama.wasm',
        'multi-thread/wllama.wasm': './wllama/esm/multi-thread/wllama.wasm',
      };

      wllamaRef.current = new Wllama(CONFIG_PATHS);

      const progressCallback = ({ loaded, total }: { loaded: number; total: number }) => {
        const progressPercentage = Math.round((loaded / total) * 100);
        setLoadProgress(progressPercentage);
        setStatus(`Loading model... ${progressPercentage}%`);
      };

      setStatus('Downloading model...');

      const config = {
        embeddings: true,
        n_ctx: 1024,
        n_batch: 1024,
        n_ubatch: 1024,
        pooling_type: 'LLAMA_POOLING_TYPE_MEAN',
        progressCallback,
      };

      await wllamaRef.current.loadModelFromUrl(modelUrl, config);

      setStatus('Model loaded successfully!');
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

      const progressCallback = ({ loaded, total }: { loaded: number; total: number }) => {
        const progressPercentage = Math.round((loaded / total) * 100);
        setLoadProgress(progressPercentage);
        setStatus(`Loading model... ${progressPercentage}%`);
      };

      setStatus('Loading model from files...');

      const config = {
        embeddings: true,
        n_ctx: 1024,
        n_batch: 1024,
        n_ubatch: 1024,
        pooling_type: 'LLAMA_POOLING_TYPE_MEAN',
        progressCallback,
      };

      const blobs = Array.from(modelFiles);
      await wllamaRef.current.loadModel(blobs, config);

      setStatus('Model loaded successfully!');
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

  // Generate embedding using Wllama
  const generateEmbedding = async (text: string): Promise<number[]> => {
    if (!wllamaRef.current) {
      throw new Error('Model not loaded');
    }

    // Use Wllama's createEmbedding method
    const embedding = await wllamaRef.current.createEmbedding(text);
    return embedding;
  };

  // Cosine similarity
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

  // Add document
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
      const embedding = await generateEmbedding(inputText);
      
      const doc: EmbeddingDocument = {
        id: Date.now().toString(),
        text: inputText,
        embedding,
        metadata: {
          timestamp: Date.now(),
        },
      };

      await saveDocumentToDB(doc);
      setDocuments(prev => [...prev, doc]);
      setInputText('');
    } catch (err: any) {
      setError('Failed to embed text: ' + (err?.message || String(err)));
    } finally {
      setIsLoading(false);
    }
  };

  // Search documents
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
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2 flex items-center gap-3">
              <Database className="w-10 h-10" />
              Vector Embeddings
            </h1>
            <p className="text-blue-200">Wllama-powered text embedding with semantic search</p>
          </div>
          <Link href="/">
            <button className="bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center gap-2">
              <Home className="w-4 h-4" />
              Back to Chat
            </button>
          </Link>
        </div>

        {/* Model Loader */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 p-6 mb-6">
          <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5" />
            Load Embedding Model
          </h2>

          <div className="mb-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
            <p className="text-blue-200 text-sm mb-2">
              <strong>Recommended Embedding Models (GGUF):</strong>
            </p>
            <ul className="text-blue-200 text-xs space-y-1 list-disc list-inside">
              <li><a href="https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf" target="_blank" className="underline">bge-base-en-v1.5-q4_k_m.gguf</a> (recommended, 85MB)</li>
              <li><a href="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF" target="_blank" className="underline">nomic-embed-text-v1.5</a> (small, fast)</li>
              <li>Use embedding models only - NOT chat models!</li>
            </ul>
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

          {status && (
            <p className="text-blue-300 text-sm mt-2">{status}</p>
          )}

          {!dbReady && (
            <p className="text-yellow-300 text-sm mt-2 flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              Initializing database...
            </p>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-500/20 border border-red-400/50 text-red-200 rounded-lg p-4 mb-6 flex items-start gap-2">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <div>{error}</div>
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
                      <p className="text-blue-300 text-xs mt-2">
                        Vector: {doc.embedding.length}D
                      </p>
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