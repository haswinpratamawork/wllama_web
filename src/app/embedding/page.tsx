'use client';

import { useState, useEffect, useRef } from 'react';
import { useEmbeddingModel } from '@/hooks/useEmbeddingModel';
import Link from 'next/link';
import { Database, Upload, Search, Trash2, Download, Loader2, FileText, Home, AlertCircle, Check, Sparkles } from 'lucide-react';

interface EmbeddingDocument {
  id: string;
  text: string;
  embedding: number[];
  // metadata?: {
  //   source?: string;
  //   timestamp: number;
  // };
}

interface SearchResult {
  document: EmbeddingDocument;
  similarity: number;
}

export default function EmbeddingPage() {
  const {
    embeddingModel,
    isLoaded: modelLoaded,
    modelCapabilities,
  } = useEmbeddingModel();

  const [documents, setDocuments] = useState<EmbeddingDocument[]>([]);
  const [inputText, setInputText] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const [dbReady, setDbReady] = useState(false);
  
  const dbRef = useRef<IDBDatabase | null>(null);

  const knowledgeFileRef = useRef<HTMLInputElement | null>(null);
  const [knowledgeBusy, setKnowledgeBusy] = useState(false);
  const [knowledgeResult, setKnowledgeResult] = useState<{ inserted: number; failed: number; errors: string[] } | null>(null);
  const [knowledgeError, setKnowledgeError] = useState<string | null>(null);

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
          // objectStore.createIndex('timestamp', 'metadata.timestamp', { unique: false });
          // objectStore.createIndex('timestamp', 'metadata.timestamp', { unique: false });
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
            failed++;
            errors.push(`Row ${idx}: not an object`);
            return;
          }
          if (!doc.id) {
            failed++;
            errors.push(`Row ${idx}: missing "id"`);
            return;
          }
          if (!doc.modified_date) {
            doc.modified_date = new Date().toISOString().slice(0, 19).replace('T', ' ');
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

      const normalized = docs.map((d, i) => {
        if (!d || typeof d !== 'object') throw new Error(`Item ${i} is not an object`);
        return d as Record<string, any>;
      });

      const res = await putManyKnowledge(normalized);
      setKnowledgeResult(res);
      loadDocumentsFromDB();
    } catch (e: any) {
      setKnowledgeError(e?.message || 'Failed to parse or store JSON');
    } finally {
      setKnowledgeBusy(false);
      if (knowledgeFileRef.current) knowledgeFileRef.current.value = '';
    }
  };

  const generateEmbedding = async (text: string): Promise<number[]> => {
    if (!embeddingModel) {
      throw new Error('Embedding model not loaded');
    }
    
    if (embeddingModel.setOptions) {
      await embeddingModel.setOptions({ embeddings: true });
    }
    
    const embedding = await embeddingModel.createEmbedding(text, {
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
        // metadata: {
        //   timestamp: Date.now(),
        // },
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
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2 flex items-center gap-3">
                <Database className="w-10 h-10" />
                Vector Embeddings & Knowledge Base
              </h1>
              <p className="text-blue-200">Manage documents and perform semantic search</p>
            </div>
            <Link href="/">
              <button className="bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center gap-2">
                <Home className="w-4 h-4" />
                Back to Chat
              </button>
            </Link>
          </div>

          <div className="bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Sparkles className={`w-6 h-6 ${modelLoaded ? 'text-green-400' : 'text-gray-500'}`} />
                <div>
                  <p className={`font-semibold ${modelLoaded ? 'text-green-300' : 'text-yellow-300'}`}>
                    {modelLoaded ? '✓ Embedding Model Active' : '⚠️ No Embedding Model'}
                  </p>
                  <p className="text-sm text-blue-200">
                    {modelLoaded ? (
                      <>
                        {modelCapabilities?.model_type} • {modelCapabilities?.n_embd}D vectors • Ready for operations
                      </>
                    ) : (
                      'Load an embedding model from the chat page Model Manager to enable features'
                    )}
                  </p>
                </div>
              </div>
              {!modelLoaded && (
                <Link href="/">
                  <button className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center gap-2">
                    <Database className="w-4 h-4" />
                    Go to Model Manager
                  </button>
                </Link>
              )}
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-500/20 border border-red-400/50 text-red-200 rounded-lg p-4 mb-6 flex items-start gap-2">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <div className="whitespace-pre-wrap">{error}</div>
          </div>
        )}

        {status && (
          <div className="bg-blue-500/20 border border-blue-400/50 text-blue-200 rounded-lg p-4 mb-6">
            {status}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-6">
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

            <div className="bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 p-6">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Upload Knowledge JSON
              </h2>

              <p className="text-blue-200 text-sm mb-3">
                Accepts an array or single object. Uses store <code className="text-white/90">embeddings</code> keyed by <code className="text-white/90">id</code>.
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

              {knowledgeResult && (
                <div className="mt-3 text-sm text-green-200">
                  Saved <b>{knowledgeResult.inserted}</b>
                  {knowledgeResult.failed ? <> • Failed <b>{knowledgeResult.failed}</b></> : null}
                  {knowledgeResult.errors?.length ? (
                    <details className="mt-2">
                      <summary className="cursor-pointer text-red-200">Errors</summary>
                      <ul className="list-disc list-inside text-red-200">
                        {knowledgeResult.errors.slice(0, 10).map((e, i) => (
                          <li key={i}>{e}</li>
                        ))}
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

          <div>
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
                      {/* {result.document.metadata?.source && (
                        <p className="text-blue-300 text-xs mt-2">
                          {result.document.metadata.source}
                        </p>
                      )} */}
                    </div>
                  ))}
                </div>
              </div>
            )}

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
                          {/* {new Date(doc.metadata?.timestamp || 0).toLocaleString()} */}
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
                        {/* {doc.metadata?.source && (
                          <p className="text-purple-300 text-xs">
                            {doc.metadata.source}
                          </p>
                        )} */}
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