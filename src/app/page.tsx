'use client';

import { useState, useRef, useEffect, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { useEmbeddingModel } from '@/hooks/useEmbeddingModel';
import { ChatModelManager } from '@/lib/chatModelManager';
import type { 
  Wllama, 
  WllamaConfig, 
  ProgressCallback, 
  Message, 
  Conversation, 
  CachedModel, 
  EmbeddingDocument 
} from '@/types/wllama';
import { Upload, Play, Loader2, Send, Trash2, User, Bot, Download, Save, Plus, MessageSquare, Menu, X, Globe, HardDrive, Database, BookOpen, Sparkles, FileStack, Check } from 'lucide-react';

export default function WllamaUI() {
  const {
    embeddingModel,
    isLoaded: embeddingModelLoaded,
    isLoading: loadingEmbeddingModel,
    loadProgress: embeddingModelProgress,
    error: embeddingModelError,
    status: embeddingModelStatus,
    modelCapabilities: embeddingModelCapabilities,
    loadEmbeddingModel,
    loadEmbeddingModelFromFiles,
    downloadEmbeddingModelToCache,
    isDownloading: embeddingModelDownloading,
    downloadProgress: embeddingDownloadProgress,
    downloadStatus: embeddingDownloadStatus,
    unloadEmbeddingModel,
  } = useEmbeddingModel();

  const router = useRouter();

  const [isLoading, setIsLoading] = useState(false);
  const [loadProgress, setLoadProgress] = useState(0);
  const [input, setInput] = useState('');
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const [nCtx, setNCtx] = useState(4096);
  const [isGenerating, setIsGenerating] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showModelManager, setShowModelManager] = useState(false);
  const [modelManagerTab, setModelManagerTab] = useState<'chat' | 'embedding' | 'cached'>('chat');
  const [modelUrl, setModelUrl] = useState('');
  const [cachedModels, setCachedModels] = useState<CachedModel[]>([]);
  const [loadMethod, setLoadMethod] = useState<'url' | 'file' | 'split'>('url');
  const [modelFile, setModelFile] = useState<FileList | null>(null);
  const [embeddingModelFiles, setEmbeddingModelFiles] = useState<File[]>([]);
  const [embeddingSplitFiles, setEmbeddingSplitFiles] = useState<File[]>([]);
  const [embeddingSplitUrls, setEmbeddingSplitUrls] = useState<string[]>(['']);
  const [embeddingSplitLoadMethod, setEmbeddingSplitLoadMethod] = useState<'local' | 'url'>('local');
  const [splitFiles, setSplitFiles] = useState<File[]>([]);
  const [splitLoadMethod, setSplitLoadMethod] = useState<'local' | 'url'>('local');
  const [splitUrls, setSplitUrls] = useState<string[]>(['']);
  const [availableFiles, setAvailableFiles] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [fetchingFiles, setFetchingFiles] = useState(false);
  const [chatTemplate, setChatTemplate] = useState<'gemma' | 'qwen' | 'llama' | 'chatml'>('gemma');
  const [useRAG, setUseRAG] = useState(false);
  const [ragTopK, setRagTopK] = useState(3);
  const [knowledgeBaseCount, setKnowledgeBaseCount] = useState(0);
  const [embeddingModelUrl, setEmbeddingModelUrl] = useState('https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf');
  const [modelLoaded, setModelLoaded] = useState(false);

  const wllamaRef = useRef<Wllama | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const embeddingFileInputRef = useRef<HTMLInputElement>(null);
  const embeddingSplitFileInputRef = useRef<HTMLInputElement>(null);
  const splitFileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const embeddingDbRef = useRef<IDBDatabase | null>(null);

  useEffect(() => {
    const savedConversations = localStorage.getItem('wllama-conversations');
    if (savedConversations) {
      try {
        const parsed = JSON.parse(savedConversations);
        const conversationsWithDates = parsed.map((conv: any) => ({
          ...conv,
          createdAt: new Date(conv.createdAt),
          updatedAt: new Date(conv.updatedAt),
          messages: conv.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }))
        }));
        setConversations(conversationsWithDates);

        const lastActiveId = localStorage.getItem('wllama-last-conversation-id');
        if (lastActiveId && conversationsWithDates.find((c: Conversation) => c.id === lastActiveId)) {
          setCurrentConversationId(lastActiveId);
        } else if (conversationsWithDates.length > 0) {
          setCurrentConversationId(conversationsWithDates[0].id);
        }
      } catch (err) {
        console.error('Failed to load conversations:', err);
      }
    }
    loadCachedModels();
    initEmbeddingDB();

    const existingModel = ChatModelManager.getModel();
    if (existingModel) {
      wllamaRef.current = existingModel;
      setModelLoaded(true);
      setStatus('Chat model restored from session');
    }

    const handleModelChange = () => {
      const model = ChatModelManager.getModel();
      wllamaRef.current = model;
      setModelLoaded(model !== null);
    };

    window.addEventListener('chat-model-changed', handleModelChange);
    
    return () => {
      window.removeEventListener('chat-model-changed', handleModelChange);
    };
  }, []);

  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem('wllama-conversations', JSON.stringify(conversations));
    }
  }, [conversations]);

  useEffect(() => {
    if (currentConversationId) {
      localStorage.setItem('wllama-last-conversation-id', currentConversationId);
    }
  }, [currentConversationId]);

  const currentConversation = conversations.find(c => c.id === currentConversationId);
  const messages = useMemo(() => currentConversation?.messages || [], [currentConversation]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const initEmbeddingDB = () => {
    const request = indexedDB.open('VectorDB', 2);

    request.onsuccess = (event) => {
      embeddingDbRef.current = (event.target as IDBOpenDBRequest).result;
      loadKnowledgeBaseCount();
    };

    request.onerror = () => {
      console.error('Failed to open embedding database');
    };

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains('embeddings')) {
        const objectStore = db.createObjectStore('embeddings', { keyPath: 'id' });
        // objectStore.createIndex('timestamp', 'metadata.timestamp', { unique: false });
      }
    };
  };

  const loadKnowledgeBaseCount = () => {
    if (!embeddingDbRef.current) return;

    const transaction = embeddingDbRef.current.transaction(['embeddings'], 'readonly');
    const objectStore = transaction.objectStore('embeddings');
    const request = objectStore.count();

    request.onsuccess = () => {
      setKnowledgeBaseCount(request.result);
    };
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

  const retrieveRelevantDocuments = async (query: string, topK: number = 3): Promise<EmbeddingDocument[]> => {
    if (!embeddingDbRef.current || !embeddingModel) {
      console.warn('Cannot retrieve documents: DB or embedding model not ready');
      return [];
    }

    try {
      setStatus('Creating query embedding...');
      
      if (embeddingModel.setOptions) {
        await embeddingModel.setOptions({ embeddings: true });
      }

      const queryEmbedding = await embeddingModel.createEmbedding(query);

      if (embeddingModel.setOptions) {
        await embeddingModel.setOptions({ embeddings: false });
      }

      setStatus('Searching knowledge base...');

      const transaction = embeddingDbRef.current.transaction(['embeddings'], 'readonly');
      const objectStore = transaction.objectStore('embeddings');
      const request = objectStore.getAll();

      return new Promise((resolve, reject) => {
        request.onsuccess = () => {
          const documents = request.result as EmbeddingDocument[];
          
          if (documents.length === 0) {
            console.warn('No documents in knowledge base');
            setStatus('');
            resolve([]);
            return;
          }

          const results = documents.map(doc => ({
            document: doc,
            similarity: cosineSimilarity(queryEmbedding, doc.embedding)
          }));

          results.sort((a, b) => b.similarity - a.similarity);
          const topResults = results.slice(0, topK).map(r => r.document);
          
          setStatus('');
          resolve(topResults);
        };

        request.onerror = () => {
          setStatus('');
          reject('Failed to retrieve documents');
        };
      });
    } catch (err) {
      console.error('Error retrieving documents:', err);
      setStatus('');
      if (embeddingModel?.setOptions) {
        try {
          await embeddingModel.setOptions({ embeddings: false });
        } catch (e) {
          console.error('Failed to reset embeddings mode:', e);
        }
      }
      return [];
    }
  };

  const loadCachedModels = async () => {
    try {
      const WllamaModule = await import('@wllama/wllama/esm/index.js');
      const { ModelManager } = WllamaModule;
      const manager = new ModelManager();
      const models = await manager.getModels();
      setCachedModels(models.map((m: any) => ({
        url: m.url,
        size: m.size,
        name: m.url.split('/').pop()?.replace('.gguf', '') || 'Unknown'
      })));
    } catch (err) {
      console.error('Failed to load cached models:', err);
    }
  };

  const fetchRepoFiles = async (repoId: string) => {
    setFetchingFiles(true);
    setAvailableFiles([]);
    setSelectedFile('');
    setError('');

    try {
      const response = await fetch(`https://huggingface.co/api/models/${repoId}/tree/main`);
      if (!response.ok) {
        throw new Error('Repository not found or inaccessible');
      }

      const data = await response.json();
      const ggufFiles = data
        .filter((item: any) => item.path.endsWith('.gguf'))
        .map((item: any) => item.path);

      if (ggufFiles.length === 0) {
        setError('No .gguf files found in this repository');
      } else {
        setAvailableFiles(ggufFiles);
        if (ggufFiles.length === 1) {
          setSelectedFile(ggufFiles[0]);
        }
      }
    } catch (err: any) {
      setError('Failed to fetch repository: ' + (err?.message || String(err)));
    } finally {
      setFetchingFiles(false);
    }
  };

  const handleRepoInputChange = (value: string) => {
    setModelUrl(value);

    const repoPattern = /^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_.-]+$/;
    if (repoPattern.test(value)) {
      fetchRepoFiles(value);
    } else {
      setAvailableFiles([]);
      setSelectedFile('');
    }
  };

  const createNewConversation = () => {
    const newConv: Conversation = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    };
    setConversations(prev => [newConv, ...prev]);
    setCurrentConversationId(newConv.id);
  };

  const updateConversationTitle = (conversationId: string, firstMessage: string) => {
    setConversations(prev => prev.map(conv => {
      if (conv.id === conversationId && conv.title === 'New Chat') {
        return {
          ...conv,
          title: firstMessage.slice(0, 30) + (firstMessage.length > 30 ? '...' : '')
        };
      }
      return conv;
    }));
  };

  const deleteConversation = (conversationId: string) => {
    setConversations(prev => prev.filter(c => c.id !== conversationId));
    if (currentConversationId === conversationId) {
      const remaining = conversations.filter(c => c.id !== conversationId);
      setCurrentConversationId(remaining.length > 0 ? remaining[0].id : null);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const hasGguf = Array.from(files).some(f => f.name.endsWith('.gguf'));
      if (hasGguf) {
        setModelFile(files);
        setError('');
      } else {
        setError('Please select at least one valid .gguf model file');
        setModelFile(null);
      }
    }
  };

  const handleEmbeddingFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) {
      setEmbeddingModelFiles([]);
      return;
    }

    const sortedFiles = Array.from(files)
      .filter(file => file.name.toLowerCase().endsWith('.gguf'))
      .sort((a, b) => a.name.localeCompare(b.name));

    if (sortedFiles.length === 0) {
      if (embeddingFileInputRef.current) {
        embeddingFileInputRef.current.value = '';
      }
      setEmbeddingModelFiles([]);
      return;
    }

    setEmbeddingModelFiles(sortedFiles);
  };

  const handleEmbeddingSplitFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) {
      setEmbeddingSplitFiles([]);
      return;
    }

    const sortedFiles = Array.from(files)
      .filter(file => file.name.toLowerCase().endsWith('.gguf'))
      .sort((a, b) => a.name.localeCompare(b.name));

    if (sortedFiles.length === 0) {
      setError('Please select valid .gguf split files');
      setEmbeddingSplitFiles([]);
      if (embeddingSplitFileInputRef.current) {
        embeddingSplitFileInputRef.current.value = '';
      }
      return;
    }

    setEmbeddingSplitFiles(sortedFiles);
    setError('');
  };

  const addEmbeddingSplitUrl = () => {
    setEmbeddingSplitUrls(prev => [...prev, '']);
  };

  const updateEmbeddingSplitUrl = (index: number, value: string) => {
    setEmbeddingSplitUrls(prev => {
      const next = [...prev];
      next[index] = value;
      return next;
    });
  };

  const removeEmbeddingSplitUrl = (index: number) => {
    setEmbeddingSplitUrls(prev => prev.filter((_, i) => i !== index));
  };

  const clearEmbeddingSplitInputs = () => {
    setEmbeddingSplitFiles([]);
    setEmbeddingSplitUrls(['']);
    if (embeddingSplitFileInputRef.current) {
      embeddingSplitFileInputRef.current.value = '';
    }
  };

  const loadEmbeddingSplitFromFiles = async () => {
    if (embeddingSplitFiles.length === 0) {
      setError('Please select split files first');
      return;
    }

    await loadEmbeddingModelFromFiles(embeddingSplitFiles);
    clearEmbeddingSplitInputs();
  };

  const loadEmbeddingSplitFromUrls = async () => {
    const urls = embeddingSplitUrls.map(u => u.trim()).filter(Boolean);
    if (urls.length === 0) {
      setError('Please enter at least one split URL');
      return;
    }

    try {
      setStatus('Downloading embedding splits...');
      const files: File[] = [];
      for (let i = 0; i < urls.length; i++) {
        const url = urls[i];
        setStatus(`Downloading embedding split ${i + 1}/${urls.length}...`);
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Failed to fetch ${url} (${response.status} ${response.statusText})`);
        }
        const blob = await response.blob();
        const filename = url.split('/').pop() || `split-${i + 1}.gguf`;
        files.push(new File([blob], filename, { type: blob.type || 'application/octet-stream' }));
      }
      setStatus('Loading embedding splits...');
      await loadEmbeddingModelFromFiles(files);
      setStatus('');
      clearEmbeddingSplitInputs();
    } catch (err: any) {
      setStatus('');
      setError('Failed to load embedding splits from URLs: ' + (err?.message || String(err)));
      console.error(err);
    }
  };

  const loadModelFromUrl = async (url: string) => {
    setIsLoading(true);
    setError('');
    setLoadProgress(0);
    setStatus('Initializing Wllama...');

    try {
      let fullUrl = url;
      if (!url.startsWith('http')) {
        if (selectedFile) {
          fullUrl = `https://huggingface.co/${modelUrl}/resolve/main/${selectedFile}`;
        } else {
          throw new Error('Please select a file from the repository');
        }
      }

      const WllamaModule = await import('@wllama/wllama/esm/index.js');
      const Wllama = WllamaModule.Wllama;

      const CONFIG_PATHS = {
        'single-thread/wllama.wasm': './wllama/esm/single-thread/wllama.wasm',
        'multi-thread/wllama.wasm': './wllama/esm/multi-thread/wllama.wasm',
      };

      wllamaRef.current = new Wllama(CONFIG_PATHS);

      const progressCallback = ({ loaded, total }: ProgressCallback) => {
        const progressPercentage = Math.round((loaded / total) * 100);
        setLoadProgress(progressPercentage);
        setStatus(`Loading model... ${progressPercentage}%`);
      };

      setStatus('Downloading chat model...');

      const start = Date.now();

      const config: WllamaConfig = {
        n_ctx: nCtx,
        n_batch: 2048,
        n_threads: navigator.hardwareConcurrency || 8,
        n_gpu_layers: 0,
        use_mlock: false,
        use_mmap: true,
        progressCallback,
      };

      await wllamaRef.current.loadModelFromUrl(fullUrl, config);

      ChatModelManager.setModel(wllamaRef.current);
      setModelLoaded(true);

      const took = Date.now() - start;
      setStatus(`Chat model loaded successfully! (${took} ms)`);
      setLoadProgress(100);
      setShowModelManager(false);

      if (conversations.length === 0) {
        createNewConversation();
      }

      await loadCachedModels();
    } catch (err: any) {
      setError('Failed to load model: ' + (err?.message || String(err)));
      setStatus('');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSplitFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const fileArray = Array.from(files);
      fileArray.sort((a, b) => a.name.localeCompare(b.name));
      
      const hasGguf = fileArray.some(f => f.name.endsWith('.gguf'));
      if (hasGguf) {
        setSplitFiles(fileArray);
        setError('');
      } else {
        setError('Please select valid .gguf model files');
        setSplitFiles([]);
      }
    }
  };

  const loadSplitModel = async (files: File[]) => {
    if (files.length === 0) {
      setError('Please select split model files first');
      return;
    }

    setIsLoading(true);
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

      const progressCallback = ({ loaded, total }: ProgressCallback) => {
        const progressPercentage = Math.round((loaded / total) * 100);
        setLoadProgress(progressPercentage);
        setStatus(`Loading split model... ${progressPercentage}%`);
      };

      setStatus('Loading split model files...');

      const start = Date.now();

      const config: WllamaConfig = {
        n_ctx: nCtx,
        n_batch: 2048,
        n_threads: navigator.hardwareConcurrency || 8,
        n_gpu_layers: 0,
        use_mlock: false,
        use_mmap: true,
        progressCallback,
      };

      await wllamaRef.current.loadModel(files, config);

      ChatModelManager.setModel(wllamaRef.current);
      setModelLoaded(true);

      const took = Date.now() - start;
      setStatus(`Split model loaded successfully! (${took} ms)`);
      setLoadProgress(100);
      setShowModelManager(false);

      if (conversations.length === 0) {
        createNewConversation();
      }
    } catch (err: any) {
      setError('Failed to load split model: ' + (err?.message || String(err)));
      setStatus('');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const loadSplitFromUrls = async () => {
    const validUrls = splitUrls.filter(url => url.trim() !== '');
    
    if (validUrls.length === 0) {
      setError('Please add at least one split model URL');
      return;
    }

    setIsLoading(true);
    setError('');
    setLoadProgress(0);
    setStatus('Downloading split model files...');

    try {
      const blobs: File[] = [];
      
      for (let i = 0; i < validUrls.length; i++) {
        const url = validUrls[i];
        setStatus(`Downloading split ${i + 1}/${validUrls.length}...`);
        
        const response = await fetch(url);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
        }
        
        const blob = await response.blob();
        const filename = url.split('/').pop() || `split-${i}.gguf`;
        const file = new File([blob], filename, { type: 'application/octet-stream' });
        blobs.push(file);
      }

      await loadSplitModel(blobs);
    } catch (err: any) {
      setError('Failed to load split models from URLs: ' + (err?.message || String(err)));
      setStatus('');
      setIsLoading(false);
    }
  };

  const addSplitUrl = () => {
    setSplitUrls([...splitUrls, '']);
  };

  const updateSplitUrl = (index: number, value: string) => {
    const updated = [...splitUrls];
    updated[index] = value;
    setSplitUrls(updated);
  };

  const removeSplitUrl = (index: number) => {
    setSplitUrls(splitUrls.filter((_, i) => i !== index));
  };

  const loadModelFromFile = async () => {
    if (!modelFile) {
      setError('Please select a model file first');
      return;
    }

    setIsLoading(true);
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

      const progressCallback = ({ loaded, total }: ProgressCallback) => {
        const progressPercentage = Math.round((loaded / total) * 100);
        setLoadProgress(progressPercentage);
        setStatus(`Loading model... ${progressPercentage}%`);
      };

      setStatus('Loading model from files...');

      const start = Date.now();

      const config: WllamaConfig = {
        n_ctx: nCtx,
        n_batch: 2048,
        n_threads: navigator.hardwareConcurrency || 8,
        n_gpu_layers: 0,
        use_mlock: false,
        use_mmap: true,
        progressCallback,
      };

      const filesToLoad = Array.from(modelFile);
      await wllamaRef.current.loadModel(filesToLoad, config);

      ChatModelManager.setModel(wllamaRef.current);
      setModelLoaded(true);

      const took = Date.now() - start;
      setStatus(`Model loaded successfully! (${took} ms)`);
      setLoadProgress(100);
      setShowModelManager(false);

      if (conversations.length === 0) {
        createNewConversation();
      }
    } catch (err: any) {
      setError('Failed to load model: ' + (err?.message || String(err)));
      setStatus('');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const unloadChatModel = () => {
    wllamaRef.current = null;
    ChatModelManager.clear();
    setModelLoaded(false);
    setStatus('Chat model unloaded');
  };

  const deleteCachedModel = async (url: string) => {
    try {
      const WllamaModule = await import('@wllama/wllama/esm/index.js');
      const { ModelManager } = WllamaModule;
      const manager = new ModelManager();
      const models = await manager.getModels();
      const model = models.find((m: any) => m.url === url);
      if (model) {
        await model.remove();
        await loadCachedModels();
        setStatus('Model deleted successfully');
      }
    } catch (err: any) {
      setError('Failed to delete model: ' + (err?.message || String(err)));
    }
  };

  const buildConversationPrompt = (messages: Message[], newUserMessage: string, ragContext?: string[]) => {
    let prompt = '';

    let contextSection = '';
    if (ragContext && ragContext.length > 0) {
      contextSection = '\n[Knowledge Base Context]\n' + ragContext.join('\n\n') + '\n[End of Context]\n\n';
    }

    switch (chatTemplate) {
      case 'gemma':
        messages.forEach(msg => {
          if (msg.role === 'user') {
            prompt += `<start_of_turn>user\n${msg.content}<end_of_turn>\n`;
          } else {
            prompt += `<start_of_turn>model\n${msg.content}<end_of_turn>\n`;
          }
        });
        prompt += `<start_of_turn>user\n${contextSection}${newUserMessage}<end_of_turn>\n<start_of_turn>model\n`;
        break;

      case 'qwen':
        messages.forEach(msg => {
          if (msg.role === 'user') {
            prompt += `<|im_start|>user\n${msg.content}<|im_end|>\n`;
          } else {
            prompt += `<|im_start|>assistant\n${msg.content}<|im_end|>\n`;
          }
        });
        prompt += `<|im_start|>user\n${contextSection}${newUserMessage}<|im_end|>\n<|im_start|>assistant\n`;
        break;

      case 'llama':
        messages.forEach(msg => {
          if (msg.role === 'user') {
            prompt += `[INST] ${msg.content} [/INST]\n`;
          } else {
            prompt += `${msg.content}\n`;
          }
        });
        prompt += `[INST] ${contextSection}${newUserMessage} [/INST]\n`;
        break;

      case 'chatml':
        messages.forEach(msg => {
          prompt += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
        });
        prompt += `<|im_start|>user\n${contextSection}${newUserMessage}<|im_end|>\n<|im_start|>assistant\n`;
        break;
    }

    return prompt;
  };

  const sendMessage = async () => {
    const model = ChatModelManager.getModel();
    if (!model) {
      setError('Please load a model first');
      return;
    }

    if (!input.trim()) {
      return;
    }

    if (!currentConversationId) {
      createNewConversation();
      return;
    }

    const userMessage = input.trim();
    setInput('');
    setIsGenerating(true);
    setError('');

    let ragContext: string[] = [];
    if (useRAG && embeddingModel && knowledgeBaseCount > 0) {
      setStatus('Retrieving relevant knowledge...');
      try {
        const relevantDocs = await retrieveRelevantDocuments(userMessage, ragTopK);
        ragContext = relevantDocs.map(doc => doc.text);
        setStatus('');
      } catch (err) {
        console.error('Failed to retrieve RAG context:', err);
        setStatus('');
      }
    }

    const newUserMessage: Message = {
      role: 'user',
      content: userMessage,
      timestamp: new Date(),
      ragContext: ragContext.length > 0 ? ragContext : undefined
    };

    setConversations(prev => prev.map(conv => {
      if (conv.id === currentConversationId) {
        const updatedMessages = [...conv.messages, newUserMessage];
        return {
          ...conv,
          messages: updatedMessages,
          updatedAt: new Date()
        };
      }
      return conv;
    }));

    if (messages.length === 0) {
      updateConversationTitle(currentConversationId, userMessage);
    }

    try {
      const assistantMessageIndex = messages.length + 1;

      setConversations(prev => prev.map(conv => {
        if (conv.id === currentConversationId) {
          return {
            ...conv,
            messages: [...conv.messages, {
              role: 'assistant',
              content: '',
              timestamp: new Date()
            }],
            updatedAt: new Date()
          };
        }
        return conv;
      }));

      const formattedPrompt = buildConversationPrompt(messages, userMessage, ragContext.length > 0 ? ragContext : undefined);

      let fullContent = '';
      let displayContent = '';

      await model.createCompletion(formattedPrompt, {
        nPredict: 512,
        sampling: {
          temp: 0.7,
          top_k: 40,
          top_p: 0.9,
          repeat_penalty: 1.1,
          repeat_last_n: 64,
        },
        onNewToken: (token: number, piece: Uint8Array, currentText: string) => {
          fullContent = currentText;

          displayContent = currentText
            .replace(/<think>[\s\S]*?<\/think>/gi, '')
            .replace(/<think>[\s\S]*$/gi, '')
            .replace(/<end_of_turn>/g, '')
            .replace(/<\|im_end\|>/g, '')
            .replace(/\[INST\]/g, '')
            .replace(/\[\/INST\]/g, '')
            .trim();

          setConversations(prev => prev.map(conv => {
            if (conv.id === currentConversationId) {
              const updated = [...conv.messages];
              if (updated[assistantMessageIndex]) {
                updated[assistantMessageIndex] = {
                  role: 'assistant',
                  content: displayContent,
                  timestamp: new Date()
                };
              }
              return { ...conv, messages: updated, updatedAt: new Date() };
            }
            return conv;
          }));
        }
      });

      const finalContent = fullContent
        .replace(/<think>[\s\S]*?<\/think>/gi, '')
        .replace(/<end_of_turn>/g, '')
        .replace(/<\|im_end\|>/g, '')
        .replace(/\[INST\]/g, '')
        .replace(/\[\/INST\]/g, '')
        .trim();

      setConversations(prev => prev.map(conv => {
        if (conv.id === currentConversationId) {
          const updated = [...conv.messages];
          if (updated[assistantMessageIndex]) {
            updated[assistantMessageIndex] = {
              role: 'assistant',
              content: finalContent,
              timestamp: new Date()
            };
          }
          return { ...conv, messages: updated, updatedAt: new Date() };
        }
        return conv;
      }));

    } catch (err: any) {
      setError('Failed to generate response: ' + (err?.message || String(err)));
      console.error(err);
    } finally {
      setIsGenerating(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const exportChat = () => {
    if (!currentConversation) return;

    const chatText = currentConversation.messages.map(msg =>
      `[${msg.timestamp.toLocaleString()}] ${msg.role.toUpperCase()}: ${msg.content}`
    ).join('\n\n');

    const blob = new Blob([chatText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentConversation.title}-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleRAGToggle = () => {
    if (!wllamaRef.current) {
      setError('Please load chat model first');
      return;
    }

    if (!embeddingModelLoaded) {
      setShowModelManager(true);
      setModelManagerTab('embedding');
      return;
    }

    setUseRAG(!useRAG);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex">
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 bg-slate-950/50 backdrop-blur-lg border-r border-white/10 flex flex-col overflow-hidden`}>
        <div className="p-4 border-b border-white/10">
          <h2 className="text-white font-bold text-lg mb-3">Wllama RAG</h2>
          <button
            onClick={createNewConversation}
            disabled={!modelLoaded}
            className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2 mb-2"
          >
            <Plus className="w-4 h-4" />
            New conversation
          </button>
          <button
            onClick={() => setShowModelManager(true)}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2 mb-2"
          >
            <HardDrive className="w-4 h-4" />
            Manage models
          </button>
          <button 
            onClick={() => router.push('/embedding')}
            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2 mb-2"
          >
            <Database className="w-4 h-4" />
            Vector Embeddings
          </button>
          
          <div className="mt-3 p-3 bg-white/5 rounded-lg border border-white/10">
            <div className="flex items-center justify-between mb-2">
              <label className="text-white text-sm font-semibold flex items-center gap-2">
                <BookOpen className="w-4 h-4" />
                RAG Mode
              </label>
              <button
                onClick={handleRAGToggle}
                disabled={!modelLoaded || (useRAG && knowledgeBaseCount === 0)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  useRAG ? 'bg-green-600' : 'bg-gray-600'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    useRAG ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
            
            {embeddingModelLoaded ? (
              <>
                <p className="text-xs text-green-200 mb-1 flex items-center gap-1">
                  <Sparkles className="w-3 h-3" />
                  Embedding model ready
                </p>
                <p className="text-xs text-blue-200 mb-2">
                  Knowledge: {knowledgeBaseCount} documents
                </p>
                {embeddingModelCapabilities && (
                  <p className="text-xs text-purple-200">
                    {embeddingModelCapabilities.n_embd}D vectors
                  </p>
                )}
              </>
            ) : (
              <>
                <p className="text-xs text-yellow-200 mb-2">
                  ⚠️ No embedding model loaded
                </p>
                <button
                  onClick={() => {
                    setShowModelManager(true);
                    setModelManagerTab('embedding');
                  }}
                  disabled={!modelLoaded}
                  className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 text-white text-xs font-semibold py-2 px-3 rounded-lg transition-colors flex items-center justify-center gap-1"
                >
                  <Download className="w-3 h-3" />
                  Load in Model Manager
                </button>
              </>
            )}
            
            {useRAG && embeddingModelLoaded && (
              <div className="mt-2">
                <label className="text-xs text-purple-200 block mb-1">
                  Top K: {ragTopK}
                </label>
                <input
                  type="range"
                  min="1"
                  max="5"
                  value={ragTopK}
                  onChange={(e) => setRagTopK(Number(e.target.value))}
                  className="w-full h-1"
                  disabled={!useRAG}
                />
              </div>
            )}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {conversations.map(conv => (
            <div
              key={conv.id}
              className={`group relative mb-1 p-3 rounded-lg cursor-pointer transition-colors ${currentConversationId === conv.id
                ? 'bg-purple-600/30 border border-purple-500/50'
                : 'bg-white/5 hover:bg-white/10'
                }`}
              onClick={() => setCurrentConversationId(conv.id)}
            >
              <div className="flex items-start gap-2">
                <MessageSquare className="w-4 h-4 text-purple-300 flex-shrink-0 mt-1" />
                <div className="flex-1 min-w-0">
                  <p className="text-white text-sm truncate">{conv.title}</p>
                  <p className="text-purple-300 text-xs">
                    {conv.messages.length} messages
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteConversation(conv.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 transition-opacity"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {showModelManager && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/20 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-white/10 flex items-center justify-between sticky top-0 bg-slate-900 z-10">
              <h2 className="text-2xl font-bold text-white">Model Manager</h2>
              <button
                onClick={() => setShowModelManager(false)}
                className="text-white hover:bg-white/10 p-2 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6">
              <div className="mb-6">
                <div className="flex items-center gap-4 p-4 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                  <Database className="w-6 h-6 text-purple-300" />
                  <div className="flex-1">
                    <h3 className="text-white font-semibold">Centralized Model Management</h3>
                    <p className="text-purple-200 text-sm">Load and manage both chat and embedding models in one place</p>
                  </div>
                </div>
              </div>

              <div className="flex gap-2 mb-6">
                <button
                  onClick={() => setModelManagerTab('chat')}
                  className={`flex-1 py-3 px-6 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                    modelManagerTab === 'chat'
                      ? 'bg-purple-600 text-white'
                      : 'bg-white/10 text-purple-200 hover:bg-white/20'
                  }`}
                >
                  <MessageSquare className="w-5 h-5" />
                  Chat Model
                </button>
                <button
                  onClick={() => setModelManagerTab('embedding')}
                  className={`flex-1 py-3 px-6 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                    modelManagerTab === 'embedding'
                      ? 'bg-blue-600 text-white'
                      : 'bg-white/10 text-blue-200 hover:bg-white/20'
                  }`}
                >
                  <Sparkles className="w-5 h-5" />
                  Embedding Model
                </button>
                <button
                  onClick={() => setModelManagerTab('cached')}
                  className={`flex-1 py-3 px-6 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                    modelManagerTab === 'cached'
                      ? 'bg-green-600 text-white'
                      : 'bg-white/10 text-green-200 hover:bg-white/20'
                  }`}
                >
                  <HardDrive className="w-5 h-5" />
                  Cache ({cachedModels.length})
                </button>
              </div>

              {modelManagerTab === 'chat' && (
                <div>
                  <div className="mb-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                    <p className="text-blue-200 text-sm mb-2">
                      <strong>💬 Chat Model:</strong> The main LLM for conversations
                    </p>
                    <ul className="text-blue-200 text-xs space-y-1 list-disc list-inside">
                      <li>Generates responses to your messages</li>
                      <li>Supports multiple chat templates (Gemma, Qwen, Llama, ChatML)</li>
                      <li>Persists across page navigation</li>
                    </ul>
                  </div>

                  {modelLoaded && (
                    <div className="mb-4 p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-green-300 font-semibold flex items-center gap-2">
                            <Check className="w-4 h-4" />
                            Chat model loaded
                          </p>
                          <p className="text-green-200 text-sm">Ready for conversations</p>
                        </div>
                        <button
                          onClick={unloadChatModel}
                          className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg text-sm"
                        >
                          Unload
                        </button>
                      </div>
                    </div>
                  )}

                  <div className="flex gap-2 mb-4">
                    <button
                      onClick={() => setLoadMethod('url')}
                      className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                        loadMethod === 'url'
                          ? 'bg-purple-600 text-white'
                          : 'bg-white/10 text-purple-200 hover:bg-white/20'
                      }`}
                    >
                      <Globe className="w-4 h-4" />
                      From URL
                    </button>
                    <button
                      onClick={() => setLoadMethod('file')}
                      className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                        loadMethod === 'file'
                          ? 'bg-purple-600 text-white'
                          : 'bg-white/10 text-purple-200 hover:bg-white/20'
                      }`}
                    >
                      <Upload className="w-4 h-4" />
                      From File
                    </button>
                    <button
                      onClick={() => setLoadMethod('split')}
                      className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                        loadMethod === 'split'
                          ? 'bg-purple-600 text-white'
                          : 'bg-white/10 text-purple-200 hover:bg-white/20'
                      }`}
                    >
                      <FileStack className="w-4 h-4" />
                      Split Model
                    </button>
                  </div>

                  {loadMethod === 'url' && (
                    <div className="mb-6">
                      <label className="block text-white font-semibold mb-3">
                        Hugging Face Repository or Direct URL
                      </label>
                      <input
                        type="text"
                        value={modelUrl}
                        onChange={(e) => handleRepoInputChange(e.target.value)}
                        placeholder="e.g., ggml-org/gemma-2-2b-it-GGUF"
                        className="w-full bg-white/10 border border-white/20 text-white placeholder-gray-400 rounded-lg p-3 mb-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
                      />

                      {fetchingFiles && (
                        <div className="flex items-center gap-2 text-purple-300 mb-3">
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span className="text-sm">Fetching repository files...</span>
                        </div>
                      )}

                      {availableFiles.length > 0 && (
                        <div className="mb-3">
                          <label className="block text-purple-200 text-sm mb-2">
                            Select a GGUF file ({availableFiles.length} available)
                          </label>
                          <select
                            value={selectedFile}
                            onChange={(e) => setSelectedFile(e.target.value)}
                            className="w-full bg-white/10 border border-white/20 text-white rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
                          >
                            <option value="">Choose a file...</option>
                            {availableFiles.map((file) => (
                              <option key={file} value={file}>
                                {file}
                              </option>
                            ))}
                          </select>
                        </div>
                      )}

                      <button
                        onClick={() => loadModelFromUrl(modelUrl)}
                        disabled={(!selectedFile && availableFiles.length > 0) || isLoading || !modelUrl}
                        className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
                      >
                        {isLoading ? (
                          <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            Loading... {loadProgress}%
                          </>
                        ) : (
                          <>
                            <Download className="w-5 h-5" />
                            {availableFiles.length > 0 ? 'Download & Load Selected' : 'Download & Load Model'}
                          </>
                        )}
                      </button>
                    </div>
                  )}

                  {loadMethod === 'file' && (
                    <div className="mb-6">
                      <label className="block text-white font-semibold mb-3">
                        Select Local .gguf File
                      </label>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".gguf"
                        multiple
                        onChange={handleFileSelect}
                        className="hidden"
                      />
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2 mb-3"
                      >
                        <Upload className="w-5 h-5" />
                        Choose File
                      </button>
                      {modelFile && (
                        <p className="text-green-300 text-sm mb-3">
                          Selected: {modelFile.length} file(s)
                        </p>
                      )}
                      <button
                        onClick={() => loadModelFromFile()}
                        disabled={!modelFile || isLoading}
                        className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
                      >
                        {isLoading ? (
                          <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            Loading... {loadProgress}%
                          </>
                        ) : (
                          <>
                            <Play className="w-5 h-5" />
                            Load Model
                          </>
                        )}
                      </button>
                    </div>
                  )}

                  {loadMethod === 'split' && (
                    <div className="mb-6">
                      <div className="flex items-center gap-2 mb-4">
                        <FileStack className="w-5 h-5 text-purple-300" />
                        <h3 className="text-white font-semibold">Load Split Model</h3>
                      </div>

                      <div className="flex gap-2 mb-4">
                        <button
                          onClick={() => setSplitLoadMethod('local')}
                          className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                            splitLoadMethod === 'local'
                              ? 'bg-purple-600 text-white'
                              : 'bg-white/10 text-purple-200 hover:bg-white/20'
                          }`}
                        >
                          <Upload className="w-4 h-4" />
                          Local Files
                        </button>
                        <button
                          onClick={() => setSplitLoadMethod('url')}
                          className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                            splitLoadMethod === 'url'
                              ? 'bg-purple-600 text-white'
                              : 'bg-white/10 text-purple-200 hover:bg-white/20'
                          }`}
                        >
                          <Globe className="w-4 h-4" />
                          From URLs
                        </button>
                      </div>

                      {splitLoadMethod === 'local' && (
                        <div>
                          <label className="block text-purple-200 text-sm mb-2">
                            Select all split model files (sorted automatically)
                          </label>
                          <input
                            ref={splitFileInputRef}
                            type="file"
                            accept=".gguf"
                            multiple
                            onChange={handleSplitFileSelect}
                            className="hidden"
                          />
                          <button
                            onClick={() => splitFileInputRef.current?.click()}
                            disabled={isLoading}
                            className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2 mb-3"
                          >
                            <Upload className="w-5 h-5" />
                            Choose Split Files
                          </button>

                          {splitFiles.length > 0 && (
                            <div className="bg-white/5 border border-white/10 rounded-lg p-3 mb-3">
                              <p className="text-green-300 text-sm font-semibold mb-2">
                                {splitFiles.length} file(s) selected:
                              </p>
                              <ul className="text-purple-200 text-xs space-y-1 max-h-32 overflow-y-auto">
                                {splitFiles.map((file, idx) => (
                                  <li key={idx} className="truncate">
                                    {idx + 1}. {file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}

                          <button
                            onClick={() => loadSplitModel(splitFiles)}
                            disabled={splitFiles.length === 0 || isLoading}
                            className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
                          >
                            {isLoading ? (
                              <>
                                <Loader2 className="w-5 h-5 animate-spin" />
                                Loading... {loadProgress}%
                              </>
                            ) : (
                              <>
                                <Play className="w-5 h-5" />
                                Load {splitFiles.length} Split Files
                              </>
                            )}
                          </button>
                        </div>
                      )}

                      {splitLoadMethod === 'url' && (
                        <div>
                          <label className="block text-purple-200 text-sm mb-2">
                            Enter URLs for each split (in order)
                          </label>
                          
                          <div className="space-y-2 mb-3 max-h-48 overflow-y-auto">
                            {splitUrls.map((url, idx) => (
                              <div key={idx} className="flex gap-2">
                                <input
                                  type="text"
                                  value={url}
                                  onChange={(e) => updateSplitUrl(idx, e.target.value)}
                                  placeholder={`Split ${idx + 1} URL`}
                                  className="flex-1 bg-white/10 border border-white/20 text-white placeholder-gray-400 rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                                  disabled={isLoading}
                                />
                                {splitUrls.length > 1 && (
                                  <button
                                    onClick={() => removeSplitUrl(idx)}
                                    disabled={isLoading}
                                    className="bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white px-3 rounded-lg transition-colors"
                                  >
                                    ✕
                                  </button>
                                )}
                              </div>
                            ))}
                          </div>

                          <button
                            onClick={addSplitUrl}
                            disabled={isLoading}
                            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors mb-3 text-sm"
                          >
                            + Add Another Split URL
                          </button>

                          <button
                            onClick={loadSplitFromUrls}
                            disabled={splitUrls.filter(u => u.trim()).length === 0 || isLoading}
                            className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
                          >
                            {isLoading ? (
                              <>
                                <Loader2 className="w-5 h-5 animate-spin" />
                                Loading... {loadProgress}%
                              </>
                            ) : (
                              <>
                                <Play className="w-5 h-5" />
                                Download & Load {splitUrls.filter(u => u.trim()).length} Splits
                              </>
                            )}
                          </button>
                        </div>
                      )}
                    </div>
                  )}

                  <div className="bg-white/5 border border-white/10 rounded-lg p-4">
                    <label className="block text-purple-200 text-sm mb-2">
                      Context Size: {nCtx}
                    </label>
                    <input
                      type="range"
                      min="128"
                      max="8192"
                      step="128"
                      value={nCtx}
                      onChange={(e) => setNCtx(Number(e.target.value))}
                      className="w-full"
                    />
                    <div className="mt-4">
                      <label className="block text-white font-semibold mb-3">
                        Chat Template
                      </label>
                      <select
                        value={chatTemplate}
                        onChange={(e) => setChatTemplate(e.target.value as any)}
                        className="w-full bg-white/10 border border-white/20 text-white rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
                      >
                        <option value="gemma">Gemma (Google)</option>
                        <option value="qwen">Qwen (Alibaba)</option>
                        <option value="llama">Llama 2/3 (Meta)</option>
                        <option value="chatml">ChatML (General)</option>
                      </select>
                    </div>
                  </div>
                </div>
              )}

              {modelManagerTab === 'embedding' && (
                <div>
                  <div className="mb-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                    <p className="text-blue-200 text-sm mb-2">
                      <strong>✨ Embedding Model:</strong> Converts text to vectors for RAG
                    </p>
                    <ul className="text-blue-200 text-xs space-y-1 list-disc list-inside">
                      <li>Required for semantic search in knowledge base</li>
                      <li>Shared across chat and embedding pages</li>
                      <li>Recommended: GGUF embeddings under ~1GB (e.g., nomic-embed-text-v1.5-Q5)</li>
                    </ul>
                  </div>

                  {embeddingModelLoaded && (
                    <div className="mb-4 p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <p className="text-green-300 font-semibold flex items-center gap-2">
                            <Check className="w-4 h-4" />
                            Embedding model loaded
                          </p>
                          <p className="text-green-200 text-sm">Ready for RAG and semantic search</p>
                        </div>
                        <button
                          onClick={() => {
                            if (window.confirm('Unload embedding model? RAG features will be disabled.')) {
                              unloadEmbeddingModel();
                            }
                          }}
                          className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg text-sm"
                        >
                          Unload
                        </button>
                      </div>
                      {embeddingModelCapabilities && (
                        <div className="grid grid-cols-2 gap-2 text-sm mt-3 p-3 bg-white/5 rounded-lg">
                          <div className="text-blue-200">
                            <span className="text-white font-medium">Type:</span> {embeddingModelCapabilities.model_type}
                          </div>
                          <div className="text-blue-200">
                            <span className="text-white font-medium">Dimensions:</span> {embeddingModelCapabilities.n_embd}D
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  <div className="space-y-3">
                    <input
                      type="text"
                      value={embeddingModelUrl}
                      onChange={(e) => setEmbeddingModelUrl(e.target.value)}
                      placeholder="https://huggingface.co/.../model.gguf"
                      className="w-full bg-white/10 border border-white/20 text-white placeholder-gray-400 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      disabled={loadingEmbeddingModel}
                    />
                    
                    <button
                      onClick={async () => {
                        await loadEmbeddingModel(embeddingModelUrl);
                      }}
                      disabled={loadingEmbeddingModel || embeddingModelLoaded || !embeddingModelUrl || embeddingModelDownloading}
                      className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
                    >
                      {loadingEmbeddingModel ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin" />
                          Loading... {embeddingModelProgress}%
                        </>
                      ) : embeddingModelLoaded ? (
                        <>
                          <Check className="w-5 h-5" />
                          Model Loaded
                        </>
                      ) : (
                        <>
                          <Download className="w-5 h-5" />
                          Download & Load Embedding Model
                        </>
                      )}
                    </button>

                    <div className="bg-white/5 border border-white/10 rounded-lg p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-white font-semibold">Cache With Model Manager</p>
                          <p className="text-blue-200 text-xs">
                            Download once and reuse from the browser cache (ideal for multi-shard GGUFs)
                          </p>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={async () => {
                          await downloadEmbeddingModelToCache(embeddingModelUrl);
                          await loadCachedModels();
                        }}
                        disabled={
                          !embeddingModelUrl ||
                          loadingEmbeddingModel ||
                          embeddingModelDownloading
                        }
                        className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                      >
                        {embeddingModelDownloading ? (
                          <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Caching... {embeddingDownloadProgress}%
                          </>
                        ) : (
                          <>
                            <HardDrive className="w-4 h-4" />
                            Download to Cache
                          </>
                        )}
                      </button>
                      {embeddingDownloadStatus && (
                        <p className="text-blue-200 text-xs whitespace-pre-wrap">
                          {embeddingDownloadStatus}
                        </p>
                      )}
                    </div>

                    <div className="bg-white/5 border border-white/10 rounded-lg p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-white font-semibold">Upload Local GGUF</p>
                          <p className="text-blue-200 text-xs">Load an embedding model directly from your device</p>
                        </div>
                        {embeddingModelFiles.length > 0 && (
                          <span className="text-green-300 text-xs">
                            {embeddingModelFiles.length} file(s)
                          </span>
                        )}
                      </div>
                      <input
                        ref={embeddingFileInputRef}
                        type="file"
                        accept=".gguf"
                        multiple
                        className="hidden"
                        onChange={handleEmbeddingFileSelect}
                      />
                      <div className="flex flex-wrap gap-2">
                        <button
                          type="button"
                          onClick={() => embeddingFileInputRef.current?.click()}
                          className="flex-1 min-w-[140px] bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                        >
                          <Upload className="w-4 h-4" />
                          Choose Files
                        </button>
                        <button
                          type="button"
                          onClick={async () => {
                            if (embeddingModelFiles.length === 0) {
                              embeddingFileInputRef.current?.click();
                              return;
                            }
                            await loadEmbeddingModelFromFiles(embeddingModelFiles);
                            setEmbeddingModelFiles([]);
                            if (embeddingFileInputRef.current) {
                              embeddingFileInputRef.current.value = '';
                            }
                          }}
                          disabled={loadingEmbeddingModel || embeddingModelLoaded || embeddingModelDownloading}
                          className="flex-1 min-w-[170px] bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                        >
                          {loadingEmbeddingModel ? (
                            <>
                              <Loader2 className="w-4 h-4 animate-spin" />
                              Loading...
                            </>
                          ) : (
                            <>
                              <Play className="w-4 h-4" />
                              Load From Files
                            </>
                          )}
                        </button>
                        {embeddingModelFiles.length > 0 && (
                          <button
                            type="button"
                            onClick={() => {
                              setEmbeddingModelFiles([]);
                              if (embeddingFileInputRef.current) {
                                embeddingFileInputRef.current.value = '';
                              }
                            }}
                            className="px-4 py-2 rounded-lg border border-white/20 text-white/80 hover:text-white hover:border-white/40 transition-colors text-sm"
                          >
                            Clear
                          </button>
                        )}
                      </div>
                      {embeddingModelFiles.length > 0 && (
                        <p className="text-blue-200 text-xs">
                          {embeddingModelFiles[0].name}
                          {embeddingModelFiles.length > 1 && ` (+${embeddingModelFiles.length - 1} more)`}
                        </p>
                      )}
                    </div>

                    <div className="bg-white/5 border border-white/10 rounded-lg p-4 space-y-3">
                      <div className="flex items-center gap-2">
                        <FileStack className="w-4 h-4 text-purple-300" />
                        <p className="text-white font-semibold">Load Split GGUF</p>
                      </div>
                      <p className="text-blue-200 text-xs">
                        Provide every shard either from local files or direct URLs, then load them together.
                      </p>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={() => setEmbeddingSplitLoadMethod('local')}
                          className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                            embeddingSplitLoadMethod === 'local'
                              ? 'bg-purple-600 text-white'
                              : 'bg-white/10 text-purple-200 hover:bg-white/20'
                          }`}
                          disabled={loadingEmbeddingModel || embeddingModelDownloading}
                        >
                          <Upload className="w-4 h-4" />
                          Local Files
                        </button>
                        <button
                          type="button"
                          onClick={() => setEmbeddingSplitLoadMethod('url')}
                          className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
                            embeddingSplitLoadMethod === 'url'
                              ? 'bg-purple-600 text-white'
                              : 'bg-white/10 text-purple-200 hover:bg-white/20'
                          }`}
                          disabled={loadingEmbeddingModel || embeddingModelDownloading}
                        >
                          <Globe className="w-4 h-4" />
                          From URLs
                        </button>
                      </div>

                      {embeddingSplitLoadMethod === 'local' && (
                        <div className="space-y-3">
                          <input
                            ref={embeddingSplitFileInputRef}
                            type="file"
                            accept=".gguf"
                            multiple
                            className="hidden"
                            onChange={handleEmbeddingSplitFileSelect}
                          />
                          <button
                            type="button"
                            onClick={() => embeddingSplitFileInputRef.current?.click()}
                            className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                            disabled={loadingEmbeddingModel || embeddingModelDownloading}
                          >
                            <Upload className="w-4 h-4" />
                            Choose Split Files
                          </button>
                          {embeddingSplitFiles.length > 0 && (
                            <p className="text-blue-200 text-xs">
                              {embeddingSplitFiles[0].name}
                              {embeddingSplitFiles.length > 1 && ` (+${embeddingSplitFiles.length - 1} more)`}
                            </p>
                          )}
                          <div className="flex flex-wrap gap-2">
                            <button
                              type="button"
                              onClick={loadEmbeddingSplitFromFiles}
                              disabled={
                                loadingEmbeddingModel ||
                                embeddingModelDownloading ||
                                embeddingSplitFiles.length === 0
                              }
                              className="flex-1 min-w-[160px] bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                            >
                              {loadingEmbeddingModel ? (
                                <>
                                  <Loader2 className="w-4 h-4 animate-spin" />
                                  Loading...
                                </>
                              ) : (
                                <>
                                  <Play className="w-4 h-4" />
                                  Load Splits
                                </>
                              )}
                            </button>
                            {embeddingSplitFiles.length > 0 && (
                              <button
                                type="button"
                                onClick={clearEmbeddingSplitInputs}
                                className="px-4 py-2 rounded-lg border border-white/20 text-white/80 hover:text-white hover:border-white/40 transition-colors text-sm"
                                disabled={loadingEmbeddingModel || embeddingModelDownloading}
                              >
                                Clear
                              </button>
                            )}
                          </div>
                        </div>
                      )}

                      {embeddingSplitLoadMethod === 'url' && (
                        <div className="space-y-3">
                          <div className="space-y-2">
                            {embeddingSplitUrls.map((url, index) => (
                              <div key={index} className="flex gap-2">
                                <input
                                  type="text"
                                  value={url}
                                  onChange={(e) => updateEmbeddingSplitUrl(index, e.target.value)}
                                  placeholder={`https://...split-${index + 1}.gguf`}
                                  className="flex-1 bg-white/10 border border-white/20 text-white placeholder-gray-400 rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                                  disabled={loadingEmbeddingModel || embeddingModelDownloading}
                                />
                                {embeddingSplitUrls.length > 1 && (
                                  <button
                                    type="button"
                                    onClick={() => removeEmbeddingSplitUrl(index)}
                                    className="px-3 py-2 rounded-lg border border-white/20 text-white/80 hover:text-white hover:border-white/40 transition-colors text-xs"
                                    disabled={loadingEmbeddingModel || embeddingModelDownloading}
                                  >
                                    Remove
                                  </button>
                                )}
                              </div>
                            ))}
                          </div>
                          <button
                            type="button"
                            onClick={addEmbeddingSplitUrl}
                            className="w-full bg-white/10 hover:bg-white/20 text-purple-200 font-semibold py-2 px-4 rounded-lg transition-colors"
                            disabled={loadingEmbeddingModel || embeddingModelDownloading}
                          >
                            Add Another URL
                          </button>
                          <div className="flex flex-wrap gap-2">
                            <button
                              type="button"
                              onClick={loadEmbeddingSplitFromUrls}
                              disabled={
                                loadingEmbeddingModel ||
                                embeddingModelDownloading ||
                                embeddingSplitUrls.every((url) => url.trim() === '')
                              }
                              className="flex-1 min-w-[160px] bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                            >
                              {loadingEmbeddingModel ? (
                                <>
                                  <Loader2 className="w-4 h-4 animate-spin" />
                                  Loading...
                                </>
                              ) : (
                                <>
                                  <Play className="w-4 h-4" />
                                  Download & Load
                                </>
                              )}
                            </button>
                            {(embeddingSplitUrls.length > 1 || embeddingSplitUrls[0].trim()) && (
                              <button
                                type="button"
                                onClick={clearEmbeddingSplitInputs}
                                className="px-4 py-2 rounded-lg border border-white/20 text-white/80 hover:text-white hover:border-white/40 transition-colors text-sm"
                                disabled={loadingEmbeddingModel || embeddingModelDownloading}
                              >
                                Clear
                              </button>
                            )}
                          </div>
                        </div>
                      )}
                    </div>

                    {embeddingModelStatus && (
                      <p className="text-blue-300 text-sm">{embeddingModelStatus}</p>
                    )}

                    {embeddingModelError && (
                      <p className="text-red-300 text-sm whitespace-pre-wrap">{embeddingModelError}</p>
                    )}
                  </div>
                </div>
              )}

              {modelManagerTab === 'cached' && (
                <div>
                  <div className="mb-4 p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
                    <p className="text-green-200 text-sm mb-2">
                      <strong>💾 Cached Models:</strong> Models stored in browser cache
                    </p>
                    <ul className="text-green-200 text-xs space-y-1 list-disc list-inside">
                      <li>Instantly load previously downloaded models</li>
                      <li>No re-download required</li>
                      <li>Clear cache to free up space or reload fresh models</li>
                    </ul>
                  </div>

                  {cachedModels.length === 0 ? (
                    <div className="text-center py-8">
                      <HardDrive className="w-16 h-16 mx-auto mb-4 text-gray-500 opacity-50" />
                      <p className="text-gray-400 mb-2">No cached models yet</p>
                      <p className="text-gray-500 text-sm">Download models from Chat or Embedding tabs to cache them</p>
                    </div>
                  ) : (
                    <>
                      <div className="mb-4 flex items-center justify-between">
                        <p className="text-white font-semibold">{cachedModels.length} Cached Model(s)</p>
                        <button
                          onClick={async () => {
                            if (window.confirm('Clear ALL cached models? This will free up space but you\'ll need to re-download models.')) {
                              for (const model of cachedModels) {
                                await deleteCachedModel(model.url);
                              }
                            }
                          }}
                          className="bg-red-600 hover:bg-red-700 text-white text-sm font-semibold py-2 px-4 rounded-lg transition-colors flex items-center gap-2"
                        >
                          <Trash2 className="w-4 h-4" />
                          Clear All Cache
                        </button>
                      </div>

                      <div className="space-y-2 max-h-[400px] overflow-y-auto">
                        {cachedModels.map((model) => (
                          <div
                            key={model.url}
                            className="bg-white/5 border border-white/10 rounded-lg p-4 flex items-center justify-between hover:bg-white/10 transition-colors"
                          >
                            <div className="flex-1 min-w-0">
                              <p className="text-white font-medium truncate">{model.name}</p>
                              <p className="text-purple-300 text-xs">
                                Size: {(model.size / 1024 / 1024).toFixed(1)} MB
                              </p>
                            </div>
                            <div className="flex gap-2 ml-4">
                              <button
                                onClick={() => loadModelFromUrl(model.url)}
                                disabled={isLoading}
                                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg text-sm whitespace-nowrap"
                              >
                                Load as Chat
                              </button>
                              <button
                                onClick={() => loadEmbeddingModel(model.url)}
                                disabled={loadingEmbeddingModel}
                                className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg text-sm whitespace-nowrap"
                              >
                                Load as Embedding
                              </button>
                              <button
                                onClick={() => deleteCachedModel(model.url)}
                                className="bg-red-600 hover:bg-red-700 text-white p-2 rounded-lg"
                                title="Delete from cache"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 flex flex-col h-screen">
        <div className="bg-white/10 backdrop-blur-lg border-b border-white/20 p-4">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-white hover:bg-white/10 p-2 rounded-lg transition-colors"
            >
              {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
            <div className="flex-1">
              <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                {currentConversation?.title || 'Wllama RAG Chatbot'}
                {useRAG && <Sparkles className="w-5 h-5 text-yellow-400" />}
              </h1>
              <p className="text-purple-200 text-sm">
                {modelLoaded ? (
                  useRAG ? `RAG Mode: ${knowledgeBaseCount} docs ready` : 'Model loaded - Ready to chat'
                ) : 'Load a model to start'}
              </p>
            </div>
          </div>
        </div>

        {status && (
          <div className="bg-blue-500/20 border-b border-blue-400/50 text-blue-100 px-6 py-3">
            {status}
          </div>
        )}

        {error && (
          <div className="bg-red-500/20 border-b border-red-400/50 text-red-100 px-6 py-3">
            {error}
          </div>
        )}

        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 && modelLoaded && (
            <div className="text-center text-purple-300 py-12">
              <Bot className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg">Start a conversation!</p>
              <p className="text-sm mt-2">Type a message below to begin chatting.</p>
              {useRAG && knowledgeBaseCount > 0 && (
                <div className="mt-4 inline-block bg-green-500/10 border border-green-500/30 rounded-lg px-4 py-2">
                  <p className="text-green-300 text-sm flex items-center gap-2">
                    <Sparkles className="w-4 h-4" />
                    RAG enabled with {knowledgeBaseCount} documents
                  </p>
                </div>
              )}
            </div>
          )}

          {!modelLoaded && (
            <div className="text-center text-purple-300 py-12">
              <HardDrive className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg">No model loaded</p>
              <p className="text-sm mt-2 mb-4">Click Manage models to get started</p>
              <button
                onClick={() => setShowModelManager(true)}
                className="bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-6 rounded-lg transition-colors"
              >
                Open Model Manager
              </button>
            </div>
          )}

          {messages.map((message, index) => (
            <div key={index}>
              <div
                className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {message.role === 'assistant' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center">
                    <Bot className="w-5 h-5 text-white" />
                  </div>
                )}

                <div
                  className={`max-w-[70%] rounded-2xl p-4 ${message.role === 'user'
                    ? 'bg-purple-600 text-white'
                    : 'bg-white/10 text-white border border-white/20'
                    }`}
                >
                  <p className="whitespace-pre-wrap break-words">{message.content}</p>
                  <p className="text-xs mt-2 opacity-60">
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>

                {message.role === 'user' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-pink-600 flex items-center justify-center">
                    <User className="w-5 h-5 text-white" />
                  </div>
                )}
              </div>

              {message.role === 'user' && message.ragContext && message.ragContext.length > 0 && (
                <div className="ml-11 mt-2 max-w-[70%]">
                  <details className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
                    <summary className="text-green-300 text-xs cursor-pointer flex items-center gap-2">
                      <BookOpen className="w-3 h-3" />
                      Retrieved {message.ragContext.length} relevant documents
                    </summary>
                    <div className="mt-2 space-y-2">
                      {message.ragContext.map((ctx, idx) => (
                        <div key={idx} className="text-green-200 text-xs bg-white/5 rounded p-2">
                          {ctx.substring(0, 150)}...
                        </div>
                      ))}
                    </div>
                  </details>
                </div>
              )}
            </div>
          ))}

          {isGenerating && (
            <div className="flex gap-3 justify-start">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="bg-white/10 text-white border border-white/20 rounded-2xl p-4">
                <Loader2 className="w-5 h-5 animate-spin" />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="bg-white/10 backdrop-blur-lg border-t border-white/20 p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-3 items-end">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder={modelLoaded ? "Type your message..." : "Load a model first..."}
                disabled={!modelLoaded || isGenerating}
                className="flex-1 bg-white/10 border border-white/20 text-white placeholder-gray-400 rounded-lg p-3 min-h-[60px] max-h-[200px] focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none disabled:opacity-50"
                rows={2}
              />
              <button
                onClick={sendMessage}
                disabled={!modelLoaded || isGenerating || !input.trim()}
                className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed text-white font-bold p-4 rounded-lg transition-all flex items-center justify-center shadow-lg"
              >
                <Send className="w-5 h-5" />
              </button>
              {messages.length > 0 && (
                <button
                  onClick={exportChat}
                  disabled={isGenerating}
                  className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold p-4 rounded-lg transition-all flex items-center justify-center"
                  title="Export chat"
                >
                  <Download className="w-5 h-5" />
                </button>
              )}
            </div>
            <div className="flex items-center justify-between mt-2">
              <p className="text-xs text-purple-300">
                Press Enter to send
                {useRAG && <span className="text-green-300"> • RAG Active</span>}
              </p>
              {conversations.length > 0 && (
                <p className="text-xs text-green-300 flex items-center gap-1">
                  <Save className="w-3 h-3" />
                  Auto-saved
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
