'use client';

import { useState, useRef, useEffect, useMemo } from 'react';
import { Upload, Play, Loader2, Send, Trash2, User, Bot, Download, Save, Plus, MessageSquare, Menu, X, Globe, HardDrive, FileStack, Database } from 'lucide-react';
import SplitModelLoader from './split-model';
import Link from 'next/link';

interface ProgressCallback {
  loaded: number;
  total: number;
}

interface WllamaConfig {
  n_ctx: number;
  n_batch: number;
  n_threads: number;
  n_gpu_layers: number;
  use_mlock: boolean;
  use_mmap: boolean;
  progressCallback: (progress: ProgressCallback) => void;
}

interface Wllama {
  loadModelFromUrl: (url: string, config: WllamaConfig) => Promise<void>;
  loadModel: (blobs: File[], config: WllamaConfig) => Promise<void>;
  createCompletion: (prompt: string, options: any) => Promise<string>;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

interface CachedModel {
  url: string;
  size: number;
  name: string;
}

export default function WllamaUI() {
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
  const [modelUrl, setModelUrl] = useState('');
  const [cachedModels, setCachedModels] = useState<CachedModel[]>([]);
  const [loadMethod, setLoadMethod] = useState<'url' | 'file' | 'split'>('url');
  const [modelFile, setModelFile] = useState<FileList | null>(null);
  const [availableFiles, setAvailableFiles] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [fetchingFiles, setFetchingFiles] = useState(false);
  const [chatTemplate, setChatTemplate] = useState<'gemma' | 'qwen' | 'llama' | 'chatml'>('gemma');
  const wllamaRef = useRef<Wllama | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Load conversations from localStorage on mount
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
      // Fetch file list from Hugging Face API
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

    // Auto-fetch files if input looks like a repo ID
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

  const loadModelFromUrl = async (url: string) => {
    setIsLoading(true);
    setError('');
    setLoadProgress(0);
    setStatus('Initializing Wllama...');

    try {
      // Construct full URL if needed
      let fullUrl = url;
      if (!url.startsWith('http')) {
        // Assume it's a repo ID with a selected file
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

      setStatus('Downloading model from URL...');

      const start = Date.now();

      const config: WllamaConfig = {
        n_ctx: nCtx,
        n_batch: 2048,
        n_threads: navigator.hardwareConcurrency || 8, // Use all available CPU threads
        n_gpu_layers: 0,
        use_mlock: false,
        use_mmap: true,
        progressCallback,
      };

      await wllamaRef.current.loadModelFromUrl(fullUrl, config);

      const took = Date.now() - start;
      setStatus(`Model loaded successfully! (${took} ms)`);
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

  const loadModelFromFile = async (blobs?: File[]) => {
    const filesToLoad = blobs || (modelFile ? Array.from(modelFile) : null);
    
    if (!filesToLoad) {
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

      await wllamaRef.current.loadModel(filesToLoad, config);

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

  const buildConversationPrompt = (messages: Message[], newUserMessage: string) => {
    let prompt = '';

    switch (chatTemplate) {
      case 'gemma':
        // Gemma format
        messages.forEach(msg => {
          if (msg.role === 'user') {
            prompt += `<start_of_turn>user\n${msg.content}<end_of_turn>\n`;
          } else {
            prompt += `<start_of_turn>model\n${msg.content}<end_of_turn>\n`;
          }
        });
        prompt += `<start_of_turn>user\n${newUserMessage}<end_of_turn>\n<start_of_turn>model\n`;
        break;

      case 'qwen':
        // Qwen format
        messages.forEach(msg => {
          if (msg.role === 'user') {
            prompt += `<|im_start|>user\n${msg.content}<|im_end|>\n`;
          } else {
            prompt += `<|im_start|>assistant\n${msg.content}<|im_end|>\n`;
          }
        });
        prompt += `<|im_start|>user\n${newUserMessage}<|im_end|>\n<|im_start|>assistant\n`;
        break;

      case 'llama':
        // Llama 2/3 format
        messages.forEach(msg => {
          if (msg.role === 'user') {
            prompt += `[INST] ${msg.content} [/INST]\n`;
          } else {
            prompt += `${msg.content}\n`;
          }
        });
        prompt += `[INST] ${newUserMessage} [/INST]\n`;
        break;

      case 'chatml':
        // ChatML format (used by many models)
        messages.forEach(msg => {
          prompt += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
        });
        prompt += `<|im_start|>user\n${newUserMessage}<|im_end|>\n<|im_start|>assistant\n`;
        break;
    }

    return prompt;
  };

  const sendMessage = async () => {
    if (!wllamaRef.current) {
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

    const newUserMessage: Message = {
      role: 'user',
      content: userMessage,
      timestamp: new Date()
    };

    // Add user message first
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
      const formattedPrompt = buildConversationPrompt(messages, userMessage);

      // Get the index for the assistant's message
      const assistantMessageIndex = messages.length + 1; // +1 because we just added user message

      // Add placeholder for assistant response
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

      let fullContent = '';
      let displayContent = '';

      // Use onNewToken callback for streaming
      await wllamaRef.current!.createCompletion(formattedPrompt, {
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

          // Remove thinking blocks in real-time
          displayContent = currentText
            .replace(/<think>[\s\S]*?<\/think>/gi, '')
            .replace(/<think>[\s\S]*$/gi, '') // Handle incomplete think blocks
            .replace(/<end_of_turn>/g, '')
            .replace(/<\|im_end\|>/g, '')
            .replace(/\[INST\]/g, '')
            .replace(/\[\/INST\]/g, '')
            .trim();

          // Update the message in real-time
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

      // Final cleanup
      const finalContent = fullContent
        .replace(/<think>[\s\S]*?<\/think>/gi, '')
        .replace(/<end_of_turn>/g, '')
        .replace(/<\|im_end\|>/g, '')
        .replace(/\[INST\]/g, '')
        .replace(/\[\/INST\]/g, '')
        .trim();

      // Update with final cleaned content
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 bg-slate-950/50 backdrop-blur-lg border-r border-white/10 flex flex-col overflow-hidden`}>
        <div className="p-4 border-b border-white/10">
          <h2 className="text-white font-bold text-lg mb-3">Wllama</h2>
          <button
            onClick={createNewConversation}
            disabled={!wllamaRef.current}
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
          <Link href="/embedding">
            <button className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2">
              <Database className="w-4 h-4" />
              Vector Embeddings
            </button>
          </Link>
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

      {/* Model Manager Modal */}
      {showModelManager && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/20 max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6 border-b border-white/10 flex items-center justify-between sticky top-0 bg-slate-900">
              <h2 className="text-2xl font-bold text-white">Manage Models</h2>
              <button
                onClick={() => setShowModelManager(false)}
                className="text-white hover:bg-white/10 p-2 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6">
              {/* Load Method Tabs */}
              <div className="flex gap-2 mb-6">
                <button
                  onClick={() => setLoadMethod('url')}
                  className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${loadMethod === 'url'
                    ? 'bg-purple-600 text-white'
                    : 'bg-white/10 text-purple-200 hover:bg-white/20'
                    }`}
                >
                  <Globe className="w-4 h-4" />
                  From URL
                </button>
                <button
                  onClick={() => setLoadMethod('file')}
                  className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${loadMethod === 'file'
                    ? 'bg-purple-600 text-white'
                    : 'bg-white/10 text-purple-200 hover:bg-white/20'
                    }`}
                >
                  <Upload className="w-4 h-4" />
                  From File
                </button>
                <button
                  onClick={() => setLoadMethod('split')}
                  className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${loadMethod === 'split'
                    ? 'bg-purple-600 text-white'
                    : 'bg-white/10 text-purple-200 hover:bg-white/20'
                    }`}
                >
                  <FileStack className="w-4 h-4" />
                  Split Model
                </button>
              </div>

              {/* Load from URL */}
              {loadMethod === 'url' && (
                <div className="mb-6">
                  <label className="block text-white font-semibold mb-3">
                    Hugging Face Repository or Direct URL
                  </label>
                  <input
                    type="text"
                    value={modelUrl}
                    onChange={(e) => handleRepoInputChange(e.target.value)}
                    placeholder="e.g., ggml-org/gemma-3-270m-it-GGUF"
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

                  <div className="mt-3 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                    <p className="text-blue-200 text-xs">
                      <strong>Examples:</strong><br />
                      • ggml-org/gemma-3-270m-it-GGUF (Gemma)<br />
                      • Qwen/Qwen2.5-0.5B-Instruct-GGUF (Qwen)<br />
                      • TheBloke/Llama-2-7B-Chat-GGUF (Llama)<br />
                      • Or paste a direct .gguf URL
                    </p>
                  </div>
                </div>
              )}

              {/* Load from File */}
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

              {/* Load Split Model */}
              {loadMethod === 'split' && (
                <SplitModelLoader
                  onLoadModel={loadModelFromFile}
                  isLoading={isLoading}
                  nCtx={nCtx}
                />
              )}

              {/* Context Size Setting */}
              <div className="mb-6 bg-white/5 border border-white/10 rounded-lg p-4">
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
              </div>

              {/* Chat Template Selection */}
              <div className="mb-6 bg-white/5 border border-white/10 rounded-lg p-4">
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
                <p className="text-purple-300 text-xs mt-2">
                  Choose the template matching your model&lsquo;s training format
                </p>
              </div>

              {/* Cached Models List */}
              <div>
                <h3 className="text-white font-semibold mb-3">Cached Models ({cachedModels.length})</h3>
                <div className="space-y-2">
                  {cachedModels.length === 0 ? (
                    <p className="text-purple-300 text-sm">No models cached yet</p>
                  ) : (
                    cachedModels.map((model) => (
                      <div
                        key={model.url}
                        className="bg-white/5 border border-white/10 rounded-lg p-4 flex items-center justify-between"
                      >
                        <div className="flex-1">
                          <p className="text-white font-medium">{model.name}</p>
                          <p className="text-purple-300 text-xs">
                            Size: {(model.size / 1024 / 1024).toFixed(1)} MB
                          </p>
                        </div>
                        <div className="flex gap-2">
                          <button
                            onClick={() => loadModelFromUrl(model.url)}
                            disabled={isLoading}
                            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg text-sm"
                          >
                            Load
                          </button>
                          <button
                            onClick={() => deleteCachedModel(model.url)}
                            className="bg-red-600 hover:bg-red-700 text-white p-2 rounded-lg"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-screen">
        {/* Header */}
        <div className="bg-white/10 backdrop-blur-lg border-b border-white/20 p-4">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-white hover:bg-white/10 p-2 rounded-lg transition-colors"
            >
              {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
            <div className="flex-1">
              <h1 className="text-2xl font-bold text-white">
                {currentConversation?.title || 'Wllama Chatbot'}
              </h1>
              <p className="text-purple-200 text-sm">
                {wllamaRef.current ? 'Model loaded - Ready to chat' : 'Load a model to start'}
              </p>
            </div>
          </div>
        </div>

        {/* Status Messages */}
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

        {/* Chat Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 && wllamaRef.current && (
            <div className="text-center text-purple-300 py-12">
              <Bot className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg">Start a conversation!</p>
              <p className="text-sm mt-2">Type a message below to begin chatting.</p>
            </div>
          )}

          {!wllamaRef.current && (
            <div className="text-center text-purple-300 py-12">
              <HardDrive className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg">No model loaded</p>
              <p className="text-sm mt-2 mb-4">Click &quot;Manage models&quot; to get started</p>
              <button
                onClick={() => setShowModelManager(true)}
                className="bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-6 rounded-lg transition-colors"
              >
                Open Model Manager
              </button>
            </div>
          )}

          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
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

        {/* Input Area */}
        <div className="bg-white/10 backdrop-blur-lg border-t border-white/20 p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-3 items-end">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder={wllamaRef.current ? "Type your message... (Shift+Enter for new line)" : "Load a model first..."}
                disabled={!wllamaRef.current || isGenerating}
                className="flex-1 bg-white/10 border border-white/20 text-white placeholder-gray-400 rounded-lg p-3 min-h-[60px] max-h-[200px] focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none disabled:opacity-50"
                rows={2}
              />
              <button
                onClick={sendMessage}
                disabled={!wllamaRef.current || isGenerating || !input.trim()}
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
                Press Enter to send &bull; Shift+Enter for new line
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