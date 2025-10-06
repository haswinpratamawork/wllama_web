'use client';

import { useState, useRef } from 'react';
import { Upload, Loader2, Play, Link, FileStack } from 'lucide-react';

interface SplitModelLoaderProps {
  onLoadModel: (blobs: File[], config: any) => Promise<void>;
  isLoading: boolean;
  nCtx: number;
}

export default function SplitModelLoader({ onLoadModel, isLoading, nCtx }: SplitModelLoaderProps) {
  const [loadMethod, setLoadMethod] = useState<'local' | 'url'>('local');
  const [modelFiles, setModelFiles] = useState<File[]>([]);
  const [splitUrls, setSplitUrls] = useState<string[]>(['']);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const fileArray = Array.from(files);
      
      // Sort files by name to ensure correct order
      fileArray.sort((a, b) => a.name.localeCompare(b.name));
      
      const hasGguf = fileArray.some(f => f.name.endsWith('.gguf'));
      if (hasGguf) {
        setModelFiles(fileArray);
        setError('');
      } else {
        setError('Please select valid .gguf model files');
        setModelFiles([]);
      }
    }
  };

  const handleLoadLocalSplits = async () => {
    if (modelFiles.length === 0) {
      setError('Please select split model files first');
      return;
    }

    setError('');

    try {
      const config = {
        n_ctx: nCtx,
        n_batch: 2048,
        n_threads: navigator.hardwareConcurrency || 8,
        n_gpu_layers: 0,
        use_mlock: false,
        use_mmap: true,
        progressCallback: ({ loaded, total }: { loaded: number; total: number }) => {
          console.log(`Loading: ${Math.round((loaded / total) * 100)}%`);
        },
      };

      await onLoadModel(modelFiles, config);
    } catch (err: any) {
      setError('Failed to load split models: ' + (err?.message || String(err)));
      console.error(err);
    }
  };

  const handleLoadUrlSplits = async () => {
    const validUrls = splitUrls.filter(url => url.trim() !== '');
    
    if (validUrls.length === 0) {
      setError('Please add at least one split model URL');
      return;
    }

    setError('');

    try {
      // Download all splits as blobs
      const blobs: File[] = [];
      
      for (let i = 0; i < validUrls.length; i++) {
        const url = validUrls[i];
        const response = await fetch(url);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
        }
        
        const blob = await response.blob();
        const filename = url.split('/').pop() || `split-${i}.gguf`;
        const file = new File([blob], filename, { type: 'application/octet-stream' });
        blobs.push(file);
      }

      const config = {
        n_ctx: nCtx,
        n_batch: 2048,
        n_threads: navigator.hardwareConcurrency || 8,
        n_gpu_layers: 0,
        use_mlock: false,
        use_mmap: true,
        progressCallback: ({ loaded, total }: { loaded: number; total: number }) => {
          console.log(`Loading: ${Math.round((loaded / total) * 100)}%`);
        },
      };

      await onLoadModel(blobs, config);
    } catch (err: any) {
      setError('Failed to load split models from URLs: ' + (err?.message || String(err)));
      console.error(err);
    }
  };

  const addUrlField = () => {
    setSplitUrls([...splitUrls, '']);
  };

  const updateUrl = (index: number, value: string) => {
    const updated = [...splitUrls];
    updated[index] = value;
    setSplitUrls(updated);
  };

  const removeUrl = (index: number) => {
    setSplitUrls(splitUrls.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 mb-4">
        <FileStack className="w-5 h-5 text-purple-300" />
        <h3 className="text-white font-semibold text-lg">Load Split Model</h3>
      </div>

      {/* Load Method Tabs */}
      <div className="flex gap-2">
        <button
          onClick={() => setLoadMethod('local')}
          className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
            loadMethod === 'local'
              ? 'bg-purple-600 text-white'
              : 'bg-white/10 text-purple-200 hover:bg-white/20'
          }`}
        >
          <Upload className="w-4 h-4" />
          Local Files
        </button>
        <button
          onClick={() => setLoadMethod('url')}
          className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
            loadMethod === 'url'
              ? 'bg-purple-600 text-white'
              : 'bg-white/10 text-purple-200 hover:bg-white/20'
          }`}
        >
          <Link className="w-4 h-4" />
          From URLs
        </button>
      </div>

      {/* Local Files Method */}
      {loadMethod === 'local' && (
        <div>
          <label className="block text-purple-200 text-sm mb-2">
            Select all split model files (they&apos;ll be sorted automatically)
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
            disabled={isLoading}
            className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2 mb-3"
          >
            <Upload className="w-5 h-5" />
            Choose Split Files
          </button>

          {modelFiles.length > 0 && (
            <div className="bg-white/5 border border-white/10 rounded-lg p-3 mb-3">
              <p className="text-green-300 text-sm font-semibold mb-2">
                {modelFiles.length} file(s) selected:
              </p>
              <ul className="text-purple-200 text-xs space-y-1">
                {modelFiles.map((file, idx) => (
                  <li key={idx} className="truncate">
                    {idx + 1}. {file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)
                  </li>
                ))}
              </ul>
            </div>
          )}

          <button
            onClick={handleLoadLocalSplits}
            disabled={modelFiles.length === 0 || isLoading}
            className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Loading Split Model...
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Load {modelFiles.length} Split Files
              </>
            )}
          </button>
        </div>
      )}

      {/* URL Method */}
      {loadMethod === 'url' && (
        <div>
          <label className="block text-purple-200 text-sm mb-2">
            Enter URLs for each split (in order)
          </label>
          
          <div className="space-y-2 mb-3">
            {splitUrls.map((url, idx) => (
              <div key={idx} className="flex gap-2">
                <input
                  type="text"
                  value={url}
                  onChange={(e) => updateUrl(idx, e.target.value)}
                  placeholder={`Split ${idx + 1} URL (e.g., https://example.com/model-${idx + 1}.gguf)`}
                  className="flex-1 bg-white/10 border border-white/20 text-white placeholder-gray-400 rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                  disabled={isLoading}
                />
                {splitUrls.length > 1 && (
                  <button
                    onClick={() => removeUrl(idx)}
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
            onClick={addUrlField}
            disabled={isLoading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors mb-3 text-sm"
          >
            + Add Another Split URL
          </button>

          <button
            onClick={handleLoadUrlSplits}
            disabled={splitUrls.filter(u => u.trim()).length === 0 || isLoading}
            className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Downloading & Loading...
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

      {/* Error Message */}
      {error && (
        <div className="bg-red-500/20 border border-red-400/50 text-red-200 rounded-lg p-3 text-sm">
          {error}
        </div>
      )}

      {/* Info Box */}
      <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
        <p className="text-blue-200 text-xs">
          <strong>ℹ️ Split Models:</strong><br />
          Wllama can load models that have been split into multiple files. Select all parts in order (e.g., model-00001.gguf, model-00002.gguf, etc.). The files will be loaded sequentially.
        </p>
      </div>
    </div>
  );
}