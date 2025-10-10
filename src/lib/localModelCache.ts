import type { ModelManager } from '@wllama/wllama';

export interface StoredLocalModelResult {
  modelUrl: string;
  totalSize: number;
  baseName: string;
}

interface StoreOptions {
  namespace?: string;
}

const createNamespace = (explicit?: string) => {
  if (explicit) return explicit;
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).slice(2, 8);
  return `local-${timestamp}-${random}`;
};

const assertOpfsSupport = () => {
  if (typeof navigator === 'undefined' || !('storage' in navigator)) {
    throw new Error('Browser tidak mendukung penyimpanan model lokal (navigator.storage tidak tersedia)');
  }
  const storageAny = navigator.storage as any;
  if (typeof storageAny?.getDirectory !== 'function') {
    throw new Error('Browser tidak mendukung penyimpanan model lokal (OPFS tidak tersedia)');
  }
  return storageAny;
};

/**
 * Persists the provided GGUF shards into wllama's internal cache storage.
 * Returns a synthetic local URL that can be passed back to ModelManager.loadModel.
 */
export async function storeLocalModelFiles(
  files: File[],
  modelManager: ModelManager,
  options: StoreOptions = {}
): Promise<StoredLocalModelResult> {
  if (!files.length) {
    throw new Error('Tidak ada file yang dipilih');
  }

  const sortedFiles = [...files].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
  );
  const baseName = sortedFiles[0]?.name ?? 'local-model.gguf';
  const namespace = createNamespace(options.namespace);
  const baseUrl = `local://${namespace}`;

  const cache = (modelManager as any).cacheManager;
  if (!cache || typeof cache.getNameFromURL !== 'function') {
    throw new Error('ModelManager cache tidak tersedia');
  }

  const storageAny = assertOpfsSupport();
  const opfsRoot: FileSystemDirectoryHandle = await storageAny.getDirectory();
  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });

  const writeToOpfs = async (name: string, data: Blob | string) => {
    const handle = await cacheDir.getFileHandle(name, { create: true });
    const writable = await handle.createWritable();
    await writable.truncate(0);
    await writable.write(data);
    await writable.close();
  };

  let totalSize = 0;
  for (const file of sortedFiles) {
    const virtualUrl = `${baseUrl}/${file.name}`;
    const cacheKey = await cache.getNameFromURL(virtualUrl);
    const metadataKey = `__metadata__${cacheKey}`;

    await writeToOpfs(cacheKey, file);
    await writeToOpfs(
      metadataKey,
      JSON.stringify({
        originalURL: virtualUrl,
        originalSize: file.size,
        etag: 'local-file',
      })
    );
    totalSize += file.size;
  }

  const modelUrl = `${baseUrl}/${sortedFiles[0].name}`;
  return {
    modelUrl,
    totalSize,
    baseName,
  };
}

