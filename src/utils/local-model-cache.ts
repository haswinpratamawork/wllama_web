import type { ModelManager } from '@wllama/wllama';

export interface StoredLocalModelResult {
  modelUrl: string;
  totalSize: number;
  baseName: string;
}

export async function storeLocalModelFiles(
  files: File[],
  modelManager: ModelManager,
  options: { namespace?: string } = {}
): Promise<StoredLocalModelResult> {
  if (!files.length) {
    throw new Error('Tidak ada file yang dipilih');
  }

  const sortedFiles = [...files].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
  );
  const baseName = sortedFiles[0]?.name ?? 'local-model.gguf';
  const namespace =
    options.namespace ?? `local-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  const baseUrl = `local://${namespace}`;

  const cache = modelManager.cacheManager;
  const storageAny = navigator.storage as any;
  if (!storageAny?.getDirectory) {
    throw new Error('Browser tidak mendukung penyimpanan model lokal (OPFS)');
  }
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
