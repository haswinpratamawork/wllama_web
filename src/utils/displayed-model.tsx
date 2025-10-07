import { Model } from '@wllama/wllama';
import { ModelState } from './types';
import { WllamaStorage } from './utils';
import { LIST_MODELS } from '../config';

export type DisplayedModelSource = 'huggingface' | 'local';

export interface DisplayedModelOptions {
  name?: string;
  source?: DisplayedModelSource;
  addedAt?: number;
}

export class DisplayedModel {
  url: string;
  size: number;
  isUserAdded: boolean;
  cachedModel?: Model;
  name?: string;
  source: DisplayedModelSource;
  addedAt?: number;

  state: ModelState = ModelState.NOT_DOWNLOADED;
  downloadPercent: number = -1; // from 0.0 to 1.0; -1 means not downloading

  constructor(
    url: string,
    size: number,
    isUserAdded: boolean,
    cachedModel?: Model,
    options: DisplayedModelOptions = {}
  ) {
    this.url = url;
    this.size = size;
    this.isUserAdded = isUserAdded;
    this.state = !!cachedModel ? ModelState.READY : ModelState.NOT_DOWNLOADED;
    this.cachedModel = cachedModel;
    this.name = options.name;
    this.source = options.source ?? (this.isLocal ? 'local' : 'huggingface');
    this.addedAt = options.addedAt;
  }

  get hfModel() {
    if (!this.url.startsWith('http')) {
      return this.displayName || 'Local model';
    }
    const parts = this.url
      .replace(/https:\/\/(huggingface.co|hf.co)\/+/, '')
      .split('/');
    if (parts.length < 2) {
      return this.url;
    }
    return `${parts[0]}/${parts[1]}`;
  }

  get hfPath() {
    if (!this.url.startsWith('http')) {
      return this.displayName || this.url;
    }
    const parts = this.url
      .replace(/https:\/\/(huggingface.co|hf.co)\/+/, '')
      .split('/');
    return parts.slice(4).join('/');
  }

  get isLocal(): boolean {
    return this.source === 'local' || this.url.startsWith('local://');
  }

  get displayName(): string {
    if (this.name) {
      return this.name;
    }
    if (this.isLocal) {
      const segments = this.url.split('/');
      return segments[segments.length - 1] || 'Local model';
    }
    const path = this.hfPath;
    return path ? path.replace(/-\d{5}-of-\d{5}/, '-(shards)') : this.url;
  }

  get displaySource(): string {
    if (this.isLocal) {
      return 'Source: Local import';
    }
    return `HF repo: ${this.hfModel}`;
  }

  clone(overwrite: Partial<DisplayedModel>): DisplayedModel {
    const obj = new DisplayedModel(
      this.url,
      this.size,
      this.isUserAdded,
      this.cachedModel,
      {
        name: this.name,
        source: this.source,
        addedAt: this.addedAt,
      }
    );
    obj.state = overwrite.state ?? this.state;
    obj.downloadPercent = overwrite.downloadPercent ?? this.downloadPercent;
    if (overwrite.name !== undefined) {
      obj.name = overwrite.name;
    }
    if (overwrite.source !== undefined) {
      obj.source = overwrite.source as DisplayedModelSource;
    }
    if (overwrite.addedAt !== undefined) {
      obj.addedAt = overwrite.addedAt;
    }
    if (overwrite.cachedModel !== undefined) {
      obj.cachedModel = overwrite.cachedModel;
      obj.state = overwrite.cachedModel ? ModelState.READY : obj.state;
    }
    return obj;
  }
}

interface UserAddedModel {
  url: string;
  size: number;
  name?: string;
  source?: DisplayedModelSource;
  addedAt?: number;
}

export function getUserAddedModels(cachedModels: Model[]): DisplayedModel[] {
  const userAddedModels: UserAddedModel[] = WllamaStorage.load(
    'custom_models',
    []
  );
  return userAddedModels.map((m: any) => {
    const cachedModel = cachedModels.find((cm) => cm.url === m.url);
    return new DisplayedModel(m.url, m.size, true, cachedModel, {
      name: m.name,
      source: m.source,
      addedAt: m.addedAt,
    });
  });
}

export function updateUserAddedModels(models: DisplayedModel[]) {
  const userAddedModels: UserAddedModel[] = models
    .filter((m) => m.isUserAdded)
    .map((m) => ({
      url: m.url,
      size: m.size,
      name: m.name,
      source: m.source,
      addedAt: m.addedAt,
    }));
  WllamaStorage.save('custom_models', userAddedModels);
}

export function getPresetModels(cachedModels: Model[]): DisplayedModel[] {
  return LIST_MODELS.map((m) => {
    const cachedModel = cachedModels.find((cm) => cm.url === m.url);
    return new DisplayedModel(m.url, m.size, false, cachedModel, {
      source: 'huggingface',
    });
  });
}

export function getDisplayedModels(cachedModels: Model[]): DisplayedModel[] {
  return [
    ...getUserAddedModels(cachedModels),
    ...getPresetModels(cachedModels),
  ];
}
