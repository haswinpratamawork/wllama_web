interface Wllama {
  loadModelFromUrl: (url: string, config: any) => Promise<void>;
  loadModel: (blobs: File[], config: any) => Promise<void>;
  createCompletion: (prompt: string, options: any) => Promise<string>;
}

class ChatModelManagerClass {
  private model: Wllama | null = null;

  setModel(model: Wllama | null) {
    this.model = model;
    if (model) {
      localStorage.setItem('wllama-chat-model-loaded', 'true');
      window.dispatchEvent(new Event('chat-model-changed'));
    } else {
      localStorage.removeItem('wllama-chat-model-loaded');
      window.dispatchEvent(new Event('chat-model-changed'));
    }
  }

  getModel(): Wllama | null {
    return this.model;
  }

  isLoaded(): boolean {
    return this.model !== null;
  }

  clear() {
    this.model = null;
    localStorage.removeItem('wllama-chat-model-loaded');
    window.dispatchEvent(new Event('chat-model-changed'));
  }
}

export const ChatModelManager = new ChatModelManagerClass();