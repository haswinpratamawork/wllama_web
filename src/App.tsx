import { MessagesProvider } from './utils/messages.context';
import { WllamaProvider } from './utils/wllama.context';
import { RagProvider } from './utils/rag.context';
import './utils/benchmark';
import { Chat } from './pages/Chat/Chat';
import EmbeddingPage from './pages/Embedding/Embedding';
import { Routes, Route } from 'react-router-dom';

function App() {
  return (
    <MessagesProvider>
      <WllamaProvider>
        <RagProvider>
          <Routes>
            <Route path="/" element={<Chat />} />
            <Route path="/embedding" element={<EmbeddingPage />} />
          </Routes>
        </RagProvider>
      </WllamaProvider>
    </MessagesProvider>
  );
}

export default App;
