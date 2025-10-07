import { useState } from 'react';
import { useMessages } from '../utils/messages.context';
import { useWllama } from '../utils/wllama.context';
import { Message, Screen } from '../utils/types';
import { formatChat } from '../utils/utils';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faStop } from '@fortawesome/free-solid-svg-icons';
import { nl2br } from '../utils/nl2br';
import ScreenWrapper from './ScreenWrapper';
import { useIntervalWhen } from '../utils/use-interval-when';
import { useRag, type RagSearchHit } from '../utils/rag.context';

export default function ChatScreen() {
  const [input, setInput] = useState('');
  const [ragContext, setRagContext] = useState<RagSearchHit[]>([]);
  const [ragNotice, setRagNotice] = useState<string | null>(null);
  const {
    currentConvId,
    isGenerating,
    createCompletion,
    navigateTo,
    loadedModel,
    getWllamaInstance,
    stopCompletion,
  } = useWllama();
  const {
    getConversationById,
    addMessageToConversation,
    editMessageInConversation,
    newConversation,
  } = useMessages();
  const { ragEnabled, toggleRag, searchByText } = useRag();

  useIntervalWhen(chatScrollToBottom, 500, isGenerating, true);

  const currConv = getConversationById(currentConvId);

  const onSubmit = async () => {
    if (isGenerating) return;

    // copy input and create messages
    const currHistory = currConv?.messages ?? [];
    const userInput = input;
    setInput('');
    setRagContext([]);
    setRagNotice(null);
    const baseId = Date.now();
    const userMsg: Message = {
      id: baseId,
      content: userInput,
      role: 'user',
    };
    const assistantMsg: Message = {
      id: baseId + 1,
      content: '',
      role: 'assistant',
    };

    // process conversation
    let convId = currConv?.id;
    if (!convId) {
      // need to create new conversation
      const newConv = newConversation(userMsg);
      convId = newConv.id;
      navigateTo(Screen.CHAT, convId);
      addMessageToConversation(convId, assistantMsg);
    } else {
      // append to current conversation
      addMessageToConversation(convId, userMsg);
      addMessageToConversation(convId, assistantMsg);
    }

    // generate response
    if (!loadedModel) {
      throw new Error('loadedModel is null');
    }
    let formattedChat: string;
    let messagesForModel: Message[] = [...currHistory, userMsg];

    try {
      if (ragEnabled) {
        try {
          const hits = await searchByText(userInput, { topK: 5 });
          if (hits.length) {
            setRagContext(hits);
            setRagNotice(
              `RAG aktif: menambahkan ${hits.length} knowledge ke prompt`
            );
            const knowledgeBlocks = hits
              .map((hit, index) => {
                const urlLine = hit.item.url ? `Sumber: ${hit.item.url}\n` : '';
                return `### Knowledge ${index + 1}\nJudul: ${hit.item.title
                  }\n${urlLine}Konten:\n${hit.item.content}`;
              })
              .join('\n\n');
            const systemMsg: Message = {
              id: baseId - 1,
              role: 'system',
              content: `Gunakan knowledge berikut sebagai konteks saat menjawab. Jika tidak relevan, abaikan dengan sopan.\n\n${knowledgeBlocks}`,
            };
            messagesForModel = [systemMsg, ...currHistory, userMsg];
          } else {
            setRagNotice('RAG aktif tetapi tidak ada knowledge yang relevan.');
          }
        } catch (err: any) {
          console.error(err);
          setRagNotice(
            'RAG dinonaktifkan untuk prompt ini: ' +
            (err?.message ?? String(err))
          );
          messagesForModel = [...currHistory, userMsg];
        }
      }
      formattedChat = await formatChat(getWllamaInstance(), messagesForModel);
      console.log("formattedChat", formattedChat)
    } catch (e) {
      alert(`Error while formatting chat: ${(e as any)?.message ?? 'unknown'}`);
      throw e;
    }
    await createCompletion(formattedChat, (newContent) => {
      // console.log("masukk", convId, assistantMsg.id, newContent)
      editMessageInConversation(convId, assistantMsg.id, newContent);
    });

  };

  return (
    <ScreenWrapper fitScreen>
      <div className="chat-messages grow overflow-auto" id="chat-history">
        <div className="h-10" />

        {currConv ? (
          <>
            {currConv.messages.map((msg) =>
              msg.role === 'user' ? (
                <div className="chat chat-end" key={msg.id}>
                  <div className="chat-bubble">{nl2br(msg.content)}</div>
                </div>
              ) : (
                <div className="chat chat-start" key={msg.id}>
                  <div className="chat-bubble bg-base-100 text-base-content">
                    {msg.content.length === 0 && isGenerating && (
                      <span className="loading loading-dots"></span>
                    )}
                    {nl2br(msg.content)}
                  </div>
                </div>
              )
            )}
          </>
        ) : (
          <div className="pt-24 text-center text-xl">Ask me something 👋</div>
        )}
      </div>
      <div className="flex flex-col input-message py-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2 text-sm text-base-content/70 mb-2">
          <label className="inline-flex items-center gap-2">
            <input
              type="checkbox"
              className="toggle toggle-xs"
              checked={ragEnabled}
              onChange={(e) => toggleRag(e.target.checked)}
            />
            <span>Gunakan knowledge base (RAG)</span>
          </label>
          {ragNotice && <span className="text-xs md:text-sm">{ragNotice}</span>}
        </div>
        {ragContext.length > 0 && (
          <div className="mb-3 bg-base-100/70 border border-base-200 rounded-lg p-3 text-xs space-y-2 max-h-40 overflow-auto">
            <div className="font-semibold text-base-content">
              Konteks yang digunakan:
            </div>
            {ragContext.slice(0, 3).map((hit) => (
              <div key={hit.item.id} className="space-y-1">
                <div className="font-medium text-base-content/90">
                  {hit.item.title}{' '}
                  <span className="text-[0.7rem] opacity-70">
                    (score {hit.score.toFixed(3)})
                  </span>
                </div>
                <div className="text-[0.7rem] text-base-content/70 line-clamp-3">
                  {hit.item.content}
                </div>
                {hit.item.url && (
                  <div className="text-[0.7rem] text-info">
                    {hit.item.url}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        {isGenerating && (
          <div className="text-center">
            <button
              className="btn btn-outline btn-sm mb-4"
              onClick={stopCompletion}
            >
              <FontAwesomeIcon icon={faStop} />
              Stop generation
            </button>
          </div>
        )}

        {loadedModel && (
          <textarea
            className="textarea textarea-bordered w-full"
            placeholder="Your message..."
            disabled={isGenerating}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.keyCode == 13 && e.shiftKey == false) {
                e.preventDefault();
                onSubmit();
              }
            }}
          />
        )}

        {!loadedModel && <WarnNoModel />}

        <small className="text-center mx-auto opacity-70 pt-2">
          wllama may generate inaccurate information. Use with your own risk.
        </small>
      </div>
    </ScreenWrapper>
  );
}

function WarnNoModel() {
  const { navigateTo } = useWllama();

  return (
    <div role="alert" className="alert">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        className="h-6 w-6 shrink-0 stroke-current"
        fill="none"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
        />
      </svg>
      <span>Model is not loaded</span>
      <div>
        <button
          className="btn btn-sm btn-primary"
          onClick={() => navigateTo(Screen.MODEL)}
        >
          Select model
        </button>
      </div>
    </div>
  );
}

const chatScrollToBottom = () => {
  const elem = document.getElementById('chat-history');
  elem?.scrollTo({
    top: elem.scrollHeight,
    behavior: 'smooth',
  });
};
