import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Sparkles, BookOpen, AlertCircle, FileText, Check, Copy } from 'lucide-react';

interface Chunk {
  chunk_id: string;
  source_file: string;
  page_number: number;
  text: string;
  score: number;
  corpus_id: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  latency?: number;
  decompTriggered?: boolean;
  unanswered?: string[];
  chunks?: Chunk[];
  error?: boolean;
}

interface ChatViewProps {
  apiBaseUrl: string;
  corpusId: string;
  groqApiKey: string | null;
}

export const ChatView: React.FC<ChatViewProps> = ({ apiBaseUrl, corpusId, groqApiKey }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: "Hello! I have finished indexing your corpus. Ask me any complex, multi-hop questions and I will reason across your documents to answer with citations.",
    }
  ]);
  const [input, setInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [loadingStep, setLoadingStep] = useState<string>('');
  const [selectedChunk, setSelectedChunk] = useState<Chunk | null>(null);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Loading animation message switcher
  useEffect(() => {
    if (!isLoading) return;

    const steps = [
      'Decomposing query into sub-questions...',
      'Running dense & sparse retrieval on Qdrant...',
      'Reranking passages via Cross-Encoder...',
      'Synthesizing structured answer using Groq...',
    ];

    let currentStep = 0;
    setLoadingStep(steps[0]);

    const interval = setInterval(() => {
      currentStep = (currentStep + 1) % steps.length;
      setLoadingStep(steps[currentStep]);
    }, 2500);

    return () => clearInterval(interval);
  }, [isLoading]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userQuery }]);
    setIsLoading(true);

    try {
      const response = await fetch(`${apiBaseUrl}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userQuery,
          corpus_id: corpusId,
          top_k: 5,
          groq_api_key: groqApiKey
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: 'Query failed' }));
        throw new Error(errData.detail || 'Internal server error occurred.');
      }

      const data = await response.json();
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer_text,
          latency: data.latency_seconds,
          decompTriggered: data.decomposition_triggered,
          unanswered: data.unanswered_aspects,
          chunks: data.retrieved_chunks,
        }
      ]);
    } catch (err: any) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `Sorry, I encountered an error: ${err.message || 'Network error occurred.'}`,
          error: true
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const parseAnswerText = (text: string, chunks?: Chunk[]) => {
    const parts = [];
    const regex = /\[(\d+)\]/g;
    let lastIndex = 0;
    let match;

    while ((match = regex.exec(text)) !== null) {
      const matchIndex = match.index;
      const citationNumber = parseInt(match[1], 10);

      if (matchIndex > lastIndex) {
        parts.push({ type: 'text', content: text.substring(lastIndex, matchIndex) });
      }

      parts.push({
        type: 'citation',
        content: match[0],
        number: citationNumber,
        chunk: chunks && chunks[citationNumber - 1] ? chunks[citationNumber - 1] : null,
      });

      lastIndex = regex.lastIndex;
    }

    if (lastIndex < text.length) {
      parts.push({ type: 'text', content: text.substring(lastIndex) });
    }

    return parts;
  };

  return (
    <div className="flex-1 flex flex-col md:flex-row h-[calc(100vh-8rem)] w-full gap-6 text-slate-100 max-w-6xl mx-auto">
      {/* Left panel: Chat Interface */}
      <div className="flex-1 flex flex-col bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">
        {/* Header */}
        <div className="bg-slate-950/60 px-6 py-4 border-b border-slate-800 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-violet-400" />
            <div>
              <span className="font-bold text-sm">Interactive RAG Session</span>
              <span className="text-[10px] bg-violet-900/30 text-violet-300 border border-violet-800/40 px-2 py-0.5 rounded-full ml-2">Active</span>
            </div>
          </div>
        </div>

        {/* Message area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
          {messages.map((msg, index) => (
            <div key={index} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
              <div
                className={`max-w-[85%] rounded-2xl px-5 py-3.5 text-sm ${
                  msg.role === 'user'
                    ? 'bg-violet-600 text-white rounded-br-none shadow-lg shadow-violet-600/10'
                    : msg.error
                    ? 'bg-red-950/40 border border-red-800/60 text-red-200 rounded-bl-none'
                    : 'bg-slate-950/50 border border-slate-800 text-slate-100 rounded-bl-none'
                }`}
              >
                {msg.role === 'assistant' && !msg.error ? (
                  <div className="space-y-3">
                    {/* Parsed Citations Answer */}
                    <p className="leading-relaxed whitespace-pre-line">
                      {parseAnswerText(msg.content, msg.chunks).map((part, pIdx) => {
                        if (part.type === 'citation') {
                          return (
                            <button
                              key={pIdx}
                              onClick={() => part.chunk && setSelectedChunk(part.chunk)}
                              disabled={!part.chunk}
                              className={`inline-flex items-center justify-center font-bold px-1.5 py-0.5 mx-0.5 text-xs rounded border transition-all ${
                                part.chunk
                                  ? 'bg-violet-950/60 border-violet-700/60 hover:bg-violet-800 text-violet-300 hover:scale-105 cursor-pointer'
                                  : 'bg-slate-800 border-slate-700 text-slate-500 cursor-not-allowed'
                              }`}
                              title={part.chunk ? `Click to inspect: ${part.chunk.source_file}` : 'No source chunk details'}
                            >
                              {part.content}
                            </button>
                          );
                        }
                        return <span key={pIdx}>{part.content}</span>;
                      })}
                    </p>

                    {/* Metadata Footer (Latency / Decomposition) */}
                    {(msg.latency !== undefined || msg.decompTriggered !== undefined) && (
                      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[10px] text-slate-500 font-mono border-t border-slate-800/50 pt-2">
                        {msg.latency !== undefined && (
                          <span>Latency: {msg.latency.toFixed(3)}s</span>
                        )}
                        {msg.decompTriggered !== undefined && (
                          <span className={msg.decompTriggered ? 'text-fuchsia-400 font-bold' : 'text-slate-500'}>
                            Multi-hop decomposition: {msg.decompTriggered ? 'Triggered' : 'Not required'}
                          </span>
                        )}
                      </div>
                    )}

                    {/* Unanswered aspects */}
                    {msg.unanswered && msg.unanswered.length > 0 && (
                      <div className="bg-amber-950/20 border border-amber-800/40 p-3 rounded-lg text-xs text-amber-300 space-y-1 flex gap-2">
                        <AlertCircle className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />
                        <div>
                          <span className="font-semibold block">Aspects not covered by your documents:</span>
                          <ul className="list-disc list-inside space-y-0.5 mt-1 font-mono text-[10px] text-amber-200">
                            {msg.unanswered.map((asp, aIdx) => (
                              <li key={aIdx}>{asp}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="leading-relaxed whitespace-pre-line">{msg.content}</p>
                )}
              </div>

              {/* Message tools */}
              {msg.role === 'assistant' && !msg.error && (
                <div className="flex gap-2.5 text-xs text-slate-500 mt-1.5 ml-2 font-mono">
                  <button 
                    onClick={() => handleCopy(msg.content, index)}
                    className="hover:text-slate-300 flex items-center gap-1 transition-all"
                  >
                    {copiedIndex === index ? (
                      <>
                        <Check className="h-3 w-3 text-emerald-400" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="h-3 w-3" />
                        Copy
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          ))}

          {/* Typing Loading Indicator */}
          {isLoading && (
            <div className="flex flex-col items-start">
              <div className="bg-slate-950/50 border border-slate-800 rounded-2xl rounded-bl-none px-5 py-4 text-sm flex items-center gap-3">
                <Loader2 className="h-4 w-4 text-violet-400 animate-spin" />
                <span className="text-slate-400 font-mono text-xs animate-pulse">{loadingStep}</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <form onSubmit={handleSend} className="bg-slate-950/60 p-4 border-t border-slate-800 flex gap-3">
          <input
            type="text"
            placeholder="Ask a question about your documents..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
            className="flex-1 px-4 py-3 bg-slate-900 border border-slate-700 focus:outline-none focus:border-violet-500 rounded-xl text-sm placeholder-slate-500 text-slate-100 transition-all disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="px-5 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 disabled:opacity-40 disabled:pointer-events-none rounded-xl flex items-center justify-center transition-all shadow-md active:scale-95"
          >
            <Send className="h-4 w-4" />
          </button>
        </form>
      </div>

      {/* Right panel: Active Citation Inspector */}
      <div className="w-full md:w-80 flex flex-col bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">
        <div className="bg-slate-950/60 px-5 py-4 border-b border-slate-800 flex items-center gap-2">
          <BookOpen className="h-5 w-5 text-fuchsia-400" />
          <h3 className="font-bold text-sm">Source Passage Inspector</h3>
        </div>

        <div className="flex-1 p-5 overflow-y-auto custom-scrollbar space-y-4">
          {selectedChunk ? (
            <div className="space-y-4 animate-fade-in">
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-xs font-semibold text-slate-400">
                  <FileText className="h-4 w-4 text-violet-400" />
                  <span>Document Details</span>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800 space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-500">File Name:</span>
                    <span className="font-mono text-slate-200 truncate max-w-[150px]" title={selectedChunk.source_file}>
                      {selectedChunk.source_file}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Page Number:</span>
                    <span className="font-mono text-slate-200">Page {selectedChunk.page_number}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Relevance Score:</span>
                    <span className="font-mono text-emerald-400 font-semibold">
                      {selectedChunk.score ? selectedChunk.score.toFixed(4) : 'N/A'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <span className="text-xs font-semibold text-slate-400 block">Cited Passage Text</span>
                <div className="bg-slate-950/40 p-4 rounded-xl border border-slate-800 text-xs leading-relaxed text-slate-300 font-sans max-h-96 overflow-y-auto custom-scrollbar whitespace-pre-line select-text">
                  {selectedChunk.text}
                </div>
              </div>

              <button
                onClick={() => setSelectedChunk(null)}
                className="w-full py-2 bg-slate-800 hover:bg-slate-700 transition-all rounded-lg text-xs font-semibold"
              >
                Clear Selection
              </button>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center text-slate-500 space-y-2 py-20">
              <BookOpen className="h-8 w-8 text-slate-700 animate-bounce" />
              <p className="text-xs">No active citation selected.</p>
              <p className="text-[10px] text-slate-600 max-w-[180px] leading-relaxed">
                Click on inline citation markers (e.g. [1]) inside the answer text to view source passages.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
