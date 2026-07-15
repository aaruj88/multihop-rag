import { useState, useEffect } from 'react';
import { UploadView } from './components/UploadView';
import { ChatView } from './components/ChatView';
import { SettingsPanel } from './components/SettingsPanel';
import { Settings, Sparkles, LogOut } from 'lucide-react';
import './App.css';

function App() {
  const [view, setView] = useState<'upload' | 'chat'>('upload');
  const [corpusId, setCorpusId] = useState<string | null>(null);
  const [groqApiKey, setGroqApiKey] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);

  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  // Load key from localStorage on mount
  useEffect(() => {
    const savedKey = localStorage.getItem('groq_api_key');
    if (savedKey) {
      setGroqApiKey(savedKey);
    }
  }, []);

  const handleSaveKey = (key: string | null) => {
    setGroqApiKey(key);
    if (key) {
      localStorage.setItem('groq_api_key', key);
    } else {
      localStorage.removeItem('groq_api_key');
    }
  };

  const handleUploadSuccess = (id: string) => {
    setCorpusId(id);
    setView('chat');
  };

  const handleResetCorpus = () => {
    setCorpusId(null);
    setView('upload');
  };

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col font-sans selection:bg-violet-600/30 selection:text-violet-200">
      {/* Navbar */}
      <header className="border-b border-slate-900 bg-slate-950/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl bg-gradient-to-tr from-violet-600 to-fuchsia-600 flex items-center justify-center shadow-lg shadow-violet-600/20">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div>
              <span className="font-extrabold text-sm tracking-tight text-white block">Multi-Hop RAG</span>
              <span className="text-[10px] text-slate-400 block -mt-1 font-mono">Academic Reasoning Engine</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {view === 'chat' && (
              <button
                onClick={handleResetCorpus}
                className="text-xs bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 font-semibold px-3.5 py-2 rounded-lg flex items-center gap-1.5 transition-all"
                title="Discard current corpus and upload new files"
              >
                <LogOut className="h-3.5 w-3.5" />
                <span>New Ingestion</span>
              </button>
            )}
            <button
              onClick={() => setIsSettingsOpen(true)}
              className="p-2 bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white rounded-lg transition-all"
              title="Open Settings & API Keys"
            >
              <Settings className="h-5 w-5" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex items-center justify-center p-6">
        {view === 'upload' ? (
          <UploadView
            apiBaseUrl={apiBaseUrl}
            onUploadSuccess={handleUploadSuccess}
            groqApiKey={groqApiKey}
          />
        ) : (
          corpusId && (
            <ChatView
              apiBaseUrl={apiBaseUrl}
              corpusId={corpusId}
              groqApiKey={groqApiKey}
            />
          )
        )}
      </main>

      {/* Settings Side Drawer */}
      <SettingsPanel
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        groqApiKey={groqApiKey}
        onSaveKey={handleSaveKey}
        onResetCorpus={handleResetCorpus}
        corpusId={corpusId}
      />

      {/* Footer */}
      <footer className="border-t border-slate-900/50 bg-slate-950/40 py-4 text-center text-xs text-slate-500 font-mono">
        Made with ❤️ by Deepmind Advanced Agentic Coding Team
      </footer>
    </div>
  );
}

export default App;
