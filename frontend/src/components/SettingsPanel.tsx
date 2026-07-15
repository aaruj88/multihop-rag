import React, { useState } from 'react';
import { Settings, X, Key, Trash2, HelpCircle } from 'lucide-react';

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
  groqApiKey: string | null;
  onSaveKey: (key: string | null) => void;
  onResetCorpus: () => void;
  corpusId: string | null;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  isOpen,
  onClose,
  groqApiKey,
  onSaveKey,
  onResetCorpus,
  corpusId,
}) => {
  const [tempKey, setTempKey] = useState<string>(groqApiKey || '');

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    const cleanKey = tempKey.trim();
    onSaveKey(cleanKey || null);
  };

  const handleClearKey = () => {
    setTempKey('');
    onSaveKey(null);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm transition-opacity" 
        onClick={onClose} 
      />

      {/* Drawer */}
      <div className="relative w-full max-w-sm h-full bg-slate-900 border-l border-slate-800 shadow-2xl p-6 flex flex-col justify-between text-slate-100 z-10 animate-slide-in">
        <div>
          <div className="flex justify-between items-center pb-4 border-b border-slate-800 mb-6">
            <h3 className="text-lg font-bold flex items-center gap-2">
              <Settings className="h-5 w-5 text-violet-400" />
              Settings & Key Management
            </h3>
            <button 
              onClick={onClose} 
              className="text-slate-400 hover:text-slate-100 transition-colors p-1 hover:bg-slate-800 rounded-md"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          <div className="space-y-6">
            {/* BYOK Section */}
            <form onSubmit={handleSave} className="space-y-3">
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <Key className="h-3.5 w-3.5 text-violet-400" />
                Custom Groq API Key (BYOK)
              </label>
              <div className="flex gap-2">
                <input
                  type="password"
                  placeholder="Paste gsk_... key"
                  value={tempKey}
                  onChange={(e) => setTempKey(e.target.value)}
                  className="flex-1 px-3 py-2 bg-slate-950/80 border border-slate-700 rounded-lg text-sm font-mono text-slate-100 placeholder-slate-600 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition-all"
                />
                <button
                  type="submit"
                  className="px-4 py-2 bg-violet-600 hover:bg-violet-500 rounded-lg text-xs font-bold transition-all"
                >
                  Save
                </button>
              </div>
              <div className="text-[10px] leading-relaxed text-slate-400 flex items-start gap-1 bg-slate-950/30 p-2.5 rounded-lg border border-slate-800/40">
                <HelpCircle className="h-4 w-4 shrink-0 text-fuchsia-400" />
                <span>
                  <strong>Privacy Note:</strong> This key is sent directly to the LLM backend for that request's Claude/Groq calls and is discarded immediately after execution. It is never logged or stored permanently on the server.
                </span>
              </div>
              {groqApiKey && (
                <div className="flex justify-between items-center text-xs bg-slate-950/40 p-2 border border-slate-800 rounded-lg">
                  <span className="font-mono text-emerald-400">Key currently active</span>
                  <button
                    type="button"
                    onClick={handleClearKey}
                    className="text-red-400 hover:text-red-300 font-semibold"
                  >
                    Remove
                  </button>
                </div>
              )}
            </form>

            {/* Corpus Management Section */}
            {corpusId && (
              <div className="space-y-3 pt-6 border-t border-slate-800">
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Active Corpus
                </label>
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800 font-mono text-[10px] break-all text-slate-300">
                  <span className="text-slate-500 block mb-1">ID:</span>
                  {corpusId}
                </div>
                <button
                  onClick={() => {
                    onResetCorpus();
                    onClose();
                  }}
                  className="w-full py-2.5 bg-red-950/30 hover:bg-red-950/50 border border-red-900/50 hover:border-red-800 text-red-300 rounded-lg text-xs font-bold flex items-center justify-center gap-2 transition-all active:scale-98"
                >
                  <Trash2 className="h-4 w-4" />
                  Discard & Start New Corpus
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="text-[10px] text-center text-slate-500 border-t border-slate-800 pt-4">
          Multi-Hop RAG API Client v1.1.0
        </div>
      </div>
    </div>
  );
};
