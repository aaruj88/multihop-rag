import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, AlertCircle, FileText, Loader2, Trash2 } from 'lucide-react';

interface UploadViewProps {
  apiBaseUrl: string;
  onUploadSuccess: (corpusId: string) => void;
  groqApiKey: string | null;
}

type IngestionStatus = 'idle' | 'uploading' | 'processing' | 'failed';

export const UploadView: React.FC<UploadViewProps> = ({ apiBaseUrl, onUploadSuccess, groqApiKey }) => {
  const [files, setFiles] = useState<File[]>([]);
  const [status, setStatus] = useState<IngestionStatus>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [progressMsg, setProgressMsg] = useState<string>('');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setErrorMsg(null);
    
    // Check combined count
    if (files.length + acceptedFiles.length > 10) {
      setErrorMsg('Maximum of 10 files allowed.');
      return;
    }

    // Check size limit: 15MB each
    const maxSize = 15 * 1024 * 1024;
    const oversized = acceptedFiles.some(f => f.size > maxSize);
    if (oversized) {
      setErrorMsg('Some files exceed the maximum size of 15MB.');
      return;
    }

    setFiles(prev => [...prev, ...acceptedFiles]);
  }, [files]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    maxFiles: 10,
  });

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const pollStatus = async (id: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`${apiBaseUrl}/corpus/${id}/status`);
        if (!res.ok) {
          throw new Error('Failed to fetch status from the server.');
        }
        const data = await res.json();
        
        if (data.status === 'ready') {
          clearInterval(pollInterval);
          setProgressMsg(`Success! Ingested ${data.file_count} files into ${data.chunk_count} chunks.`);
          setTimeout(() => {
            onUploadSuccess(id);
          }, 1000);
        } else if (data.status === 'failed') {
          clearInterval(pollInterval);
          setStatus('failed');
          setErrorMsg('The backend failed to process the uploaded documents. Please try again with valid PDFs.');
        } else {
          setProgressMsg(`Processing documents... (Files: ${data.file_count || 0}, Chunks: ${data.chunk_count || 0})`);
        }
      } catch (err: any) {
        clearInterval(pollInterval);
        setStatus('failed');
        setErrorMsg(err.message || 'Error occurred while checking ingestion progress.');
      }
    }, 2000);
  };

  const handleUploadSubmit = async () => {
    if (files.length === 0) return;
    setStatus('uploading');
    setErrorMsg(null);
    setProgressMsg('Uploading documents...');

    const formData = new FormData();
    files.forEach(f => {
      formData.append('files', f);
    });

    if (groqApiKey) {
      formData.append('groq_api_key', groqApiKey);
    }

    try {
      const res = await fetch(`${apiBaseUrl}/corpus`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(errData.detail || `Upload failed with status code ${res.status}`);
      }

      const data = await res.json();
      setStatus('processing');
      setProgressMsg('Upload complete! Ingesting and vectorizing chunks...');
      pollStatus(data.corpus_id);
    } catch (err: any) {
      setStatus('failed');
      setErrorMsg(err.message || 'Network error occurred during upload.');
    }
  };

  const resetUpload = () => {
    setFiles([]);
    setStatus('idle');
    setErrorMsg(null);
    setProgressMsg('');
  };

  return (
    <div className="w-full max-w-2xl mx-auto bg-slate-900 border border-slate-800 rounded-2xl p-8 shadow-2xl text-slate-100">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-violet-400 to-fuchsia-400 bg-clip-text text-transparent">
          Ingest Your Corpus
        </h2>
        <p className="text-sm text-slate-400 mt-2">
          Upload up to 10 academic PDFs (Max 15MB each) to index into the multi-hop RAG pipeline.
        </p>
      </div>

      {status === 'idle' && (
        <div className="space-y-6">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300 ${
              isDragActive
                ? 'border-violet-500 bg-violet-950/20 scale-102'
                : 'border-slate-700 hover:border-slate-600 bg-slate-950/40 hover:bg-slate-950/60'
            }`}
          >
            <input {...getInputProps()} />
            <UploadCloud className="mx-auto h-12 w-12 text-slate-400 mb-4 animate-pulse" />
            <p className="text-base font-semibold">Drag & drop your PDF documents here</p>
            <p className="text-xs text-slate-500 mt-1">or click to browse from files</p>
          </div>

          {files.length > 0 && (
            <div className="bg-slate-950/60 rounded-xl p-4 border border-slate-800">
              <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                <FileText className="h-4 w-4 text-violet-400" />
                Selected Files ({files.length}/10)
              </h3>
              <ul className="space-y-2 max-h-48 overflow-y-auto pr-2 custom-scrollbar">
                {files.map((file, idx) => (
                  <li
                    key={idx}
                    className="flex justify-between items-center text-xs bg-slate-900/80 p-2.5 rounded-lg border border-slate-800 group"
                  >
                    <span className="truncate max-w-[80%] font-mono text-slate-300">{file.name}</span>
                    <div className="flex items-center gap-3">
                      <span className="text-[10px] text-slate-500">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                      <button
                        onClick={() => removeFile(idx)}
                        className="text-slate-500 hover:text-red-400 transition-colors"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {errorMsg && (
            <div className="flex items-center gap-3 p-4 bg-red-950/30 border border-red-800/50 rounded-xl text-red-300 text-xs">
              <AlertCircle className="h-5 w-5 shrink-0 text-red-400" />
              <span>{errorMsg}</span>
            </div>
          )}

          <button
            onClick={handleUploadSubmit}
            disabled={files.length === 0}
            className="w-full py-3.5 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 disabled:opacity-40 disabled:pointer-events-none rounded-xl text-sm font-bold tracking-wide transition-all shadow-lg hover:shadow-violet-600/20 active:scale-98"
          >
            Process & Vectorize Documents
          </button>
        </div>
      )}

      {status !== 'idle' && (
        <div className="flex flex-col items-center justify-center py-10 space-y-6 text-center">
          {status === 'uploading' && (
            <>
              <Loader2 className="h-16 w-16 text-violet-500 animate-spin" />
              <div className="space-y-2">
                <p className="text-lg font-bold text-slate-100">{progressMsg}</p>
                <p className="text-xs text-slate-500">Uploading file payload to API endpoints...</p>
              </div>
            </>
          )}

          {status === 'processing' && (
            <>
              <div className="relative">
                <Loader2 className="h-16 w-16 text-fuchsia-500 animate-spin" />
                <FileText className="absolute inset-0 m-auto h-6 w-6 text-slate-300" />
              </div>
              <div className="space-y-2">
                <p className="text-lg font-bold text-slate-100">{progressMsg}</p>
                <p className="text-xs text-slate-500">FastAPI parses pages, generates sentence chunks, and indexes embeddings into Qdrant collection...</p>
              </div>
            </>
          )}

          {status === 'failed' && (
            <>
              <AlertCircle className="h-16 w-16 text-red-500" />
              <div className="space-y-2 max-w-md">
                <p className="text-lg font-bold text-red-400">Ingestion Failed</p>
                <p className="text-xs text-slate-400">{errorMsg}</p>
              </div>
              <button
                onClick={resetUpload}
                className="px-6 py-2.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-xs font-semibold tracking-wide transition-all active:scale-95"
              >
                Retry Upload
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
};
