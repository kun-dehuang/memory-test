import React, { useState, useEffect } from 'react';
import ResultCard from './components/ResultCard';
import DetailModal from './components/DetailModal';

function App() {
  // API Configuration
  const [apiBaseUrl, setApiBaseUrl] = useState('');

  // Load API URL from localStorage or env
  useEffect(() => {
    const saved = localStorage.getItem('apiBaseUrl');
    const defaultUrl = import.meta.env.VITE_API_BASE_URL || '';
    setApiBaseUrl(saved || defaultUrl);
  }, []);

  const handleApiUrlChange = (value) => {
    setApiBaseUrl(value);
    localStorage.setItem('apiBaseUrl', value);
  };

  // State for init operation
  const [initializing, setInitializing] = useState(false);
  const [initResult, setInitResult] = useState(null);

  // State for sync operation
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState(null);

  // State for reset operation
  const [resetting, setResetting] = useState(false);

  // State for search
  const [query, setQuery] = useState('');
  const [provider, setProvider] = useState('zep');
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState([]);

  // State for detail modal
  const [selectedResult, setSelectedResult] = useState(null);
  const [showModal, setShowModal] = useState(false);

  // Normalize API URL (remove trailing slash)
  const normalizedApiUrl = apiBaseUrl.replace(/\/+$/, '');

  // API helper
  const api = {
    init: async () => {
      const response = await fetch(`${normalizedApiUrl}/api/init`, {
        method: 'POST',
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'Init failed');
      }
      return response.json();
    },
    sync: async (forceRefresh = false) => {
      const response = await fetch(`${normalizedApiUrl}/api/sync?force_refresh=${forceRefresh}`, {
        method: 'POST',
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'Sync failed');
      }
      return response.json();
    },
    reset: async () => {
      const response = await fetch(`${normalizedApiUrl}/api/reset`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'Reset failed');
      }
      return response.json();
    },
    search: async (query, provider, limit = 10) => {
      const response = await fetch(`${normalizedApiUrl}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, provider, limit }),
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'Search failed');
      }
      return response.json();
    },
  };

  // Handlers
  const handleInit = async () => {
    if (!apiBaseUrl) {
      alert('Please configure API Base URL first');
      return;
    }

    setInitializing(true);
    setInitResult(null);
    try {
      const result = await api.init();
      setInitResult(result);
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setInitializing(false);
    }
  };

  const handleSync = async () => {
    if (!apiBaseUrl) {
      alert('Please configure API Base URL first');
      return;
    }

    const shouldForce = window.confirm(
      'Sync photos. Choose:\n' +
      'OK - Force refresh (clears existing data first)\n' +
      'Cancel - Incremental sync (adds new photos only)'
    );

    setSyncing(true);
    setSyncResult(null);
    setResults([]);
    try {
      const result = await api.sync(shouldForce);
      setSyncResult(result);
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setSyncing(false);
    }
  };

  const handleReset = async () => {
    if (!apiBaseUrl) {
      alert('Please configure API Base URL first');
      return;
    }

    if (!confirm('Are you sure you want to clear all memories from both Mem0 and Zep?')) return;

    setResetting(true);
    try {
      await api.reset();
      setInitResult(null);
      setSyncResult(null);
      setResults([]);
      alert('All memories cleared');
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setResetting(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!apiBaseUrl) {
      alert('Please configure API Base URL first');
      return;
    }
    if (!query.trim()) return;

    setSearching(true);
    try {
      const data = await api.search(query, provider, 20);
      setResults(data);
    } catch (error) {
      alert(`Error: ${error.message}`);
      setResults([]);
    } finally {
      setSearching(false);
    }
  };

  const handleCardClick = (result) => {
    setSelectedResult(result);
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setSelectedResult(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 tracking-tight">
                Vision Memory Lab
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                Compare Mem0 and Zep with VLM + Face Recognition + EXIF
              </p>
            </div>
            <div className="text-xs text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
              v2.0.0
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Configuration Section */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c.94-1.543-.826-3.31 2.37-2.37.996.608 2.296.07 3-2.37-.996-.608-.07-2.296-2.37-3z" />
            </svg>
            Configuration
          </h2>
          <div>
            <label htmlFor="api-url" className="block text-sm font-medium text-gray-700 mb-2">
              Backend API Base URL
            </label>
            <input
              id="api-url"
              type="url"
              value={apiBaseUrl}
              onChange={(e) => handleApiUrlChange(e.target.value)}
              placeholder="https://your-backend.railway.app"
              className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
            />
            <p className="text-xs text-gray-500 mt-2">
              Enter your Railway deployment URL or local development server
            </p>
          </div>
        </section>

        {/* Control Section */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4 0m-4 0a2 2 0 110 4 0m-4 0v6m0 0v6m0-6a2 2 0 100 4 0m0 0v-6m0 0V8m0-6a2 2 0 100-4 0" />
            </svg>
            Memory Controls
          </h2>
          <div className="flex flex-wrap gap-4">
            <button
              onClick={handleInit}
              disabled={initializing}
              className="px-5 py-2.5 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:bg-emerald-300 disabled:cursor-not-allowed transition-colors font-medium flex items-center gap-2"
            >
              {initializing ? (
                <>
                  <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12c0 2.209-.797 4.218-2.192 5.959L16 17V4h-2.091z"></path>
                  </svg>
                  Initializing...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0m0 0v5m0-6a2 2 0 110 4m0 0a2 2 0 010 4m-6 6a2 2 0 01-2-2H5a2 2 0 01-2-2v6a2 2 0 002 2z" />
                  </svg>
                  Initialize Identity
                </>
              )}
            </button>
            <button
              onClick={handleSync}
              disabled={syncing}
              className="px-5 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed transition-colors font-medium flex items-center gap-2"
            >
              {syncing ? (
                <>
                  <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12c0 2.209-.797 4.218-2.192 5.959L16 17V4h-2.091z"></path>
                  </svg>
                  Syncing...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4 8l-4-4m0 0l4-4m-4 4" />
                  </svg>
                  Sync Photos
                </>
              )}
            </button>
            <button
              onClick={handleReset}
              disabled={resetting}
              className="px-5 py-2.5 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-red-300 disabled:cursor-not-allowed transition-colors font-medium flex items-center gap-2"
            >
              {resetting ? (
                <>
                  <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12c0 2.209-.797 4.218-2.192 5.959L16 17V4h-2.091z"></path>
                  </svg>
                  Resetting...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6" />
                  </svg>
                  Reset Memories
                </>
              )}
            </button>
          </div>

          {/* Init Result */}
          {initResult && (
            <div className="mt-4 p-4 bg-emerald-50 border border-emerald-200 rounded-lg">
              <p className="text-emerald-800 font-medium flex items-center gap-2">
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414-1.414L11 9.414l3.293 3.293a1 1 0 001.414-1.414L11 11.586l3.293-3.293z" clipRule="evenodd" />
                </svg>
                Identity Initialized
              </p>
              <div className="mt-2 text-sm text-emerald-700 space-y-1">
                <p>VLM Features: {initResult.vlm_features ? '✓ Extracted' : '✗ Not available'}</p>
                <p>Face Encoding: {initResult.face_encoding ? '✓ Loaded' : '✗ Not available'}</p>
              </div>
            </div>
          )}

          {/* Sync Result */}
          {syncResult && (
            <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-blue-800 font-medium flex items-center gap-2">
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414-1.414L11 9.414l3.293 3.293a1 1 0 001.414-1.414L11 11.586l3.293-3.293z" clipRule="evenodd" />
                </svg>
                Sync Complete
              </p>
              <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                <div className="bg-white rounded p-2">
                  <p className="text-gray-500 text-xs">Total Photos</p>
                  <p className="text-blue-900 font-semibold">{syncResult.total_photos}</p>
                </div>
                <div className="bg-white rounded p-2">
                  <p className="text-gray-500 text-xs">Zep Stored</p>
                  <p className="text-blue-900 font-semibold">{syncResult.zep_stored}</p>
                </div>
                <div className="bg-white rounded p-2">
                  <p className="text-gray-500 text-xs">Mem0 Stored</p>
                  <p className="text-blue-900 font-semibold">{syncResult.mem0_stored}</p>
                </div>
                <div className="bg-white rounded p-2">
                  <p className="text-gray-500 text-xs">Protagonist</p>
                  <p className="text-purple-900 font-semibold">{syncResult.protagonist_photos}</p>
                </div>
              </div>
              {syncResult.face_recognition && (
                <p className="text-xs text-blue-600 mt-2">
                  ✓ Face recognition enabled
                </p>
              )}
            </div>
          )}
        </section>

        {/* Search Section */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            Search Memories
          </h2>
          <form onSubmit={handleSearch}>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="md:col-span-2">
                <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
                  Search Query
                </label>
                <input
                  id="query"
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder='e.g., "我在海边拍照" or "landscape photos"'
                  className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                />
                <p className="text-xs text-gray-500 mt-1.5">
                  Queries with "我" will be rewritten to "【主角】" for protagonist filtering
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Memory Provider
                </label>
                <div className="space-y-2">
                  <label className="flex items-center p-3 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors">
                    <input
                      type="radio"
                      name="provider"
                      value="zep"
                      checked={provider === 'zep'}
                      onChange={() => setProvider('zep')}
                      className="w-4 h-4 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-3 flex items-center gap-2">
                      <span className="font-medium text-gray-700">Zep</span>
                      <span className="text-xs text-gray-500">v1 HTTP API</span>
                    </span>
                  </label>
                  <label className="flex items-center p-3 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors">
                    <input
                      type="radio"
                      name="provider"
                      value="mem0"
                      checked={provider === 'mem0'}
                      onChange={() => setProvider('mem0')}
                      className="w-4 h-4 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-3 flex items-center gap-2">
                      <span className="font-medium text-gray-700">Mem0</span>
                      <span className="text-xs text-gray-500">Cloud API</span>
                    </span>
                  </label>
                </div>
              </div>
            </div>

            <div className="mt-4">
              <button
                type="submit"
                disabled={searching || !query.trim()}
                className="w-full px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-indigo-300 disabled:cursor-not-allowed transition-colors font-medium flex items-center justify-center gap-2"
              >
                {searching ? (
                  <>
                    <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12c0 2.209-.797 4.218-2.192 5.959L16 17V4h-2.091z"></path>
                    </svg>
                    Searching...
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    Search Memories
                  </>
                )}
              </button>
            </div>
          </form>
        </section>

        {/* Results Section - Waterfall Grid */}
        {results.length > 0 && (
          <section>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">
                Search Results ({results.length})
              </h2>
              <span className="text-sm text-gray-500">
                Provider: <span className="font-medium text-gray-700 uppercase">{provider}</span>
              </span>
            </div>
            <div className="columns-1 sm:columns-2 lg:columns-3 xl:columns-4 gap-4 space-y-4">
              {results.map((result, index) => (
                <div
                  key={index}
                  onClick={() => handleCardClick(result)}
                  className="break-inside-avoid cursor-pointer"
                >
                  <ResultCard result={result} />
                </div>
              ))}
            </div>
          </section>
        )}

        {results.length === 0 && !searching && query && (
          <div className="text-center text-gray-500 py-12 bg-white rounded-xl border border-gray-200">
            <svg className="w-16 h-16 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6 0h6m2 5a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2h6l-4-4 6 6 0 11-8-8V4" />
            </svg>
            <p className="text-lg font-medium">No results found</p>
            <p className="text-sm mt-1">Try a different search query or sync photos first</p>
          </div>
        )}
      </main>

      {/* Detail Modal */}
      {showModal && selectedResult && (
        <DetailModal result={selectedResult} onClose={handleCloseModal} />
      )}

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6 text-center text-sm text-gray-500">
          <p>Vision Memory Lab - Local evaluation tool for Mem0 and Zep</p>
          <p className="mt-1">Powered by Gemini 1.5 Flash + face_recognition + exifread</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
