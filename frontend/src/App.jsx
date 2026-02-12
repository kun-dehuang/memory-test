import React, { useState, useEffect } from 'react';
import ResultCard from './components/ResultCard';

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

  // State for scan operation
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);

  // State for reset operation
  const [resetting, setResetting] = useState(false);

  // State for search
  const [query, setQuery] = useState('');
  const [provider, setProvider] = useState('mem0');
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState([]);

  // API helper
  const api = {
    scan: async () => {
      const response = await fetch(`${apiBaseUrl}/api/scan`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error('Scan failed');
      return response.json();
    },
    reset: async () => {
      const response = await fetch(`${apiBaseUrl}/api/reset`, {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error('Reset failed');
      return response.json();
    },
    search: async (query, provider) => {
      const response = await fetch(`${apiBaseUrl}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, provider }),
      });
      if (!response.ok) throw new Error('Search failed');
      return response.json();
    },
  };

  // Handlers
  const handleScan = async () => {
    if (!apiBaseUrl) {
      alert('Please configure API Base URL first');
      return;
    }

    setScanning(true);
    setScanResult(null);
    setResults([]);
    try {
      const result = await api.scan();
      setScanResult(result);
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setScanning(false);
    }
  };

  const handleReset = async () => {
    if (!apiBaseUrl) {
      alert('Please configure API Base URL first');
      return;
    }

    if (!confirm('Are you sure you want to clear all memories?')) return;

    setResetting(true);
    try {
      await api.reset();
      setScanResult(null);
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
      const data = await api.search(query, provider);
      setResults(data);
    } catch (error) {
      alert(`Error: ${error.message}`);
      setResults([]);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-gray-900">
            Memory Solution Evaluation Tool
          </h1>
          <p className="text-sm text-gray-600 mt-1">
            Compare performance between Mem0 and Zep
          </p>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Configuration Section */}
        <section className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h2 className="text-lg font-semibold mb-4">Configuration</h2>
          <div>
            <label htmlFor="api-url" className="block text-sm font-medium text-gray-700 mb-2">
              Backend API Base URL
            </label>
            <input
              id="api-url"
              type="url"
              value={apiBaseUrl}
              onChange={(e) => handleApiUrlChange(e.target.value)}
              placeholder="https://your-app.railway.app"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            <p className="text-xs text-gray-500 mt-1">
              Enter your Railway deployment URL here
            </p>
          </div>
        </section>

        {/* Control Section */}
        <section className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h2 className="text-lg font-semibold mb-4">Memory Controls</h2>
          <div className="flex flex-wrap gap-4">
            <button
              onClick={handleScan}
              disabled={scanning}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed transition-colors"
            >
              {scanning ? 'Scanning...' : 'Re-scan Photos'}
            </button>
            <button
              onClick={handleReset}
              disabled={resetting}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:bg-red-300 disabled:cursor-not-allowed transition-colors"
            >
              {resetting ? 'Resetting...' : 'Reset All Memories'}
            </button>
          </div>

          {scanResult && (
            <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-md">
              <p className="text-green-800 font-medium">
                Scan Complete
              </p>
              <p className="text-green-700 text-sm mt-1">
                Processed {scanResult.total_photos} photos
                {' · '}
                Mem0: {scanResult.mem0_stored} stored
                {' · '}
                Zep: {scanResult.zep_stored} stored
              </p>
            </div>
          )}
        </section>

        {/* Search Section */}
        <section className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h2 className="text-lg font-semibold mb-4">Search Memories</h2>
          <form onSubmit={handleSearch}>
            <div className="mb-4">
              <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
                Search Query
              </label>
              <input
                id="query"
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., 'photos taken at the beach' or 'landscape photos'"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Memory Provider
              </label>
              <div className="flex gap-6">
                <label className="flex items-center">
                  <input
                    type="radio"
                    name="provider"
                    value="mem0"
                    checked={provider === 'mem0'}
                    onChange={() => setProvider('mem0')}
                    className="w-4 h-4 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-gray-700">Mem0</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="radio"
                    name="provider"
                    value="zep"
                    checked={provider === 'zep'}
                    onChange={() => setProvider('zep')}
                    className="w-4 h-4 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-gray-700">Zep</span>
                </label>
              </div>
            </div>

            <button
              type="submit"
              disabled={searching || !query.trim()}
              className="w-full px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:bg-indigo-300 disabled:cursor-not-allowed transition-colors"
            >
              {searching ? 'Searching...' : 'Search'}
            </button>
          </form>
        </section>

        {/* Results Section */}
        {results.length > 0 && (
          <section>
            <h2 className="text-lg font-semibold mb-4">
              Search Results ({results.length})
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {results.map((result, index) => (
                <ResultCard key={index} result={result} />
              ))}
            </div>
          </section>
        )}

        {results.length === 0 && !searching && query && (
          <div className="text-center text-gray-500 py-8">
            No results found. Try a different search query or scan photos first.
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
