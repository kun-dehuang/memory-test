import React from 'react';

const ResultCard = ({ result }) => {
  const hasError = 'error' in result;

  if (hasError) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-600 text-sm">{result.error}</p>
      </div>
    );
  }

  // Format metadata for display
  const formatMetadata = (metadata) => {
    const displayItems = [];

    if (metadata.filename) {
      displayItems.push({ label: 'Filename', value: metadata.filename });
    }
    if (metadata.gps) {
      const { latitude, longitude } = metadata.gps;
      displayItems.push({
        label: 'GPS',
        value: `${latitude.toFixed(5)}, ${longitude.toFixed(5)}`
      });
    }
    if (metadata.width && metadata.height) {
      displayItems.push({
        label: 'Dimensions',
        value: `${metadata.width} × ${metadata.height}`
      });
    }
    if (metadata.format) {
      displayItems.push({ label: 'Format', value: metadata.format });
    }

    return displayItems;
  };

  const metadataItems = formatMetadata(result.metadata || {});
  const hasProtagonist = result.metadata?.has_protagonist || false;

  // Clean content for display (remove protagonist tag for cleaner UI)
  const cleanContent = result.content?.replace(/【主角】/g, '').trim() || result.content;

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-200">
      {/* Image Section */}
      {result.image_url && (
        <div className="aspect-video w-full bg-gray-100 flex items-center justify-center relative">
          <img
            src={result.image_url}
            alt={result.content}
            className="max-w-full max-h-full object-contain"
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.nextSibling.style.display = 'flex';
            }}
          />
          <div className="hidden w-full h-full items-center justify-center text-gray-400 text-sm">
            Image not available
          </div>

          {/* Protagonist Badge */}
          {hasProtagonist && (
            <div className="absolute top-2 left-2 bg-purple-600 text-white px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1">
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
              </svg>
              Protagonist
            </div>
          )}
        </div>
      )}

      {/* Content Section */}
      <div className="p-4">
        <p className="text-gray-800 mb-3 line-clamp-3">{cleanContent}</p>

        {/* Score Badge */}
        <div className="mb-3 flex items-center gap-2">
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
            Score: {(result.score * 100).toFixed(1)}%
          </span>
          {result.metadata?.scene && (
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-700">
              {result.metadata.scene.slice(0, 30)}...
            </span>
          )}
        </div>

        {/* Metadata */}
        {metadataItems.length > 0 && (
          <div className="border-t pt-3 mt-3">
            <dl className="grid grid-cols-2 gap-2 text-sm">
              {metadataItems.map((item) => (
                <div key={item.label}>
                  <dt className="text-gray-500 text-xs">{item.label}</dt>
                  <dd className="text-gray-900 truncate" title={item.value}>
                    {item.value}
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultCard;
