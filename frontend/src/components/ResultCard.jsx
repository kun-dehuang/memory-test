import React from 'react';

const ResultCard = ({ result }) => {
  const hasError = 'error' in result;

  if (hasError) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-4">
        <p className="text-red-600 text-sm">{result.error}</p>
      </div>
    );
  }

  const metadata = result.metadata || {};
  const hasProtagonist = metadata.has_protagonist || false;
  const faceCount = metadata.face_count || 0;
  const exif = metadata.exif || null;

  // Format metadata for quick display
  const displayItems = [];

  if (metadata.filename) {
    displayItems.push({ label: 'File', value: metadata.filename });
  }
  if (exif && exif.gps_latitude !== undefined && exif.gps_longitude !== undefined) {
    displayItems.push({
      label: 'GPS',
      value: `${exif.gps_latitude.toFixed(4)}, ${exif.gps_longitude.toFixed(4)}`
    });
  }
  if (metadata.width && metadata.height) {
    displayItems.push({
      label: 'Size',
      value: `${metadata.width} × ${metadata.height}`
    });
  }

  // Clean content for display (remove protagonist tag for cleaner UI)
  const cleanContent = result.content
    ? result.content.replace(/【主角】/g, '').trim()
    : result.content || '';

  const getScoreLabel = (score) => {
    if (score >= 0.8) return 'High Match';
    if (score >= 0.5) return 'Medium Match';
    return 'Low Match';
  };

  const formatDate = (dateStr) => {
    try {
      return new Date(dateStr.replace(':', '-')).toLocaleDateString();
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden hover:shadow-xl transition-all duration-200 cursor-pointer group">
      {/* Image Section */}
      {result.image_url && (
        <div className="aspect-video w-full bg-gray-100 relative overflow-hidden">
          <img
            src={result.image_url}
            alt={cleanContent}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
            onError={(e) => {
              e.target.style.display = 'none';
              if (e.target.nextSibling) {
                e.target.nextSibling.style.display = 'flex';
              }
            }}
          />
          <div className="hidden w-full h-full items-center justify-center text-gray-400 text-sm bg-gray-50">
            <div className="text-center p-4">
              <svg className="w-12 h-12 mx-auto mb-2 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              Image not available
            </div>
          </div>

          {/* Protagonist Badge */}
          {hasProtagonist && (
            <div className="absolute top-2 left-2 bg-purple-600 text-white px-2.5 py-1 rounded-full text-xs font-medium flex items-center gap-1.5 shadow-md">
              <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
              </svg>
              Protagonist
            </div>
          )}

          {/* Face Count Badge */}
          {faceCount > 0 && !hasProtagonist && (
            <div className="absolute top-2 left-2 bg-blue-600 text-white px-2.5 py-1 rounded-full text-xs font-medium flex items-center gap-1.5 shadow-md">
              <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.324.07-.66.07-1a6.97 6.97 0 00-1.593-1.383 6.975 6.975 0 01-2.774-1.393 6.97 6.97 0 00-1.593 1.383c-.036.34-.056.68-.07 1zM12 4.5a2.5 2.5 0 110-5 2.5 2.5 0 010 5zm3.5 3a2.5 2.5 0 110-5 2.5 2.5 0 010 5zM5.5 9.5a2.5 2.5 0 110-5 2.5 2.5 0 010 5z" />
              </svg>
              {faceCount} {faceCount === 1 ? 'Face' : 'Faces'}
            </div>
          )}

          {/* Score Overlay */}
          <div className="absolute bottom-2 right-2 bg-black/70 text-white px-2.5 py-1 rounded-lg text-xs font-medium backdrop-blur-sm">
            {result.score !== undefined && Math.round(result.score * 100)}%
          </div>
        </div>
      )}

      {/* Content Section */}
      <div className="p-4">
        <p className="text-gray-800 text-sm mb-3 line-clamp-3 leading-relaxed min-h-[4rem]">
          {cleanContent || 'No description available'}
        </p>

        {/* Metadata Tags */}
        <div className="flex flex-wrap gap-1.5 mb-3">
          {result.score !== undefined && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium bg-blue-100 text-blue-700">
              {getScoreLabel(result.score)}
            </span>
          )}
          {exif && exif.datetime_original && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium bg-gray-100 text-gray-600">
              {formatDate(exif.datetime_original)}
            </span>
          )}
        </div>

        {/* Quick Info */}
        {displayItems.length > 0 && (
          <div className="border-t border-gray-100 pt-3 mt-3 grid grid-cols-2 gap-2 text-xs">
            {displayItems.slice(0, 4).map((item) => (
              <div key={item.label} className="truncate" title={item.value}>
                <span className="text-gray-500">{item.label}:</span>{' '}
                <span className="text-gray-900 font-medium">{item.value}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultCard;
