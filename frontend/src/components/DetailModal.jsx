import React from 'react';

const DetailModal = ({ result, onClose }) => {
  if (!result) return null;

  const formatPrimitiveValue = (value) => {
    if (typeof value === 'boolean') {
      return value ? 'Yes' : 'No';
    }
    if (typeof value === 'number') {
      return value.toLocaleString();
    }
    return String(value);
  };

  const renderExifData = (exif) => {
    if (!exif) return null;

    const items = [];

    if (exif.gps_latitude !== undefined) {
      items.push({
        label: 'GPS Latitude',
        value: `${exif.gps_latitude?.toFixed(6)}°`
      });
    }
    if (exif.gps_longitude !== undefined) {
      items.push({
        label: 'GPS Longitude',
        value: `${exif.gps_longitude?.toFixed(6)}°`
      });
    }
    if (exif.datetime_original) {
      items.push({
        label: 'Date/Time',
        value: exif.datetime_original
      });
    }
    if (exif.camera_make || exif.camera_model) {
      items.push({
        label: 'Camera',
        value: `${exif.camera_make || ''} ${exif.camera_model || ''}`.trim()
      });
    }
    if (exif.lens_model) {
      items.push({
        label: 'Lens',
        value: exif.lens_model
      });
    }
    if (exif.focal_length) {
      items.push({
        label: 'Focal Length',
        value: exif.focal_length
      });
    }
    if (exif.iso) {
      items.push({
        label: 'ISO',
        value: exif.iso
      });
    }
    if (exif.aperture) {
      items.push({
        label: 'Aperture',
        value: exif.aperture
      });
    }
    if (exif.shutter_speed) {
      items.push({
        label: 'Shutter Speed',
        value: exif.shutter_speed
      });
    }

    if (items.length === 0) return null;

    return (
      <div className="space-y-2">
        <h4 className="font-semibold text-gray-900 text-sm border-b border-gray-200 pb-1">
          EXIF Data
        </h4>
        {items.map((item) => (
          <div key={item.label} className="flex">
            <span className="text-gray-500 text-xs w-32 flex-shrink-0">{item.label}:</span>
            <span className="text-gray-800 text-xs">{item.value}</span>
          </div>
        ))}
      </div>
    );
  };

  const metadata = result.metadata || {};
  const hasProtagonist = metadata.has_protagonist || false;
  const faceCount = metadata.face_count || 0;
  const score = result.score || 0;
  const exifData = metadata.exif || null;
  const cleanContent = result.content || '';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-white rounded-2xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">
            Memory Detail
          </h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left Column - Image */}
            <div className="space-y-4">
              {result.image_url && (
                <div className="aspect-video bg-gray-100 rounded-xl overflow-hidden">
                  <img
                    src={result.image_url}
                    alt="Memory"
                    className="w-full h-full object-contain"
                  />
                </div>
              )}

              {/* Score Badge */}
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Match Score</span>
                  <span className="text-2xl font-bold text-blue-600">
                    {(score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="mt-2 bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all"
                    style={{ width: `${score * 100}%` }}
                  />
                </div>
              </div>

              {/* Quick Stats */}
              <div className="grid grid-cols-2 gap-3">
                {hasProtagonist && (
                  <div className="bg-purple-50 rounded-lg p-3 text-center">
                    <svg className="w-6 h-6 mx-auto text-purple-600 mb-1" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
                    </svg>
                    <span className="text-xs text-purple-800 font-medium">Protagonist</span>
                  </div>
                )}
                {faceCount > 0 && (
                  <div className="bg-blue-50 rounded-lg p-3 text-center">
                    <svg className="w-6 h-6 mx-auto text-blue-600 mb-1" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.324.07-.66.07-1a6.97 6.97 0 00-1.593-1.383 6.975 6.975 0 01-2.774-1.393 6.97 6.97 0 00-1.593 1.383c-.036.34-.056.68-.07 1zM12 4.5a2.5 2.5 0 110-5 2.5 2.5 0 010 5zm3.5 3a2.5 2.5 0 110-5 2.5 2.5 0 010 5zM5.5 9.5a2.5 2.5 0 110-5 2.5 2.5 0 010 5z" />
                    </svg>
                    <span className="text-xs text-blue-800 font-medium">{faceCount} Face{faceCount > 1 ? 's' : ''}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column - Details */}
            <div className="space-y-4">
              {/* Description */}
              <div className="bg-gray-50 rounded-xl p-4">
                <h4 className="font-semibold text-gray-900 text-sm mb-2">
                  VLM Description
                </h4>
                <p className="text-gray-700 text-sm leading-relaxed whitespace-pre-wrap">
                  {cleanContent}
                </p>
              </div>

              {/* EXIF Data */}
              {exifData && (
                <div className="bg-gray-50 rounded-xl p-4">
                  {renderExifData(exifData)}
                </div>
              )}

              {/* Full Metadata JSON */}
              <details className="bg-gray-900 rounded-xl overflow-hidden">
                <summary className="px-4 py-3 cursor-pointer text-white text-sm font-medium hover:bg-gray-800 transition-colors">
                  Raw JSON Data
                </summary>
                <pre className="p-4 text-xs text-green-400 overflow-x-auto">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </details>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200 bg-gray-50">
          <button
            onClick={onClose}
            className="w-full px-4 py-2 bg-white border border-gray-300 rounded-lg text-gray-700 font-medium hover:bg-gray-50 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default DetailModal;
