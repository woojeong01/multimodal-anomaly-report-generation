import React from 'react';

export function LoadingState() {
  return (
    <div className="flex flex-col items-center justify-center py-16">
      <div className="w-12 h-12 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
      <p className="text-sm text-gray-500">데이터 로딩 중...</p>
    </div>
  );
}
