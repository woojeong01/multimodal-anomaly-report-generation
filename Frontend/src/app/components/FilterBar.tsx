import React from 'react';
import { Calendar, Factory, Package, AlertTriangle, Filter, X } from 'lucide-react';

export interface FilterState {
  dateRange: string;
  line: string;
  productGroup: string;
  defectType: string;
  decision: string;
  scoreRange: [number, number];
}

interface FilterBarProps {
  filters: FilterState;
  onFilterChange: (filters: FilterState) => void;
}

export function FilterBar({ filters, onFilterChange }: FilterBarProps) {
  const hasActiveFilters = 
    filters.line !== 'all' ||
    filters.productGroup !== 'all' ||
    filters.defectType !== 'all' ||
    filters.decision !== 'all' ||
    filters.scoreRange[0] > 0 ||
    filters.scoreRange[1] < 1;
  
  const clearFilters = () => {
    onFilterChange({
      dateRange: 'today',
      line: 'all',
      productGroup: 'all',
      defectType: 'all',
      decision: 'all',
      scoreRange: [0, 1]
    });
  };
  
  return (
    <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
      <div className="px-8 py-4">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2 text-sm text-gray-700">
            <Filter className="w-4 h-4" />
            <span className="font-medium">필터</span>
          </div>
          
          {/* Date Range */}
          <select
            value={filters.dateRange}
            onChange={(e) => onFilterChange({ ...filters, dateRange: e.target.value })}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="today">오늘</option>
            <option value="yesterday">어제</option>
            <option value="week">최근 7일</option>
            <option value="month">최근 30일</option>
          </select>
          
          {/* Line */}
          <select
            value={filters.line}
            onChange={(e) => onFilterChange({ ...filters, line: e.target.value })}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">전체 라인</option>
            <option value="LINE-A-01">LINE-A-01</option>
            <option value="LINE-B-02">LINE-B-02</option>
            <option value="LINE-C-03">LINE-C-03</option>
          </select>
          
          {/* Product Group */}
          <select
            value={filters.productGroup}
            onChange={(e) => onFilterChange({ ...filters, productGroup: e.target.value })}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">전체 제품군</option>
            <option value="Food">Food</option>
            <option value="Household">Household</option>
          </select>
          
          {/* Defect Type */}
          <select
            value={filters.defectType}
            onChange={(e) => onFilterChange({ ...filters, defectType: e.target.value })}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">전체 결함 타입</option>
            <option value="seal_issue">실링 불량</option>
            <option value="contamination">오염</option>
            <option value="crack">파손/균열</option>
            <option value="dent">찌그러짐</option>
            <option value="scratch">스크래치</option>
          </select>
          
          {/* Decision */}
          <select
            value={filters.decision}
            onChange={(e) => onFilterChange({ ...filters, decision: e.target.value })}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">전체 판정</option>
            <option value="OK">OK</option>
            <option value="NG">NG</option>
            <option value="REVIEW">REVIEW</option>
          </select>
          
          {hasActiveFilters && (
            <button
              onClick={clearFilters}
              className="flex items-center gap-1 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <X className="w-4 h-4" />
              <span>초기화</span>
            </button>
          )}
        </div>
        
        {hasActiveFilters && (
          <div className="flex items-center gap-2 mt-3">
            <span className="text-xs text-gray-500">활성 필터:</span>
            {filters.line !== 'all' && (
              <span className="px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded">
                라인: {filters.line}
              </span>
            )}
            {filters.productGroup !== 'all' && (
              <span className="px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded">
                제품군: {filters.productGroup}
              </span>
            )}
            {filters.defectType !== 'all' && (
              <span className="px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded">
                결함: {filters.defectType}
              </span>
            )}
            {filters.decision !== 'all' && (
              <span className="px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded">
                판정: {filters.decision}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
