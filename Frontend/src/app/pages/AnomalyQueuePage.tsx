import React, { useState, useMemo } from 'react';
import { Badge } from '../components/Badge';
import { EmptyState } from '../components/EmptyState';
import { ChevronDown, ChevronUp, X, ZoomIn } from 'lucide-react';
import { AnomalyCase } from '../data/mockData';

interface AnomalyQueuePageProps {
  cases: AnomalyCase[];
  onCaseClick: (caseId: string) => void;
}

type SortField = 'timestamp' | 'anomaly_score' | 'product_class' | 'decision';
type SortDirection = 'asc' | 'desc';

export function AnomalyQueuePage({ cases, onCaseClick }: AnomalyQueuePageProps) {
  const [sortField, setSortField] = useState<SortField>('timestamp');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [previewCase, setPreviewCase] = useState<AnomalyCase | null>(null);
  
  const sortedCases = useMemo(() => {
    return [...cases].sort((a, b) => {
      let comparison = 0;
      
      switch (sortField) {
        case 'timestamp':
          comparison = a.timestamp.getTime() - b.timestamp.getTime();
          break;
        case 'anomaly_score':
          comparison = a.anomaly_score - b.anomaly_score;
          break;
        case 'product_class':
          comparison = a.product_class.localeCompare(b.product_class);
          break;
        case 'decision':
          comparison = a.decision.localeCompare(b.decision);
          break;
      }
      
      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [cases, sortField, sortDirection]);
  
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };
  
  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return sortDirection === 'asc' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />;
  };
  
  if (cases.length === 0) {
    return (
      <div className="p-8">
        <EmptyState
          title="조건에 맞는 케이스가 없습니다"
          description="필터 조건을 변경하거나 날짜 범위를 확대해보세요."
        />
      </div>
    );
  }
  
  return (
    <div className="flex h-full">
      {/* Main Table */}
      <div className={`flex-1 p-8 ${previewCase ? 'pr-4' : ''}`}>
        <div className="mb-6">
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">이상 큐</h1>
          <p className="text-sm text-gray-600">
            총 <span className="font-medium text-gray-900">{cases.length}</span>개 케이스
          </p>
        </div>
        
        <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th
                    onClick={() => handleSort('timestamp')}
                    className="text-left py-3 px-4 text-sm font-medium text-gray-600 cursor-pointer hover:bg-gray-100"
                  >
                    <div className="flex items-center gap-1">
                      시간
                      <SortIcon field="timestamp" />
                    </div>
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">케이스 ID</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">라인</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">교대</th>
                  <th
                    onClick={() => handleSort('product_class')}
                    className="text-left py-3 px-4 text-sm font-medium text-gray-600 cursor-pointer hover:bg-gray-100"
                  >
                    <div className="flex items-center gap-1">
                      제품
                      <SortIcon field="product_class" />
                    </div>
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">배치 ID</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">결함 타입</th>
                  <th
                    onClick={() => handleSort('anomaly_score')}
                    className="text-left py-3 px-4 text-sm font-medium text-gray-600 cursor-pointer hover:bg-gray-100"
                  >
                    <div className="flex items-center gap-1">
                      Score
                      <SortIcon field="anomaly_score" />
                    </div>
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">위치</th>
                  <th
                    onClick={() => handleSort('decision')}
                    className="text-left py-3 px-4 text-sm font-medium text-gray-600 cursor-pointer hover:bg-gray-100"
                  >
                    <div className="flex items-center gap-1">
                      판정
                      <SortIcon field="decision" />
                    </div>
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">심각도</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-600"></th>
                </tr>
              </thead>
              <tbody>
                {sortedCases.map((anomaly) => (
                  <tr
                    key={anomaly.id}
                    className={`border-b border-gray-100 hover:bg-gray-50 cursor-pointer transition-colors ${
                      previewCase?.id === anomaly.id ? 'bg-blue-50' : ''
                    }`}
                    onClick={() => setPreviewCase(anomaly)}
                  >
                    <td className="py-3 px-4 text-sm text-gray-700">
                      {anomaly.timestamp.toLocaleString('ko-KR', { 
                        month: 'short', 
                        day: 'numeric', 
                        hour: '2-digit', 
                        minute: '2-digit' 
                      })}
                    </td>
                    <td className="py-3 px-4 text-sm font-mono text-blue-600">{anomaly.id}</td>
                    <td className="py-3 px-4 text-sm text-gray-700">{anomaly.line_id}</td>
                    <td className="py-3 px-4 text-sm text-gray-700">{anomaly.shift}</td>
                    <td className="py-3 px-4 text-sm text-gray-700">
                      <div>
                        <div className="font-medium">{anomaly.product_class}</div>
                        <div className="text-xs text-gray-500">{anomaly.product_group}</div>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-sm font-mono text-gray-600">{anomaly.batch_id}</td>
                    <td className="py-3 px-4 text-sm text-gray-700">
                      {anomaly.defect_type === 'seal_issue' ? '실링 불량' :
                       anomaly.defect_type === 'contamination' ? '오염' :
                       anomaly.defect_type === 'crack' ? '파손/균열' :
                       anomaly.defect_type === 'dent' ? '찌그러짐' :
                       anomaly.defect_type === 'scratch' ? '스크래치' :
                       anomaly.defect_type === 'none' ? '-' : anomaly.defect_type}
                    </td>
                    <td className="py-3 px-4 text-sm">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-gray-200 rounded-full h-2 w-16">
                          <div
                            className={`h-2 rounded-full ${
                              anomaly.anomaly_score >= 0.8 ? 'bg-red-500' :
                              anomaly.anomaly_score >= 0.65 ? 'bg-orange-500' :
                              'bg-green-500'
                            }`}
                            style={{ width: `${anomaly.anomaly_score * 100}%` }}
                          ></div>
                        </div>
                        <span className="font-medium text-gray-900 w-10">
                          {anomaly.anomaly_score.toFixed(2)}
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-700">
                      {anomaly.location === 'top-left' ? '상단좌' :
                       anomaly.location === 'top-right' ? '상단우' :
                       anomaly.location === 'bottom-left' ? '하단좌' :
                       anomaly.location === 'bottom-right' ? '하단우' :
                       anomaly.location === 'center' ? '중앙' : '-'}
                    </td>
                    <td className="py-3 px-4">
                      <Badge variant={anomaly.decision} size="sm">{anomaly.decision}</Badge>
                    </td>
                    <td className="py-3 px-4">
                      <Badge variant={anomaly.severity} size="sm">
                        {anomaly.severity === 'high' ? '높음' : anomaly.severity === 'med' ? '중간' : '낮음'}
                      </Badge>
                    </td>
                    <td className="py-3 px-4">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onCaseClick(anomaly.id);
                        }}
                        className="text-blue-600 hover:text-blue-700"
                      >
                        <ZoomIn className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      
      {/* Preview Panel */}
      {previewCase && (
        <div className="w-96 bg-white border-l border-gray-200 p-6 overflow-y-auto">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">빠른 미리보기</h3>
            <button
              onClick={() => setPreviewCase(null)}
              className="p-1 hover:bg-gray-100 rounded"
            >
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>
          
          {/* Image Placeholder */}
          <div className="bg-gray-100 rounded-lg mb-4 aspect-square flex items-center justify-center">
            <div className="text-center">
              <div className="w-16 h-16 bg-gray-300 rounded-lg mx-auto mb-2"></div>
              <p className="text-sm text-gray-500">이미지 샘플</p>
              <p className="text-xs text-gray-400 mt-1">{previewCase.image_id}</p>
            </div>
          </div>
          
          {/* Details */}
          <div className="space-y-4">
            <div>
              <label className="text-xs font-medium text-gray-500 uppercase">케이스 ID</label>
              <p className="text-sm font-mono text-gray-900 mt-1">{previewCase.id}</p>
            </div>
            
            <div>
              <label className="text-xs font-medium text-gray-500 uppercase">판정 결과</label>
              <div className="mt-1">
                <Badge variant={previewCase.decision}>{previewCase.decision}</Badge>
              </div>
            </div>
            
            <div>
              <label className="text-xs font-medium text-gray-500 uppercase">Anomaly Score</label>
              <div className="flex items-center gap-2 mt-1">
                <div className="flex-1 bg-gray-200 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full ${
                      previewCase.anomaly_score >= 0.8 ? 'bg-red-500' :
                      previewCase.anomaly_score >= 0.65 ? 'bg-orange-500' :
                      'bg-green-500'
                    }`}
                    style={{ width: `${previewCase.anomaly_score * 100}%` }}
                  ></div>
                </div>
                <span className="text-sm font-medium">{previewCase.anomaly_score.toFixed(2)}</span>
              </div>
              <p className="text-xs text-gray-500 mt-1">Threshold: {previewCase.threshold.toFixed(2)}</p>
            </div>
            
            <div>
              <label className="text-xs font-medium text-gray-500 uppercase">AI 요약</label>
              <p className="text-sm text-gray-700 mt-1">{previewCase.llm_summary}</p>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-medium text-gray-500 uppercase">결함 타입</label>
                <p className="text-sm text-gray-900 mt-1">
                  {previewCase.defect_type === 'seal_issue' ? '실링 불량' :
                   previewCase.defect_type === 'contamination' ? '오염' :
                   previewCase.defect_type === 'crack' ? '파손/균열' :
                   previewCase.defect_type === 'dent' ? '찌그러짐' :
                   previewCase.defect_type === 'scratch' ? '스크래치' : '-'}
                </p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500 uppercase">심각도</label>
                <div className="mt-1">
                  <Badge variant={previewCase.severity} size="sm">
                    {previewCase.severity === 'high' ? '높음' : previewCase.severity === 'med' ? '중간' : '낮음'}
                  </Badge>
                </div>
              </div>
            </div>
            
            <div>
              <label className="text-xs font-medium text-gray-500 uppercase">모델 정보</label>
              <p className="text-sm text-gray-900 mt-1">{previewCase.model_name} {previewCase.model_version}</p>
              <p className="text-xs text-gray-500">추론 시간: {previewCase.inference_time_ms}ms</p>
            </div>
            
            <button
              onClick={() => onCaseClick(previewCase.id)}
              className="w-full py-2 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              상세 보기
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
