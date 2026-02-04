import React, { useState } from 'react';
import { Badge } from '../components/Badge';
import { ChevronRight, Download, MessageSquare, UserPlus, CheckCircle, XCircle, RotateCcw, FileText } from 'lucide-react';
import { AnomalyCase } from '../data/mockData';

interface CaseDetailPageProps {
  caseData: AnomalyCase;
  onBack: () => void;
}

export function CaseDetailPage({ caseData, onBack }: CaseDetailPageProps) {
  const [activeTab, setActiveTab] = useState<'original' | 'heatmap' | 'overlay'>('original');
  const [showJson, setShowJson] = useState(false);
  const [note, setNote] = useState('');
  
  const handleAction = (action: string) => {
    alert(`${action} 액션이 실행되었습니다.`);
  };
  
  return (
    <div className="p-8">
      {/* Breadcrumbs */}
      <div className="flex items-center gap-2 text-sm text-gray-600 mb-6">
        <button onClick={onBack} className="hover:text-gray-900">개요</button>
        <ChevronRight className="w-4 h-4" />
        <button onClick={onBack} className="hover:text-gray-900">이상 큐</button>
        <ChevronRight className="w-4 h-4" />
        <span className="text-gray-900 font-medium">케이스 상세</span>
      </div>
      
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">{caseData.id}</h1>
          <p className="text-sm text-gray-600">
            {caseData.timestamp.toLocaleString('ko-KR')} · {caseData.line_id} · {caseData.shift}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant={caseData.decision}>{caseData.decision}</Badge>
          <Badge variant={caseData.severity}>
            {caseData.severity === 'high' ? '높음' : caseData.severity === 'med' ? '중간' : '낮음'}
          </Badge>
        </div>
      </div>
      
      <div className="grid grid-cols-3 gap-6">
        {/* Left Column - Image Viewer */}
        <div className="col-span-2">
          <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-medium text-gray-900">검사 이미지</h2>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setActiveTab('original')}
                  className={`px-3 py-1.5 text-sm rounded ${
                    activeTab === 'original'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  원본
                </button>
                <button
                  onClick={() => setActiveTab('heatmap')}
                  className={`px-3 py-1.5 text-sm rounded ${
                    activeTab === 'heatmap'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Heatmap
                </button>
                <button
                  onClick={() => setActiveTab('overlay')}
                  className={`px-3 py-1.5 text-sm rounded ${
                    activeTab === 'overlay'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Overlay
                </button>
              </div>
            </div>
            
            {/* Image Placeholder */}
            <div className="bg-gray-100 rounded-lg aspect-[4/3] flex items-center justify-center mb-4">
              <div className="text-center">
                <div className="w-24 h-24 bg-gray-300 rounded-lg mx-auto mb-3"></div>
                <p className="font-medium text-gray-700">이미지 {activeTab}</p>
                <p className="text-sm text-gray-500 mt-1">{caseData.image_id}</p>
                {activeTab === 'heatmap' && (
                  <p className="text-xs text-gray-400 mt-2">
                    이상 영역 히트맵 시각화
                  </p>
                )}
                {activeTab === 'overlay' && (
                  <p className="text-xs text-gray-400 mt-2">
                    검출 영역 오버레이 표시
                  </p>
                )}
              </div>
            </div>
            
            {/* Image Info */}
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-500">이미지 ID:</span>
                <span className="ml-2 font-mono text-gray-900">{caseData.image_id}</span>
              </div>
              <div>
                <span className="text-gray-500">배치 ID:</span>
                <span className="ml-2 font-mono text-gray-900">{caseData.batch_id}</span>
              </div>
              <div>
                <span className="text-gray-500">영향 영역:</span>
                <span className="ml-2 font-medium text-gray-900">{caseData.affected_area_pct.toFixed(1)}%</span>
              </div>
            </div>
          </div>
          
          {/* Evidence Summary */}
          <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">근거 요약</h2>
            
            {/* Anomaly Score */}
            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">Anomaly Score</span>
                <span className="text-2xl font-semibold text-gray-900">{caseData.anomaly_score.toFixed(3)}</span>
              </div>
              <div className="relative">
                <div className="h-4 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${
                      caseData.anomaly_score >= 0.8 ? 'bg-red-500' :
                      caseData.anomaly_score >= 0.65 ? 'bg-orange-500' :
                      'bg-green-500'
                    }`}
                    style={{ width: `${caseData.anomaly_score * 100}%` }}
                  ></div>
                </div>
                {/* Threshold marker */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-gray-900"
                  style={{ left: `${caseData.threshold * 100}%` }}
                >
                  <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs text-gray-600 whitespace-nowrap">
                    임계값: {caseData.threshold.toFixed(2)}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Defect Details */}
            <div className="grid grid-cols-2 gap-6 mb-6">
              <div>
                <label className="text-sm font-medium text-gray-500 uppercase block mb-2">결함 타입</label>
                <p className="text-lg text-gray-900">
                  {caseData.defect_type === 'seal_issue' ? '실링 불량' :
                   caseData.defect_type === 'contamination' ? '오염' :
                   caseData.defect_type === 'crack' ? '파손/균열' :
                   caseData.defect_type === 'dent' ? '찌그러짐' :
                   caseData.defect_type === 'scratch' ? '스크래치' : '-'}
                </p>
                <p className="text-sm text-gray-500 mt-1">신뢰도: {(caseData.defect_confidence * 100).toFixed(1)}%</p>
              </div>
              <div>
                <label className="text-sm font-medium text-gray-500 uppercase block mb-2">위치</label>
                <p className="text-lg text-gray-900">
                  {caseData.location === 'top-left' ? '상단 좌측' :
                   caseData.location === 'top-right' ? '상단 우측' :
                   caseData.location === 'bottom-left' ? '하단 좌측' :
                   caseData.location === 'bottom-right' ? '하단 우측' :
                   caseData.location === 'center' ? '중앙' : '-'}
                </p>
                <p className="text-sm text-gray-500 mt-1">영향 면적: {caseData.affected_area_pct.toFixed(1)}%</p>
              </div>
            </div>
            
            {/* AI Summary */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <label className="text-sm font-medium text-blue-900 uppercase block mb-2">AI 분석 요약</label>
              <p className="text-sm text-blue-900">{caseData.llm_summary}</p>
            </div>
            
            {/* Structured JSON */}
            <div className="mt-4">
              <button
                onClick={() => setShowJson(!showJson)}
                className="text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1"
              >
                <FileText className="w-4 h-4" />
                {showJson ? '구조화된 분석 결과 숨기기' : '구조화된 분석 결과 보기'}
              </button>
              {showJson && (
                <pre className="mt-3 p-4 bg-gray-900 text-green-400 rounded-lg text-xs overflow-x-auto">
                  {JSON.stringify(caseData.llm_structured_json, null, 2)}
                </pre>
              )}
            </div>
          </div>
          
          {/* Action Log */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">작업 로그</h2>
            <div className="space-y-3">
              {caseData.action_log.map((log, idx) => (
                <div key={idx} className="flex items-start gap-3 pb-3 border-b border-gray-100 last:border-0 last:pb-0">
                  <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-sm font-medium text-blue-700">{log.who[0]}</span>
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-gray-900">{log.what}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      {log.who} · {log.when.toLocaleString('ko-KR')}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        
        {/* Right Column - Actions & Info */}
        <div className="space-y-6">
          {/* Actions */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">액션</h2>
            <div className="space-y-2">
              <button
                onClick={() => handleAction('승인(OK)')}
                className="w-full flex items-center gap-2 px-4 py-2.5 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
              >
                <CheckCircle className="w-4 h-4" />
                <span>승인 (OK)</span>
              </button>
              <button
                onClick={() => handleAction('불량 확정(NG)')}
                className="w-full flex items-center gap-2 px-4 py-2.5 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                <XCircle className="w-4 h-4" />
                <span>불량 확정 (NG)</span>
              </button>
              <button
                onClick={() => handleAction('재검 요청')}
                className="w-full flex items-center gap-2 px-4 py-2.5 bg-amber-600 text-white rounded-lg hover:bg-amber-700 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                <span>재검 요청</span>
              </button>
            </div>
          </div>
          
          {/* Add Note */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">메모 추가</h2>
            <textarea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="케이스에 대한 메모를 입력하세요..."
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
              rows={4}
            />
            <button
              onClick={() => {
                handleAction('메모 추가');
                setNote('');
              }}
              className="mt-3 w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
            >
              <MessageSquare className="w-4 h-4" />
              <span>메모 저장</span>
            </button>
            {caseData.operator_note && (
              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <p className="text-xs font-medium text-gray-500 mb-1">기존 메모</p>
                <p className="text-sm text-gray-700">{caseData.operator_note}</p>
              </div>
            )}
          </div>
          
          {/* Product Info */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">제품 정보</h2>
            <div className="space-y-3 text-sm">
              <div>
                <span className="text-gray-500 block mb-1">제품군</span>
                <span className="font-medium text-gray-900">{caseData.product_group}</span>
              </div>
              <div>
                <span className="text-gray-500 block mb-1">제품 클래스</span>
                <span className="font-medium text-gray-900">{caseData.product_class}</span>
              </div>
              <div>
                <span className="text-gray-500 block mb-1">생산 라인</span>
                <span className="font-medium text-gray-900">{caseData.line_id}</span>
              </div>
              <div>
                <span className="text-gray-500 block mb-1">교대조</span>
                <span className="font-medium text-gray-900">{caseData.shift}</span>
              </div>
              <div>
                <span className="text-gray-500 block mb-1">배치 ID</span>
                <span className="font-mono text-sm text-gray-900">{caseData.batch_id}</span>
              </div>
            </div>
          </div>
          
          {/* Model Info */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">모델 정보</h2>
            <div className="space-y-3 text-sm">
              <div>
                <span className="text-gray-500 block mb-1">모델명</span>
                <span className="font-medium text-gray-900">{caseData.model_name}</span>
              </div>
              <div>
                <span className="text-gray-500 block mb-1">버전</span>
                <span className="font-mono text-sm text-gray-900">{caseData.model_version}</span>
              </div>
              <div>
                <span className="text-gray-500 block mb-1">추론 시간</span>
                <span className="font-medium text-gray-900">{caseData.inference_time_ms}ms</span>
              </div>
            </div>
          </div>
          
          {/* Export */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">내보내기</h2>
            <div className="space-y-2">
              <button
                onClick={() => handleAction('PDF 내보내기')}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
              >
                <Download className="w-4 h-4" />
                <span>PDF 다운로드</span>
              </button>
              <button
                onClick={() => handleAction('JSON 내보내기')}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
              >
                <Download className="w-4 h-4" />
                <span>JSON 다운로드</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
