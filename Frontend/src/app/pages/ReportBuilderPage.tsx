import React, { useState, useMemo } from 'react';
import { Badge } from '../components/Badge';
import { Calendar, Download, FileText, TrendingDown, TrendingUp } from 'lucide-react';
import { AnomalyCase } from '../data/mockData';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ReportBuilderPageProps {
  cases: AnomalyCase[];
}

export function ReportBuilderPage({ cases }: ReportBuilderPageProps) {
  const [reportPeriod, setReportPeriod] = useState('today');
  const [selectedLine, setSelectedLine] = useState('all');
  const [selectedGroup, setSelectedGroup] = useState('all');
  
  // Calculate report metrics
  const reportMetrics = useMemo(() => {
    const filteredCases = cases.filter(c => {
      if (selectedLine !== 'all' && c.line_id !== selectedLine) return false;
      if (selectedGroup !== 'all' && c.product_group !== selectedGroup) return false;
      return true;
    });
    
    const total = filteredCases.length;
    const ngCount = filteredCases.filter(c => c.decision === 'NG').length;
    const reviewCount = filteredCases.filter(c => c.decision === 'REVIEW').length;
    const okCount = filteredCases.filter(c => c.decision === 'OK').length;
    
    const ngRate = total > 0 ? ((ngCount / total) * 100).toFixed(2) : '0.00';
    const reviewRate = total > 0 ? ((reviewCount / total) * 100).toFixed(2) : '0.00';
    const passRate = total > 0 ? ((okCount / total) * 100).toFixed(2) : '0.00';
    
    const avgScore = total > 0 
      ? (filteredCases.reduce((sum, c) => sum + c.anomaly_score, 0) / total).toFixed(3)
      : '0.000';
    
    const avgInferenceTime = total > 0
      ? (filteredCases.reduce((sum, c) => sum + c.inference_time_ms, 0) / total).toFixed(1)
      : '0.0';
    
    // Defect type breakdown
    const defectBreakdown: { [key: string]: number } = {};
    filteredCases.filter(c => c.decision === 'NG').forEach(c => {
      defectBreakdown[c.defect_type] = (defectBreakdown[c.defect_type] || 0) + 1;
    });
    
    // Representative cases (worst cases)
    const representativeCases = filteredCases
      .filter(c => c.decision === 'NG')
      .sort((a, b) => b.anomaly_score - a.anomaly_score)
      .slice(0, 5);
    
    return {
      total,
      ngCount,
      reviewCount,
      okCount,
      ngRate,
      reviewRate,
      passRate,
      avgScore,
      avgInferenceTime,
      defectBreakdown,
      representativeCases,
      filteredCases
    };
  }, [cases, selectedLine, selectedGroup]);
  
  // Hourly trend
  const hourlyTrend = useMemo(() => {
    const hourlyData: { [key: number]: { total: number; ng: number; review: number; ok: number } } = {};
    
    reportMetrics.filteredCases.forEach(c => {
      const hour = c.timestamp.getHours();
      if (!hourlyData[hour]) {
        hourlyData[hour] = { total: 0, ng: 0, review: 0, ok: 0 };
      }
      hourlyData[hour].total++;
      if (c.decision === 'NG') hourlyData[hour].ng++;
      else if (c.decision === 'REVIEW') hourlyData[hour].review++;
      else hourlyData[hour].ok++;
    });
    
    return Object.entries(hourlyData)
      .map(([hour, data]) => ({
        hour: `${hour}시`,
        불량: data.ng,
        재검: data.review,
        정상: data.ok
      }))
      .sort((a, b) => parseInt(a.hour) - parseInt(b.hour));
  }, [reportMetrics.filteredCases]);
  
  // Defect breakdown chart data
  const defectChartData = useMemo(() => {
    const typeLabels: { [key: string]: string } = {
      seal_issue: '실링 불량',
      contamination: '오염',
      crack: '파손/균열',
      dent: '찌그러짐',
      scratch: '스크래치'
    };
    
    return Object.entries(reportMetrics.defectBreakdown).map(([type, count]) => ({
      type: typeLabels[type] || type,
      count
    }));
  }, [reportMetrics.defectBreakdown]);
  
  const handleExport = (format: string) => {
    alert(`리포트를 ${format} 형식으로 내보냅니다.`);
  };
  
  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">리포트 빌더</h1>
        <p className="text-sm text-gray-600">기간별 품질 분석 리포트 자동 생성</p>
      </div>
      
      {/* Report Configuration */}
      <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">리포트 설정</h2>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">기간</label>
            <select
              value={reportPeriod}
              onChange={(e) => setReportPeriod(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="today">오늘</option>
              <option value="yesterday">어제</option>
              <option value="week">최근 7일</option>
              <option value="month">최근 30일</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">생산 라인</label>
            <select
              value={selectedLine}
              onChange={(e) => setSelectedLine(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">전체 라인</option>
              <option value="LINE-A-01">LINE-A-01</option>
              <option value="LINE-B-02">LINE-B-02</option>
              <option value="LINE-C-03">LINE-C-03</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">제품군</label>
            <select
              value={selectedGroup}
              onChange={(e) => setSelectedGroup(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">전체 제품군</option>
              <option value="Food">Food</option>
              <option value="Household">Household</option>
            </select>
          </div>
        </div>
      </div>
      
      {/* Executive Summary */}
      <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">요약 (Executive Summary)</h2>
        <div className="grid grid-cols-5 gap-6 mb-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-600 mb-1">총 검사 수</p>
            <p className="text-3xl font-semibold text-gray-900">{reportMetrics.total}</p>
          </div>
          <div className="bg-red-50 rounded-lg p-4">
            <p className="text-sm text-red-700 mb-1">불량 (NG)</p>
            <p className="text-3xl font-semibold text-red-700">{reportMetrics.ngCount}</p>
            <p className="text-xs text-red-600 mt-1">{reportMetrics.ngRate}%</p>
          </div>
          <div className="bg-amber-50 rounded-lg p-4">
            <p className="text-sm text-amber-700 mb-1">재검 (REVIEW)</p>
            <p className="text-3xl font-semibold text-amber-700">{reportMetrics.reviewCount}</p>
            <p className="text-xs text-amber-600 mt-1">{reportMetrics.reviewRate}%</p>
          </div>
          <div className="bg-green-50 rounded-lg p-4">
            <p className="text-sm text-green-700 mb-1">정상 (OK)</p>
            <p className="text-3xl font-semibold text-green-700">{reportMetrics.okCount}</p>
            <p className="text-xs text-green-600 mt-1">{reportMetrics.passRate}%</p>
          </div>
          <div className="bg-blue-50 rounded-lg p-4">
            <p className="text-sm text-blue-700 mb-1">평균 Score</p>
            <p className="text-3xl font-semibold text-blue-700">{reportMetrics.avgScore}</p>
            <p className="text-xs text-blue-600 mt-1">{reportMetrics.avgInferenceTime}ms</p>
          </div>
        </div>
        
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-medium text-blue-900 mb-2">핵심 인사이트</h3>
          <ul className="space-y-2 text-sm text-blue-900">
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-0.5">•</span>
              <span>
                전체 불량률은 <strong>{reportMetrics.ngRate}%</strong>로 
                {parseFloat(reportMetrics.ngRate) > 2.5 ? ' 목표치(2.5%)를 초과했습니다.' : ' 목표치(2.5%) 이하를 유지하고 있습니다.'}
              </span>
            </li>
            {Object.entries(reportMetrics.defectBreakdown).length > 0 && (
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-0.5">•</span>
                <span>
                  가장 빈번한 결함 유형은 <strong>
                    {Object.entries(reportMetrics.defectBreakdown)
                      .sort((a, b) => b[1] - a[1])[0][0] === 'seal_issue' ? '실링 불량' :
                     Object.entries(reportMetrics.defectBreakdown)
                      .sort((a, b) => b[1] - a[1])[0][0] === 'contamination' ? '오염' :
                     Object.entries(reportMetrics.defectBreakdown)
                      .sort((a, b) => b[1] - a[1])[0][0] === 'crack' ? '파손/균열' :
                     Object.entries(reportMetrics.defectBreakdown)
                      .sort((a, b) => b[1] - a[1])[0][0] === 'dent' ? '찌그러짐' : '스크래치'}
                  </strong>입니다 ({Object.entries(reportMetrics.defectBreakdown).sort((a, b) => b[1] - a[1])[0][1]}건).
                </span>
              </li>
            )}
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-0.5">•</span>
              <span>평균 AI 추론 시간은 <strong>{reportMetrics.avgInferenceTime}ms</strong>로 안정적인 성능을 보이고 있습니다.</span>
            </li>
          </ul>
        </div>
      </div>
      
      {/* Charts */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* Hourly Trend */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">시간대별 검사 결과</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={hourlyTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="hour" tick={{ fontSize: 12 }} stroke="#6b7280" />
              <YAxis tick={{ fontSize: 12 }} stroke="#6b7280" />
              <Tooltip />
              <Legend />
              <Bar dataKey="불량" fill="#ef4444" stackId="a" />
              <Bar dataKey="재검" fill="#f59e0b" stackId="a" />
              <Bar dataKey="정상" fill="#10b981" stackId="a" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        
        {/* Defect Type Breakdown */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">결함 타입별 분포</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={defectChartData} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis type="number" tick={{ fontSize: 12 }} stroke="#6b7280" />
              <YAxis dataKey="type" type="category" tick={{ fontSize: 12 }} stroke="#6b7280" width={100} />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Representative Cases */}
      <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">대표 사례 (Top 5 심각 케이스)</h2>
        {reportMetrics.representativeCases.length > 0 ? (
          <div className="space-y-4">
            {reportMetrics.representativeCases.map((caseData, idx) => (
              <div key={caseData.id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center justify-center w-6 h-6 bg-red-100 text-red-700 rounded-full text-xs font-medium">
                        {idx + 1}
                      </span>
                      <span className="font-mono text-sm text-blue-600">{caseData.id}</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      {caseData.timestamp.toLocaleString('ko-KR')} · {caseData.line_id}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={caseData.decision} size="sm">{caseData.decision}</Badge>
                    <Badge variant={caseData.severity} size="sm">
                      {caseData.severity === 'high' ? '높음' : caseData.severity === 'med' ? '중간' : '낮음'}
                    </Badge>
                  </div>
                </div>
                
                <div className="grid grid-cols-4 gap-4 mb-3 text-sm">
                  <div>
                    <span className="text-gray-500 text-xs">제품</span>
                    <p className="font-medium text-gray-900">{caseData.product_class}</p>
                  </div>
                  <div>
                    <span className="text-gray-500 text-xs">결함 타입</span>
                    <p className="font-medium text-gray-900">
                      {caseData.defect_type === 'seal_issue' ? '실링 불량' :
                       caseData.defect_type === 'contamination' ? '오염' :
                       caseData.defect_type === 'crack' ? '파손/균열' :
                       caseData.defect_type === 'dent' ? '찌그러짐' : '스크래치'}
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-500 text-xs">Score</span>
                    <p className="font-medium text-red-600">{caseData.anomaly_score.toFixed(3)}</p>
                  </div>
                  <div>
                    <span className="text-gray-500 text-xs">영향 면적</span>
                    <p className="font-medium text-gray-900">{caseData.affected_area_pct.toFixed(1)}%</p>
                  </div>
                </div>
                
                <div className="bg-gray-50 rounded p-3">
                  <p className="text-sm text-gray-700">{caseData.llm_summary}</p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-500 text-center py-8">선택한 조건에서 불량 케이스가 없습니다.</p>
        )}
      </div>
      
      {/* Export Actions */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">리포트 내보내기</h2>
        <div className="flex items-center gap-3">
          <button
            onClick={() => handleExport('PDF')}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Download className="w-5 h-5" />
            <span>PDF로 내보내기</span>
          </button>
          <button
            onClick={() => handleExport('Excel')}
            className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
          >
            <Download className="w-5 h-5" />
            <span>Excel로 내보내기</span>
          </button>
          <button
            onClick={() => handleExport('JSON')}
            className="flex items-center gap-2 px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            <Download className="w-5 h-5" />
            <span>JSON으로 내보내기</span>
          </button>
        </div>
        <p className="text-sm text-gray-500 mt-4">
          리포트에는 위의 모든 차트, 통계, 대표 사례가 포함됩니다.
        </p>
      </div>
    </div>
  );
}
