import React, { useMemo } from 'react';
import { KPICard } from '../components/KPICard';
import { Badge } from '../components/Badge';
import { EmptyState } from '../components/EmptyState';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity, AlertTriangle, Clock, Zap, AlertCircle, TrendingUp } from 'lucide-react';
import { AnomalyCase, Alert } from '../data/mockData';
import { FilterState } from '../components/FilterBar';

interface OverviewPageProps {
  cases: AnomalyCase[];
  alerts: Alert[];
  filters: FilterState;
  onCaseClick: (caseId: string) => void;
  onFilterUpdate: (filters: Partial<FilterState>) => void;
}

export function OverviewPage({ cases, alerts, filters, onCaseClick, onFilterUpdate }: OverviewPageProps) {
  // Calculate KPIs
  const kpis = useMemo(() => {
    const total = cases.length;
    const ngCount = cases.filter(c => c.decision === 'NG').length;
    const reviewCount = cases.filter(c => c.decision === 'REVIEW').length;
    const avgInferenceTime = cases.reduce((sum, c) => sum + c.inference_time_ms, 0) / total;
    
    const ngRate = ((ngCount / total) * 100).toFixed(1);
    const reviewRate = ((reviewCount / total) * 100).toFixed(1);
    
    return {
      total,
      ngRate: `${ngRate}%`,
      reviewRate: `${reviewRate}%`,
      avgInferenceTime: `${avgInferenceTime.toFixed(0)}ms`
    };
  }, [cases]);
  
  // Trend data - hourly defect rate
  const trendData = useMemo(() => {
    const hourlyData: { [key: string]: { total: number; ng: number } } = {};
    
    cases.forEach(c => {
      const hour = c.timestamp.getHours();
      const key = `${hour}:00`;
      if (!hourlyData[key]) {
        hourlyData[key] = { total: 0, ng: 0 };
      }
      hourlyData[key].total++;
      if (c.decision === 'NG') {
        hourlyData[key].ng++;
      }
    });
    
    return Object.entries(hourlyData).map(([time, data]) => ({
      time,
      불량률: ((data.ng / data.total) * 100).toFixed(1)
    })).sort((a, b) => parseInt(a.time) - parseInt(b.time));
  }, [cases]);
  
  // Top defects by product class
  const productDefects = useMemo(() => {
    const defectsByClass: { [key: string]: number } = {};
    
    cases.filter(c => c.decision === 'NG').forEach(c => {
      defectsByClass[c.product_class] = (defectsByClass[c.product_class] || 0) + 1;
    });
    
    return Object.entries(defectsByClass)
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
  }, [cases]);
  
  // Defect type distribution
  const defectTypeData = useMemo(() => {
    const types: { [key: string]: number } = {};
    
    cases.filter(c => c.decision === 'NG').forEach(c => {
      types[c.defect_type] = (types[c.defect_type] || 0) + 1;
    });
    
    const typeLabels: { [key: string]: string } = {
      seal_issue: '실링 불량',
      contamination: '오염',
      crack: '파손/균열',
      dent: '찌그러짐',
      scratch: '스크래치'
    };
    
    return Object.entries(types).map(([type, value]) => ({
      name: typeLabels[type] || type,
      value,
      type
    }));
  }, [cases]);
  
  const COLORS = ['#ef4444', '#f97316', '#eab308', '#84cc16', '#06b6d4', '#8b5cf6'];
  
  // Recent anomalies
  const recentAnomalies = useMemo(() => {
    return cases
      .filter(c => c.decision === 'NG' || c.decision === 'REVIEW')
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, 8);
  }, [cases]);
  
  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">개요</h1>
        <p className="text-sm text-gray-600">실시간 품질 검사 현황 및 이상 탐지 결과</p>
      </div>
      
      {/* KPI Cards */}
      <div className="grid grid-cols-4 gap-6 mb-8">
        <KPICard
          title="검사 수"
          value={kpis.total}
          subtitle="오늘"
          icon={Activity}
        />
        <KPICard
          title="불량률"
          value={kpis.ngRate}
          trend={{ value: 12, isPositive: true }}
          icon={AlertTriangle}
        />
        <KPICard
          title="재검률"
          value={kpis.reviewRate}
          trend={{ value: 5, isPositive: true }}
          icon={AlertCircle}
        />
        <KPICard
          title="평균 추론 시간"
          value={kpis.avgInferenceTime}
          subtitle="AI 모델"
          icon={Zap}
        />
      </div>
      
      {/* Alerts */}
      <div className="bg-white border border-gray-200 rounded-lg p-6 mb-8">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="w-5 h-5 text-gray-700" />
          <h2 className="text-lg font-medium text-gray-900">오늘의 알림</h2>
        </div>
        <div className="space-y-3">
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className={`flex items-start gap-3 p-4 rounded-lg border ${
                alert.severity === 'high'
                  ? 'bg-red-50 border-red-200'
                  : alert.severity === 'med'
                  ? 'bg-orange-50 border-orange-200'
                  : 'bg-gray-50 border-gray-200'
              }`}
            >
              <AlertCircle className={`w-5 h-5 mt-0.5 ${
                alert.severity === 'high'
                  ? 'text-red-600'
                  : alert.severity === 'med'
                  ? 'text-orange-600'
                  : 'text-gray-600'
              }`} />
              <div className="flex-1">
                <div className="flex items-start justify-between mb-1">
                  <h3 className="font-medium text-gray-900">{alert.title}</h3>
                  <Badge variant={alert.severity} size="sm">{alert.severity.toUpperCase()}</Badge>
                </div>
                <p className="text-sm text-gray-700 mb-2">{alert.description}</p>
                <p className="text-xs text-gray-500">
                  {alert.timestamp.toLocaleString('ko-KR')}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Charts */}
      <div className="grid grid-cols-2 gap-6 mb-8">
        {/* Trend Chart */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">불량률 추이</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="time" tick={{ fontSize: 12 }} stroke="#6b7280" />
              <YAxis tick={{ fontSize: 12 }} stroke="#6b7280" />
              <Tooltip />
              <Line type="monotone" dataKey="불량률" stroke="#ef4444" strokeWidth={2} dot={{ fill: '#ef4444', r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        {/* Defect Type Distribution */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">결함 타입 분포</h3>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie
                data={defectTypeData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {defectTypeData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={COLORS[index % COLORS.length]}
                    className="cursor-pointer hover:opacity-80"
                    onClick={() => onFilterUpdate({ defectType: entry.type })}
                  />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Top Products by Defects */}
      <div className="bg-white border border-gray-200 rounded-lg p-6 mb-8">
        <h3 className="text-lg font-medium text-gray-900 mb-4">제품별 불량 Top 10</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={productDefects}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} stroke="#6b7280" />
            <YAxis tick={{ fontSize: 12 }} stroke="#6b7280" />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      {/* Recent Anomalies */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900">최근 이상 케이스</h3>
          <button
            onClick={() => onFilterUpdate({ decision: 'NG' })}
            className="text-sm text-blue-600 hover:text-blue-700"
          >
            전체 보기 →
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">케이스 ID</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">시간</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">라인</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">제품</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">결함 타입</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Score</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">판정</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">심각도</th>
              </tr>
            </thead>
            <tbody>
              {recentAnomalies.map((anomaly) => (
                <tr
                  key={anomaly.id}
                  onClick={() => onCaseClick(anomaly.id)}
                  className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer transition-colors"
                >
                  <td className="py-3 px-4 text-sm font-mono text-blue-600">{anomaly.id}</td>
                  <td className="py-3 px-4 text-sm text-gray-700">
                    {anomaly.timestamp.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}
                  </td>
                  <td className="py-3 px-4 text-sm text-gray-700">{anomaly.line_id}</td>
                  <td className="py-3 px-4 text-sm text-gray-700">{anomaly.product_class}</td>
                  <td className="py-3 px-4 text-sm text-gray-700">
                    {anomaly.defect_type === 'seal_issue' ? '실링 불량' :
                     anomaly.defect_type === 'contamination' ? '오염' :
                     anomaly.defect_type === 'crack' ? '파손/균열' :
                     anomaly.defect_type === 'dent' ? '찌그러짐' :
                     anomaly.defect_type === 'scratch' ? '스크래치' : anomaly.defect_type}
                  </td>
                  <td className="py-3 px-4 text-sm">
                    <span className={`font-medium ${
                      anomaly.anomaly_score >= 0.8 ? 'text-red-600' :
                      anomaly.anomaly_score >= 0.65 ? 'text-orange-600' :
                      'text-gray-600'
                    }`}>
                      {anomaly.anomaly_score.toFixed(2)}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    <Badge variant={anomaly.decision} size="sm">{anomaly.decision}</Badge>
                  </td>
                  <td className="py-3 px-4">
                    <Badge variant={anomaly.severity} size="sm">
                      {anomaly.severity === 'high' ? '높음' : anomaly.severity === 'med' ? '중간' : '낮음'}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
