import React, { useState, useMemo } from 'react';
import { Sidebar } from './components/Sidebar';
import { FilterBar, FilterState } from './components/FilterBar';
import { OverviewPage } from './pages/OverviewPage';
import { AnomalyQueuePage } from './pages/AnomalyQueuePage';
import { CaseDetailPage } from './pages/CaseDetailPage';
import { ReportBuilderPage } from './pages/ReportBuilderPage';
import { SettingsPage } from './pages/SettingsPage';
import { mockCases, mockAlerts } from './data/mockData';

export default function App() {
  const [currentPage, setCurrentPage] = useState<string>('overview');
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  
  const [filters, setFilters] = useState<FilterState>({
    dateRange: 'today',
    line: 'all',
    productGroup: 'all',
    defectType: 'all',
    decision: 'all',
    scoreRange: [0, 1]
  });
  
  // Filter cases based on current filters
  const filteredCases = useMemo(() => {
    return mockCases.filter(c => {
      if (filters.line !== 'all' && c.line_id !== filters.line) return false;
      if (filters.productGroup !== 'all' && c.product_group !== filters.productGroup) return false;
      if (filters.defectType !== 'all' && c.defect_type !== filters.defectType) return false;
      if (filters.decision !== 'all' && c.decision !== filters.decision) return false;
      if (c.anomaly_score < filters.scoreRange[0] || c.anomaly_score > filters.scoreRange[1]) return false;
      return true;
    });
  }, [filters]);
  
  // Handle navigation
  const handleNavigate = (page: string) => {
    setCurrentPage(page);
    setSelectedCaseId(null);
  };
  
  // Handle case click - navigate to detail
  const handleCaseClick = (caseId: string) => {
    setSelectedCaseId(caseId);
    setCurrentPage('detail');
  };
  
  // Handle filter updates from charts or other components
  const handleFilterUpdate = (newFilters: Partial<FilterState>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
    setCurrentPage('queue');
  };
  
  // Get current case data
  const currentCase = selectedCaseId 
    ? mockCases.find(c => c.id === selectedCaseId) 
    : null;
  
  // Render current page
  const renderPage = () => {
    if (currentPage === 'detail' && currentCase) {
      return (
        <CaseDetailPage 
          caseData={currentCase} 
          onBack={() => setCurrentPage('queue')} 
        />
      );
    }
    
    switch (currentPage) {
      case 'overview':
        return (
          <OverviewPage 
            cases={filteredCases}
            alerts={mockAlerts}
            filters={filters}
            onCaseClick={handleCaseClick}
            onFilterUpdate={handleFilterUpdate}
          />
        );
      case 'queue':
        return (
          <AnomalyQueuePage 
            cases={filteredCases}
            onCaseClick={handleCaseClick}
          />
        );
      case 'report':
        return (
          <ReportBuilderPage 
            cases={filteredCases}
          />
        );
      case 'settings':
        return <SettingsPage />;
      default:
        return (
          <OverviewPage 
            cases={filteredCases}
            alerts={mockAlerts}
            filters={filters}
            onCaseClick={handleCaseClick}
            onFilterUpdate={handleFilterUpdate}
          />
        );
    }
  };
  
  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <Sidebar currentPage={currentPage} onNavigate={handleNavigate} />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Filter Bar - only show on overview and queue pages */}
        {(currentPage === 'overview' || currentPage === 'queue') && (
          <FilterBar filters={filters} onFilterChange={setFilters} />
        )}
        
        {/* Page Content */}
        <div className="flex-1 overflow-y-auto">
          {renderPage()}
        </div>
      </div>
    </div>
  );
}