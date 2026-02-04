import React from 'react';
import { LucideIcon } from 'lucide-react';

interface KPICardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  icon?: LucideIcon;
}

export function KPICard({ title, value, subtitle, trend, icon: Icon }: KPICardProps) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm text-gray-600 mb-1">{title}</p>
          <p className="text-3xl font-semibold text-gray-900 mb-2">{value}</p>
          {subtitle && (
            <p className="text-sm text-gray-500">{subtitle}</p>
          )}
          {trend && (
            <div className={`inline-flex items-center text-sm mt-2 ${
              trend.isPositive ? 'text-green-600' : 'text-red-600'
            }`}>
              <span>{trend.isPositive ? '↓' : '↑'} {Math.abs(trend.value)}%</span>
              <span className="ml-1 text-gray-500">vs 전일</span>
            </div>
          )}
        </div>
        {Icon && (
          <div className="p-3 bg-gray-50 rounded-lg">
            <Icon className="w-6 h-6 text-gray-600" />
          </div>
        )}
      </div>
    </div>
  );
}
