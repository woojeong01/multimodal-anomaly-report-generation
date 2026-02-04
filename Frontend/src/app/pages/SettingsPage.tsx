import React from "react";
import { Settings, Users, Bell, Database } from "lucide-react";

export function SettingsPage() {
  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">
          설정
        </h1>
        <p className="text-sm text-gray-600">
          시스템 설정 및 환경 구성
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Model Settings */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Database className="w-5 h-5 text-blue-600" />
            </div>
            <h2 className="text-lg font-medium text-gray-900">
              모델 설정
            </h2>
          </div>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                기본 임계값 (Threshold)
              </label>
              <input
                type="number"
                step="0.01"
                defaultValue="0.65"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">
                이상 탐지 기준 점수 (0.0 - 1.0)
              </p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                활성 모델
              </label>
              <select className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500">
                <option value="efficientad">
                  EfficientAD v3.1.0
                </option>
                <option value="efficientad">
                  PatchCore v3.1.0
                </option>
              </select>
            </div>
          </div>
        </div>

        {/* Notification Settings */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-amber-100 rounded-lg">
              <Bell className="w-5 h-5 text-amber-600" />
            </div>
            <h2 className="text-lg font-medium text-gray-900">
              알림 설정
            </h2>
          </div>
          <div className="space-y-3">
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                defaultChecked
                className="w-4 h-4 text-blue-600 rounded"
              />
              <span className="text-sm text-gray-700">
                불량률 급증 알림
              </span>
            </label>
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                defaultChecked
                className="w-4 h-4 text-blue-600 rounded"
              />
              <span className="text-sm text-gray-700">
                새로운 결함 패턴 감지
              </span>
            </label>
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                className="w-4 h-4 text-blue-600 rounded"
              />
              <span className="text-sm text-gray-700">
                라인별 이상 알림
              </span>
            </label>
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                defaultChecked
                className="w-4 h-4 text-blue-600 rounded"
              />
              <span className="text-sm text-gray-700">
                일일 리포트 이메일
              </span>
            </label>
          </div>
        </div>

        {/* User Management */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-green-100 rounded-lg">
              <Users className="w-5 h-5 text-green-600" />
            </div>
            <h2 className="text-lg font-medium text-gray-900">
              사용자 관리
            </h2>
          </div>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div>
                <p className="text-sm font-medium text-gray-900">
                  손우정
                </p>
                <p className="text-xs text-gray-500">
                  품질관리팀 · 관리자
                </p>
              </div>
              <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                활성
              </span>
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div>
                <p className="text-sm font-medium text-gray-900">
                  노성호
                </p>
                <p className="text-xs text-gray-500">
                  품질관리팀 · 검사원
                </p>
              </div>
              <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                활성
              </span>
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div>
                <p className="text-sm font-medium text-gray-900">
                  문국현
                </p>
                <p className="text-xs text-gray-500">
                  생산팀 · 검사원
                </p>
              </div>
              <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                활성
              </span>
            </div>
          </div>
        </div>

        {/* System Info */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Settings className="w-5 h-5 text-purple-600" />
            </div>
            <h2 className="text-lg font-medium text-gray-900">
              시스템 정보
            </h2>
          </div>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">버전</span>
              <span className="font-mono text-gray-900">
                v2.3.1
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">
                마지막 업데이트
              </span>
              <span className="text-gray-900">2026-02-01</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">
                데이터베이스 크기
              </span>
              <span className="text-gray-900">2.4 GB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">활성 라인</span>
              <span className="text-gray-900">3개</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}