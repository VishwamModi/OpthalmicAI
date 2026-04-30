'use client';

import { useState } from 'react';
import { HomePage } from './home-page';
import { LoginPage } from './login-page';
import { ClinicalDashboard } from './clinical-dashboard-scroll';

type AppView = 'home' | 'login' | 'dashboard';

export function RetinaAIApp() {
  const [currentView, setCurrentView] = useState<AppView>('home');

  const handleNavigate = (view: AppView) => {
    setCurrentView(view);
  };

  return (
    <>
      {currentView === 'home' && <HomePage onNavigate={handleNavigate} />}
      {currentView === 'login' && <LoginPage onNavigate={handleNavigate} />}
      {currentView === 'dashboard' && <ClinicalDashboard onLogout={() => handleNavigate('home')} />}
    </>
  );
}
