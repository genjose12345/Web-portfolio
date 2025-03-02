import React from 'react';
import { ParticleBackground } from './Background';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  return (
    <div className="min-h-screen bg-gray-900 text-white relative">
      <ParticleBackground />
      {children}
    </div>
  );
};

export default Layout; 