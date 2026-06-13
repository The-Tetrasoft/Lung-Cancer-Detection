import React, { ReactNode } from 'react';

interface MobileLayoutProps {
  children: ReactNode;
  className?: string;
  withSafeArea?: boolean;
  withPadding?: boolean;
}

export function MobileLayout({
  children,
  className = '',
  withSafeArea = true,
  withPadding = true
}: MobileLayoutProps) {
  return (
    <div
      className={`
        w-full min-h-screen
        ${withSafeArea ? 'mobile-safe-area' : ''}
        ${withPadding ? 'mobile-spacing' : ''}
        ${className}
      `}
    >
      {children}
    </div>
  );
}

// Mobile-optimized container component
export function MobileContainer({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <div className={`max-w-sm mx-auto px-4 py-6 ${className}`}>
      {children}
    </div>
  );
}

// Mobile-friendly card component
export function MobileCard({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <div className={`
      bg-white/80 backdrop-blur-sm rounded-xl border-blue-100 shadow-md p-6 
      mobile-spacing touch-manipulation
      ${className}
    `}>
      {children}
    </div>
  );
}

// Mobile-optimized button group
export function MobileButtonGroup({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <div className={`flex flex-col space-y-3 w-full ${className}`}>
      {children}
    </div>
  );
}

// Mobile-friendly form layout
export function MobileForm({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <form className={`space-y-4 w-full ${className}`}>
      {children}
    </form>
  );
}