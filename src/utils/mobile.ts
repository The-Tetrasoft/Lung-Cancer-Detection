// Mobile utility functions for Android compatibility

// Type declarations for browser APIs
interface NavigatorWithMSTouch extends Navigator {
  msMaxTouchPoints?: number;
}

interface NavigatorWithStandalone extends Navigator {
  standalone?: boolean;
}

interface NavigatorWithConnection extends Navigator {
  connection?: {
    effectiveType?: string;
    downlink?: number;
    rtt?: number;
  };
  mozConnection?: {
    effectiveType?: string;
    downlink?: number;
    rtt?: number;
  };
  webkitConnection?: {
    effectiveType?: string;
    downlink?: number;
    rtt?: number;
  };
}

interface NavigatorWithHaptics extends Navigator {
  haptics?: {
    vibrate: (type: string) => void;
  };
}

/**
 * Detects if the device is mobile
 */
export function isMobileDevice(): boolean {
  if (typeof window === 'undefined') return false;

  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
  ) || window.innerWidth <= 768;
}

/**
 * Detects if the device is Android
 */
export function isAndroidDevice(): boolean {
  if (typeof window === 'undefined') return false;

  return /Android/i.test(navigator.userAgent);
}

/**
 * Detects if the device supports touch
 */
export function isTouchDevice(): boolean {
  if (typeof window === 'undefined') return false;

  const nav = navigator as NavigatorWithMSTouch;
  return 'ontouchstart' in window ||
    navigator.maxTouchPoints > 0 ||
    (nav.msMaxTouchPoints ?? 0) > 0;
}

/**
 * Gets the device's safe area insets
 */
export function getSafeAreaInsets() {
  if (typeof window === 'undefined') {
    return { top: 0, right: 0, bottom: 0, left: 0 };
  }

  const style = getComputedStyle(document.documentElement);

  return {
    top: parseInt(style.getPropertyValue('env(safe-area-inset-top)') || '0'),
    right: parseInt(style.getPropertyValue('env(safe-area-inset-right)') || '0'),
    bottom: parseInt(style.getPropertyValue('env(safe-area-inset-bottom)') || '0'),
    left: parseInt(style.getPropertyValue('env(safe-area-inset-left)') || '0'),
  };
}

/**
 * Prevents the default behavior of certain mobile gestures
 */
export function preventMobileGestures() {
  if (typeof window === 'undefined') return;

  // Prevent pull-to-refresh
  document.body.style.overscrollBehavior = 'none';

  // Prevent zoom on double tap
  let lastTouchEnd = 0;
  document.addEventListener('touchend', (event) => {
    const now = new Date().getTime();
    if (now - lastTouchEnd <= 300) {
      event.preventDefault();
    }
    lastTouchEnd = now;
  }, { passive: false });

  // Prevent context menu on long press
  document.addEventListener('contextmenu', (event) => {
    if (isTouchDevice()) {
      event.preventDefault();
    }
  });
}

/**
 * Adds haptic feedback if supported
 */
export function hapticFeedback(type: 'light' | 'medium' | 'heavy' = 'light') {
  if (typeof window === 'undefined') return;

  // Web Vibration API
  if ('vibrate' in navigator) {
    const duration = type === 'light' ? 10 : type === 'medium' ? 20 : 50;
    navigator.vibrate(duration);
  }

  // Future: Web Haptics API when available
  const nav = navigator as NavigatorWithHaptics;
  if (nav.haptics) {
    nav.haptics.vibrate(type);
  }
}

/**
 * Sets up mobile app-like behavior
 */
export function setupMobileApp() {
  if (typeof window === 'undefined') return;

  // Prevent gestures
  preventMobileGestures();

  // Add mobile meta tags if not present
  const addMetaTag = (name: string, content: string) => {
    if (!document.querySelector(`meta[name="${name}"]`)) {
      const meta = document.createElement('meta');
      meta.name = name;
      meta.content = content;
      document.head.appendChild(meta);
    }
  };

  addMetaTag('viewport', 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover');
  addMetaTag('mobile-web-app-capable', 'yes');
  addMetaTag('apple-mobile-web-app-capable', 'yes');
  addMetaTag('apple-mobile-web-app-status-bar-style', 'black-translucent');
  addMetaTag('theme-color', '#ffffff');

  // Add touch-action CSS
  document.documentElement.style.touchAction = 'manipulation';
}

/**
 * Mobile-optimized scroll behavior
 */
export function setupMobileScroll() {
  if (typeof window === 'undefined') return;

  // Smooth scrolling for mobile
  document.documentElement.style.scrollBehavior = 'smooth';

  // Momentum scrolling for iOS
  (document.body.style as any).webkitOverflowScrolling = 'touch';

  // Prevent overscroll
  document.body.style.overscrollBehavior = 'none';
}

/**
 * Gets the current device orientation
 */
export function getDeviceOrientation(): 'portrait' | 'landscape' {
  if (typeof window === 'undefined') return 'portrait';

  return window.innerHeight > window.innerWidth ? 'portrait' : 'landscape';
}

/**
 * Checks if the device is in standalone mode (PWA)
 */
export function isStandaloneMode(): boolean {
  if (typeof window === 'undefined') return false;

  const nav = navigator as NavigatorWithStandalone;
  return nav.standalone === true ||
    window.matchMedia('(display-mode: standalone)').matches;
}

/**
 * Mobile-friendly localStorage with error handling
 */
export const mobileStorage = {
  setItem: (key: string, value: string): boolean => {
    try {
      localStorage.setItem(key, value);
      return true;
    } catch (error) {
      console.warn('Failed to save to localStorage:', error);
      return false;
    }
  },

  getItem: (key: string): string | null => {
    try {
      return localStorage.getItem(key);
    } catch (error) {
      console.warn('Failed to read from localStorage:', error);
      return null;
    }
  },

  removeItem: (key: string): boolean => {
    try {
      localStorage.removeItem(key);
      return true;
    } catch (error) {
      console.warn('Failed to remove from localStorage:', error);
      return false;
    }
  }
};

/**
 * Mobile network status detection
 */
export function getNetworkStatus() {
  if (typeof navigator === 'undefined') {
    return { online: true, effectiveType: 'unknown' };
  }

  const nav = navigator as NavigatorWithConnection;
  const connection = nav.connection || nav.mozConnection || nav.webkitConnection;

  return {
    online: navigator.onLine,
    effectiveType: connection?.effectiveType || 'unknown',
    downlink: connection?.downlink || 0,
    rtt: connection?.rtt || 0
  };
}