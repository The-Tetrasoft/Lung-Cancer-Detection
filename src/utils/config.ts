import { Capacitor } from '@capacitor/core';

// Backend base URL - detects if running on Android Emulator or Web
// Notes:
// - On Android emulators the host machine is reachable at 10.0.2.2 (Android Emulator) or 10.0.3.2 (Genymotion).
// - For physical devices use an IP accessible by the device and set VITE_BACKEND_BASE accordingly.
const platform = Capacitor.getPlatform();
const envAny = (import.meta as any).env ?? {};

// Allow an explicit Android backend override (useful when running on device)
const androidOverride = envAny.VITE_BACKEND_BASE_ANDROID;

export const BACKEND_BASE = (() => {
    if (platform === 'android') {
        // Use local IP for physical device support
        return 'http://192.168.0.100:8000';
    }

    // Web / desktop fallback
    return envAny.VITE_BACKEND_BASE ?? 'http://localhost:8000';
})();
