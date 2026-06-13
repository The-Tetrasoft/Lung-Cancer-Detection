// Ambient declarations to satisfy TS when compiling under Node with moduleResolution Bundler
declare module 'multer';

// Provide minimal typings for global fetch-related classes used at runtime (Node 18+)
// These are available at runtime via undici; typings here prevent TS errors.
interface Blob {}
interface FormData {
  append(name: string, value: any, fileName?: string): void;
}
declare const Blob: {
  new (parts?: any[], options?: any): Blob;
};
declare const FormData: {
  new (): FormData;
};