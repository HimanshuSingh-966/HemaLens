import { defineConfig } from 'vite';
import path from 'path';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],

  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },

  build: {
    outDir: 'dist',
    sourcemap: false,
  },

  server: {
    host: true,
    allowedHosts: [
      "5173-01khth4np36nvzmea2psj0r2xr.cloudspaces.litng.ai"
    ],
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },

  assetsInclude: ['**/*.svg', '**/*.csv'],
});