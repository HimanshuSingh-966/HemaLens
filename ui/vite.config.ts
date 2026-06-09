import { defineConfig } from 'vite';
import path from 'path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],

  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },

  server: {
    host: true,  // ⬅ REQUIRED for cloud IDEs
    allowedHosts: [
      "5173-01khth4np36nvzmea2psj0r2xr.cloudspaces.litng.ai"  // ⬅ add your cloud host here
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