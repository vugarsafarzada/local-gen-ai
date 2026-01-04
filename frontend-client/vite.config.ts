import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: '.', // Ensure root is correct
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000', // Your FastAPI backend
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/outputs': { // New rule for images
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        about: resolve(__dirname, 'about/index.html'),
        generator: resolve(__dirname, 'image-generator/index.html'),
      },
    },
  },
});