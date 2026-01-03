import { defineConfig } from 'vite'

export default defineConfig({
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
})
