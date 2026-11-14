import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// 5180 because 5173 is often taken. /api is proxied to the FastAPI backend on 8010.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    strictPort: true,
    proxy: { '/api': 'http://localhost:8010' },
  },
})
