import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import cesium from 'vite-plugin-cesium';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), cesium()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
        // The app uses no cookies, but the browser sends every localhost
        // cookie from other projects. A large Cookie header exceeds the
        // backend WebSocket server's 8 KB header-line limit and the
        // handshake is rejected with 400, so strip it here.
        configure: (proxy) => {
          proxy.on('proxyReqWs', (proxyReq) => proxyReq.removeHeader('cookie'));
        },
      },
    },
  },
})
