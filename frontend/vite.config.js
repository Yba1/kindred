import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import { fileURLToPath } from 'node:url'

// The village viz is workstream D's file. We mount it, we never edit it.
const VILLAGE = fileURLToPath(new URL('../viz/village.html', import.meta.url))
const VILLAGE_PATHS = ['/village', '/village/', '/village.html']

function villageRoute() {
  return {
    name: 'kindred-village-route',
    // dev: serve ../viz/village.html verbatim at /village
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = req.url.split('?')[0]
        if (!VILLAGE_PATHS.includes(path)) return next()
        res.setHeader('Content-Type', 'text/html; charset=utf-8')
        res.end(fs.readFileSync(VILLAGE, 'utf8'))
      })
    },
    // build: copy it into dist so /village works in preview + deploy
    generateBundle() {
      const source = fs.readFileSync(VILLAGE, 'utf8')
      this.emitFile({ type: 'asset', fileName: 'village.html', source })
      this.emitFile({ type: 'asset', fileName: 'village/index.html', source })
    },
  }
}

export default defineConfig({
  plugins: [react(), villageRoute()],
  server: {
    port: 5173,
    // Backend (workstream A) runs on :8000. Until it exists, the app falls back
    // to public/sample_graph.json on its own — see src/data/loadGraph.js.
    proxy: {
      '/profile': 'http://localhost:8000',
      '/graph': 'http://localhost:8000',
      '/agent-stream': 'http://localhost:8000',
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.js'],
  },
})
