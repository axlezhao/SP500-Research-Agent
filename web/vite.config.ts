import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the API runs separately (`sp500-web --reload`); Vite proxies /api to it.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
  build: {
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (/node_modules\/(recharts|d3-|victory-vendor|lightweight-charts)/.test(id)) return "charts";
          if (/node_modules\/(react-markdown|remark-|micromark|mdast-|unified|hast-|unist-)/.test(id)) return "markdown";
          return undefined;
        },
      },
    },
  },
});
