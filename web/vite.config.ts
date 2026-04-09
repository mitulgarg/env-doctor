import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../src/env_doctor/web",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://localhost:8765",
    },
  },
});
