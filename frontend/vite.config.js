import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  test: {
    // jsdom rather than a browser: these tests are about what the components
    // decide -- which request goes out, which verdict is shown -- not about
    // how they paint. The browser check that the app talks to a real backend
    // stays a manual step, because faking it here would test the fake.
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.js"],
  },
});
