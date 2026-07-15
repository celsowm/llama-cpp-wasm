import { defineConfig } from "vite";
import { resolve } from "node:path";

function normalizeBasePath(value: string | undefined): string {
  if (!value || value === "/") {
    return "/";
  }

  const path = value.replace(/^\/+|\/+$/g, "");
  return path.length > 0 ? `/${path}/` : "/";
}

export default defineConfig(({ command }) => ({
  // actions/configure-pages supplies the repository path at build time.
  // Local development remains available at the origin root.
  base:
    command === "build"
      ? normalizeBasePath(process.env.VITE_BASE_PATH)
      : "/",

  root: resolve(__dirname, "demo"),
  publicDir: resolve(__dirname, "demo/public"),

  server: {
    port: 5173,
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp"
    },
    fs: {
      allow: [resolve(__dirname)]
    }
  },

  build: {
    outDir: resolve(__dirname, "dist-demo"),
    emptyOutDir: true
  }
}));
