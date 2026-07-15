import { chromium } from "playwright";
import http from "node:http";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("../../dist-demo/", import.meta.url));
const MODEL = process.env.MODEL_PATH ?? "";

if (!MODEL) {
  throw new Error(
    "Set MODEL_PATH to a local GGUF file before running the smoke test, " +
      "e.g. MODEL_PATH=/path/to/LFM2.5-230M-Q4_K_M.gguf npm run test:browser"
  );
}

const consoleErrors = [];
const pageErrors = [];
const crashEvents = [];

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".wasm": "application/wasm",
  ".json": "application/json; charset=utf-8",
  ".gguf": "application/octet-stream"
};

function startServer() {
  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url, "http://localhost");
      let pathname = decodeURIComponent(url.pathname);
      if (pathname === "/") pathname = "/index.html";
      const rel = pathname.replace(/^\/+/, "");
      const filePath = normalize(join(ROOT, rel));
      console.log("REQ", pathname, "->", filePath, "exists:", existsSync(filePath));
      if (!filePath.startsWith(ROOT) || !existsSync(filePath)) {
        res.writeHead(404);
        res.end("Not found");
        return;
      }
      const body = await readFile(filePath);
      res.writeHead(200, {
        "Content-Type": MIME[extname(filePath)] ?? "application/octet-stream",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Resource-Policy": "same-origin"
      });
      res.end(body);
    } catch (err) {
      res.writeHead(500);
      res.end(String(err));
    }
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      resolve({ server, port });
    });
  });
}

async function main() {
  if (!existsSync(MODEL)) {
    throw new Error(`Model not found: ${MODEL}`);
  }

  const { server, port } = await startServer();
  const base = `http://127.0.0.1:${port}/`;
  console.log(`Serving ${ROOT} at ${base}`);

  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-dev-shm-usage"]
  });

  const page = await browser.newPage();

  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (err) => pageErrors.push(String(err)));
  page.on("crash", () => crashEvents.push("page crashed"));

  const memorySamples = [];
  const memoryTimer = setInterval(async () => {
    try {
      const used = await page.evaluate(() => performance.memory?.usedJSHeapSize ?? 0);
      memorySamples.push(used);
    } catch {
      /* ignore */
    }
  }, 250);

  const t0 = Date.now();
  await page.goto(base, { waitUntil: "load" });

  console.log("DEBUG url:", page.url());
  console.log("DEBUG has #model:", await page.evaluate(() => !!document.getElementById("model")));
  console.log("DEBUG content length:", (await page.content()).length);
  console.log("DEBUG title:", await page.title());

  await page.setInputFiles("#model", MODEL);
  await page.click("#load");

  try {
    await page.waitForFunction(
      () => /Loaded/.test(document.getElementById("status")?.textContent ?? ""),
      { timeout: 180000 }
    );
  } catch (err) {
    console.log("LOAD FAILED. status:", await page.textContent("#status"));
    throw err;
  }
  const loadMs = Date.now() - t0;
  const loadStatus = await page.textContent("#status");

  const genStart = Date.now();
  await page.fill("#prompt", "What is a GGUF file? Answer in one sentence.");
  await page.click(process.env.CHAT_MODE ? "#generate" : "#generateRaw");

  try {
    await page.waitForFunction(
      () => /Generation complete/.test(document.getElementById("status")?.textContent ?? ""),
      { timeout: 180000 }
    );
  } catch (err) {
    console.log("GENERATE FAILED. status:", await page.textContent("#status"));
    console.log("partial output:", JSON.stringify((await page.textContent("#output")) ?? ""));
    throw err;
  }
  const genMs = Date.now() - genStart;
  const output = await page.textContent("#output");
  clearInterval(memoryTimer);

  await browser.close();
  server.close();

  const peakHeap = Math.max(0, ...memorySamples);
  console.log("\n===== SMOKE TEST RESULT =====");
  console.log("model load status:", loadStatus?.trim());
  console.log("model load time (ms):", loadMs);
  console.log("generation time (ms):", genMs);
  console.log("output chars:", output?.length ?? 0);
  console.log("output sample:", JSON.stringify((output ?? "").slice(0, 200)));
  console.log("peak JS heap (bytes):", peakHeap);
  console.log("console errors:", consoleErrors.length);
  consoleErrors.slice(0, 10).forEach((e) => console.log("  !", e));
  console.log("page errors:", pageErrors.length);
  pageErrors.slice(0, 10).forEach((e) => console.log("  !", e));
  console.log("crash events:", crashEvents.length);
  crashEvents.forEach((e) => console.log("  !", e));

  const realErrors = consoleErrors.filter((e) => !e.startsWith("[llama.cpp]"));
  console.log("real console errors (excl. llama.cpp logs):", realErrors.length);
  realErrors.slice(0, 20).forEach((e) => console.log("  !", e));

  const ok =
    (output?.length ?? 0) > 0 &&
    pageErrors.length === 0 &&
    crashEvents.length === 0 &&
    realErrors.length === 0;
  console.log("RESULT:", ok ? "PASS" : "FAIL");
  process.exit(ok ? 0 : 1);
}

main().catch((err) => {
  console.error("Smoke test failed:", err);
  console.log("console errors:", consoleErrors.length);
  consoleErrors
    .filter((e) => /map::|wasm|llama_|lcw_|LCW_|TEMPLATE| at |Exception|abort|Assertion|\.cpp/i.test(e))
    .slice(0, 40)
    .forEach((e) => console.log("  !", e));
  console.log("page errors:", pageErrors.length);
  pageErrors.slice(0, 10).forEach((e) => console.log("  !", e));
  console.log("crash events:", crashEvents.length);
  crashEvents.forEach((e) => console.log("  !", e));
  process.exit(1);
});
