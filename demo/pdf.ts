import {
  GlobalWorkerOptions,
  getDocument,
  type PDFDocumentProxy
} from "pdfjs-dist";

// Vite resolves the worker bundle to a URL at build time. Pointing
// GlobalWorkerOptions at it keeps text extraction off the main thread.
import workerSrc from "pdfjs-dist/build/pdf.worker.min.mjs?url";

GlobalWorkerOptions.workerSrc = workerSrc;

export interface ExtractedPdf {
  text: string;
  pageCount: number;
  fileName: string;
}

// Pull text out of every page of a PDF. pdf.js decodes each page on its own
// worker, so this stays responsive even on multi-megabyte documents. We cap the
// returned length so a huge scan-heavy PDF cannot blow the model context.
const MAX_CHARS = 200_000;

export async function extractPdfText(file: File): Promise<ExtractedPdf> {
  const buffer = await file.arrayBuffer();
  const document: PDFDocumentProxy = await getDocument({ data: buffer }).promise;

  const parts: string[] = [];
  let total = 0;
  let truncated = false;

  for (let pageNumber = 1; pageNumber <= document.numPages; pageNumber += 1) {
    const page = await document.getPage(pageNumber);
    const content = await page.getTextContent();
    const pageText = content.items
      .map(item => ("str" in item ? item.str : ""))
      .join(" ");
    parts.push(pageText);
    total += pageText.length + 1;
    if (total >= MAX_CHARS) {
      truncated = true;
      break;
    }
  }

  await document.destroy();

  let text = parts.join("\n\n");
  if (truncated) {
    text = text.slice(0, MAX_CHARS) +
      "\n\n[PDF truncated to stay within context limits]";
  }

  return {
    text,
    pageCount: document.numPages,
    fileName: file.name
  };
}
