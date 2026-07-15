const ESCAPE_MAP: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;"
};

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, char => ESCAPE_MAP[char] ?? char);
}

function sanitizeUrl(url: string): string {
  const trimmed = url.trim();
  if (/^(https?:|mailto:)/i.test(trimmed) || trimmed.startsWith("/") || trimmed.startsWith("#")) {
    return trimmed;
  }
  return "#";
}

function renderInline(text: string): string {
  const codeSpans: string[] = [];

  let out = text.replace(/`([^`]+)`/g, (_match, code: string) => {
    codeSpans.push("<code>" + escapeHtml(code) + "</code>");
    return " " + (codeSpans.length - 1) + " ";
  });

  out = escapeHtml(out);

  out = out.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_match, label: string, url: string) => {
    return '<a href="' + escapeHtml(sanitizeUrl(url)) + '" target="_blank" rel="noopener noreferrer">' + label + "</a>";
  });

  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/\*([^*]+)\*/g, "<em>$1</em>");

  out = out.replace(/ (\d+) /g, (_match, index: string) => codeSpans[Number(index)] ?? "");

  return out;
}

export function renderMarkdown(markdown: string): string {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const blocks: string[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];

    const fence = line.match(/^```(\w*)\s*$/);
    if (fence) {
      const language = fence[1];
      const buffer: string[] = [];
      index += 1;
      while (index < lines.length && !/^```\s*$/.test(lines[index])) {
        buffer.push(lines[index]);
        index += 1;
      }
      index += 1;
      const className = language ? ' class="language-' + escapeHtml(language) + '"' : "";
      blocks.push("<pre><code" + className + ">" + escapeHtml(buffer.join("\n")) + "</code></pre>");
      continue;
    }

    if (line.trim() === "") {
      index += 1;
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      const level = heading[1].length;
      blocks.push("<h" + level + ">" + renderInline(heading[2]) + "</h" + level + ">");
      index += 1;
      continue;
    }

    if (/^>\s?/.test(line)) {
      const buffer: string[] = [];
      while (index < lines.length && /^>\s?/.test(lines[index])) {
        buffer.push(lines[index].replace(/^>\s?/, ""));
        index += 1;
      }
      blocks.push("<blockquote>" + renderInline(buffer.join(" ")) + "</blockquote>");
      continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) {
        items.push("<li>" + renderInline(lines[index].replace(/^\s*[-*]\s+/, "")) + "</li>");
        index += 1;
      }
      blocks.push("<ul>" + items.join("") + "</ul>");
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\s*\d+\.\s+/.test(lines[index])) {
        items.push("<li>" + renderInline(lines[index].replace(/^\s*\d+\.\s+/, "")) + "</li>");
        index += 1;
      }
      blocks.push("<ol>" + items.join("") + "</ol>");
      continue;
    }

    const paragraph: string[] = [];
    while (
      index < lines.length &&
      lines[index].trim() !== "" &&
      !/^```/.test(lines[index]) &&
      !/^(#{1,6})\s+/.test(lines[index]) &&
      !/^>\s?/.test(lines[index]) &&
      !/^\s*[-*]\s+/.test(lines[index]) &&
      !/^\s*\d+\.\s+/.test(lines[index])
    ) {
      paragraph.push(lines[index]);
      index += 1;
    }
    blocks.push("<p>" + renderInline(paragraph.join(" ")) + "</p>");
  }

  return blocks.join("\n");
}
