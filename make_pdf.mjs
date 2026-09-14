#!/usr/bin/env node
// Compile the published pages of the built site (__site) into one printable PDF.
//
// "Published" = the pages linked from the side-bar menu in __site/index.html,
// in menu order (entries commented out in _layout/pgwrap.html are skipped).
//
// Usage:
//   node make_pdf.mjs [-o enve160b.pdf] [--a4] [--no-urls]
//
// Build the site first (e.g. julia -e 'using Franklin; optimize()').
// Needs Node >= 18 and Google Chrome / Chromium (set CHROME=/path/to/chrome to override).

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath, pathToFileURL } from "node:url";

const run = promisify(execFile);
const root = path.dirname(fileURLToPath(import.meta.url));
const site = path.join(root, "__site");

// ------------------------------------------------------------------ options
const argv = process.argv.slice(2);
const opt = { out: path.join(root, "enve160b.pdf"), paper: "letter", urls: true };
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === "-o" || a === "--output") opt.out = path.resolve(argv[++i]);
  else if (a === "--a4") opt.paper = "A4";
  else if (a === "--no-urls") opt.urls = false;
  else if (a === "-h" || a === "--help") {
    console.log("usage: node make_pdf.mjs [-o file.pdf] [--a4] [--no-urls]");
    process.exit(0);
  } else {
    console.error(`unknown argument: ${a}`);
    process.exit(1);
  }
}

// ------------------------------------------------------------------ config
const config = fs.readFileSync(path.join(root, "config.md"), "utf8");
const cfg = (re, fallback) => (config.match(re) || [])[1] ?? fallback;
const prepath = cfg(/@def\s+prepath\s*=\s*"([^"]*)"/, "");
const siteUrl = cfg(/website_url\s*=\s*"([^"]*)"/, "").replace(/\/?$/, "/");
const author = cfg(/author\s*=\s*"([^"]*)"/, "");
const pre = prepath ? `/${prepath}/` : "/";

// ------------------------------------------------------------------ pages
const indexHtml = fs.readFileSync(path.join(site, "index.html"), "utf8");
const menu = (indexHtml.match(/<ul class="menu-list">([\s\S]*?)<\/ul>/) || [])[1];
if (!menu) throw new Error("could not find the side-bar menu in __site/index.html");

const pages = [...menu.replace(/<!--[\s\S]*?-->/g, "").matchAll(/<a\s[^>]*href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/g)]
  .map(([, href, label]) => {
    const rel = href.startsWith(pre) ? href.slice(pre.length) : href.replace(/^\//, "");
    const dir = rel.replace(/\/$/, "");
    return {
      slug: dir || "index",
      dir,
      label: label.replace(/<[^>]+>/g, "").trim(),
      file: path.join(site, dir, "index.html"),
    };
  });

const published = new Set(pages.map((p) => p.dir));
const fileUrl = (p) => pathToFileURL(p).href;
const escapeHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

// Map a site-absolute URL (e.g. /enve160b/chamber/#methods) to a target in the PDF.
function resolveHref(href) {
  const [p, hash = ""] = href.slice(pre.length).split("#");
  const dir = p.replace(/\/?(index\.html)?$/, "");
  if (published.has(dir)) return `#${dir || "index"}--${hash || "top"}`;
  return siteUrl + href.slice(pre.length); // unpublished page or downloadable asset
}

function chapterHtml(page) {
  const html = fs.readFileSync(page.file, "utf8");
  const start = html.indexOf('<div class="franklin-content">');
  if (start < 0) throw new Error(`no franklin-content in ${page.file}`);
  let end = html.indexOf('<div class="page-foot">', start);
  if (end < 0) end = html.indexOf("<!-- CONTENT ENDS HERE -->", start);
  let body = html.slice(start + '<div class="franklin-content">'.length, end);

  // Embedded videos cannot be printed: replace them with a link.
  body = body.replace(/<iframe\s([^>]*)>[\s\S]*?<\/iframe>/g, (_, attrs) => {
    let src = (attrs.match(/src="([^"]*)"/) || [])[1] || "";
    const title = (attrs.match(/title="([^"]*)"/) || [])[1] || "Embedded content";
    const yt = src.match(/youtube\.com\/embed\/([\w-]+)/);
    if (yt) src = `https://www.youtube.com/watch?v=${yt[1]}`;
    return `<span class="embed">▶ Video: <a href="${src}">${title}</a></span>`;
  });

  // Keep anchors unique across pages and point cross-page links into the PDF.
  const pageUrl = siteUrl + (page.dir ? `${page.dir}/` : "");
  body = body
    .replace(/\sid="([^"]+)"/g, ` id="${page.slug}--$1"`)
    .replace(/\shref="#([^"]*)"/g, ` href="#${page.slug}--$1"`)
    .replace(/\shref="(\/[^"]*)"/g, (_, h) => ` href="${h.startsWith(pre) ? resolveHref(h) : siteUrl + h.slice(1)}"`)
    .replace(/\shref="(?![a-z][\w+.-]*:|#|\/)([^"]+)"/gi, (_, h) => ` href="${new URL(h, pageUrl).href}"`)
    .replace(/\ssrc="(\/[^"]*)"/g, (_, s) => ` src="${fileUrl(path.join(site, s.startsWith(pre) ? s.slice(pre.length) : s.slice(1)))}"`);

  // Short table cells (dates, percentages) should not wrap in narrow columns.
  body = body.replace(/<(td|th)(\s[^>]*)?>([\s\S]*?)<\/\1>/g, (m, tag, attrs = "", inner) =>
    inner.replace(/<[^>]+>/g, "").trim().length <= 8 ? `<${tag} class="nowrap"${attrs}>${inner}</${tag}>` : m);

  // Don't print the URL after a link whose text already is the URL.
  body = body.replace(/<a\s([^>]*)>([\s\S]*?)<\/a>/g, (m, attrs, inner) => {
    const href = (attrs.match(/href="([^"]*)"/) || [])[1] || "";
    const text = inner.replace(/<[^>]+>/g, "").trim();
    const bare = (u) => u.replace(/^https?:\/\/(www\.)?/, "").replace(/\/$/, "");
    if (/^https?:/.test(href) && bare(text) === bare(href)) return `<a class="bare" ${attrs}>${inner}</a>`;
    return m;
  });

  return `<section class="chapter" id="${page.slug}--top">
  <div class="chapter-label">${escapeHtml(page.label)}</div>
  <div class="franklin-content">${body}</div>
</section>`;
}

// ------------------------------------------------------------------ print styles
const css = `
@page {
  size: ${opt.paper};
  margin: 0.75in 0.8in 0.8in 0.8in;
  @bottom-left  { content: "ENVE 160B · Environmental Engineering Laboratory"; font: 8pt Helvetica, Arial, sans-serif; color: #777; }
  @bottom-right { content: counter(page); font: 8pt Helvetica, Arial, sans-serif; color: #777; }
}
@page :first { @bottom-left { content: none; } @bottom-right { content: none; } }

html { font-size: 10.5pt; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  margin: 0; padding: 0; background: #fff; color: #222;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 1rem; line-height: 1.45;
}

/* ----- content geometry (override the screen layout from jtd.css) ----- */
.franklin-content { padding: 0; width: auto; margin: 0; line-height: 1.45; }
.franklin-content p { margin: 0 0 0.7em; orphans: 3; widows: 3; }
.franklin-content ul, .franklin-content ol { line-height: 1.45; padding-left: 1.4em; margin: 0 0 0.7em; }
.franklin-content li { margin: 0.2em 0; line-height: 1.45; }
.franklin-content li p { margin: 0.2em 0; }

/* ----- headings ----- */
.franklin-content h1, .franklin-content h2, .franklin-content h3,
.franklin-content h4, .franklin-content h5, .franklin-content h6 {
  color: #111; font-weight: 600; line-height: 1.2;
  break-after: avoid; page-break-after: avoid; break-inside: avoid;
}
.franklin-content h1 { font-size: 19pt; margin: 0 0 0.8em; padding-bottom: 0.35em; border-bottom: 2px solid #bbb; }
.franklin-content h2 { font-size: 14pt; margin: 1.5em 0 0.6em; padding-bottom: 0.2em; border-bottom: 0.75pt solid #ccc; }
.franklin-content h3 { font-size: 12pt; margin: 1.2em 0 0.5em; }
.franklin-content h4, .franklin-content h5, .franklin-content h6 { font-size: 10.5pt; margin: 1em 0 0.4em; }
.franklin-content h1 a, .franklin-content h2 a, .franklin-content h3 a,
.franklin-content h4 a, .franklin-content h5 a, .franklin-content h6 a { color: inherit; }

/* ----- links ----- */
.franklin-content a { color: #0b3a8c; white-space: normal; overflow: visible; text-overflow: clip; }
${opt.urls ? `.franklin-content a[href^="http"]:not(.bare)::after {
  content: " <" attr(href) ">"; font-size: 8pt; color: #555; overflow-wrap: anywhere;
}` : ""}
.franklin-content a.bare { overflow-wrap: anywhere; }

/* ----- tables ----- */
.franklin-content table { font-size: 9pt; margin: 0.5em auto 1em; break-inside: auto; }
.franklin-content th, .franklin-content td,
.franklin-content table tbody tr td { font-size: 9pt; line-height: 1.3; padding: 3pt 5pt; border: 0.75pt solid #444; }
.franklin-content th { background: #eee; }
.franklin-content tr { break-inside: avoid; page-break-inside: avoid; }
.franklin-content thead { display: table-header-group; }

/* ----- images, code, math ----- */
.franklin-content img { break-inside: avoid; max-width: 100%; }
.franklin-content code { font-size: 8.5pt; }
.franklin-content pre { white-space: pre-wrap; break-inside: avoid; }
.hljs { font-size: 8.5pt; }
.katex { font-size: 1.05em !important; }
.katex-display { break-inside: avoid; }

/* ----- coloured boxes ----- */
.concept, .note, .learning, .warning, .exercise, .outline { margin: 1em 0; }
.concept .title, .note .title, .learning .title, .warning .title, .exercise .title, .outline .title {
  font-size: 10.5pt; padding: 2pt 7pt; break-after: avoid;
}
.concept .content, .note .content, .learning .content, .warning .content, .exercise .content, .outline .content {
  padding: 6pt 9pt;
}
.franklin-content .nowrap { white-space: nowrap; }
.embed { display: block; font-style: italic; margin: 0 0 0.7em; }

/* ----- chapters ----- */
.chapter { counter-reset: eqnum; }
.chapter + .chapter { break-before: page; page-break-before: always; }
.chapter-label {
  font-size: 8.5pt; letter-spacing: 0.08em; text-transform: uppercase; color: #777;
  margin-bottom: 0.4em;
}

/* ----- cover & contents ----- */
.cover { height: 8.5in; display: flex; flex-direction: column; justify-content: center; break-after: page; }
.cover .course { font-size: 34pt; font-weight: 700; color: #111; margin: 0; }
.cover .subtitle { font-size: 18pt; color: #333; margin: 0.2em 0 1.4em; }
.cover .meta { font-size: 10.5pt; color: #444; line-height: 1.6; }
.cover .license { margin-top: 3em; font-size: 8.5pt; color: #666; }
.toc { break-after: page; }
.toc h1 { font-size: 19pt; font-weight: 600; margin: 0 0 1em; padding-bottom: 0.35em; border-bottom: 2px solid #bbb; color: #111; }
.toc ol { list-style: none; padding: 0; margin: 0; }
.toc li { display: flex; align-items: baseline; margin: 0 0 0.6em; font-size: 11.5pt; }
.toc a { color: #111; }
.toc .leader { flex: 1; border-bottom: 1pt dotted #999; margin: 0 0.4em; }
`;

// ------------------------------------------------------------------ document
function documentHtml({ chapters, front, pageNumbers }) {
  const today = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
  const frontMatter = !front ? "" : `
<section class="cover">
  <p class="course">ENVE 160B</p>
  <p class="subtitle">Environmental Engineering Laboratory</p>
  <div class="meta">
    ${escapeHtml(author)}<br>
    University of California, Riverside<br>
    ${siteUrl}<br><br>
    Printed ${today}
  </div>
  <div class="license">Licensed CC BY-NC 4.0 — https://creativecommons.org/licenses/by-nc/4.0/</div>
</section>
<section class="toc">
  <h1>Contents</h1>
  <ol>${pages.map((p, i) => `
    <li><a href="#${p.slug}--top">${escapeHtml(p.label)}</a><span class="leader"></span><span>${pageNumbers ? pageNumbers[i] : "000"}</span></li>`).join("")}
  </ol>
</section>`;
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ENVE 160B — Environmental Engineering Laboratory</title>
<link rel="stylesheet" href="${fileUrl(path.join(site, "libs/katex/katex.min.css"))}">
<link rel="stylesheet" href="${fileUrl(path.join(site, "css/jtd.css"))}">
<style>${css}</style>
</head>
<body>
${frontMatter}
${chapters.join("\n")}
</body>
</html>`;
}

// ------------------------------------------------------------------ chrome
function findChrome() {
  const candidates = [
    process.env.CHROME,
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
  ].filter(Boolean);
  const dirs = (process.env.PATH || "").split(path.delimiter);
  for (const c of candidates) {
    if (path.isAbsolute(c)) { if (fs.existsSync(c)) return c; continue; }
    for (const d of dirs) if (fs.existsSync(path.join(d, c))) return path.join(d, c);
  }
  throw new Error("Chrome/Chromium not found; set CHROME=/path/to/chrome");
}

const chrome = findChrome();
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "enve160b-pdf-"));

async function printPdf(name, html, outline = false) {
  const htmlFile = path.join(tmp, `${name}.html`);
  const pdfFile = path.join(tmp, `${name}.pdf`);
  fs.writeFileSync(htmlFile, html);
  await run(chrome, [
    "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
    "--allow-file-access-from-files", "--no-pdf-header-footer",
    ...(outline ? ["--generate-pdf-document-outline"] : []),
    `--user-data-dir=${path.join(tmp, `profile-${name}`)}`,
    `--print-to-pdf=${pdfFile}`,
    fileUrl(htmlFile),
  ], { timeout: 120_000 });
  return pdfFile;
}

// Chrome writes each page as a "/Type /Page" dictionary (the page tree is "/Type /Pages").
const countPages = (pdfFile) => (fs.readFileSync(pdfFile, "latin1").match(/\/Type\s*\/Page(?![a-zA-Z])/g) || []).length;

// ------------------------------------------------------------------ main
try {
  const chapters = pages.map(chapterHtml);

  // Pass 1: print the front matter and every chapter on its own to learn page counts.
  const [frontPages, ...chapterPages] = (await Promise.all([
    printPdf("front", documentHtml({ chapters: [], front: true })),
    ...chapters.map((c, i) => printPdf(`chapter-${i}`, documentHtml({ chapters: [c], front: false }))),
  ])).map(countPages);

  let next = frontPages + 1;
  const pageNumbers = chapterPages.map((n) => { const start = next; next += n; return start; });

  // Pass 2: the full document with page numbers in the table of contents.
  const full = await printPdf("full", documentHtml({ chapters, front: true, pageNumbers }), true);
  const total = countPages(full);
  if (total !== next - 1) console.warn(`warning: expected ${next - 1} pages but got ${total}; contents page numbers may be off`);

  fs.copyFileSync(full, opt.out);
  pages.forEach((p, i) => console.log(`  p. ${String(pageNumbers[i]).padStart(3)}  ${p.label}`));
  console.log(`wrote ${path.relative(process.cwd(), opt.out) || opt.out} (${total} pages)`);
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}
