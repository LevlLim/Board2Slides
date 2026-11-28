// Render one validated adaptive-visual model output as an editable PowerPoint deck.
// Usage: node renderer/adaptive_visual_renderer.js input.json output.pptx
const fs = require("fs");
const path = require("path");
const PptxGenJS = require(path.join(__dirname, "..", "node_modules", "pptxgenjs"));
const JSZip = require(path.join(__dirname, "..", "node_modules", "jszip"));
const { assess: assessSlidePlanSafety } = require("./slide_plan_safety");

const inputPath = process.argv[2];
const outputPath = process.argv[3];
if (!inputPath || !outputPath) throw new Error("Usage: node renderer/adaptive_visual_renderer.js input.json output.pptx");
const envelope = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const model = envelope.model_output;
const visual = model && model.visual;
const reviewedDemo = envelope.generation_mode === "reviewed_demo";
const allowed = new Set(["none", "process", "comparison_table", "formula", "line_chart", "bar_chart"]);
if (!model || !visual || !allowed.has(visual.type)) throw new Error("Input needs a validated adaptive model_output");
if (!Array.isArray(model.key_points) || model.key_points.length < 2 || model.key_points.length > 4) {
  throw new Error("Expected 2 to 4 key points");
}
const included = new Set(model.included_slides ||
  (visual.type === "none" ? ["cover", "overview", "example"] : ["cover", "overview", "visual", "example"]));
if (!included.size || [...included].some(slide => !["cover", "overview", "visual", "example"].includes(slide))) {
  throw new Error("Keep at least one valid slide in the deck");
}

function numericTokens(value) {
  return (String(value).replaceAll(",", "").match(/[-+]?\d+(?:\.\d+)?/g) || []).map(Number);
}
function normalize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}
function contentConsistencyIssues() {
  const issues = [];
  const ocrText = (envelope.ocr && envelope.ocr.regions || []).map(r => String(r.text)).join(" \n ");
  const teachingSegments = [model.summary, ...model.key_points.map(p => `${p.heading}: ${p.explanation}`)];
  const teachingText = teachingSegments.join(". ");
  if (["line_chart", "bar_chart"].includes(visual.type)) {
    const labels = visual.data.labels.map(v => String(v).toLowerCase());
    for (const series of visual.data.series) {
      const values = series.values.map(Number);
      for (const sentence of teachingSegments) {
        const lower = sentence.toLowerCase();
        const tokens = numericTokens(sentence);
        if (/\b(largest|highest|maximum)\b/.test(lower) && !/increase/.test(lower) && !/next largest/.test(lower)) {
          const maximum = Math.max(...values), label = labels[values.indexOf(maximum)];
          if (!tokens.includes(maximum) || !lower.includes(label)) issues.push(`maximum claim conflicts with chart: ${sentence}`);
        }
        if (/\b(smallest|lowest|minimum)\b/.test(lower)) {
          const minimum = Math.min(...values), label = labels[values.indexOf(minimum)];
          if (!tokens.includes(minimum) || !lower.includes(label)) issues.push(`minimum claim conflicts with chart: ${sentence}`);
        }
        if (/next largest/.test(lower) && new Set(values).size > 1) {
          const second = [...new Set(values)].sort((a, b) => b - a)[1];
          const label = labels[values.indexOf(second)];
          if (!tokens.includes(second) || !lower.includes(label)) issues.push(`second-largest claim conflicts with chart: ${sentence}`);
        }
        if (/largest increase/.test(lower) && values.length > 1) {
          const changes = values.slice(1).map((v, i) => v - values[i]);
          const index = changes.indexOf(Math.max(...changes));
          if (!lower.includes(labels[index]) || !lower.includes(labels[index + 1])) issues.push(`largest-increase claim conflicts with chart: ${sentence}`);
        }
      }
      const range = teachingText.match(/increases? from\s+([-+]?\d+(?:\.\d+)?)\s+to\s+([-+]?\d+(?:\.\d+)?)/i);
      if (range && (Number(range[1]) !== values[0] || Number(range[2]) !== values.at(-1))) {
        issues.push("start/end claim conflicts with chart");
      }
    }
  }
  if (visual.type === "comparison_table") {
    const source = normalize(ocrText);
    for (const cell of [...visual.data.columns, ...visual.data.rows.flat()]) {
      if (!source.includes(normalize(cell))) issues.push(`table cell is not supported by OCR: ${cell}`);
    }
  }
  if (visual.type === "process") {
    const source = normalize(ocrText);
    for (const step of visual.data.steps) {
      if (!source.includes(normalize(step))) issues.push(`process step is not supported by OCR: ${step}`);
    }
  }
  if (visual.type === "formula") {
    const equations = (envelope.ocr.regions || []).map(r => String(r.text)).filter(t => t.includes("=")).map(normalize);
    if (equations.length && !equations.includes(normalize(visual.data.expression))) issues.push("formula expression does not match the OCR equation");
    const mappings = new Map();
    for (const region of (envelope.ocr.regions || [])) {
      const text = String(region.text);
      if (!text.includes(":")) continue;
      const splitAt = text.indexOf(":");
      mappings.set(normalize(text.slice(0, splitAt)), normalize(text.slice(splitAt + 1)));
    }
    for (const variable of (visual.data.variables || [])) {
      const symbol = normalize(variable.symbol), meaning = normalize(variable.meaning);
      if (mappings.has(symbol) && mappings.get(symbol) !== meaning) {
        issues.push(`formula meaning conflicts with OCR for ${variable.symbol}`);
      }
    }
  }
  const ocrNumbers = new Set(numericTokens(ocrText));
  const exampleText = `${model.example.title} ${model.example.description}`;
  const extra = numericTokens(exampleText).filter(v => !ocrNumbers.has(v));
  if (extra.length && !/\b(hypothetical|suppose|assume|if|for example)\b/i.test(exampleText)) {
    issues.push(`new example numbers are not labelled hypothetical: ${[...new Set(extra)].join(", ")}`);
  }
  return issues;
}
const consistencyIssues = [...contentConsistencyIssues(), ...assessSlidePlanSafety(envelope)];
if (consistencyIssues.length) {
  throw new Error("Content consistency guardrail blocked rendering:\n- " + consistencyIssues.join("\n- "));
}

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Whiteboard OCR-to-slides";
pptx.subject = "Adaptive visual lesson generated from OCR text";
pptx.title = String(model.title);
pptx.lang = "en-SG";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-SG",
};

const W = 13.333, H = 7.5;
const F = "Aptos";
const C = {
  navy: "16324A", teal: "087F7A", aqua: "DDF3EF", ink: "263746",
  muted: "657989", paper: "F7FAFC", white: "FFFFFF", line: "D9E4EA",
  blue: "2F6B9A", orange: "DB7C38", gold: "E3B341", green: "4E9A72",
};

function rect(slide, x, y, w, h, fill, line = fill, radius = false) {
  slide.addShape(radius ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, {
    x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: line, width: line === fill ? 0 : 1 },
  });
}
function addText(slide, value, x, y, w, h, options = {}) {
  slide.addText(String(value), {
    x, y, w, h, fontFace: F, fontSize: 18, color: C.ink, margin: 0,
    valign: "mid", breakLine: false, fit: "shrink", ...options,
  });
}
function provenance(extra = "") {
  const ocr = (envelope.ocr && envelope.ocr.regions || []).map(r => `${r.id}: ${r.text}`).join("\n");
  return [
    (reviewedDemo ? "Draft source: " : "Model condition: ") + (envelope.model_condition || "adaptive v3"),
    reviewedDemo ? "Visible text is copied from the user-reviewed draft without rewriting." : "Visible teaching text is copied from model_output without rewriting.",
    `Visual source region IDs: ${(visual.source_region_ids || []).join(", ")}`,
    extra,
    "OCR input:", ocr,
  ].filter(Boolean).join("\n");
}
function base(title, label, page) {
  const slide = pptx.addSlide();
  slide.background = { color: C.paper };
  rect(slide, 0, 0, W, 0.1, C.teal);
  addText(slide, label.toUpperCase(), 0.66, 0.35, 5.8, 0.28, { fontSize: 10, bold: true, color: C.teal, charSpacing: 1.3 });
  addText(slide, title, 0.66, 0.78, 12.0, 0.72, { fontFace: "Aptos Display", fontSize: 30, bold: true, color: C.navy });
  rect(slide, 0.66, 7.02, 12.0, 0.012, C.line);
  addText(slide, String(page).padStart(2, "0"), 12.12, 7.08, 0.4, 0.18, { fontSize: 9, color: C.muted, align: "right" });
  return slide;
}

let page = 0;
if (included.has("cover")) {
const cover = pptx.addSlide();
cover.background = { color: C.navy };
rect(cover, 0.72, 0.9, 0.12, 5.7, C.teal);
addText(cover, "WHITEBOARD TO LESSON", 1.18, 1.25, 10.6, 0.3, { fontSize: 13, bold: true, color: "7BDBD2", charSpacing: 1.8 });
addText(cover, model.title, 1.16, 2.02, 11.35, 1.5, { fontFace: "Aptos Display", fontSize: 44, bold: true, color: C.white });
addText(cover, reviewedDemo ? "From reviewed board notes" : `Adaptive visual: ${visual.type.replaceAll("_", " ")}`, 1.17, 4.27, 10.7, 0.42, { fontSize: 18, color: "C9D9E4" });
addText(cover, envelope.model_condition || "Qwen3-1.7B with adaptive visual LoRA v2", 1.17, 5.0, 10.7, 0.34, { fontSize: 13, color: "8FA8B9" });
cover.addNotes(provenance("Cover uses model_output.title and model_output.visual.type."));
}

if (included.has("overview")) {
const overview = base("Lesson overview", reviewedDemo ? "Reviewed board notes" : "Model generated content", ++page);
addText(overview, model.summary, 0.68, 1.72, 12.0, 0.78, { fontSize: 22, bold: true, color: C.navy, valign: "top" });
const pointCount = model.key_points.length;
const pointH = 3.7 / pointCount;
model.key_points.forEach((point, i) => {
  const y = 2.76 + i * pointH;
  addText(overview, String(i + 1).padStart(2, "0"), 0.72, y, 0.44, 0.32, { fontSize: 13, bold: true, color: C.teal });
  addText(overview, point.heading, 1.27, y - 0.02, 3.15, 0.38, { fontSize: 18, bold: true, color: C.navy });
  addText(overview, point.explanation, 4.48, y - 0.03, 7.9, Math.min(0.76, pointH - 0.1), { fontSize: 17, valign: "top" });
  if (i < pointCount - 1) rect(overview, 1.27, y + pointH - 0.12, 11.1, 0.01, C.line);
});
overview.addNotes(provenance("Slide uses model_output.summary and model_output.key_points."));
}

function addComparison(slide, data) {
  const rows = [
    data.columns.map(v => ({ text: String(v), options: { bold: true, color: C.white, fill: { color: C.navy }, valign: "mid" } })),
    ...data.rows.map((row, ri) => row.map(v => ({ text: String(v), options: { fill: { color: ri % 2 ? C.white : "EEF5F7" }, valign: "mid" } }))),
  ];
  const colW = data.columns.map((_, i) => i === 0 ? 2.55 : (9.45 / (data.columns.length - 1)));
  slide.addTable(rows, {
    x: 0.68, y: 2.3, w: 12.0, h: 3.6, colW, rowH: 0.72,
    fontFace: F, fontSize: 17, color: C.ink, margin: 0.12,
    border: { type: "solid", color: C.line, pt: 1 },
  });
}
function addChart(slide, data, type) {
  const chartType = type === "line_chart" ? pptx.ChartType.line : pptx.ChartType.bar;
  const numericValues = data.series.flatMap(s => s.values).map(Number);
  const numberFormat = numericValues.some(v => !Number.isInteger(v)) ? "0.0" : "0";
  slide.addChart(chartType, data.series.map(s => ({ name: String(s.name), labels: data.labels.map(String), values: s.values })), {
    x: 0.78, y: 2.1, w: 11.75, h: 4.35,
    chartColors: [C.teal, C.blue, C.orange],
    showLegend: data.series.length > 1, legendPos: "b",
    showTitle: false, showValue: true, showCatName: false,
    catAxisLabelFontFace: F, catAxisLabelFontSize: 13,
    valAxisLabelFontFace: F, valAxisLabelFontSize: 12,
    showCatName: false, showValAxisTitle: false, showCatAxisTitle: false,
    showGridLines: true, gridLine: { color: C.line, width: 1 },
    lineDataSymbol: "circle", lineSize: 3,
    dataLabelFormatCode: numberFormat,
    valAxisNumFmt: numberFormat,
    dataLabelPosition: type === "bar_chart" ? "outEnd" : "t",
  });
  addText(slide, `Unit: ${data.unit || "not specified"}`, 10.0, 6.47, 2.5, 0.24, { fontSize: 11, color: C.muted, align: "right" });
}
function addProcess(slide, data) {
  const steps = data.steps;
  const gap = 0.18, totalW = 12.0, boxW = (totalW - gap * (steps.length - 1)) / steps.length;
  steps.forEach((step, i) => {
    const x = 0.68 + i * (boxW + gap);
    rect(slide, x, 2.45, boxW, 3.25, C.white, C.line, true);
    rect(slide, x + 0.22, 2.72, 0.56, 0.56, C.teal, C.teal, true);
    addText(slide, i + 1, x + 0.22, 2.84, 0.56, 0.24, { fontSize: 16, bold: true, color: C.white, align: "center" });
    addText(slide, step, x + 0.24, 3.58, boxW - 0.48, 1.42, { fontSize: steps.length > 4 ? 17 : 20, bold: true, color: C.navy, align: "center", valign: "mid" });
    if (i < steps.length - 1) addText(slide, "›", x + boxW - 0.01, 3.7, 0.2, 0.42, { fontSize: 26, bold: true, color: C.teal, align: "center" });
  });
}
function addFormula(slide, data) {
  rect(slide, 0.7, 2.0, 12.0, 1.05, C.aqua, C.aqua, true);
  addText(slide, data.expression, 1.0, 2.21, 11.4, 0.56, { fontFace: "Cambria Math", fontSize: 28, bold: true, color: C.navy, align: "center" });
  const variableRows = [["Symbol", "Meaning"], ...data.variables.map(v => [v.symbol, v.meaning])];
  slide.addTable(variableRows, {
    x: 0.72, y: 3.45, w: 5.45, h: 2.35, colW: [1.55, 3.9],
    fontFace: F, fontSize: 16, color: C.ink, margin: 0.1,
    border: { type: "solid", color: C.line, pt: 1 },
    fill: C.white, bold: false,
  });
  addText(slide, "Worked steps", 6.72, 3.46, 5.7, 0.36, { fontSize: 18, bold: true, color: C.navy });
  data.worked_steps.forEach((step, i) => {
    addText(slide, `${i + 1}.`, 6.76, 4.04 + i * 0.58, 0.34, 0.27, { fontSize: 15, bold: true, color: C.teal });
    addText(slide, step, 7.18, 3.96 + i * 0.58, 5.12, 0.46, { fontSize: 16, valign: "top" });
  });
}

if (included.has("visual")) {
const visualSlide = base(visual.title || model.title, reviewedDemo ? "Board recap" : visual.type.replaceAll("_", " "), ++page);
addText(visualSlide, visual.reason, 0.7, 1.55, 12.0, 0.42, { fontSize: 16, color: C.muted, italic: true });
if (visual.type === "comparison_table") addComparison(visualSlide, visual.data);
else if (visual.type === "line_chart" || visual.type === "bar_chart") addChart(visualSlide, visual.data, visual.type);
else if (visual.type === "process") addProcess(visualSlide, visual.data);
else if (visual.type === "formula") addFormula(visualSlide, visual.data);
else addText(visualSlide, model.summary, 1.0, 2.55, 11.2, 2.0, { fontSize: 28, bold: true, color: C.navy, align: "center" });
visualSlide.addNotes(provenance("Native editable visual uses model_output.visual.data."));
}

if (included.has("example")) {
const exampleSlide = base(model.example.title, reviewedDemo ? "Review activity" : "Model generated example", ++page);
rect(exampleSlide, 0.78, 2.05, 11.75, 3.85, C.white, C.line, true);
addText(exampleSlide, model.example.description, 1.25, 2.65, 10.8, 2.4, { fontSize: 27, color: C.navy, align: "center", valign: "mid" });
addText(exampleSlide, reviewedDemo
  ? "Drafted from reviewed board text; check and edit before sharing"
  : "Generated from OCR text and the model's pretrained knowledge",
  1.25, 5.37, 10.8, 0.28, { fontSize: 11, color: C.muted, align: "center" });
exampleSlide.addNotes(provenance("Slide uses model_output.example without rewriting."));
}

async function removeUnusedMasterDeclarations(filePath) {
  const zip = await JSZip.loadAsync(fs.readFileSync(filePath));
  let contentTypes = await zip.file("[Content_Types].xml").async("string");
  contentTypes = contentTypes.replace(
    /<Override\b[^>]*PartName="\/ppt\/slideMasters\/(slideMaster\d+\.xml)"[^>]*\/>/g,
    (entry, fileName) => zip.file(`ppt/slideMasters/${fileName}`) ? entry : ""
  );
  zip.file("[Content_Types].xml", contentTypes);
  fs.writeFileSync(filePath, await zip.generateAsync({ type: "nodebuffer" }));
}

pptx.writeFile({ fileName: outputPath }).then(async () => {
  await removeUnusedMasterDeclarations(outputPath);
  console.log(`Created ${outputPath}`);
});
