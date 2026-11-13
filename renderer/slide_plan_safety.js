// Conservative evidence checks for model-generated slide plans.
// Usage: const { assess } = require('./slide_plan_safety');
// A blocked visual should be reviewed or replaced with an explanation layout.
const numberPattern = /[-+]?\d+(?:\.\d+)?/g;

function numbers(text) {
  return (String(text).match(numberPattern) || []).map(Number);
}
function escapeRegex(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
function numericLabel(label) {
  const match = String(label).match(/[-+]?\d+(?:\.\d+)?/);
  return match ? match[0] : null;
}
function numberToken(value) {
  const number = Number(value);
  const literal = escapeRegex(String(number));
  const flexible = Number.isInteger(number) ? `${literal}(?:\\.0+)?` : literal;
  return `(?<![\\d.])${flexible}(?!\\d|\\.\\d)`;
}
function numericPairInText(label, value, text) {
  const x = numericLabel(label);
  if (x === null || !Number.isFinite(Number(value))) return false;
  const xToken = numberToken(x);
  const yToken = numberToken(value);
  // Examples: "0 / 80", "0 min: 80 C", "day 0 | 80".
  const labelFirst = new RegExp(`${xToken}\\s*(?:min(?:ute)?s?|h(?:ours?)?|days?|weeks?)?\\s*[:|/=,-]\\s*${yToken}`, 'i');
  const labelFirstSpace = new RegExp(`${xToken}\\s*(?:min(?:ute)?s?|h(?:ours?)?|days?|weeks?)\\s+${yToken}`, 'i');
  // Examples: "80 C at 0 minutes", "2 cm on day 0".
  const valueFirst = new RegExp(`${yToken}\\s*(?:°?C|V|mm|cm|kg|g|m|%)?\\s*(?:at|on)\\s*(?:day|time|hour|minute|week)?\\s*${xToken}`, 'i');
  return labelFirst.test(text) || labelFirstSpace.test(text) || valueFirst.test(text);
}
function categoricalPairInText(label, value, text) {
  const word = escapeRegex(String(label).trim());
  if (!word || !Number.isFinite(Number(value))) return false;
  const number = numberToken(value);
  const labelFirst = new RegExp(`\\b${word}\\b\\s*(?:[:|=,-]\\s*)?${number}`, 'i');
  const valueFirst = new RegExp(`${number}\\s*(?:[A-Za-z%]+\\s+){0,3}\\b${word}\\b`, 'i');
  return labelFirst.test(text) || valueFirst.test(text);
}
function editDistance(a, b) {
  const rows = Array.from({ length: a.length + 1 }, (_, i) => [i]);
  for (let j = 1; j <= b.length; j++) rows[0][j] = j;
  for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) {
    rows[i][j] = Math.min(rows[i - 1][j] + 1, rows[i][j - 1] + 1,
      rows[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  }
  return rows[a.length][b.length];
}
function fuzzyCategoricalPair(label, value, text) {
  const name = String(label).trim().toLowerCase();
  if (!/^[a-z]{3,20}$/.test(name)) return false;
  const match = text.match(new RegExp(numberToken(value)));
  if (!match) return false;
  const near = text.slice(Math.max(0, match.index - 18), match.index + match[0].length + 18);
  return (near.match(/[A-Za-z]{3,20}/g) || []).some(word =>
    editDistance(word.toLowerCase(), name) <= (name.length >= 6 ? 2 : 1));
}
function derivedNumber(value, sourceNumbers, segment) {
  if (!/\b(sum|total|difference|gains?|drops?|increase|decrease|more|less|minus|divide|divid|mean|average|exceed(?:s|ed)?)\b/i.test(segment)) return false;
  const close = (a, b) => Math.abs(a - b) < 1e-6;
  for (const a of sourceNumbers) for (const b of sourceNumbers) {
    if (close(a + b, value) || close(Math.abs(a - b), value) || (b && close(a / b, value))) return true;
  }
  if (/\b(sum|total|mean|average|divide)\b/i.test(segment) && sourceNumbers.length <= 15) {
    for (const a of sourceNumbers) for (const b of sourceNumbers) for (const c of sourceNumbers) {
      if (close(a + b + c, value)) return true;
    }
  }
  return false;
}
function assess(envelope) {
  const issues = [];
  const regions = Array.isArray(envelope?.ocr?.regions) ? envelope.ocr.regions : [];
  const output = envelope?.model_output;
  if (!output || typeof output !== 'object') return ['model_output is missing'];
  const source = regions.map(r => String(r.text || '')).join('\n');
  const sourceNumbers = [...new Set(numbers(source))];
  const visual = output.visual;
  if (!visual || typeof visual !== 'object') return ['visual is missing'];

  const prose = [output.summary, ...(Array.isArray(output.key_points) ? output.key_points.map(p => p.explanation) : []),
    output.example?.description];
  for (const segment of prose) {
    if (typeof segment !== 'string') continue;
    const extra = [...new Set(numbers(segment).filter(n =>
      !sourceNumbers.includes(n) && !derivedNumber(n, sourceNumbers, segment)))];
    if (extra.length && !/\b(hypothetical|suppose|assume|if|for example)\b/i.test(segment)) {
      issues.push(`unsupported prose number(s): ${extra.join(', ')}`);
    }
  }

  if (visual.type === 'formula') {
    const cited = Array.isArray(visual.source_region_ids) ? new Set(visual.source_region_ids) : new Set();
    const equations = regions.filter(r => cited.has(r.id) && String(r.text || '').includes('='))
      .map(r => String(r.text).toLowerCase().replace(/\s+/g, ''));
    const expression = String(visual.data?.expression || '').toLowerCase().replace(/\s+/g, '');
    if (!equations.length) issues.push('formula lacks an explicit cited OCR equation');
    else if (!expression || !equations.some(eq => eq.includes(expression))) {
      issues.push('formula expression does not match a cited OCR equation');
    }
  }

  if (visual.type === 'line_chart' || visual.type === 'bar_chart') {
    const data = visual.data || {};
    const labels = data.labels, series = data.series;
    if (!Array.isArray(labels) || labels.length < 3 || labels.length > 8 ||
        !Array.isArray(series) || series.length < 1 || series.length > 3) {
      issues.push('chart requires 3–8 labels and 1–3 series');
      return issues;
    }
    const cited = Array.isArray(visual.source_region_ids) ? new Set(visual.source_region_ids) : new Set();
    const citedLines = regions.filter(r => cited.has(r.id)).map(r => String(r.text || ''));
    if (!citedLines.length) issues.push('chart has no cited OCR regions');
    for (const item of series) {
      if (!Array.isArray(item.values) || item.values.length !== labels.length) {
        issues.push('chart label/value lengths differ');
        continue;
      }
      for (let i = 0; i < labels.length; i++) {
        const label = labels[i], value = item.values[i];
        const supported = citedLines.some(line => visual.type === 'line_chart'
          ? numericPairInText(label, value, line)
          : categoricalPairInText(label, value, line) || fuzzyCategoricalPair(label, value, line));
        if (!supported) issues.push(`chart pair lacks direct OCR evidence: ${label} → ${value}`);
      }
    }
  }
  return issues;
}

module.exports = { assess };
