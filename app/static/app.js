const state = { job: null, step: "upload", busy: false, dirty: false, originalVisual: null,
  ocrReady: false, models: [], files: [], activePhotoId: null, draftByPhoto: new Map() };
const stepNames = ["upload", "prepare", "review", "plan", "export"];
const regionColor = (index) => `hsl(${Math.round(index * 137.508) % 360} 67% 37%)`;
const el = (id) => document.getElementById(id);

function notify(message, isError) {
  const box = el("toast");
  box.textContent = message;
  box.style.background = isError ? "#873e38" : "#183d45";
  box.hidden = false;
  clearTimeout(notify.timer);
  notify.timer = setTimeout(() => { box.hidden = true; }, isError ? 6500 : 3400);
}

async function request(path, options) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    let detail = data && data.detail ? data.detail : "Something went wrong. Please try again.";
    if (Array.isArray(detail)) detail = detail.map((item) => item.msg).join("; ");
    throw new Error(String(detail));
  }
  return data;
}

function showStep(step) {
  state.step = step;
  document.body.classList.toggle("workflow-active", step !== "upload");
  el("workspace-bar").hidden = !state.job;
  if (!state.job) {
    el("share-panel").hidden = true;
    el("share-toggle").setAttribute("aria-expanded", "false");
  }
  stepNames.forEach((name, index) => {
    el("step-" + name).hidden = name !== step;
    const marker = document.querySelector('.step[data-step="' + name + '"]');
    marker.classList.toggle("active", name === step);
    marker.classList.toggle("done", index < stepNames.indexOf(step));
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setDirty(dirty) {
  state.dirty = dirty;
  el("save-state").textContent = dirty
    ? "Unsaved changes on this device"
    : "Saved on this laptop server";
  el("refresh-job").textContent = dirty
    ? "Discard edits & refresh"
    : "Refresh from server";
}

function updateShare() {
  if (!state.job) return;
  const link = new URL("/", window.location.origin);
  link.searchParams.set("job", state.job.id);
  el("share-url").value = link.href;
  el("share-qr").src = "/api/jobs/" + state.job.id + "/qr";
  const local = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
  el("share-network-note").textContent = local
    ? "This link uses localhost. For a phone, reopen the app using your laptop's Wi-Fi IP address before copying or scanning."
    : "Both devices must be able to reach this server at " + window.location.host + ".";
}

function setCurrentJob(job) {
  state.job = job;
  if (job.photos && job.photos.length) {
    if (!job.photos.some((photo) => photo.id === state.activePhotoId)) state.activePhotoId = job.photos[0].id;
    state.draftByPhoto = new Map(job.photos.map((photo) => [photo.id,
      job.ocr.regions.filter((region) => region.photo_id === photo.id).map((region) => ({ ...region }))]));
    if (job.photos[0].preparation) el("prepare-mode").value = job.photos[0].preparation.mode;
    el("ocr-model").value = job.selected_model || "tuned";
  } else {
    state.activePhotoId = null;
    state.draftByPhoto = new Map();
  }
  localStorage.setItem("board2slides_job", job.id);
  history.replaceState(null, "", "/?job=" + encodeURIComponent(job.id));
  updateShare();
  setDirty(false);
}

function currentPhoto() {
  return state.job?.photos?.find((photo) => photo.id === state.activePhotoId) || null;
}

function photoTabs(target, callback) {
  const wrap = el(target);
  wrap.replaceChildren();
  if (!state.job?.photos?.length) { wrap.hidden = true; return; }
  wrap.hidden = state.job.photos.length < 2;
  state.job.photos.forEach((photo, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "photo-tab" + (photo.id === state.activePhotoId ? " active" : "");
    button.textContent = `Photo ${index + 1}`;
    button.title = photo.name;
    button.setAttribute("aria-pressed", String(photo.id === state.activePhotoId));
    button.addEventListener("click", () => callback(photo.id));
    wrap.append(button);
  });
}

function renderPrepare() {
  if (!state.job?.photos?.length) return;
  photoTabs("prepare-photo-tabs", (photoId) => { state.activePhotoId = photoId; renderPrepare(); });
  const photo = currentPhoto();
  const version = `?v=${state.job.revision}`;
  el("prepare-original").src = photo.original_url;
  el("prepare-processed").src = (photo.preparation ? photo.processed_url : photo.original_url) + version;
  const decision = el("prepare-decision");
  decision.replaceChildren();
  const summary = document.createElement("p");
  summary.textContent = photo.preparation
    ? (photo.preparation.applied ? "Changed: " : "Retained: ") + photo.preparation.reason
    : "The prepared preview will appear after you run photo preparation.";
  decision.append(summary);
  if (photo.preparation?.decisions?.length) {
    const list = document.createElement("ul");
    list.className = "preparation-decisions";
    photo.preparation.decisions.forEach((item) => {
      const row = document.createElement("li");
      const label = document.createElement("strong");
      label.textContent = `${item.name[0].toUpperCase()}${item.name.slice(1)}: ${item.applied ? "Applied" : "Skipped"}`;
      row.append(label, document.createTextNode(` — ${item.reason}`));
      list.append(row);
    });
    decision.append(list);
  }
  if (photo.preparation && photo.preparation.mode !== el("prepare-mode").value) {
    const note = document.createElement("p");
    note.textContent = "This preview uses the previous mode. Press Prepare photos to see the selected mode.";
    decision.append(note);
  }
  const ready = state.job.photos.every((item) => item.preparation);
  const matchingMode = ready && state.job.photos.every((item) => item.preparation.mode === el("prepare-mode").value);
  el("run-ocr-button").disabled = !matchingMode || !state.models.some((model) => model.id === el("ocr-model").value && model.available);
  el("skip-ocr-button").disabled = !matchingMode;
}

async function prepareAll() {
  if (!state.job || state.busy) return;
  if (state.job.ocr.regions.length && !window.confirm("Prepare again? This replaces the OCR text and any slide draft for this lesson.")) return;
  const button = el("prepare-button");
  setBusy(button, true, "Preparing photos…");
  try {
    const job = await request(`/api/jobs/${state.job.id}/prepare`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision: state.job.revision, mode: el("prepare-mode").value }),
    });
    setCurrentJob(job);
    renderPrepare();
    notify("Photo preparation finished. Review the result, then run OCR.");
  } catch (error) { notify(error.message, true); }
  finally { setBusy(button, false); }
}

async function runSelectedOcr() {
  if (!state.job || state.busy) return;
  if (state.job.ocr.regions.length && !window.confirm("Run OCR again? This replaces your reviewed lines and any slide draft.")) return;
  const model = el("ocr-model").value;
  const button = el("run-ocr-button");
  setBusy(button, true, "Reading photos; this may take a minute…");
  try {
    const job = await request(`/api/jobs/${state.job.id}/ocr/run`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision: state.job.revision, model }),
    });
    setCurrentJob(job);
    renderOcr();
    showStep("review");
    notify(job.ocr_message, job.ocr_status === "error");
  } catch (error) { notify(error.message, true); }
  finally { setBusy(button, false); }
}

function setBusy(button, busy, label) {
  state.busy = busy;
  if (button) {
    if (!button.dataset.original) button.dataset.original = button.innerHTML;
    button.disabled = busy;
    button.innerHTML = busy ? label : button.dataset.original;
  }
}

function makeOcrRow(region, index, confidence) {
  const row = document.createElement("div");
  row.className = "ocr-row";
  row.dataset.regionId = region.id || "";
  row.dataset.photoId = region.photo_id || state.activePhotoId || "";
  row.style.setProperty("--region-color", regionColor(index));
  const number = document.createElement("span");
  number.className = "ocr-number";
  number.textContent = String(index + 1).padStart(2, "0");
  const input = document.createElement("input");
  input.type = "text";
  input.maxLength = 180;
  input.value = region.text || "";
  input.placeholder = "Write one board line here";
  input.setAttribute("aria-label", "Board line " + (index + 1));
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "icon-button";
  remove.textContent = "×";
  remove.title = "Remove this line";
  remove.setAttribute("aria-label", "Remove board line " + (index + 1));
  remove.addEventListener("click", () => {
    row.remove();
    renumberRows();
    setDirty(true);
    renderBoxes();
  });
  row.append(number, input);
  if (typeof confidence === "number" && Number.isFinite(confidence)) {
    const score = document.createElement("span");
    score.className = "ocr-confidence";
    score.textContent = Math.round(confidence * 100) + "% OCR";
    score.title = "Original model confidence before your edits; check the text even when this is high";
    row.append(score);
  }
  row.append(remove);
  row.addEventListener("mouseenter", () => highlightRegion(row.dataset.regionId));
  row.addEventListener("focusin", () => highlightRegion(row.dataset.regionId));
  return row;
}

function renumberRows() {
  [...el("ocr-lines").children].forEach((row, index) => {
    row.querySelector(".ocr-number").textContent = String(index + 1).padStart(2, "0");
    row.style.setProperty("--region-color", regionColor(index));
    row.querySelector("input").setAttribute("aria-label", "Board line " + (index + 1));
    row.querySelector("button").setAttribute("aria-label", "Remove board line " + (index + 1));
  });
}

function renderOcr() {
  if (!state.job) return;
  photoTabs("review-photo-tabs", (photoId) => {
    saveVisibleDraft();
    state.activePhotoId = photoId;
    renderOcr();
  });
  const photo = currentPhoto();
  el("board-preview").src = photo ? photo.processed_url + `?v=${state.job.revision}` : state.job.image_url;
  el("photo-tag").textContent = photo ? photo.name : state.job.mode === "sample" ? "Sample lesson" : "Your photo";
  const status = state.job.ocr_status || "manual";
  el("ocr-status").textContent = status === "auto" ? "PaddleOCR draft" :
    status === "sample" ? "Sample text" : status === "reviewed" ? "Reviewed text" :
      status === "partial" ? "Partial OCR" : status === "empty" ? "No text found" : "Manual text";
  el("ocr-help").textContent = state.job.mode === "sample"
    ? "These sample lines are editable. Change them to see how the deck responds."
    : (state.job.ocr_message || "Type the visible board lines below.") +
      " Compare every line with the photo. Percentages are the original OCR scores, even after edits.";
  const evidence = new Map((state.job.region_evidence || []).map((item) => [item.id, item]));
  const stored = photo ? (state.draftByPhoto.get(photo.id) || []) : state.job.ocr.regions;
  const regions = stored.length ? stored :
    (photo ? [{ id: manualId(photo.id), photo_id: photo.id, text: "" }] :
      [{ text: "" }, { text: "" }, { text: "" }]);
  el("ocr-lines").replaceChildren(...regions.map((region, i) =>
    makeOcrRow(region, i, evidence.get(region.id)?.confidence)));
  el("ocr-run").hidden = state.job.mode === "sample" || !state.ocrReady;
  const plannerLines = el("planner-lines");
  plannerLines.replaceChildren(...(state.job.planner_input?.regions || []).map((line) => {
    const item = document.createElement("li");
    const index = state.job.photos?.findIndex((photo) => photo.id === line.photo_id) ?? -1;
    item.textContent = (index >= 0 ? `Photo ${index + 1}: ` : "") + line.text;
    return item;
  }));
  renderBoxes();
}

function manualId(photoId) {
  return `${photoId || "r"}m${Math.random().toString(36).slice(2, 10)}`;
}

function saveVisibleDraft() {
  const photo = currentPhoto();
  if (!photo) return;
  state.draftByPhoto.set(photo.id, [...el("ocr-lines").children].map((row) => ({
    id: row.dataset.regionId || manualId(photo.id), photo_id: photo.id,
    text: row.querySelector("input").value,
  })));
}

function highlightRegion(regionId) {
  document.querySelectorAll(".ocr-row, #ocr-boxes polygon").forEach((node) =>
    node.classList.toggle("region-active", Boolean(regionId) &&
      (node.dataset.regionId === regionId)));
}

function renderBoxes() {
  const svg = el("ocr-boxes");
  svg.replaceChildren();
  const photo = currentPhoto();
  if (!photo?.preparation) return;
  svg.setAttribute("viewBox", `0 0 ${photo.preparation.width} ${photo.preparation.height}`);
  const evidence = new Map((state.job.region_evidence || []).map((item) => [item.id, item]));
  [...el("ocr-lines").children].forEach((row, index) => {
    const item = evidence.get(row.dataset.regionId);
    if (!item?.polygon?.length) return;
    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    polygon.setAttribute("points", item.polygon.map((point) => point.join(",")).join(" "));
    polygon.setAttribute("stroke", regionColor(index));
    polygon.setAttribute("stroke-width", Math.max(2, photo.preparation.width / 350));
    polygon.setAttribute("tabindex", "0");
    polygon.setAttribute("role", "button");
    polygon.setAttribute("aria-label", `Text region ${index + 1}: ${row.querySelector("input").value}`);
    polygon.dataset.regionId = row.dataset.regionId;
    polygon.addEventListener("click", () => { row.scrollIntoView({ block: "center", behavior: "smooth" }); row.querySelector("input").focus(); });
    polygon.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); polygon.click(); }
    });
    polygon.addEventListener("mouseenter", () => highlightRegion(row.dataset.regionId));
    svg.append(polygon);
  });
}

function renderSelectedFiles() {
  const list = el("selected-photos");
  list.replaceChildren();
  state.files.forEach((file, index) => {
    const row = document.createElement("div");
    row.className = "selected-photo";
    const name = document.createElement("span");
    name.textContent = `${index + 1}. ${file.name}`;
    const actions = document.createElement("div");
    for (const [label, delta] of [["↑", -1], ["↓", 1]]) {
      const button = document.createElement("button");
      button.type = "button"; button.textContent = label;
      button.setAttribute("aria-label", `Move photo ${index + 1} ${delta < 0 ? "up" : "down"}`);
      button.disabled = index + delta < 0 || index + delta >= state.files.length;
      button.addEventListener("click", () => {
        [state.files[index], state.files[index + delta]] = [state.files[index + delta], state.files[index]];
        renderSelectedFiles();
      });
      actions.append(button);
    }
    const remove = document.createElement("button");
    remove.type = "button"; remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove photo ${index + 1}`);
    remove.addEventListener("click", () => { state.files.splice(index, 1); renderSelectedFiles(); });
    actions.append(remove);
    row.append(name, actions); list.append(row);
  });
  el("start-upload").hidden = !state.files.length;
}

function queueFiles(files) {
  for (const file of files) {
    if (!file.type.startsWith("image/")) { notify("Choose JPG, PNG or WebP photos.", true); continue; }
    if (state.files.length >= 4) { notify("Use up to four photos for one lesson.", true); break; }
    state.files.push(file);
  }
  renderSelectedFiles();
}

async function begin(sample) {
  if (state.busy) return;
  if (!sample && !state.files.length) { notify("Take or choose at least one photo.", true); return; }
  const button = sample ? el("sample-button") : null;
  setBusy(button, true, "Opening lesson…");
  el("camera-button").disabled = true;
  el("file-button").disabled = true;
  el("upload-status").hidden = sample;
  try {
    const form = new FormData();
    form.append("sample", sample ? "true" : "false");
    if (!sample) state.files.forEach((file) => form.append("images", file));
    const nextJob = await request("/api/jobs", { method: "POST", body: form });
    setCurrentJob(nextJob);
    state.files = []; renderSelectedFiles();
    if (sample) { renderOcr(); showStep("review"); }
    else { renderPrepare(); showStep("prepare"); }
  } catch (error) {
    notify(error.message, true);
  } finally {
    setBusy(button, false);
    el("camera-button").disabled = false;
    el("file-button").disabled = false;
    el("upload-status").hidden = true;
    el("camera-input").value = "";
    el("file-input").value = "";
  }
}

async function rerunOcr() {
  if (!state.job || state.busy) return;
  if (state.job.mode === "photos") {
    if (state.dirty) { notify("Save your text edits before changing the OCR model.", true); return; }
    renderPrepare(); showStep("prepare"); return;
  }
  if (!window.confirm("Run OCR again? This replaces the current board lines and any slide draft.")) return;
  const button = el("ocr-run");
  setBusy(button, true, "Reading photo…");
  try {
    const job = await request("/api/jobs/" + state.job.id + "/ocr/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision: state.job.revision }),
    });
    setCurrentJob(job);
    renderOcr();
    notify(job.ocr_message, job.ocr_status !== "auto");
  } catch (error) {
    notify(error.message, true);
  } finally {
    setBusy(button, false);
  }
}

function boardLines() {
  if (state.job?.mode === "photos") {
    saveVisibleDraft();
    return state.job.photos.flatMap((photo) => (state.draftByPhoto.get(photo.id) || [])
      .filter((region) => region.text.trim()).map((region) => ({ ...region, text: region.text.trim() })));
  }
  return [...el("ocr-lines").querySelectorAll("input")]
    .map((input) => input.value.trim()).filter(Boolean);
}

async function persistOcr() {
  const lines = boardLines();
  if (!lines.length) throw new Error("Add at least one board line before saving.");
  const job = await request("/api/jobs/" + state.job.id + "/ocr", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      revision: state.job.revision,
      regions: state.job.mode === "photos" ? lines : lines.map((text, i) => ({ id: "r" + (i + 1), text })),
    }),
  });
  setCurrentJob(job);
}

async function saveBoardText() {
  if (!state.job || state.busy) return;
  if (!state.dirty) { notify("Board text is already saved."); return; }
  const button = el("save-ocr");
  setBusy(button, true, "Saving…");
  try {
    await persistOcr();
    renderOcr();
    notify("Board text saved. Refresh on the other device to see it.");
  } catch (error) {
    notify(error.message, true);
  } finally {
    setBusy(button, false);
  }
}

async function createDraft() {
  if (!state.job || state.busy) return;
  const lines = boardLines();
  if (lines.length < 2) {
    notify("Add at least two board lines before creating slides.", true);
    return;
  }
  if (state.job.mode === "photos" && state.job.photos.some((photo) => !lines.some((line) => line.photo_id === photo.id))) {
    notify("Check every photo and add at least one line for each before creating slides.", true);
    return;
  }
  const button = el("review-next");
  const mode = el("planner-mode").value;
  setBusy(button, true, mode === "llm" ? "Running the LLM…" : "Creating demo draft…");
  try {
    if (state.dirty || !state.job.ocr.regions.length) await persistOcr();
    const job = await request("/api/jobs/" + state.job.id + "/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision: state.job.revision, mode }),
    });
    setCurrentJob(job);
    renderPlan();
    showStep("plan");
  } catch (error) {
    notify(error.message, true);
  } finally {
    setBusy(button, false);
  }
}

function makePoint(point, index) {
  const card = document.createElement("div");
  card.className = "point-item";
  const label = document.createElement("strong");
  label.textContent = "POINT " + String(index + 1).padStart(2, "0");
  const headingLabel = document.createElement("label");
  const headingText = document.createElement("span");
  headingText.textContent = "Heading";
  const heading = document.createElement("input");
  heading.maxLength = 70;
  heading.required = true;
  heading.value = point.heading;
  headingLabel.append(headingText, heading);
  const explanationLabel = document.createElement("label");
  const explanationText = document.createElement("span");
  explanationText.textContent = "Explanation";
  const explanation = document.createElement("textarea");
  explanation.rows = 2;
  explanation.maxLength = 240;
  explanation.required = true;
  explanation.value = point.explanation;
  explanationLabel.append(explanationText, explanation);
  card.append(label, headingLabel, explanationLabel);
  return card;
}

function renderPlan() {
  const plan = state.job && state.job.plan;
  if (!plan) return;
  el("plan-title-input").value = plan.title;
  el("plan-summary").value = plan.summary;
  el("point-list").replaceChildren(...plan.key_points.map(makePoint));
  el("example-title").value = plan.example.title;
  el("example-description").value = plan.example.description;
  state.originalVisual = plan.visual || { type: "none", title: "Explanation only", reason: "No visual selected.", source_region_ids: [], data: {} };
  el("visual-choice").value = state.originalVisual.type === "none" ? "none" : "model";
  el("planner-tag").textContent = state.job.planner_mode === "llm" ? "Qwen + LoRA v3" : "Demo planner";
  el("plan-notes").textContent = (state.job.plan_notes || []).join(" ");
  el("plan-notes").hidden = !(state.job.plan_notes || []).length;
  renderVisual();
  updatePreview();
  setDirty(false);
}

function renderVisual() {
  const visual = state.originalVisual;
  const useVisual = el("visual-choice").value === "model" && visual.type !== "none";
  el("visual-reason").textContent = useVisual
    ? visual.type.replaceAll("_", " ") + " · " + (visual.reason || "Review this visual against the board.")
    : "Explanation only: the visual slide will use text from the slide plan.";
  const box = el("visual-data");
  box.replaceChildren();
  if (!useVisual) return;
  const heading = document.createElement("strong");
  heading.textContent = visual.title || "Selected visual";
  const details = document.createElement("pre");
  details.textContent = JSON.stringify(visual.data, null, 2);
  const evidence = document.createElement("small");
  evidence.textContent = "Board lines: " + (visual.source_region_ids || []).join(", ");
  box.append(heading, details, evidence);
}

function updatePreview() {
  el("preview-title").textContent = el("plan-title-input").value || "Your board topic";
  el("preview-summary").textContent = el("plan-summary").value || "Your reviewed board notes will appear here.";
}

function collectPlan() {
  const plan = {
    revision: state.job.revision,
    title: el("plan-title-input").value.trim(),
    summary: el("plan-summary").value.trim(),
    key_points: [...el("point-list").children].map((card) => ({
      heading: card.querySelector("input").value.trim(),
      explanation: card.querySelector("textarea").value.trim(),
    })),
    example: {
      title: el("example-title").value.trim(),
      description: el("example-description").value.trim(),
    },
    visual: el("visual-choice").value === "model" && state.originalVisual.type !== "none"
      ? state.originalVisual
      : { type: "none", title: "Explanation only", reason: "Selected during human review.",
          source_region_ids: state.job.ocr.regions.map((region) => region.id), data: {} },
    uncertain_ocr_terms: state.job.plan.uncertain_ocr_terms || [],
  };
  if (!plan.title || !plan.summary || !plan.example.title || !plan.example.description ||
      plan.key_points.some((point) => !point.heading || !point.explanation)) {
    throw new Error("Complete every slide field before exporting.");
  }
  return plan;
}

async function persistPlan() {
  const plan = collectPlan();
  const job = await request("/api/jobs/" + state.job.id + "/plan", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(plan),
  });
  setCurrentJob(job);
}

async function saveDraft() {
  if (!state.job || state.busy) return;
  if (!state.dirty) { notify("Slide draft is already saved."); return; }
  const button = el("save-plan");
  setBusy(button, true, "Saving…");
  try {
    await persistPlan();
    notify("Slide draft saved. Refresh on the other device to see it.");
  } catch (error) {
    notify(error.message, true);
  } finally {
    setBusy(button, false);
  }
}

async function exportDeck() {
  if (!state.job || state.busy) return;
  const button = el("export-button");
  try { collectPlan(); } catch (error) { notify(error.message, true); return; }
  setBusy(button, true, "Building deck…");
  try {
    if (state.dirty) await persistPlan();
    const job = await request("/api/jobs/" + state.job.id + "/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision: state.job.revision }),
    });
    setCurrentJob(job);
    el("download-link").href = job.download_url;
    showStep("export");
  } catch (error) {
    notify(error.message, true);
  } finally {
    setBusy(button, false);
  }
}

async function startOver() {
  state.job = null;
  state.files = [];
  state.draftByPhoto = new Map();
  renderSelectedFiles();
  localStorage.removeItem("board2slides_job");
  history.replaceState(null, "", "/");
  setDirty(false);
  showStep("upload");
  notify("New board ready. Your previous draft remains on the server at its share link.");
}

function showCurrentJob(job) {
  setCurrentJob(job);
  if (job.download_url) {
    el("download-link").href = job.download_url;
    showStep("export");
  } else if (job.plan) {
    renderPlan();
    showStep("plan");
  } else if (job.mode === "photos" && job.ocr_status === "pending") {
    renderPrepare();
    showStep("prepare");
  } else {
    renderOcr();
    showStep("review");
  }
}

async function refreshJob() {
  if (!state.job || state.busy) return;
  const button = el("refresh-job");
  setBusy(button, true, "Refreshing…");
  try {
    const before = state.job.revision;
    const hadLocalEdits = state.dirty;
    const job = await request("/api/jobs/" + state.job.id + "?v=" + Date.now());
    showCurrentJob(job);
    notify(hadLocalEdits
      ? "Local edits discarded; saved draft loaded."
      : job.revision === before ? "This is already the latest saved draft." : "Latest saved draft loaded.");
  } catch (error) {
    notify(error.message, true);
  } finally {
    setBusy(button, false);
  }
}

async function copyShareLink() {
  const input = el("share-url");
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(input.value);
    } else {
      input.select();
      if (!document.execCommand("copy")) throw new Error("Copy was blocked.");
    }
    notify("Link copied. Open it on the other device.");
  } catch (_) {
    input.focus();
    input.select();
    notify("Select and copy the link manually.", true);
  }
}

async function restore() {
  const sharedId = new URLSearchParams(window.location.search).get("job");
  const id = sharedId || localStorage.getItem("board2slides_job");
  if (!id) return;
  try {
    const job = await request("/api/jobs/" + encodeURIComponent(id) + "?v=" + Date.now());
    showCurrentJob(job);
  } catch (error) {
    localStorage.removeItem("board2slides_job");
    history.replaceState(null, "", "/");
    if (sharedId) notify("Could not open this shared draft. Check that the server is running and the link is complete.", true);
  }
}

el("camera-button").addEventListener("click", () => el("camera-input").click());
el("file-button").addEventListener("click", () => el("file-input").click());
el("camera-input").addEventListener("change", (event) => {
  queueFiles([...event.target.files]); event.target.value = "";
});
el("file-input").addEventListener("change", (event) => {
  queueFiles([...event.target.files]); event.target.value = "";
});
el("start-upload").addEventListener("click", () => begin(false));
el("sample-button").addEventListener("click", () => begin(true));
el("prepare-button").addEventListener("click", prepareAll);
el("prepare-mode").addEventListener("change", renderPrepare);
el("ocr-model").addEventListener("change", renderPrepare);
el("run-ocr-button").addEventListener("click", runSelectedOcr);
el("skip-ocr-button").addEventListener("click", () => {
  if (!state.job?.photos?.every((photo) => photo.preparation)) return;
  renderOcr(); showStep("review");
  notify("Enter at least one board line for every photo, then save.");
});
el("prepare-back").addEventListener("click", () => showStep("upload"));
el("ocr-lines").addEventListener("input", () => setDirty(true));
el("add-line").addEventListener("click", () => {
  const list = el("ocr-lines");
  if (list.children.length >= 60) { notify("Use up to 60 board lines in one deck.", true); return; }
  const photo = currentPhoto();
  const row = makeOcrRow({ id: photo ? manualId(photo.id) : "", photo_id: photo?.id, text: "" }, list.children.length);
  list.append(row);
  setDirty(true);
  row.querySelector("input").focus();
});
el("save-ocr").addEventListener("click", saveBoardText);
el("ocr-run").addEventListener("click", rerunOcr);
el("review-back").addEventListener("click", () => {
  if (state.dirty) { notify("Save board text before leaving this step.", true); return; }
  if (state.job?.mode === "photos") { renderPrepare(); showStep("prepare"); }
  else showStep("upload");
});
el("review-next").addEventListener("click", createDraft);
el("plan-back").addEventListener("click", () => {
  if (state.dirty) { notify("Save the slide draft before editing board text.", true); return; }
  renderOcr(); showStep("review");
});
el("plan-form").addEventListener("input", () => { updatePreview(); setDirty(true); });
el("visual-choice").addEventListener("change", () => { renderVisual(); setDirty(true); });
el("plan-form").addEventListener("submit", (event) => event.preventDefault());
el("save-plan").addEventListener("click", saveDraft);
el("export-button").addEventListener("click", exportDeck);
el("export-back").addEventListener("click", () => { renderPlan(); showStep("plan"); });
el("start-over").addEventListener("click", startOver);
el("refresh-job").addEventListener("click", refreshJob);
el("share-toggle").addEventListener("click", () => {
  const open = el("share-panel").hidden;
  el("share-panel").hidden = !open;
  el("share-toggle").setAttribute("aria-expanded", String(open));
  if (open) updateShare();
});
el("copy-share").addEventListener("click", copyShareLink);
restore();

request("/api/health").then((health) => {
  state.ocrReady = Boolean(health.ocr_ready);
  state.models = health.ocr_models || [];
  if (!state.ocrReady) {
    el("prepare-mode").value = "original";
    [...el("prepare-mode").options].forEach((option) => { if (option.value !== "original") option.disabled = true; });
  }
  const select = el("ocr-model");
  select.replaceChildren(...state.models.map((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.label + (model.available ? "" : " · Coming soon");
    option.disabled = !model.available;
    return option;
  }));
  select.value = state.job?.selected_model || "tuned";
  if (state.job?.mode === "photos" && state.step === "prepare") renderPrepare();
  if (state.job && state.step === "review" && !state.dirty) renderOcr();
  if (!health.llm_ready) {
    el("planner-mode").value = "demo";
    el("planner-mode").querySelector('option[value="llm"]').disabled = true;
  }
}).catch(() => {});
