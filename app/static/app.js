const state = { job: null, step: "upload", busy: false, dirty: false, originalVisual: null,
  ocrReady: false, models: [], files: [], activePhotoId: null, draftByPhoto: new Map(),
  plannerDraft: [], plannerDirty: false, previewIndex: 0, previewJobId: null,
  includedSlides: new Set() };
const stepNames = ["upload", "prepare", "review", "plan", "export"];
const slideOrder = ["cover", "overview", "visual", "example"];
const slideLabels = { cover: "Cover", overview: "Key points", visual: "Visual", example: "Activity" };
const regionColor = (index) => `hsl(${Math.round(index * 137.508) % 360} 67% 37%)`;
const el = (id) => document.getElementById(id);

// In-page replacement for window.confirm(). Same yes/no answer, styled to match the app.
// Cancel is focused first and Esc cancels, so the safe choice stays the default.
function askConfirm({ title, message, confirmLabel, cancelLabel = "Cancel" }) {
  let dialog = el("confirm-dialog");
  if (!dialog) {
    dialog = document.createElement("dialog");
    dialog.id = "confirm-dialog";
    dialog.className = "confirm-dialog";
    dialog.setAttribute("aria-labelledby", "confirm-title");
    dialog.setAttribute("aria-describedby", "confirm-message");
    dialog.innerHTML = '<h3 id="confirm-title"></h3><p id="confirm-message"></p>' +
      '<div class="confirm-actions"><button type="button" class="button secondary" data-answer="no"></button>' +
      '<button type="button" class="button primary danger" data-answer="yes"></button></div>';
    document.body.append(dialog);
  }
  if (typeof dialog.showModal !== "function") return Promise.resolve(window.confirm(`${title}\n\n${message}`));
  el("confirm-title").textContent = title;
  el("confirm-message").textContent = message;
  const yes = dialog.querySelector('[data-answer="yes"]');
  const no = dialog.querySelector('[data-answer="no"]');
  yes.textContent = confirmLabel;
  no.textContent = cancelLabel;
  return new Promise((resolve) => {
    const finish = (answer) => {
      dialog.removeEventListener("cancel", onCancel);
      yes.onclick = null; no.onclick = null;
      dialog.close();
      resolve(answer);
    };
    const onCancel = (event) => { event.preventDefault(); finish(false); };
    yes.onclick = () => finish(true);
    no.onclick = () => finish(false);
    dialog.addEventListener("cancel", onCancel);
    dialog.showModal();
    no.focus();
  });
}

// Before preparation runs, the "Prepared for OCR" pane shows a placeholder, not the original.
function preparedPlaceholder(image) {
  let note = image.parentElement.querySelector(".prep-empty");
  if (!note) {
    note = document.createElement("p");
    note.className = "prep-empty";
    note.textContent = "Not prepared yet. Press Prepare photos to see what the model will read.";
    image.parentElement.append(note);
  }
  return note;
}

// Outline the kept area on the original after a crop. Skipped when perspective was corrected,
// because the crop is then measured on the warped image and would not line up with the original.
function showCropOutline(image, preparation) {
  let stage = image.parentElement;
  if (!stage.classList.contains("prep-stage")) {
    stage = document.createElement("span");
    stage.className = "prep-stage";
    image.replaceWith(stage);
    stage.append(image);
  }
  let box = stage.querySelector(".crop-outline");
  if (!box) {
    box = document.createElement("span");
    box.className = "crop-outline";
    box.setAttribute("aria-hidden", "true");
    stage.append(box);
  }
  box.hidden = true;
  image.onload = null;
  const region = preparation?.region;
  const warped = preparation?.mode === "perspective" ||
    (preparation?.decisions || []).some((item) => item.name === "perspective" && item.applied);
  if (warped || !Array.isArray(region) || region.length !== 4) return;
  const place = () => {
    const width = image.naturalWidth, height = image.naturalHeight;
    if (!width || !height) return;
    const [x0, y0, x1, y1] = region;
    if (x0 <= 0 && y0 <= 0 && x1 >= width && y1 >= height) return;
    box.style.left = `${(x0 / width) * 100}%`;
    box.style.top = `${(y0 / height) * 100}%`;
    box.style.width = `${((x1 - x0) / width) * 100}%`;
    box.style.height = `${((y1 - y0) / height) * 100}%`;
    box.hidden = false;
  };
  image.onload = place;
  if (image.complete && image.naturalWidth) place();
}

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
  updateReviewEditLocks();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setDirty(dirty) {
  state.dirty = dirty;
  const anyDirty = dirty || state.plannerDirty;
  el("save-state").textContent = anyDirty
    ? "Unsaved changes on this device"
    : "Saved on this laptop server";
  el("refresh-job").textContent = anyDirty
    ? "Discard edits & refresh"
    : "Refresh from server";
  updateReviewEditLocks();
}

function setPlannerDirty(dirty) {
  state.plannerDirty = dirty;
  setDirty(state.dirty);
}

function updateReviewEditLocks() {
  if (state.step !== "review") return;
  el("ocr-lines").querySelectorAll("input, button").forEach((node) => { node.disabled = state.plannerDirty; });
  el("add-line").disabled = state.plannerDirty;
  el("save-ocr").disabled = state.plannerDirty;
  el("planner-lines").querySelectorAll("textarea, button").forEach((node) => { node.disabled = state.dirty; });
  el("save-planner-input").disabled = state.dirty;
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
  state.plannerDraft = (job.planner_input?.regions || []).map((line) => ({
    ...line, source_region_ids: [...(line.source_region_ids || [line.id])],
  }));
  state.plannerDirty = false;
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
  const original = el("prepare-original");
  const processed = el("prepare-processed");
  const prepared = Boolean(photo.preparation);
  original.src = photo.original_url;
  processed.hidden = !prepared;
  if (prepared) processed.src = photo.processed_url + version;
  else processed.removeAttribute("src");
  preparedPlaceholder(processed).hidden = prepared;
  showCropOutline(original, photo.preparation);
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
  if (state.job.ocr.regions.length && !(await askConfirm({
    title: "Prepare the photos again?",
    message: "This replaces the OCR text and any slide draft for this lesson.",
    confirmLabel: "Prepare again", cancelLabel: "Keep current text",
  }))) return;
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
  if (state.job.ocr.regions.length && !(await askConfirm({
    title: "Run OCR again?",
    message: "This replaces your reviewed lines and any slide draft.",
    confirmLabel: "Replace and re-read", cancelLabel: "Keep my edits",
  }))) return;
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
  renderPlannerInput();
  renderBoxes();
}

function renderPlannerInput() {
  const list = el("planner-lines");
  list.replaceChildren(...state.plannerDraft.map((line, index) => {
    const card = document.createElement("div");
    card.className = "planner-line";
    const head = document.createElement("div");
    head.className = "planner-line-head";
    const photoIndex = state.job.photos?.findIndex((photo) => photo.id === line.photo_id) ?? -1;
    const label = document.createElement("span");
    label.textContent = `Group ${index + 1}${photoIndex >= 0 ? ` · Photo ${photoIndex + 1}` : ""}`;
    const source = document.createElement("span");
    source.textContent = `${line.source_region_ids.length} source box${line.source_region_ids.length === 1 ? "" : "es"}`;
    head.append(label, source);
    const textarea = document.createElement("textarea");
    textarea.maxLength = 350;
    textarea.value = line.text;
    textarea.setAttribute("aria-label", `Joined text group ${index + 1}`);
    textarea.addEventListener("input", () => { line.text = textarea.value; setPlannerDirty(true); });
    const actions = document.createElement("div");
    actions.className = "planner-line-actions";
    const action = (title, enabled, callback) => {
      const button = document.createElement("button");
      button.type = "button"; button.textContent = title; button.disabled = !enabled;
      button.addEventListener("click", callback); actions.append(button);
    };
    const previous = state.plannerDraft[index - 1];
    const next = state.plannerDraft[index + 1];
    action("Split into boxes", line.source_region_ids.length > 1, async () => {
      const originals = line.source_region_ids.map((id) => state.job.ocr.regions.find((row) => row.id === id));
      if (line.text.trim() !== originals.map((row) => row.text).join(" ").trim() && !(await askConfirm({
        title: "Split this group?",
        message: "Splitting restores the saved text in each source box, so your edits to this group are lost.",
        confirmLabel: "Split and restore", cancelLabel: "Keep group",
      }))) return;
      state.plannerDraft.splice(index, 1, ...originals.map((row) => ({
        id: row.id, text: row.text, photo_id: row.photo_id || null, source_region_ids: [row.id],
      })));
      setPlannerDirty(true); renderPlannerInput();
    });
    action("Merge with previous", Boolean(previous && previous.photo_id === line.photo_id &&
      previous.text.length + line.text.length + 1 <= 350), () => {
      previous.text = `${previous.text.trim()} ${line.text.trim()}`.trim();
      previous.source_region_ids.push(...line.source_region_ids);
      state.plannerDraft.splice(index, 1);
      setPlannerDirty(true); renderPlannerInput();
    });
    action("Move up", Boolean(previous && previous.photo_id === line.photo_id), () => {
      [state.plannerDraft[index - 1], state.plannerDraft[index]] = [line, previous];
      setPlannerDirty(true); renderPlannerInput();
    });
    action("Move down", Boolean(next && next.photo_id === line.photo_id), () => {
      [state.plannerDraft[index + 1], state.plannerDraft[index]] = [line, next];
      setPlannerDirty(true); renderPlannerInput();
    });
    card.append(head, textarea, actions);
    return card;
  }));
  updateReviewEditLocks();
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
  if (state.plannerDirty) { notify("Save the joined text before changing the OCR model.", true); return; }
  if (state.job.mode === "photos") {
    if (state.dirty) { notify("Save your text edits before changing the OCR model.", true); return; }
    renderPrepare(); showStep("prepare"); return;
  }
  if (!(await askConfirm({
    title: "Run OCR again?",
    message: "This replaces the current board lines and any slide draft.",
    confirmLabel: "Replace and re-read", cancelLabel: "Keep my edits",
  }))) return;
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

async function persistPlannerInput() {
  const regions = state.plannerDraft.map((line) => ({
    id: line.source_region_ids[0], text: line.text.trim(),
    photo_id: line.photo_id || null, source_region_ids: line.source_region_ids,
  }));
  if (regions.some((line) => !line.text)) throw new Error("Complete every joined text group before saving.");
  const job = await request("/api/jobs/" + state.job.id + "/planner-input", {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ revision: state.job.revision, regions }),
  });
  setCurrentJob(job);
}

async function savePlannerInput() {
  if (!state.job || state.busy) return;
  if (!state.plannerDirty) { notify("Joined text is already saved."); return; }
  const button = el("save-planner-input");
  setBusy(button, true, "Saving…");
  try {
    await persistPlannerInput();
    renderOcr();
    notify("Joined text saved. The LLM will use this version.");
  } catch (error) { notify(error.message, true); }
  finally { setBusy(button, false); }
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
    if (state.plannerDirty) await persistPlannerInput();
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
  if (state.previewJobId !== state.job.id) state.previewIndex = 0;
  state.previewJobId = state.job.id;
  state.includedSlides = new Set(plan.included_slides ||
    (plan.visual?.type === "none" ? ["cover", "overview", "example"] : slideOrder));
  el("plan-title-input").value = plan.title;
  el("plan-summary").value = plan.summary;
  el("point-list").replaceChildren(...plan.key_points.map(makePoint));
  el("example-title").value = plan.example.title;
  el("example-description").value = plan.example.description;
  state.originalVisual = plan.visual || { type: "none", title: "Explanation only", reason: "No visual selected.", source_region_ids: [], data: {} };
  el("visual-choice").value = state.originalVisual.type === "none" ? "none" : "model";
  el("planner-tag").textContent = state.job.planner_mode === "llm" ? "Qwen + LoRA v3" : "Demo planner";
  const notes = state.job.plan_notes || [];
  el("plan-notes").replaceChildren(...notes.map((note) => {
    const item = document.createElement("li"); item.textContent = note; return item;
  }));
  el("plan-warning-card").hidden = !notes.length;
  el("regenerate-plan").hidden = state.job.planner_mode !== "llm";
  renderVisual();
  updatePreview();
  setDirty(false);
}

function renderVisual() {
  const visual = state.originalVisual;
  const useVisual = el("visual-choice").value === "model" && visual.type !== "none";
  el("visual-reason").textContent = useVisual
    ? visual.type.replaceAll("_", " ") + " · " + (visual.reason || "Review this visual against the board.")
    : "Explanation only. The recap slide repeats the summary, so it starts removed from the deck.";
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
  if (!state.job?.plan) return;
  const kind = slideOrder[state.previewIndex];
  const included = state.includedSlides.has(kind);
  const title = el("plan-title-input").value.trim() || "Your board topic";
  const summary = el("plan-summary").value.trim() || "Your reviewed board notes will appear here.";
  const points = [...el("point-list").children].map((card) => ({
    heading: card.querySelector("input").value.trim(),
    explanation: card.querySelector("textarea").value.trim(),
  }));
  const slide = el("preview-slide");
  slide.replaceChildren();
  slide.classList.toggle("preview-removed", !included);
  const accent = document.createElement("div"); accent.className = "preview-accent";
  const label = document.createElement("p"); label.className = "preview-label";
  label.textContent = kind === "cover" ? "STUDY DECK" : slideLabels[kind].toUpperCase();
  const heading = document.createElement("h3");
  const body = document.createElement("div"); body.className = "preview-body";
  const paragraph = (value, className = "") => {
    const node = document.createElement("p"); node.textContent = value; node.className = className;
    body.append(node);
  };
  if (kind === "cover") {
    heading.textContent = title;
    paragraph(summary);
  } else if (kind === "overview") {
    heading.textContent = "Lesson overview";
    paragraph(summary, "preview-lead");
    points.forEach((point, index) => paragraph(`${index + 1}. ${point.heading}: ${point.explanation}`));
  } else if (kind === "example") {
    heading.textContent = el("example-title").value.trim() || "Review activity";
    paragraph(el("example-description").value.trim() || "Your activity prompt appears here.", "preview-lead");
  } else {
    const visual = el("visual-choice").value === "model" && state.originalVisual.type !== "none"
      ? state.originalVisual : { type: "none", title: "No additional visual", data: {} };
    heading.textContent = visual.title || "Visual";
    if (visual.type === "none") paragraph("This slide would repeat the opening summary. You can leave it removed.");
    else if (["bar_chart", "line_chart"].includes(visual.type)) {
      paragraph(`${visual.type === "bar_chart" ? "Bar" : "Line"} chart · ${visual.data.unit || "values"}`, "preview-lead");
      (visual.data.labels || []).forEach((item, index) => paragraph(`${item}: ${(visual.data.series || []).map(series => series.values[index]).join(" / ")}`));
    } else if (visual.type === "process") {
      (visual.data.steps || []).forEach((step, index) => paragraph(`${index + 1}. ${step}`));
    } else if (visual.type === "comparison_table") {
      paragraph((visual.data.columns || []).join(" · "), "preview-lead");
      (visual.data.rows || []).forEach(row => paragraph(row.join(" · ")));
    } else if (visual.type === "formula") {
      paragraph(visual.data.expression || "Formula", "preview-lead");
      (visual.data.worked_steps || []).forEach((step, index) => paragraph(`${index + 1}. ${step}`));
    }
  }
  const rule = document.createElement("div"); rule.className = "preview-rule";
  const footer = document.createElement("span"); footer.textContent = "Board2Slides";
  slide.append(accent, label, heading, rule, body, footer);
  el("preview-count").textContent = `${String(state.previewIndex + 1).padStart(2, "0")} / 04`;
  el("preview-prev").disabled = state.previewIndex === 0;
  el("preview-next").disabled = state.previewIndex === slideOrder.length - 1;
  el("preview-toggle").textContent = included ? "Remove slide" : "Restore slide";
  el("preview-status").textContent = included
    ? `${slideLabels[kind]} is included · ${state.includedSlides.size} slide${state.includedSlides.size === 1 ? "" : "s"} in the PowerPoint.`
    : `${slideLabels[kind]} is removed from the PowerPoint. Restore it any time.`;
  const list = el("preview-slide-list");
  list.replaceChildren(...slideOrder.map((id, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preview-tab" + (index === state.previewIndex ? " active" : "") +
      (state.includedSlides.has(id) ? "" : " removed");
    button.textContent = `${index + 1}. ${slideLabels[id]}`;
    button.setAttribute("aria-label", `${slideLabels[id]} slide, ${state.includedSlides.has(id) ? "included" : "removed"}`);
    if (index === state.previewIndex) button.setAttribute("aria-current", "step");
    button.addEventListener("click", () => { state.previewIndex = index; updatePreview(); });
    return button;
  }));
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
          source_region_ids: state.job.planner_mode === "llm"
            ? state.job.planner_input.regions.map((region) => region.id)
            : state.job.ocr.regions.map((region) => region.id), data: {} },
    uncertain_ocr_terms: state.job.plan.uncertain_ocr_terms || [],
    included_slides: slideOrder.filter((slide) => state.includedSlides.has(slide)),
  };
  if (!plan.included_slides.length) throw new Error("Keep at least one slide in the deck.");
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
    renderPlan();
    notify("Slide draft saved. Refresh on the other device to see it.");
  } catch (error) {
    notify(error.message, true);
  } finally {
    setBusy(button, false);
  }
}

async function regenerateDraft() {
  if (!state.job || state.busy || state.job.planner_mode !== "llm") return;
  if (state.dirty) {
    notify("Save your slide edits before regenerating; regeneration replaces the current draft.", true);
    return;
  }
  if (!(await askConfirm({
    title: "Regenerate the LLM draft?",
    message: "The new result will replace your saved slide edits.",
    confirmLabel: "Regenerate", cancelLabel: "Keep this draft",
  }))) return;
  const button = el("regenerate-plan");
  setBusy(button, true, "Regenerating…");
  try {
    const job = await request("/api/jobs/" + state.job.id + "/plan", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision: state.job.revision, mode: "llm", regenerate: true }),
    });
    setCurrentJob(job); renderPlan();
    notify("New LLM draft ready. Review every slide before export.");
  } catch (error) { notify(error.message, true); }
  finally { setBusy(button, false); }
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
  state.plannerDraft = [];
  state.plannerDirty = false;
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
    const hadLocalEdits = state.dirty || state.plannerDirty;
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
el("save-planner-input").addEventListener("click", savePlannerInput);
el("ocr-run").addEventListener("click", rerunOcr);
el("review-back").addEventListener("click", () => {
  if (state.dirty || state.plannerDirty) { notify("Save board and joined text before leaving this step.", true); return; }
  if (state.job?.mode === "photos") { renderPrepare(); showStep("prepare"); }
  else showStep("upload");
});
el("review-next").addEventListener("click", createDraft);
el("plan-back").addEventListener("click", () => {
  if (state.dirty) { notify("Save the slide draft before editing board text.", true); return; }
  renderOcr(); showStep("review");
});
el("plan-form").addEventListener("input", () => { updatePreview(); setDirty(true); });
el("visual-choice").addEventListener("change", () => {
  if (el("visual-choice").value === "none") state.includedSlides.delete("visual");
  else if (state.originalVisual.type !== "none") state.includedSlides.add("visual");
  renderVisual(); updatePreview(); setDirty(true);
});
el("preview-prev").addEventListener("click", () => { state.previewIndex--; updatePreview(); });
el("preview-next").addEventListener("click", () => { state.previewIndex++; updatePreview(); });
el("preview-toggle").addEventListener("click", () => {
  const kind = slideOrder[state.previewIndex];
  if (state.includedSlides.has(kind)) {
    if (state.includedSlides.size === 1) { notify("Keep at least one slide in the deck.", true); return; }
    state.includedSlides.delete(kind);
  } else state.includedSlides.add(kind);
  updatePreview(); setDirty(true);
});
el("plan-form").addEventListener("submit", (event) => event.preventDefault());
el("save-plan").addEventListener("click", saveDraft);
el("regenerate-plan").addEventListener("click", regenerateDraft);
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
  if (state.job && state.step === "review" && !state.dirty && !state.plannerDirty) renderOcr();
  if (!health.llm_ready) {
    el("planner-mode").value = "demo";
    el("planner-mode").querySelector('option[value="llm"]').disabled = true;
  }
}).catch(() => {});
