const API_BASE = window.location.origin.includes("http")
  ? window.location.origin
  : "http://127.0.0.1:8000";

let state = {
  activeFormat: "T20",
  cameraViewMode: "raw",
  // Mode can be "server_folder", "browser_folder", or "manual_file"
  sourceMode: "server_folder",
  watchedFolderPath: "samples",
  lastFolderSignature: null,
  autoWatchEnabled: true,
  watchIntervalId: null,
  autoCycleEnabled: false,
  autoCycleIntervalId: null,

  // Server folder file lists
  creaseFiles: [],
  waistFiles: [],
  activeCreaseFile: "",
  activeWaistFile: "",

  // Browser folder File objects map (if user used Browse Folder...)
  browserCreaseMap: {},
  browserWaistMap: {},

  // Manual or browser blobs
  footBlob: null,
  footFilename: "",
  footRawUrl: null,
  footGradcamUrl: null,
  waistBlob: null,
  waistFilename: "",
  waistRawUrl: null,
  waistGradcamUrl: null,

  historyCache: [],
};

document.addEventListener("DOMContentLoaded", async () => {
  setupDragAndDrop();
  renderDefaultComplianceMatrix();
  updateFieldingRadar();
  await checkBackendStatus();
  await loadModelMetrics();
  await refreshHistoryLedger();

  // Initialize folder watcher and evaluate the first detected pair in /crease and /waist
  await syncWatchedFolder(true);
  startFolderWatcherLoop();
});

function switchTab(tabId) {
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-section").forEach((sec) => {
    sec.classList.toggle("active", sec.id === `tab-${tabId}`);
  });
  if (tabId === "analytics") {
    refreshHistoryLedger();
  } else if (tabId === "models") {
    loadModelMetrics();
  }
}

async function checkBackendStatus() {
  const dot = document.getElementById("apiStatusDot");
  const text = document.getElementById("apiStatusText");
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    const data = await res.json();
    dot.className = "status-indicator online";
    text.textContent = `SYSTEM ONLINE | ${data.device.toUpperCase()}`;
  } catch (e) {
    dot.className = "status-indicator";
    text.textContent = "BACKEND OFFLINE";
  }
}

/* ============================================================================
   AUTOMATED FOLDER WATCHER (/crease & /waist SUBFOLDER INGESTION)
   ============================================================================ */

function startFolderWatcherLoop() {
  if (state.watchIntervalId) clearInterval(state.watchIntervalId);
  state.watchIntervalId = setInterval(async () => {
    if (state.autoWatchEnabled && state.sourceMode === "server_folder") {
      await syncWatchedFolder(false);
    }
  }, 1500);
}

function toggleAutoWatch() {
  state.autoWatchEnabled = !state.autoWatchEnabled;
  const btn = document.getElementById("btnToggleAutoWatch");
  const text = document.getElementById("autoWatchText");
  btn.classList.toggle("active", state.autoWatchEnabled);
  text.textContent = state.autoWatchEnabled ? "AUTO-WATCH ACTIVE" : "AUTO-WATCH PAUSED";
}

async function applyWatchFolderPath() {
  const input = document.getElementById("inputWatchFolderPath");
  const folderPath = (input.value || "samples").trim();
  if (!folderPath) return;

  state.sourceMode = "server_folder";
  updateQuickFolderChips(folderPath);

  try {
    const res = await fetch(`${API_BASE}/api/folder/select`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_path: folderPath }),
    });
    const data = await res.json();
    applyServerFolderState(data, true);
  } catch (err) {
    console.error("Failed to connect watched folder:", err);
  }
}

async function quickSelectFolder(folderName) {
  const input = document.getElementById("inputWatchFolderPath");
  input.value = folderName;
  updateQuickFolderChips(folderName);
  await applyWatchFolderPath();
}

function updateQuickFolderChips(folderPath) {
  const norm = folderPath.replace(/\\/g, "/").toLowerCase();
  const chipSamples = document.getElementById("chipSamples");
  const chipLive = document.getElementById("chipLiveFeed");
  if (chipSamples) {
    chipSamples.classList.toggle("active", norm.endsWith("samples") || norm === "samples/");
  }
  if (chipLive) {
    chipLive.classList.toggle("active", norm.endsWith("live_feed") || norm === "live_feed/");
  }
}

async function syncWatchedFolder(forceEvaluate = false) {
  try {
    const res = await fetch(`${API_BASE}/api/folder/status`);
    const data = await res.json();
    applyServerFolderState(data, forceEvaluate);
  } catch (err) {
    console.error("Error polling watched folder:", err);
  }
}

function applyServerFolderState(data, forceEvaluate = false) {
  if (!data) return;

  const sigChanged = data.signature !== state.lastFolderSignature;
  if (!sigChanged && !forceEvaluate) {
    return;
  }

  state.lastFolderSignature = data.signature;
  state.watchedFolderPath = data.folder_path;
  state.creaseFiles = data.crease_files || [];
  state.waistFiles = data.waist_files || [];

  const pathInput = document.getElementById("inputWatchFolderPath");
  if (pathInput && document.activeElement !== pathInput) {
    pathInput.value = data.folder_path;
    updateQuickFolderChips(data.folder_path);
  }

  document.getElementById("creaseFolderCount").textContent = `${data.crease_count || 0} images in /crease`;
  document.getElementById("waistFolderCount").textContent = `${data.waist_count || 0} images in /waist`;

  // Populate /crease selector (sorted newest-first by backend)
  const creaseSelect = document.getElementById("selectCreaseQueue");
  if (state.creaseFiles.length > 0) {
    creaseSelect.innerHTML = state.creaseFiles
      .map((f) => `<option value="${f.name}">${f.name}</option>`)
      .join("");
    state.activeCreaseFile = data.latest_crease || state.creaseFiles[0].name;
    creaseSelect.value = state.activeCreaseFile;
  } else {
    creaseSelect.innerHTML = `<option value="">No images in /crease</option>`;
    state.activeCreaseFile = "";
  }

  // Populate /waist selector (with optional "NONE" choice)
  const waistSelect = document.getElementById("selectWaistQueue");
  if (state.waistFiles.length > 0) {
    waistSelect.innerHTML =
      state.waistFiles.map((f) => `<option value="${f.name}">${f.name}</option>`).join("") +
      `<option value="NONE">-- Disable Waist Feed --</option>`;
    state.activeWaistFile = data.latest_waist || state.waistFiles[0].name;
    waistSelect.value = state.activeWaistFile;
  } else {
    waistSelect.innerHTML = `<option value="NONE">No images in /waist</option>`;
    state.activeWaistFile = "NONE";
  }

  // Automatically load preview images and run evaluation
  loadSelectedSubfolderPreviewsAndEvaluate();
}

function handleBrowserDirectoryPick(event) {
  const files = Array.from(event.target.files || []);
  if (!files.length) return;

  const creaseMap = {};
  const waistMap = {};
  let rootFolderName = "Selected Folder";

  files.forEach((file) => {
    const rel = (file.webkitRelativePath || file.name).replace(/\\/g, "/");
    const parts = rel.split("/");
    if (parts.length > 1) {
      rootFolderName = parts[0];
    }
    const lowerRel = rel.toLowerCase();
    const isImage = /\.(jpg|jpeg|png|webp|bmp)$/i.test(file.name);
    if (!isImage) return;

    if (lowerRel.includes("/crease/")) {
      creaseMap[file.name] = file;
    } else if (lowerRel.includes("/waist/")) {
      waistMap[file.name] = file;
    }
  });

  const creaseNames = Object.keys(creaseMap).sort();
  const waistNames = Object.keys(waistMap).sort();

  if (creaseNames.length === 0 && waistNames.length === 0) {
    alert(
      "No images found under '/crease' or '/waist' subfolders in the selected directory. Please choose a parent folder that contains 'crease' and 'waist' subfolders."
    );
    return;
  }

  state.sourceMode = "browser_folder";
  state.browserCreaseMap = creaseMap;
  state.browserWaistMap = waistMap;
  state.creaseFiles = creaseNames.map((name) => ({ name }));
  state.waistFiles = waistNames.map((name) => ({ name }));

  document.getElementById("inputWatchFolderPath").value = rootFolderName;
  document.getElementById("creaseFolderCount").textContent = `${creaseNames.length} images in /crease`;
  document.getElementById("waistFolderCount").textContent = `${waistNames.length} images in /waist`;

  const creaseSelect = document.getElementById("selectCreaseQueue");
  creaseSelect.innerHTML = creaseNames.map((n) => `<option value="${n}">${n}</option>`).join("");
  state.activeCreaseFile = creaseNames[0] || "";

  const waistSelect = document.getElementById("selectWaistQueue");
  waistSelect.innerHTML =
    waistNames.map((n) => `<option value="${n}">${n}</option>`).join("") +
    `<option value="NONE">-- Disable Waist Feed --</option>`;
  state.activeWaistFile = waistNames[0] || "NONE";

  loadSelectedSubfolderPreviewsAndEvaluate();
}

async function onSubfolderFileChange() {
  const creaseSelect = document.getElementById("selectCreaseQueue");
  const waistSelect = document.getElementById("selectWaistQueue");
  state.activeCreaseFile = creaseSelect.value;
  state.activeWaistFile = waistSelect.value;

  if (state.sourceMode === "manual_file") {
    state.sourceMode = "server_folder";
  }

  await loadSelectedSubfolderPreviewsAndEvaluate();
}

async function stepNextFolderPair() {
  const creaseSelect = document.getElementById("selectCreaseQueue");
  const waistSelect = document.getElementById("selectWaistQueue");

  if (creaseSelect && creaseSelect.options.length > 0) {
    creaseSelect.selectedIndex = (creaseSelect.selectedIndex + 1) % creaseSelect.options.length;
    state.activeCreaseFile = creaseSelect.value;
  }

  if (waistSelect && waistSelect.options.length > 1) {
    // Cycle among actual waist files (excluding the last "NONE" option)
    const realCount = waistSelect.options.length - 1;
    const nextIdx = (waistSelect.selectedIndex + 1) % realCount;
    waistSelect.selectedIndex = nextIdx;
    state.activeWaistFile = waistSelect.value;
  }

  // Also increment ball counter for realistic match progression
  const ballEl = document.getElementById("inputBallIndex");
  const overEl = document.getElementById("inputOver");
  let nextBall = parseInt(ballEl.value || "1", 10) + 1;
  ballEl.value = nextBall;
  overEl.value = Math.floor((nextBall - 1) / 6);
  updateFieldingRadar();

  await loadSelectedSubfolderPreviewsAndEvaluate();
}

function toggleAutoCycleQueue() {
  state.autoCycleEnabled = !state.autoCycleEnabled;
  const btn = document.getElementById("btnAutoCycle");
  if (state.autoCycleEnabled) {
    btn.textContent = "Stop Auto-Play Queue";
    btn.style.borderColor = "#10b981";
    btn.style.color = "#34d399";
    state.autoCycleIntervalId = setInterval(() => {
      stepNextFolderPair();
    }, 3500);
  } else {
    btn.textContent = "Auto-Play Folder Queue";
    btn.style.borderColor = "";
    btn.style.color = "";
    if (state.autoCycleIntervalId) {
      clearInterval(state.autoCycleIntervalId);
      state.autoCycleIntervalId = null;
    }
  }
}

async function loadSelectedSubfolderPreviewsAndEvaluate() {
  if (state.sourceMode === "browser_folder") {
    const cFile = state.browserCreaseMap[state.activeCreaseFile];
    if (cFile) {
      state.footBlob = cFile;
      state.footFilename = cFile.name;
      state.footRawUrl = URL.createObjectURL(cFile);
      state.footGradcamUrl = null;
      document.getElementById("footFileName").textContent = `crease/${cFile.name}`;
    }

    const wFile =
      state.activeWaistFile && state.activeWaistFile !== "NONE"
        ? state.browserWaistMap[state.activeWaistFile]
        : null;
    if (wFile) {
      state.waistBlob = wFile;
      state.waistFilename = wFile.name;
      state.waistRawUrl = URL.createObjectURL(wFile);
      state.waistGradcamUrl = null;
      document.getElementById("waistFileName").textContent = `waist/${wFile.name}`;
    } else {
      clearWaistImageStateOnly();
    }

    refreshCameraImages();
    await runMultipartEvaluation();
    return;
  }

  // Server folder mode: load directly from /api/folder/image/{sub}
  if (state.activeCreaseFile) {
    const ts = Date.now();
    state.footRawUrl = `${API_BASE}/api/folder/image/crease?file=${encodeURIComponent(
      state.activeCreaseFile
    )}&t=${ts}`;
    state.footFilename = state.activeCreaseFile;
    state.footGradcamUrl = null;
    document.getElementById("footFileName").textContent = `crease/${state.activeCreaseFile}`;
  } else {
    state.footRawUrl = null;
    document.getElementById("footFileName").textContent = "No image in /crease";
  }

  if (state.activeWaistFile && state.activeWaistFile !== "NONE") {
    const ts = Date.now();
    state.waistRawUrl = `${API_BASE}/api/folder/image/waist?file=${encodeURIComponent(
      state.activeWaistFile
    )}&t=${ts}`;
    state.waistFilename = state.activeWaistFile;
    state.waistGradcamUrl = null;
    document.getElementById("waistFileName").textContent = `waist/${state.activeWaistFile}`;
  } else {
    clearWaistImageStateOnly();
  }

  refreshCameraImages();
  await runServerFolderEvaluation();
}

async function triggerCurrentEvaluation() {
  if (state.sourceMode === "server_folder") {
    await runServerFolderEvaluation();
  } else {
    await runMultipartEvaluation();
  }
}

async function runServerFolderEvaluation() {
  if (!state.activeCreaseFile) return;

  const btn = document.getElementById("btnEvaluateDelivery");
  btn.disabled = true;
  btn.innerHTML = `<span>Analyzing Subfolders...</span>`;

  const formData = new FormData();
  formData.append("crease_file", state.activeCreaseFile);
  formData.append("waist_file", state.activeWaistFile || "NONE");
  appendMatchContextToForm(formData);

  try {
    const res = await fetch(`${API_BASE}/api/folder/evaluate`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const result = await res.json();
    renderDeliveryVerdict(result);
    await refreshHistoryLedger();
  } catch (err) {
    console.error("Folder evaluation error:", err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span>Evaluate Delivery</span>`;
  }
}

async function runMultipartEvaluation() {
  if (!state.footBlob) {
    return;
  }

  const btn = document.getElementById("btnEvaluateDelivery");
  btn.disabled = true;
  btn.innerHTML = `<span>Analyzing Frame...</span>`;

  const formData = new FormData();
  formData.append("foot_image", state.footBlob, state.footFilename || "foot.jpg");
  if (state.waistBlob) {
    formData.append("waist_image", state.waistBlob, state.waistFilename || "waist.jpg");
  }
  appendMatchContextToForm(formData);

  try {
    const res = await fetch(`${API_BASE}/check-delivery`, {
      method: "POST",
      body: formData,
    });
    const result = await res.json();
    renderDeliveryVerdict(result);
    await refreshHistoryLedger();
  } catch (err) {
    console.error("Multipart evaluation error:", err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span>Evaluate Delivery</span>`;
  }
}

function appendMatchContextToForm(formData) {
  formData.append("match_format", state.activeFormat);
  formData.append("over", document.getElementById("inputOver").value || "0");
  formData.append("ball_index_in_innings", document.getElementById("inputBallIndex").value || "1");
  formData.append(
    "bouncers_in_over_before",
    document.getElementById("sliderBouncersPrior").value || "0"
  );
  formData.append(
    "is_current_ball_bouncer",
    document.getElementById("chkCurrentBouncer").checked
  );
  formData.append("is_beamer", document.getElementById("chkManualBeamer").checked);
  formData.append("fielders_leg_side", document.getElementById("sliderLegSide").value || "0");
  formData.append(
    "fielders_behind_square",
    document.getElementById("sliderBehindSquare").value || "0"
  );
  formData.append(
    "fielders_outside_circle",
    document.getElementById("sliderOutsideCircle").value || "0"
  );
  formData.append("bowler_name", document.getElementById("inputBowler").value || "Bowler");
  formData.append("batter_name", document.getElementById("inputBatter").value || "Batter");
}

/* ============================================================================
   MANUAL FILE OVERRIDE & DRAG-AND-DROP
   ============================================================================ */

function setupDragAndDrop() {
  [
    { zoneId: "creaseDropZone", type: "foot" },
    { zoneId: "waistDropZone", type: "waist" },
  ].forEach(({ zoneId, type }) => {
    const zone = document.getElementById(zoneId);
    if (!zone) return;
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.style.borderColor = "#38bdf8";
    });
    zone.addEventListener("dragleave", () => {
      zone.style.borderColor = "";
    });
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.style.borderColor = "";
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        loadUserFile(e.dataTransfer.files[0], type);
      }
    });
  });
}

function handleFileSelect(event, type) {
  if (event.target.files && event.target.files[0]) {
    loadUserFile(event.target.files[0], type);
  }
}

async function loadUserFile(file, type) {
  // If we were in server_folder mode and the user overrides one camera manually,
  // ensure the other camera has a valid Blob if needed
  if (state.sourceMode === "server_folder") {
    if (type === "waist" && !state.footBlob && state.footRawUrl) {
      const r = await fetch(state.footRawUrl);
      state.footBlob = await r.blob();
    } else if (type === "foot" && !state.waistBlob && state.waistRawUrl) {
      const r = await fetch(state.waistRawUrl);
      state.waistBlob = await r.blob();
    }
  }

  state.sourceMode = "manual_file";

  if (type === "foot") {
    state.footBlob = file;
    state.footFilename = file.name;
    state.footRawUrl = URL.createObjectURL(file);
    state.footGradcamUrl = null;
    document.getElementById("footFileName").textContent = file.name;
  } else {
    state.waistBlob = file;
    state.waistFilename = file.name;
    state.waistRawUrl = URL.createObjectURL(file);
    state.waistGradcamUrl = null;
    document.getElementById("waistFileName").textContent = file.name;
  }
  refreshCameraImages();
  await runMultipartEvaluation();
}

function clearWaistImageStateOnly() {
  state.waistBlob = null;
  state.waistFilename = "";
  state.waistRawUrl = null;
  state.waistGradcamUrl = null;
  document.getElementById("waistFileName").textContent = "Waist feed disabled";
  document.getElementById("beamerProbText").textContent = "N/A";
  document.getElementById("beamerProbBar").style.width = "0%";
}

function clearWaistImage() {
  clearWaistImageStateOnly();
  const waistSelect = document.getElementById("selectWaistQueue");
  if (waistSelect) {
    waistSelect.value = "NONE";
    state.activeWaistFile = "NONE";
  }
  refreshCameraImages();
  triggerCurrentEvaluation();
}

function setCameraViewMode(mode) {
  state.cameraViewMode = mode;
  document.getElementById("btnViewRaw").classList.toggle("active", mode === "raw");
  document.getElementById("btnViewGradcam").classList.toggle("active", mode === "gradcam");
  refreshCameraImages();
}

function refreshCameraImages() {
  const footImg = document.getElementById("footPreviewImg");
  const footPh = document.getElementById("footPlaceholder");
  if (state.footRawUrl) {
    const src =
      state.cameraViewMode === "gradcam" && state.footGradcamUrl
        ? state.footGradcamUrl
        : state.footRawUrl;
    footImg.src = src;
    footImg.classList.remove("hidden");
    footPh.classList.add("hidden");
  } else {
    footImg.classList.add("hidden");
    footPh.classList.remove("hidden");
  }

  const waistImg = document.getElementById("waistPreviewImg");
  const waistPh = document.getElementById("waistPlaceholder");
  if (state.waistRawUrl) {
    const src =
      state.cameraViewMode === "gradcam" && state.waistGradcamUrl
        ? state.waistGradcamUrl
        : state.waistRawUrl;
    waistImg.src = src;
    waistImg.classList.remove("hidden");
    waistPh.classList.add("hidden");
  } else {
    waistImg.classList.add("hidden");
    waistPh.classList.remove("hidden");
  }
}

/* ============================================================================
   ICC / MCC RULE ENGINE & INTERACTIVE PITCH FIELDING RADAR
   ============================================================================ */

function selectFormat(fmt, triggerEval = true) {
  state.activeFormat = fmt;
  document.querySelectorAll(".fmt-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.fmt === fmt);
  });
  updateFieldingRadar();
  if (triggerEval) {
    triggerCurrentEvaluation();
  }
}

function getFormatLimits(fmt, over, ballIdx) {
  const f = fmt.toUpperCase();
  if (f === "T20" || f === "IPL") {
    return {
      phase: over < 6 ? `${f} Powerplay (Overs 1-6)` : `${f} Non-Powerplay (Overs 7-20)`,
      maxOutside: over < 6 ? 2 : 5,
      maxBouncers: f === "IPL" ? 2 : 1,
    };
  }
  if (f === "ODI") {
    if (over < 10) return { phase: "ODI Powerplay 1 (Overs 1-10)", maxOutside: 2, maxBouncers: 2 };
    if (over < 40) return { phase: "ODI Powerplay 2 (Overs 11-40)", maxOutside: 4, maxBouncers: 2 };
    return { phase: "ODI Powerplay 3 (Overs 41-50)", maxOutside: 5, maxBouncers: 2 };
  }
  if (f === "100-BALL") {
    return {
      phase: ballIdx <= 25 ? "100-Ball Powerplay (1-25)" : "100-Ball Standard Play",
      maxOutside: ballIdx <= 25 ? 2 : 5,
      maxBouncers: 1,
    };
  }
  return { phase: "Test Match Playing Conditions", maxOutside: 9, maxBouncers: 2 };
}

function updateFieldingRadar() {
  const over = parseInt(document.getElementById("inputOver").value || "0", 10);
  const ballIdx = parseInt(document.getElementById("inputBallIndex").value || "1", 10);
  const outside = parseInt(document.getElementById("sliderOutsideCircle").value, 10);
  const legSide = parseInt(document.getElementById("sliderLegSide").value, 10);
  const behindSq = parseInt(document.getElementById("sliderBehindSquare").value, 10);
  const bouncersPrior = parseInt(document.getElementById("sliderBouncersPrior").value, 10);

  const limits = getFormatLimits(state.activeFormat, over, ballIdx);
  document.getElementById("activePhaseBadge").textContent = limits.phase;

  document.getElementById("valOutsideCircle").textContent =
    state.activeFormat.toUpperCase() === "TEST"
      ? `${outside} (Unrestricted)`
      : `${outside} / ${limits.maxOutside} max`;
  document.getElementById("valLegSide").textContent = `${legSide} / 5 max`;
  document.getElementById("valBehindSquare").textContent = `${behindSq} / 2 max`;
  document.getElementById("valBouncersPrior").textContent = `${bouncersPrior} / ${limits.maxBouncers} max`;

  const circleViolated = state.activeFormat.toUpperCase() !== "TEST" && outside > limits.maxOutside;
  const legViolated = legSide > 5;
  const behindViolated = behindSq > 2;
  const anyFieldViolation = circleViolated || legViolated || behindViolated;

  const badge = document.getElementById("radarRuleBadge");
  badge.textContent = anyFieldViolation ? "FIELDING VIOLATION" : "FIELDING LEGAL";
  badge.className = `radar-status ${anyFieldViolation ? "fail" : "pass"}`;

  document.getElementById("thirtyYardCircle").setAttribute(
    "stroke",
    circleViolated ? "#F43F5E" : "#38BDF8"
  );
  document.getElementById("legSideZonePath").setAttribute(
    "fill",
    legViolated ? "rgba(244,63,94,0.22)" : "rgba(59,130,246,0.06)"
  );
  document.getElementById("behindSquareZonePath").setAttribute(
    "fill",
    behindViolated ? "rgba(244,63,94,0.32)" : "rgba(245,158,11,0.08)"
  );

  const layer = document.getElementById("radarFieldersLayer");
  const dots = [];

  dots.push({ x: 160, y: 102, color: "#38BDF8", label: "WK" });
  dots.push({ x: 160, y: 215, color: "#38BDF8", label: "BWL" });

  const bsCoords = [
    [202, 92],
    [238, 72],
    [215, 52],
    [262, 98],
    [188, 65],
  ];
  for (let i = 0; i < behindSq; i++) {
    const [x, y] = bsCoords[i % bsCoords.length];
    dots.push({
      x,
      y,
      color: i >= 2 ? "#F43F5E" : "#FBBF24",
    });
  }

  const remLeg = Math.max(0, legSide - behindSq);
  const legInCoords = [
    [205, 152],
    [216, 185],
    [194, 212],
    [228, 140],
  ];
  const legOutCoords = [
    [275, 155],
    [268, 218],
    [238, 265],
    [286, 190],
  ];

  let outsideRemaining = outside;
  for (let i = 0; i < remLeg; i++) {
    const isOut = outsideRemaining > 0;
    if (isOut) outsideRemaining--;
    const pool = isOut ? legOutCoords : legInCoords;
    const [x, y] = pool[i % pool.length];
    dots.push({
      x,
      y,
      color: legViolated || (isOut && circleViolated) ? "#F43F5E" : "#10B981",
    });
  }

  const totalPlaced = behindSq + remLeg;
  const offCount = Math.max(0, 9 - totalPlaced);
  const offInCoords = [
    [122, 112],
    [108, 148],
    [114, 186],
    [132, 210],
    [102, 126],
  ];
  const offOutCoords = [
    [48, 135],
    [56, 198],
    [82, 258],
    [76, 75],
    [42, 168],
  ];

  for (let i = 0; i < offCount; i++) {
    const isOut = outsideRemaining > 0;
    if (isOut) outsideRemaining--;
    const pool = isOut ? offOutCoords : offInCoords;
    const [x, y] = pool[i % pool.length];
    dots.push({
      x,
      y,
      color: isOut && circleViolated ? "#F43F5E" : "#10B981",
    });
  }

  layer.innerHTML = dots
    .map(
      (d) => `
      <circle cx="${d.x}" cy="${d.y}" r="${d.label ? 5.2 : 5}" fill="${d.color}" stroke="#060a12" stroke-width="1.5" />
    `
    )
    .join("");
}

/* ============================================================================
   VERDICT RENDERING & TELEMETRY LEDGER
   ============================================================================ */

function renderDeliveryVerdict(res) {
  if (res.telemetry) {
    state.footGradcamUrl = res.telemetry.foot_gradcam || null;
    state.waistGradcamUrl = res.telemetry.waist_gradcam || null;
    refreshCameraImages();
  }

  const overstepPct = ((res.ai_overstep_probability || 0) * 100).toFixed(1);
  const footBar = document.getElementById("overstepProbBar");
  document.getElementById("overstepProbText").textContent = `${overstepPct}%`;
  footBar.style.width = `${Math.min(100, overstepPct)}%`;
  footBar.style.backgroundColor = overstepPct > 50 ? "#f43f5e" : "#10b981";

  const waistBar = document.getElementById("beamerProbBar");
  if (res.ai_beamer_probability !== null && res.ai_beamer_probability !== undefined) {
    const beamerPct = (res.ai_beamer_probability * 100).toFixed(1);
    document.getElementById("beamerProbText").textContent = `${beamerPct}%`;
    waistBar.style.width = `${Math.min(100, beamerPct)}%`;
    waistBar.style.backgroundColor = beamerPct > 50 ? "#f43f5e" : "#10b981";
  } else {
    document.getElementById("beamerProbText").textContent = "N/A";
    waistBar.style.width = "0%";
  }

  const banner = document.getElementById("verdictBanner");
  const title = document.getElementById("verdictTitle");
  const sub = document.getElementById("verdictSub");
  const freeHit = document.getElementById("freeHitPill");

  banner.className = `verdict-banner ${res.is_no_ball ? "noball" : "legal"}`;
  title.textContent = res.is_no_ball ? "NO-BALL SIGNALLED" : "LEGAL DELIVERY";
  sub.textContent = res.is_no_ball
    ? `${res.reasons.length} law infringement(s) detected across ${res.foot_file || "crease"}${
        res.waist_file ? " & " + res.waist_file : ""
      }.`
    : `All optical computer vision checks and ICC/MCC playing conditions satisfied (${res.foot_file || "crease"}).`;

  freeHit.classList.toggle("hidden", !res.free_hit_awarded);
  document.getElementById("metaDeliveryId").textContent = `#${res.delivery_id || "LIVE"}`;
  document.getElementById("metaLatency").textContent = `${res.inference_time_ms || 12} ms`;

  const reasonsBox = document.getElementById("reasonsContainer");
  const reasonsList = document.getElementById("reasonsList");
  if (res.reasons && res.reasons.length > 0) {
    reasonsBox.classList.remove("hidden");
    reasonsList.innerHTML = res.reasons.map((r) => `<li>${r}</li>`).join("");
  } else {
    reasonsBox.classList.add("hidden");
  }

  if (res.rule_details && res.rule_details.checks) {
    document.getElementById("activePhaseBadge").textContent = res.rule_details.powerplay_phase;
    const grid = document.getElementById("complianceGrid");
    grid.innerHTML = res.rule_details.checks
      .map(
        (c) => `
      <div class="law-card ${c.passed ? "pass" : "fail"}">
        <div class="law-top">
          <span class="law-code">${c.law}</span>
          <span class="law-status ${c.passed ? "pass" : "fail"}">${c.passed ? "PASS" : "VIOLATION"}</span>
        </div>
        <div class="law-title">${c.title}</div>
        <div class="law-metric">${c.metric}</div>
      </div>
    `
      )
      .join("");
  }
}

function renderDefaultComplianceMatrix() {
  const defaultChecks = [
    { law: "MCC Law 21.5", title: "Front-Foot Popping Crease", metric: "Awaiting frame" },
    { law: "MCC Law 41.7", title: "Waist-High Full Toss", metric: "Awaiting frame" },
    { law: "MCC Law 41.6", title: "Short-Pitched Bouncer Quota", metric: "0 / 1 in over" },
    { law: "ICC Clause 28.7", title: "30-Yard Powerplay Circle", metric: "2 / 2 outside" },
    { law: "MCC Law 28.4", title: "On-Side (Leg-Side) Limit", metric: "3 / 5 on leg side" },
    { law: "MCC Law 28.4", title: "Behind Square Leg Limit", metric: "1 / 2 behind square" },
  ];
  document.getElementById("complianceGrid").innerHTML = defaultChecks
    .map(
      (c) => `
    <div class="law-card pass">
      <div class="law-top">
        <span class="law-code">${c.law}</span>
        <span class="law-status pass">READY</span>
      </div>
      <div class="law-title">${c.title}</div>
      <div class="law-metric">${c.metric}</div>
    </div>
  `
    )
    .join("");
}

async function refreshHistoryLedger() {
  try {
    const res = await fetch(`${API_BASE}/api/history`);
    const data = await res.json();
    state.historyCache = data.deliveries || [];

    const s = data.summary || {};
    document.getElementById("navHistoryCount").textContent = s.total_deliveries || 0;
    document.getElementById("kpiTotalDeliveries").textContent = s.total_deliveries || 0;
    document.getElementById("kpiLegalDeliveries").textContent = s.legal_deliveries || 0;
    document.getElementById("kpiNoBalls").textContent = s.no_balls || 0;
    document.getElementById("kpiNoBallRate").textContent = `${s.no_ball_rate_pct || 0}% infringement rate`;
    document.getElementById("kpiAiDetections").textContent =
      (s.ai_oversteps || 0) + (s.beamers_flagged || 0);

    const tbody = document.getElementById("historyTableBody");
    if (state.historyCache.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" class="empty-table-cell">No deliveries evaluated in this session yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = state.historyCache
      .map(
        (d) => `
      <tr>
        <td class="mono-val">#${d.delivery_id}</td>
        <td class="mono-val">${d.timestamp}</td>
        <td><strong>${d.match_format}</strong> | Ov ${d.over_ball}</td>
        <td>${d.bowler_name} vs ${d.batter_name}</td>
        <td class="mono-val">${(d.ai_overstep_probability * 100).toFixed(1)}%</td>
        <td class="mono-val">${
          d.ai_beamer_probability !== null ? (d.ai_beamer_probability * 100).toFixed(1) + "%" : "--"
        }</td>
        <td>
          <span class="law-status ${d.is_no_ball ? "fail" : "pass"}">
            ${d.verdict}${d.free_hit_awarded ? " (FREE HIT)" : ""}
          </span>
        </td>
        <td>${d.reasons && d.reasons.length ? d.reasons.join(" | ") : "All ICC checks passed"}</td>
        <td class="mono-val">${d.inference_time_ms} ms</td>
      </tr>
    `
      )
      .join("");
  } catch (e) {
    console.error("Failed loading history:", e);
  }
}

async function clearMatchHistory() {
  await fetch(`${API_BASE}/api/reset-history`, { method: "POST" });
  await refreshHistoryLedger();
}

function exportMatchHistoryJson() {
  const blob = new Blob([JSON.stringify(state.historyCache, null, 2)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `CreaseGuard_Official_Report_${Date.now()}.json`;
  a.click();
}

function exportMatchHistoryCsv() {
  if (!state.historyCache.length) return;
  const headers = [
    "delivery_id",
    "timestamp",
    "match_format",
    "over_ball",
    "bowler_name",
    "batter_name",
    "foot_file",
    "waist_file",
    "verdict",
    "ai_overstep_probability",
    "ai_beamer_probability",
    "inference_time_ms",
  ];
  const rows = state.historyCache.map((d) =>
    headers.map((h) => JSON.stringify(d[h] ?? "")).join(",")
  );
  const csv = [headers.join(","), ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `CreaseGuard_Ledger_${Date.now()}.csv`;
  a.click();
}

async function loadModelMetrics() {
  const grid = document.getElementById("modelsBenchmarkGrid");
  try {
    const res = await fetch(`${API_BASE}/api/metrics`);
    const data = await res.json();
    const models = data.models || {};
    const keys = [
      {
        key: "crease_classifier",
        title: "Crease Classifier (Front-Foot Overstep Model)",
        subtitle: "MCC Law 21.5 | Real Broadcast + Domain-Adapted Crease Dataset",
      },
      {
        key: "waist_classifier",
        title: "Waist Classifier (Full-Toss Beamer Model)",
        subtitle: "MCC Law 41.7 | Hybrid Real-Background & Ball Trajectory Dataset",
      },
    ];

    grid.innerHTML = keys
      .map(({ key, title, subtitle }) => {
        const m = models[key];
        if (!m) {
          return `<div class="glass-card"><h3>${title}</h3><p class="muted-hint">Training metrics unavailable.</p></div>`;
        }
        const cm = m.confusion_matrix || { tp: 0, tn: 0, fp: 0, fn: 0 };
        return `
        <div class="glass-card">
          <span class="eyebrow">${m.architecture || "Residual SE-CNN"}</span>
          <h3>${title}</h3>
          <p class="brand-sub" style="margin-bottom:0.8rem;">${subtitle}</p>

          <div class="model-metric-row">
            <div class="m-stat">
              <span class="m-stat-label">VAL ACCURACY</span>
              <span class="m-stat-val">${m.best_val_accuracy}%</span>
            </div>
            <div class="m-stat">
              <span class="m-stat-label">F1-SCORE</span>
              <span class="m-stat-val">${m.f1_score}%</span>
            </div>
            <div class="m-stat">
              <span class="m-stat-label">PRECISION</span>
              <span class="m-stat-val">${m.precision}%</span>
            </div>
            <div class="m-stat">
              <span class="m-stat-label">RECALL</span>
              <span class="m-stat-val">${m.recall}%</span>
            </div>
          </div>

          <div class="radar-top-row" style="margin-top:0.5rem;">
            <span class="radar-title">DATASET SAMPLES: ${m.total_samples} (${m.train_samples} TRAIN / ${m.val_samples} VAL)</span>
            <span class="radar-status pass">LOSS: ${m.best_val_loss}</span>
          </div>

          <div class="cm-grid">
            <div class="cm-cell"><span>True Negative (Legal)</span><strong style="color:#34d399">${cm.tn}</strong></div>
            <div class="cm-cell"><span>True Positive (Illegal)</span><strong style="color:#34d399">${cm.tp}</strong></div>
            <div class="cm-cell"><span>False Positive</span><strong style="color:#fb7185">${cm.fp}</strong></div>
            <div class="cm-cell"><span>False Negative</span><strong style="color:#fb7185">${cm.fn}</strong></div>
          </div>
        </div>
      `;
      })
      .join("");
  } catch (e) {
    grid.innerHTML = `<div class="glass-card"><p class="muted-hint">Metrics loading...</p></div>`;
  }
}

function copyApiSnippet() {
  const code = document.getElementById("apiCodeSnippet").innerText;
  navigator.clipboard.writeText(code);
}
