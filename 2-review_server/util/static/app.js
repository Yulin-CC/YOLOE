const $ = (id) => document.getElementById(id);

const state = {
  config: null,
  files: [],
  tasks: [],
  task: null,
  itemIndex: 0,
  image: null,
  poll: null,
  view: { zoom: 1, centerX: 0.5, centerY: 0.5 },
  drag: null,
  hiddenByModel: {},
  taskCache: {},
};

function esc(s) {
  return String(s ?? "").replace(/[&<>"'`]/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;", "`": "&#96;",
  }[c]));
}

async function api(path, opt) {
  const res = await fetch(path, opt);
  const data = await res.json();
  if (!data.ok) {
    throw new Error((data.error && data.error.message) || res.statusText);
  }
  return data.data;
}

function setMsg(text, isError) {
  const el = $("form-msg");
  el.textContent = text || "";
  el.style.color = isError ? "var(--danger)" : "var(--muted)";
}

function collectFiles(list) {
  const out = [];
  for (const f of list) {
    if (f && f.type && f.type.startsWith("image/")) out.push(f);
    else if (f && /\.(jpe?g|png|bmp|webp|tiff?)$/i.test(f.name)) out.push(f);
  }
  out.sort((a, b) => (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name));
  return out;
}

function filesSize() {
  return state.files.reduce((s, f) => s + (f.size || 0), 0);
}

function fmtMB(bytes) {
  return (bytes / (1024 * 1024)).toFixed(1);
}

function renderFiles() {
  const n = state.files.length;
  if (!n) {
    $("file-hint").textContent = "未选择文件";
    return;
  }
  const mb = fmtMB(filesSize());
  const lim = (state.config && state.config.maxUploadMB) || 64;
  $("file-hint").textContent = `已选 ${n} 张 · ${mb} MB（上限 ${lim} MB）`;
}

function fillConfig(cfg) {
  state.config = cfg;
  const sel = $("model");
  sel.innerHTML = "";
  (cfg.models || []).forEach((m) => {
    if (!m.available) return;
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.name;
    sel.appendChild(opt);
  });
  const firstOk = (cfg.models || []).find((m) => m.available);
  if (firstOk) sel.value = firstOk.id;
  $("names").value = (cfg.defaults && cfg.defaults.names) || "";
  $("conf").value = (cfg.defaults && cfg.defaults.conf) || 0.25;
  $("server-meta").textContent = `device ${cfg.device} · 单次最多 ${cfg.maxFilesPerTask} 张 · ${cfg.maxUploadMB} MB`;
}

function statusLabel(s) {
  return ({
    running: "进行中",
    queued: "排队",
    completed: "完成",
    partial_failed: "部分失败",
    failed: "失败",
    interrupted: "中断",
    success: "成功",
  })[s] || s;
}

function hiddenSet(modelId) {
  const id = modelId || "_";
  if (!state.hiddenByModel[id]) state.hiddenByModel[id] = new Set();
  return state.hiddenByModel[id];
}

function currentModelId() {
  return (state.task && state.task.model && state.task.model.id) || "";
}

function isLabelHidden(label) {
  const mid = currentModelId();
  if (!mid) return false;
  return hiddenSet(mid).has(String(label || ""));
}

function modelGroups() {
  const byId = new Map();
  (state.tasks || []).forEach((t) => {
    const id = (t.model && t.model.id) || "unknown";
    if (!byId.has(id)) {
      byId.set(id, {
        id,
        name: (t.model && t.model.name) || id,
        tasks: [],
      });
    }
    byId.get(id).tasks.push(t);
  });
  const out = [];
  ((state.config && state.config.models) || []).forEach((m) => {
    if (byId.has(m.id)) out.push(byId.get(m.id));
  });
  byId.forEach((g, id) => {
    if (!out.some((x) => x.id === id)) out.push(g);
  });
  return out;
}

function namesForGroup(g) {
  const seen = [];
  const push = (n) => {
    const s = String(n || "").trim();
    if (!s || seen.includes(s)) return;
    seen.push(s);
  };
  (g.tasks || []).slice().reverse().forEach((t) => {
    ((t.params && t.params.names) || []).forEach(push);
    const full = (state.taskCache && state.taskCache[t.id]) || (state.task && state.task.id === t.id ? state.task : null);
    ((full && full.params && full.params.names) || []).forEach(push);
  });
  if (currentModelId() === g.id) {
    ((state.task && state.task.params && state.task.params.names) || []).forEach(push);
    mergedRegions().forEach((r) => push(r.label));
  }
  return seen;
}

function mergedRegions() {
  const item = currentItem();
  if (!item) return [];
  const fallback = (item.result && item.result.regions) || [];
  const g = modelGroups().find((x) => x.id === currentModelId());
  if (!g || !g.tasks.length) return fallback;
  const picked = new Map();
  g.tasks.forEach((t) => {
    const full = (state.taskCache && state.taskCache[t.id]) || (state.task && state.task.id === t.id ? state.task : null);
    if (!full) return;
    const it = (full.items || []).find((x) => x.name === item.name);
    const regs = (it && it.result && it.result.regions) || [];
    const labels = new Set(regs.map((r) => String(r.label || "")));
    labels.forEach((lab) => {
      if (!lab || picked.has(lab)) return;
      picked.set(lab, regs.filter((r) => String(r.label || "") === lab));
    });
  });
  const out = [];
  picked.forEach((rs) => { rs.forEach((r) => out.push(r)); });
  return out.length ? out : fallback;
}

async function ensureGroupTasks(modelId) {
  const g = modelGroups().find((x) => x.id === modelId);
  if (!g) return;
  if (!state.taskCache) state.taskCache = {};
  const jobs = [];
  g.tasks.forEach((t) => {
    if (state.task && t.id === state.task.id) {
      state.taskCache[t.id] = state.task;
      return;
    }
    const cached = state.taskCache[t.id];
    if (cached && cached.updatedAt === t.updatedAt) return;
    jobs.push(
      api(`/api/tasks/${t.id}`).then((full) => { state.taskCache[t.id] = full; }).catch(() => {})
    );
  });
  if (jobs.length) await Promise.all(jobs);
}

function renderModelChips() {
  const root = $("model-chips");
  const scroll = root.scrollTop;
  root.innerHTML = "";
  const groups = modelGroups();
  if (!groups.length) {
    root.innerHTML = `<p class="hint">尚未推理。提交后按模型显示描述词。</p>`;
    return;
  }
  const activeId = currentModelId();
  groups.forEach((g) => {
    const latest = g.tasks[0];
    const names = namesForGroup(g);
    const hidden = hiddenSet(g.id);
    const counts = {};
    if (activeId === g.id) {
      mergedRegions().forEach((r) => {
        const k = String(r.label || "");
        counts[k] = (counts[k] || 0) + 1;
      });
    }
    const chips = names.length
      ? names.map((name) => {
        const on = !hidden.has(name);
        const n = activeId === g.id ? (counts[name] || 0) : "";
        const color = chipColor(name);
        const bg = on ? chipFill(name) : "";
        return `<button type="button" class="phrase-chip ${on ? "on" : "off"}" data-model="${esc(g.id)}" data-ph="${esc(name)}" style="border-color:${color};${on ? `color:${color};background:${bg}` : ""}">${esc(name)}${n === "" ? "" : `<span class="n">${n}</span>`}</button>`;
      }).join("")
      : `<span class="hint">暂无描述词</span>`;
    const card = document.createElement("article");
    card.className = "model-card" + (activeId === g.id ? " active" : "");
    card.setAttribute("data-open-model", g.id);
    card.innerHTML = `
      <div class="model-hd">
        <div class="model-open">
          <div class="model-name">${esc(g.name)}</div>
          <div class="hint"><span class="status ${esc(latest.status)}">${esc(statusLabel(latest.status))}</span> · ${latest.okCount || 0}/${latest.itemCount || 0}</div>
        </div>
        <button class="del" type="button" data-del-model="${esc(g.id)}" title="删除该模型全部记录">删除</button>
      </div>
      <div class="phrase-list">${chips}</div>`;
    root.appendChild(card);
  });
  root.scrollTop = scroll;
}

async function refreshTasks() {
  state.tasks = await api("/api/tasks");
  renderModelChips();
}

async function openTask(id) {
  state.task = await api(`/api/tasks/${id}`);
  state.taskCache[id] = state.task;
  const items = state.task.items || [];
  if (state.itemIndex >= items.length) state.itemIndex = Math.max(0, items.length - 1);
  await ensureGroupTasks(currentModelId());
  renderModelChips();
  await showItem();
  const st = state.task.status;
  if (st === "running" || st === "queued") startPoll();
  else stopPoll();
}

async function openModel(modelId) {
  const g = modelGroups().find((x) => x.id === modelId);
  if (!g || !g.tasks.length) return;
  const curName = currentItem() && currentItem().name;
  await openTask(g.tasks[0].id);
  if (!curName || !state.task) return;
  const idx = (state.task.items || []).findIndex((it) => it.name === curName);
  if (idx >= 0 && idx !== state.itemIndex) {
    state.itemIndex = idx;
    await showItem();
  }
}

function startPoll() {
  stopPoll();
  state.poll = setInterval(() => {
    if (state.task) openTask(state.task.id).catch((e) => setMsg(e.message, true));
    refreshTasks().catch(() => {});
  }, 1000);
}

function stopPoll() {
  if (state.poll) {
    clearInterval(state.poll);
    state.poll = null;
  }
}

function currentItem() {
  const items = (state.task && state.task.items) || [];
  return items[state.itemIndex] || null;
}

async function showItem() {
  const item = currentItem();
  const empty = $("empty");
  const err = $("item-error");
  if (!item) {
    empty.classList.remove("hidden");
    $("item-meta").textContent = "尚未加载任务";
    err.hidden = true;
    draw();
    return;
  }
  empty.classList.add("hidden");
  const n = (state.task.items || []).length;
  const cost = item.costMs != null ? `${item.costMs} ms` : "-";
  const regs = mergedRegions();
  let extra = "";
  if (item.status === "queued" || item.status === "running") {
    extra = "  首次会加载 MobileCLIP，请稍候";
  }
  $("item-meta").textContent = `${item.name}  ${state.itemIndex + 1}/${n}  ${statusLabel(item.status)}  ${regs.length} 个目标  ${cost}${extra}`;
  renderModelChips();
  if (item.error && item.error.message) {
    err.hidden = false;
    err.textContent = item.error.message;
  } else {
    err.hidden = true;
  }
  const url = item.inputs && item.inputs[0] && item.inputs[0].url;
  if (!url) {
    state.image = null;
    draw();
    return;
  }
  const img = new Image();
  img.onload = () => {
    if (currentItem() !== item) return;
    state.image = img;
    draw();
  };
  img.onerror = () => {
    state.image = null;
    draw();
  };
  img.src = url;
}

function labelHue(label) {
  let h = 0;
  for (const ch of String(label)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return h % 360;
}

function colorFor(label, alpha) {
  const hue = labelHue(label);
  if (alpha == null) return `hsl(${hue}, 80%, 55%)`;
  return `hsla(${hue}, 80%, 55%, ${alpha})`;
}

function chipColor(label) {
  const hue = labelHue(label);
  const light = (hue >= 50 && hue <= 165) ? 24 : 30;
  return `hsl(${hue}, 72%, ${light}%)`;
}

function chipFill(label) {
  const hue = labelHue(label);
  return `hsl(${hue}, 40%, 92%)`;
}

function viewMetrics() {
  const canvas = $("view");
  const img = state.image;
  if (!img) return null;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  const fit = Math.min(cssW / img.naturalWidth, cssH / img.naturalHeight);
  const scale = fit * state.view.zoom;
  const drawW = img.naturalWidth * scale;
  const drawH = img.naturalHeight * scale;
  const offsetX = cssW / 2 - state.view.centerX * drawW;
  const offsetY = cssH / 2 - state.view.centerY * drawH;
  return { cssW, cssH, scale, offsetX, offsetY, drawW, drawH };
}

function draw() {
  const canvas = $("view");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  canvas.width = Math.max(1, Math.round(cssW * dpr));
  canvas.height = Math.max(1, Math.round(cssH * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  if (!state.image) return;
  const m = viewMetrics();
  ctx.drawImage(state.image, m.offsetX, m.offsetY, m.drawW, m.drawH);
  const item = currentItem();
  const regions = mergedRegions();
  const showBox = $("show-box").checked;
  const showMask = $("show-mask").checked;
  const showLabel = $("show-label").checked;
  const showConf = $("show-conf").checked;
  regions.forEach((r) => {
    if (isLabelHidden(r.label)) return;
    const color = colorFor(r.label);
    if (showMask && r.points && r.points.length >= 3) {
      ctx.beginPath();
      r.points.forEach((p, i) => {
        const x = m.offsetX + p.x * state.image.naturalWidth * m.scale;
        const y = m.offsetY + p.y * state.image.naturalHeight * m.scale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fillStyle = colorFor(r.label, 0.28);
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    if (showBox && r.box) {
      const x = m.offsetX + r.box.x * state.image.naturalWidth * m.scale;
      const y = m.offsetY + r.box.y * state.image.naturalHeight * m.scale;
      const w = r.box.w * state.image.naturalWidth * m.scale;
      const h = r.box.h * state.image.naturalHeight * m.scale;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
    }
    if (!showLabel && !showConf) return;
    const parts = [];
    if (showLabel && r.label) parts.push(r.label);
    if (showConf) parts.push(Number(r.score || 0).toFixed(2));
    const text = parts.join(" ");
    if (!text || !r.box) return;
    const x = m.offsetX + r.box.x * state.image.naturalWidth * m.scale;
    const y = m.offsetY + r.box.y * state.image.naturalHeight * m.scale;
    ctx.font = "12px sans-serif";
    const tw = ctx.measureText(text).width + 8;
    ctx.fillStyle = color;
    ctx.fillRect(x, Math.max(0, y - 16), tw, 16);
    ctx.fillStyle = "#111";
    ctx.fillText(text, x + 4, Math.max(12, y - 4));
  });
}

function clientToNorm(ev) {
  const m = viewMetrics();
  if (!m) return null;
  const rect = $("view").getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const y = ev.clientY - rect.top;
  return {
    nx: (x - m.offsetX) / (state.image.naturalWidth * m.scale),
    ny: (y - m.offsetY) / (state.image.naturalHeight * m.scale),
  };
}

async function submit() {
  if (!state.files.length) {
    setMsg("请先选择图片", true);
    return;
  }
  const limMB = (state.config && state.config.maxUploadMB) || 64;
  if (filesSize() / (1024 * 1024) > limMB) {
    setMsg(`选中 ${fmtMB(filesSize())} MB，超过上限 ${limMB} MB，请少选几张`, true);
    return;
  }
  const model = $("model").value;
  if (!model) {
    setMsg("没有可用模型，请检查 config.json 权重路径", true);
    return;
  }
  const fd = new FormData();
  fd.append("model", model);
  fd.append("names", $("names").value);
  fd.append("conf", $("conf").value);
  state.files.forEach((f) => fd.append("files", f, f.name));
  $("submit").disabled = true;
  setMsg("提交中…");
  try {
    const created = await api("/api/tasks", { method: "POST", body: fd });
    const parsed = (created.params && created.params.names) || [];
    setMsg(`已排队 · ${parsed.length} 个词：${parsed.join(" | ")}`);
    state.itemIndex = 0;
    state.view = { zoom: 1, centerX: 0.5, centerY: 0.5 };
    await refreshTasks();
    await openTask(created.id);
  } catch (e) {
    setMsg(e.message, true);
  } finally {
    $("submit").disabled = false;
  }
}

function bind() {
  $("files").addEventListener("change", (e) => {
    state.files = collectFiles(e.target.files);
    renderFiles();
  });
  $("folder").addEventListener("change", (e) => {
    state.files = collectFiles(e.target.files);
    renderFiles();
  });
  $("submit").addEventListener("click", submit);
  $("prev").addEventListener("click", () => {
    if (!state.task) return;
    state.itemIndex = Math.max(0, state.itemIndex - 1);
    showItem();
  });
  $("next").addEventListener("click", () => {
    if (!state.task) return;
    state.itemIndex = Math.min((state.task.items || []).length - 1, state.itemIndex + 1);
    showItem();
  });
  $("reset-view").addEventListener("click", () => {
    state.view = { zoom: 1, centerX: 0.5, centerY: 0.5 };
    draw();
  });
  $("show-box").addEventListener("change", draw);
  $("show-mask").addEventListener("change", draw);
  $("show-label").addEventListener("change", draw);
  $("show-conf").addEventListener("change", draw);
  $("model-chips").addEventListener("click", async (e) => {
    const del = e.target.closest("[data-del-model]");
    const ph = e.target.closest("[data-ph]");
    const open = e.target.closest("[data-open-model]");
    if (del) {
      const modelId = del.getAttribute("data-del-model");
      const g = modelGroups().find((x) => x.id === modelId);
      const ids = (g && g.tasks || []).map((t) => t.id);
      if (!ids.length) return;
      if (!confirm(`删除「${(g && g.name) || modelId}」的 ${ids.length} 条记录？`)) return;
      try {
        for (const id of ids) {
          await api(`/api/tasks/${id}`, { method: "DELETE" });
          delete state.taskCache[id];
        }
        if (state.task && ids.includes(state.task.id)) {
          state.task = null;
          state.image = null;
          stopPoll();
          draw();
        }
        await refreshTasks();
        await showItem();
      } catch (err) {
        setMsg(err.message, true);
      }
      return;
    }
    if (ph) {
      const modelId = ph.getAttribute("data-model");
      const name = ph.getAttribute("data-ph");
      const set = hiddenSet(modelId);
      if (set.has(name)) set.delete(name);
      else set.add(name);
      if (currentModelId() !== modelId) {
        openModel(modelId).catch((err) => setMsg(err.message, true));
      } else {
        renderModelChips();
        draw();
      }
      return;
    }
    if (open) {
      openModel(open.getAttribute("data-open-model")).catch((err) => setMsg(err.message, true));
    }
  });

  const canvas = $("view");
  canvas.addEventListener("wheel", (ev) => {
    if (!state.image) return;
    ev.preventDefault();
    const before = clientToNorm(ev);
    const factor = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
    state.view.zoom = Math.min(8, Math.max(1, state.view.zoom * factor));
    if (before) {
      const after = clientToNorm(ev);
      if (after) {
        state.view.centerX += before.nx - after.nx;
        state.view.centerY += before.ny - after.ny;
      }
    }
    state.view.centerX = Math.min(1.2, Math.max(-0.2, state.view.centerX));
    state.view.centerY = Math.min(1.2, Math.max(-0.2, state.view.centerY));
    draw();
  }, { passive: false });
  canvas.addEventListener("pointerdown", (ev) => {
    if (!state.image) return;
    state.drag = { x: ev.clientX, y: ev.clientY, cx: state.view.centerX, cy: state.view.centerY };
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (!state.drag || !state.image) return;
    const m = viewMetrics();
    state.view.centerX = state.drag.cx - (ev.clientX - state.drag.x) / m.drawW;
    state.view.centerY = state.drag.cy - (ev.clientY - state.drag.y) / m.drawH;
    draw();
  });
  canvas.addEventListener("pointerup", () => { state.drag = null; });
  window.addEventListener("resize", draw);
  window.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowLeft") $("prev").click();
    if (ev.key === "ArrowRight") $("next").click();
  });
  new ResizeObserver(draw).observe($("stage"));
  initResizers();
}

const SIDE_W_DEFAULT = 320;
const SIDE_W_MIN = 220;
const SIDE_W_MAX = 720;

function loadLayout() {
  try {
    const s = JSON.parse(localStorage.getItem("yoloe_review_layout") || "{}");
    const w = parseInt(s.sideW, 10);
    if (w) {
      const clamped = Math.min(SIDE_W_MAX, Math.max(SIDE_W_MIN, w));
      document.documentElement.style.setProperty("--side-w", clamped + "px");
    }
  } catch (_) {}
}

function saveLayout() {
  const css = getComputedStyle(document.documentElement);
  localStorage.setItem("yoloe_review_layout", JSON.stringify({
    sideW: parseInt(css.getPropertyValue("--side-w"), 10) || SIDE_W_DEFAULT,
  }));
}

function bindResizer(el, onMove) {
  el.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    el.setPointerCapture(e.pointerId);
    el.classList.add("active");
    const move = (ev) => {
      if (ev.pointerId !== e.pointerId) return;
      onMove(ev);
    };
    const up = (ev) => {
      if (ev.pointerId !== e.pointerId) return;
      el.classList.remove("active");
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", up);
      saveLayout();
      draw();
    };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
  });
}

function initResizers() {
  loadLayout();
  const el = $("res-side");
  if (!el) return;
  bindResizer(el, (ev) => {
    const layout = document.querySelector(".layout").getBoundingClientRect();
    const w = Math.min(SIDE_W_MAX, Math.max(SIDE_W_MIN, layout.right - ev.clientX - 12));
    document.documentElement.style.setProperty("--side-w", w + "px");
    draw();
  });
}

async function main() {
  bind();
  try {
    fillConfig(await api("/api/config"));
    await refreshTasks();
  } catch (e) {
    $("server-meta").textContent = e.message;
  }
}

main();
