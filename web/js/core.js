// Shared runtime: i18n, API client, session, toast, haptics, update gate.
export const BUILD = Number(document.documentElement.dataset.build || 0);

// ---------- i18n ----------
let strings = {};
export const lang = (navigator.language || "en").toLowerCase().startsWith("sv") ? "sv" : "en";
export async function loadI18n() {
  const r = await fetch(`/i18n/${lang}.json?v=${BUILD}`);
  strings = await r.json();
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.placeholder = t(el.dataset.i18nPh); });
}
export const t = (key, vars = {}) =>
  (strings[key] ?? key).replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? "");

// ---------- session ----------
const KEY = "sportscam.session";
export function session() { try { return JSON.parse(localStorage.getItem(KEY) || "null"); } catch { return null; } }
export function saveSession(s) { localStorage.setItem(KEY, JSON.stringify(s)); }
export function clearSession() { localStorage.removeItem(KEY); }
export const codeFromUrl = () => new URLSearchParams(location.search).get("c")?.toUpperCase() || "";

// ---------- API ----------
export class ApiError extends Error { constructor(status, detail) { super(detail); this.status = status; } }
export async function api(path, { method = "GET", body, token, raw } = {}) {
  const headers = {};
  if (token) headers["X-Token"] = token;
  if (body && !raw) headers["Content-Type"] = "application/json";
  if (raw) headers["Content-Type"] = "application/sdp";
  let r;
  try {
    r = await fetch(path, { method, headers, body: raw ? body : body ? JSON.stringify(body) : undefined });
  } catch {
    throw new ApiError(0, "offline");
  }
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch { /* keep */ }
    throw new ApiError(r.status, detail);
  }
  if (r.status === 204) return null;
  return raw ? r : r.json();
}

// ---------- feedback ----------
export function haptic(kind = "light") {
  if (!navigator.vibrate) return;
  navigator.vibrate(kind === "light" ? 8 : kind === "medium" ? 18 : [20, 30, 40]);
}
let toastEl;
export function toast(text, { action, onAction, sticky } = {}) {
  if (!toastEl) { toastEl = document.createElement("div"); toastEl.className = "toast"; document.body.appendChild(toastEl); }
  toastEl.innerHTML = "";
  const span = document.createElement("span"); span.textContent = text; toastEl.appendChild(span);
  if (action) {
    const b = document.createElement("button"); b.className = "btn primary"; b.style.minHeight = "40px";
    b.textContent = action; b.onclick = onAction; toastEl.appendChild(b);
  }
  toastEl.classList.add("show");
  clearTimeout(toastEl._t);
  if (!sticky) toastEl._t = setTimeout(() => toastEl.classList.remove("show"), 3500);
}

// ---------- game clock ----------
export function fmtClock(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

// ---------- update gate (viccom-auto-update) ----------
// Web assets ship with the hub, so "new hub build" == "reload the page". A blocked hub shows a
// full screen; a nudge shows once per version. Offline never blocks.
export async function updateGate() {
  let v;
  try { v = await api("/api/version"); } catch { return; }
  if (v.build !== BUILD && BUILD) {
    toast(t("update.reload"), { action: t("update.cta"), onAction: () => location.reload(), sticky: true });
  }
  const u = v.update || {};
  const msg = lang === "sv" ? u.message_sv : u.message_en;
  if (u.status === "blocked") {
    const el = document.createElement("div"); el.className = "block-screen";
    el.innerHTML = `<div class="stack" style="max-width:420px"><h1></h1><p class="dim"></p><a class="btn primary block" target="_blank" rel="noopener"></a></div>`;
    el.querySelector("h1").textContent = t("update.blocked");
    el.querySelector("p").textContent = msg || "";
    const a = el.querySelector("a"); a.textContent = t("update.cta"); a.href = u.store_url || "https://viccom.se";
    document.body.appendChild(el);
  } else if (u.status === "nudge") {
    const seen = localStorage.getItem("sportscam.nudged");
    if (seen !== String(u.latest_version_code)) {
      localStorage.setItem("sportscam.nudged", String(u.latest_version_code));
      toast(msg || t("update.nudge"), { action: t("update.cta"), onAction: () => window.open(u.store_url || "https://viccom.se", "_blank"), sticky: true });
    }
  }
}
export function watchForNewBuild() {
  const check = () => updateGate();
  document.addEventListener("visibilitychange", () => { if (!document.hidden) check(); });
  setInterval(check, 30000);
}
