/* Yamanote dashboard — vanilla JS, live via Server-Sent Events. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const SVG = "http://www.w3.org/2000/svg";

let S = null;                 // last /api/state
let depFilter = "line";
let arrFilter = "all";
let openId = null;            // item id shown in the drawer
let detail = null;            // last /api/items/:id
let detailTab = null;
const expandedRuns = new Set();
const runSteps = new Map();   // run id -> steps[]
const feed = [];              // line announcements, newest first
let lastEventId = 0;

/* ── formatting ───────────────────────────────────────── */
const money = (v) => v == null ? "—" : v < 0.01 ? (v === 0 ? "$0" : "$" + v.toFixed(4)) : "$" + v.toFixed(2);
const pct = (v) => v == null ? "—" : Math.round(v * 100) + "%";
const hm = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
const hms = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour12: false });
const ago = (ts) => dur(Date.now() / 1000 - ts);
function dur(s) {
  if (s == null || isNaN(s)) return "—";
  s = Math.max(0, Math.round(s));
  if (s < 60) return s + "s";
  if (s < 3600) return Math.floor(s / 60) + "m" + String(s % 60).padStart(2, "0");
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h < 48 ? `${h}h${String(m).padStart(2, "0")}` : Math.floor(h / 24) + "d";
}
const tokens = (n) => n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(n || 0);
const base = (p) => (p || "").split("/").filter(Boolean).pop() || p;
const CLASS_SUFFIX = { local: "G", rapid: "K", express: "E", shinkansen: "S" };
// Train numbers keep the class the train first departed in, even if it escalates later.
const trainNo = (it) => String(it.id).padStart(4, "0") + (CLASS_SUFFIX[it.first_class || it.service_class] || "");
const station = (key) => (S?.stations || []).find((s) => s.key === key) || { key, code: "JY--", name: key };

// "provider/model" with a break opportunity after the slash, never mid-name.
function modelId(id) {
  const i = String(id || "").indexOf("/");
  if (i < 0) return `<code class="model-id">${esc(id)}</code>`;
  return `<code class="model-id"><span class="prov">${esc(id.slice(0, i + 1))}</span><wbr>${esc(id.slice(i + 1))}</code>`;
}

function svcBadge(cls) {
  const c = S?.service_classes?.[cls];
  if (!c) return `<span class="svc svc-none">—</span>`;
  return `<span class="svc svc-${esc(cls)}" title="${esc(c.label)} · ${esc(c.model)}"><span class="kanji">${esc(c.kanji)}</span>${esc(c.label === "Limited Express" ? "Ltd Exp" : c.label)}</span>`;
}

const VERB = { triage: "TRIAGING", spec: "SPECIFYING", build: "BUILDING", inspect: "INSPECTING", verify: "VERIFYING", merge: "MERGING", deploy: "DEPLOYING", retro: "REFLECTING" };
function boardStatus(it) {
  const now = Date.now() / 1000;
  if (it.status === "waiting" && it.gate === "board") return ["st-signal", "PROPOSED · BOARD?"];
  if (it.status === "waiting") return ["st-signal", `SIGNAL · ${it.gate === "spec" ? "SPEC" : "MERGE"} OK?`];
  if (it.status === "held") return ["st-held", "HELD"];
  if (it.status === "running") return ["st-run", VERB[it.station] || "RUNNING"];
  if (it.station === "retro") return ["st-held", (it.pending_status === "done" ? "ARRIVED · " : "ENDED · ") + "RETRO"];
  if (it.hold_until && it.hold_until > now) return ["st-delay", "DELAYED " + dur(it.hold_until - now)];
  if (["build", "inspect", "verify", "merge", "deploy"].includes(it.station) && !it.train) return ["st-dim", "AWAITING TRAIN"];
  if (it.attempt > 1 && it.station === "build") return ["st-delay", "REWORK " + (it.attempt)];
  if (it.station === "triage" || it.station === "spec") return ["st-dim", "BOARDING"];
  return ["st-ok", "ON TIME"];
}
const ARR_STATUS = { done: ["st-ok", "ARRIVED ✓"], failed: ["st-bad", "TERMINATED"], rejected: ["st-dim", "NOT IN SERVICE"], cancelled: ["st-dim", "CANCELLED"] };

/* ── API ──────────────────────────────────────────────── */
function token() { try { return localStorage.getItem("yamanote-token") || ""; } catch { return ""; } }
function setToken(t) {
  try { localStorage.setItem("yamanote-token", t); } catch {}
  // EventSource can't send headers; the server also accepts this cookie.
  document.cookie = "yamanote_token=" + encodeURIComponent(t) + "; path=/; SameSite=Strict";
}
let askedForToken = false;
async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json", "X-Yamanote": "1" };
  if (token()) headers.Authorization = "Bearer " + token();
  const r = await fetch(path, { ...opts, headers: { ...headers, ...(opts.headers || {}) } });
  const data = await r.json().catch(() => ({}));
  if (r.status === 401 && !askedForToken) {
    askedForToken = true;
    const t = prompt("This dashboard needs its access token (AGENT_TEAM_DASHBOARD_TOKEN):");
    askedForToken = false;
    if (t) { setToken(t.trim()); return api(path, opts); }
  }
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body || {}) });

async function refresh() {
  try {
    S = await api("/api/state");
    $("#conn-lost").hidden = true;
    render();
  } catch (e) {
    $("#conn-lost").hidden = false;
  }
}
let refreshTimer = null;
function refreshSoon(ms = 350) { clearTimeout(refreshTimer); refreshTimer = setTimeout(refresh, ms); }
let detailTimer = null;
function detailSoon(ms = 500) { if (!openId) return; clearTimeout(detailTimer); detailTimer = setTimeout(() => loadDetail(openId, false), ms); }

/* ── render: top bar, KPIs ────────────────────────────── */
function render() {
  if (!S) return;
  renderTopbar();
  renderLoop();
  renderKpis();
  renderBoards();
  renderCrew();
  renderRouting();
  renderLearning();
  diagramSoon();
}

function renderTopbar() {
  const pill = $("#line-status");
  const ap = S.autopilot || {};
  if (S.paused) { pill.className = "pill pill-warn"; pill.textContent = "■ Paused · 運転見合わせ"; }
  else if (S.suspended) { pill.className = "pill pill-bad"; pill.textContent = "■ Suspended"; }
  else if (ap.on) { pill.className = "pill pill-auto"; pill.textContent = "◐ Autopilot · 自動運転"; }
  else { pill.className = "pill pill-ok"; pill.textContent = "● Supervised · 運転中"; }
  const sw = $("#btn-autopilot");
  sw.setAttribute("aria-checked", ap.on ? "true" : "false");
  const next = ap.config?.schedule_enabled ? (ap.on ? ap.preview?.next_off : ap.preview?.next_on) : null;
  sw.title = (ap.on ? `Autopilot on since ${hm(ap.since)} (${ap.source})` : "Autopilot off — supervised")
    + (next ? ` · scheduled to switch ${new Date(next * 1000).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false })}` : "");
  document.body.classList.toggle("autopilot", !!ap.on);
  renderAwayBanner();
  $("#btn-pause").textContent = S.paused ? "Resume line" : "Pause line";
  const b = S.budget;
  $("#budget-spent").textContent = money(b.spent_today);
  $("#budget-limit").textContent = "$" + b.daily_limit.toFixed(0);
  const frac = Math.min(1, b.spent_today / Math.max(0.01, b.daily_limit));
  const fill = $("#budget-fill");
  fill.style.width = (frac * 100).toFixed(1) + "%";
  fill.style.background = frac > 0.9 ? "var(--led-red)" : frac > 0.7 ? "var(--amber)" : "var(--line)";
  const banner = $("#suspended-banner");
  banner.hidden = !S.suspended;
  if (S.suspended) banner.textContent = "Service suspended: " + S.suspended + ". Running jobs finish; no new departures.";
}

function renderAwayBanner() {
  const el = $("#away-banner"), r = S.autopilot_report;
  let dismissed = "";
  try { dismissed = localStorage.getItem("yamanote-away-dismissed") || ""; } catch {}
  if (!r || S.autopilot?.on || String(r.id) === dismissed || Date.now() / 1000 - r.ts > 2 * 86400) { el.hidden = true; return; }
  const d = r.data || {};
  const links = (list, label) => list && list.length ? ` ${label}: ` + list.map((i) => `<a data-id="${i.id}">#${i.id}</a>`).join(", ") + "." : "";
  el.innerHTML = `<span class="away-body"><b>While you were away</b> (${esc(hm(d.start))}–${esc(hm(d.end))}) — ${esc(r.message.replace(/^Autopilot ran [^:]+: /, ""))}${links(d.failed, "Failed")}${links(d.waiting, "Waiting")}</span><button aria-label="Dismiss" title="Dismiss">×</button>`;
  el.hidden = false;
  el.querySelector("button").onclick = () => { try { localStorage.setItem("yamanote-away-dismissed", String(r.id)); } catch {} el.hidden = true; };
}

function renderKpis() {
  const st = S.stats || {};
  const onLine = S.items.filter((i) => i.train).length;
  const waiting = S.items.filter((i) => !i.train && i.status !== "held").length;
  const kpis = [
    ["Trains in service", `${onLine}/${S.trains.length}`, `${waiting} on the platform`],
    ["Arrived (24h)", st.done_24h ?? 0, `${st.failed || 0} terminated · ${st.rejected || 0} not in service (all time)`],
    ["Avg journey", st.lead_time_avg_24h ? dur(st.lead_time_avg_24h) : "—", "build → deploy, 24h"],
    ["Fare per arrival", money(st.cost_per_merge_24h), "24h average"],
    ["Satisfaction", pct(st.satisfaction_7d), "holdout scenarios, 7d"],
    ["Spend (24h)", money(st.spend_24h), `${(st.by_model_24h || []).length} models used`],
  ];
  $("#kpis").innerHTML = kpis.map(([k, v, n]) => `<div class="kpi"><div class="kpi-label">${esc(k)}</div><div class="kpi-value">${esc(v)}</div><div class="kpi-note">${esc(n)}</div></div>`).join("");
  $("#center-count").textContent = onLine;
  $("#center-sub").textContent = onLine === 1 ? "train running" : "trains running";
}

/* ── render: the loop ─────────────────────────────────── */
const trackLen = () => $("#track").getTotalLength();
function pointAt(len) {
  const L = trackLen();
  const l = ((len % L) + L) % L;
  const p = $("#track").getPointAtLength(l);
  const q = $("#track").getPointAtLength((l + 2) % L);
  let angle = Math.atan2(q.y - p.y, q.x - p.x) * 180 / Math.PI;
  return { x: p.x, y: p.y, angle };
}
function outward(p) {
  const cx = Math.min(770, Math.max(230, p.x));
  const dx = p.x - cx, dy = p.y - 200;
  const n = Math.hypot(dx, dy) || 1;
  return { x: dx / n, y: dy / n };
}
const stationLen = (i) => trackLen() * (i / S.stations.length) + 40;

function renderLoop() {
  const gS = $("#stations"), gT = $("#trains");
  gS.innerHTML = ""; gT.innerHTML = "";
  const byStation = {};
  for (const it of S.items) {
    if (it.status === "held") continue;
    (byStation[it.station] ||= []).push(it);
  }
  S.stations.forEach((st, i) => {
    const p = pointAt(stationLen(i));
    const o = outward(p);
    const count = (byStation[st.key] || []).length;
    const dot = document.createElementNS(SVG, "circle");
    dot.setAttribute("cx", p.x); dot.setAttribute("cy", p.y); dot.setAttribute("r", 9);
    dot.setAttribute("class", "station-dot" + (count ? " busy" : ""));
    gS.appendChild(dot);
    const sx = p.x + o.x * 58, sy = p.y + o.y * 46;
    const g = document.createElementNS(SVG, "g");
    g.setAttribute("class", "station-sign");
    g.setAttribute("transform", `translate(${sx} ${sy})`);
    let extra = "";
    if (st.key === "intake") {
      const jobs = Object.entries(S.line_jobs).filter(([, on]) => on).map(([k]) => k);
      extra = jobs.length ? jobs.join(" · ") + " active" : "";
    }
    const sub = count ? `${count} train${count > 1 ? "s" : ""}` : extra;
    g.innerHTML = `<title>${esc(st.code + " " + st.name + " — " + st.desc)}</title>
      <rect class="sign-bg" x="-48" y="-22" width="96" height="${sub ? 50 : 40}"></rect>
      <text class="sign-code" x="0" y="-8" text-anchor="middle">${esc(st.code)}</text>
      <text class="sign-name" x="0" y="11" text-anchor="middle">${esc(st.name)}</text>
      ${sub ? `<text class="sign-count" x="0" y="23" text-anchor="middle">${esc(sub)}</text>` : ""}`;
    gS.appendChild(g);

    // Trains: running ones just past the platform (departing), others waiting before it.
    const items = (byStation[st.key] || []).sort((a, b) => a.id - b.id);
    const moving = items.filter((t) => t.status === "running");
    const parked = items.filter((t) => t.status !== "running");
    moving.forEach((t, k) => drawTrain(gT, t, stationLen(i) + 34 + k * 40));
    parked.forEach((t, k) => drawTrain(gT, t, stationLen(i) - 34 - k * 40));
  });
  renderLineList(byStation);
}

function renderLineList(byStation) {
  $("#line-list").innerHTML = S.stations.map((st) => {
    const trains = (byStation[st.key] || []).map((it) =>
      `<button class="ll-train ${esc(it.service_class || "none")} ${esc(it.status)}" data-train="${it.id}">${esc(trainNo(it))}<span>${esc(it.title)}</span></button>`).join("");
    return `<li><span class="ll-dot"></span><div><div class="ll-name"><span class="code">${esc(st.code)}</span>${esc(st.name)}</div>${trains ? `<div class="ll-trains">${trains}</div>` : ""}</div></li>`;
  }).join("");
}

function drawTrain(layer, it, len) {
  const p = pointAt(len);
  let a = p.angle;
  if (a > 90 || a < -90) a += 180; // keep labels upright
  const cls = it.service_class || "none";
  const color = { local: "var(--c-local)", rapid: "var(--c-rapid)", express: "var(--c-express)", shinkansen: "var(--c-shinkansen)" }[cls] || "#888";
  const g = document.createElementNS(SVG, "g");
  g.setAttribute("class", `train ${it.status}`);
  g.setAttribute("transform", `translate(${p.x} ${p.y}) rotate(${a})`);
  g.dataset.id = it.id;
  g.innerHTML = `<rect class="car" x="-19" y="-9" width="38" height="18" rx="7" fill="${color}"></rect>
    <rect class="car-window" x="-15" y="-5" width="6" height="4" rx="1"></rect>
    <rect class="car-window" x="9" y="-5" width="6" height="4" rx="1"></rect>
    <text class="car-label" x="0" y="6" text-anchor="middle">${it.id}</text>
    ${it.status === "waiting" ? `<circle class="signal" cx="0" cy="-17" r="5"></circle>` : ""}`;
  g.addEventListener("click", () => openItem(it.id));
  g.addEventListener("mouseenter", (e) => showTip(e, it));
  g.addEventListener("mousemove", (e) => moveTip(e));
  g.addEventListener("mouseleave", hideTip);
  layer.appendChild(g);
}
function showTip(e, it) {
  const [, label] = boardStatus(it);
  const tip = $("#train-tip");
  tip.innerHTML = `<b>${esc(trainNo(it))}</b> ${esc(it.title)}<br>${esc(station(it.station).name)} · ${esc(label)}<br>${money(it.cost_usd)} · attempt ${it.attempt || 0}${it.run ? `<br>${esc(it.run.role)} on ${esc(it.run.model)} · ${it.run.steps} steps` : ""}`;
  tip.hidden = false; moveTip(e);
}
function moveTip(e) {
  const tip = $("#train-tip"), box = $(".loop-wrap").getBoundingClientRect();
  tip.style.left = Math.min(box.width - 290, e.clientX - box.left + 14) + "px";
  tip.style.top = (e.clientY - box.top + 14) + "px";
}
function hideTip() { $("#train-tip").hidden = true; }

/* ── render: boards ───────────────────────────────────── */
function renderBoards() {
  const LINE = ["build", "inspect", "verify", "merge", "deploy"];
  const items = S.items.filter((it) => {
    if (depFilter === "held") return it.status === "held";
    if (it.status === "held") return false;
    const onLine = !!it.train || it.station === "retro";
    return depFilter === "line" ? onLine : !onLine;
  }).sort((a, b) => (LINE.indexOf(b.station) - LINE.indexOf(a.station)) || a.id - b.id);
  const counts = {
    line: S.items.filter((i) => (i.train || i.station === "retro") && i.status !== "held").length,
    waiting: S.items.filter((i) => !i.train && i.station !== "retro" && i.status !== "held").length,
    held: S.items.filter((i) => i.status === "held").length,
  };
  $$(".tab[data-filter]").forEach((t) => {
    const n = counts[t.dataset.filter];
    t.textContent = { line: "On the line", waiting: "Platform", held: "Held" }[t.dataset.filter] + (n ? ` ${n}` : "");
  });
  $("#departures").innerHTML = items.length ? items.map((it) => {
    const [cls, label] = boardStatus(it);
    const st = station(it.station);
    const elapsed = it.started_at ? Date.now() / 1000 - it.started_at : Date.now() / 1000 - it.created_at;
    return `<div class="board-row item${openId === it.id ? " selected" : ""}" data-id="${it.id}">
      <span class="trainno">${esc(trainNo(it))}</span>
      <span>${svcBadge(it.service_class)}</span>
      <span class="dest" title="${esc(it.title)}">${esc(it.title)}</span>
      <span class="nowat"><span class="code">${esc(st.code)}</span>${esc(st.name)}</span>
      <span class="status ${cls}">${esc(label)}</span>
      <span class="num">${esc(dur(elapsed))}</span>
      <span class="num">${esc(money(it.cost_usd))}</span></div>`;
  }).join("") : `<div class="board-empty">${depFilter === "line" ? "NO TRAINS IN SERVICE" : depFilter === "held" ? "NO HELD TRAINS" : "PLATFORM EMPTY"}</div>`;

  const arr = S.arrivals.filter((it) => arrFilter === "all" || (arrFilter === "done" ? it.status === "done" : it.status !== "done"));
  $("#arrivals").innerHTML = arr.length ? arr.map((it) => {
    const [cls, label] = ARR_STATUS[it.status] || ["st-dim", it.status.toUpperCase()];
    const trip = it.started_at && it.finished_at ? it.finished_at - it.started_at : null;
    return `<div class="board-row arr item${openId === it.id ? " selected" : ""}" data-id="${it.id}">
      <span class="trainno">${esc(trainNo(it))}</span>
      <span class="status ${cls}">${esc(label)}</span>
      <span class="dest" title="${esc(it.title)}">${esc(it.title)}</span>
      <span class="num">${esc(trip ? dur(trip) : "—")}</span>
      <span class="num">${esc(money(it.cost_usd))}</span>
      <span class="num">${it.finished_at ? esc(hm(it.finished_at)) : ""}</span></div>`;
  }).join("") : `<div class="board-empty">NO ARRIVALS YET</div>`;
}

/* ── render: crew, routing ────────────────────────────── */
function renderCrew() {
  const lj = S.line_jobs, jev = S.jev;
  const nextDispatch = S.dispatcher_next > 0 ? "next survey in " + dur(S.dispatcher_next) : "ready when a train is free";
  const rows = [
    ["Dispatcher", lj.dispatcher ? ["badge-on", "SURVEYING"] : ["", "IDLE"], "Proposes the next most valuable change · " + nextDispatch,
      `<button class="btn btn-sm" id="btn-dispatch">Survey now</button>`],
    ["Signal", lj.signal ? ["badge-on", "ANALYSING"] : ["", "WATCHING"], "Tails app logs; Jev screens error bursts before an LLM files a bug", ""],
    ["Operations", lj.ops ? ["badge-on", "WRITING"] : ["", "IDLE"], "Hourly digest of the line with recommendations", ""],
    ["Jev (decide)", jev.available ? ["badge-on", "ON"] : ["badge-warn", "OFF"],
      jev.available ? `${jev.model} — difficulty → service class, triage pre-screen, file relevance, log screening` : `Unavailable: ${jev.error || "unknown"}. LLMs take over its decisions.`, ""],
  ];
  let html = rows.map(([name, [bcls, btxt], desc, action]) => `<div class="crew-row"><span class="crew-name">${esc(name)}</span><span class="badge ${bcls}">${esc(btxt)}</span><span class="crew-desc">${desc.includes("<") ? desc : esc(desc)} ${action}</span></div>`).join("");
  const projects = S.projects.map((p) => `<span class="chip" title="${esc(p.path)}">${esc(p.name)}${p.paused ? " · paused" : ""}${p.schedule ? " · " + esc(p.schedule) + "h" : ""}</span>`).join(" ");
  html += `<div class="crew-row"><span class="crew-name">Projects</span><span></span><span class="crew-desc">${projects || "None configured — set AGENT_TEAM_DEFAULT_PROJECT or projects.json"}</span></div>`;
  for (const p of S.projects) {
    const c = S.checks[p.path] || {};
    const w = S.watches[p.path];
    html += `<div class="crew-row"><span class="crew-name">${esc(p.name)} checks</span><span class="badge ${c.test ? "badge-on" : "badge-warn"}">${c.test ? "CI ON" : "NO TESTS"}</span>
      <span class="crew-desc">${c.setup ? `setup <code class="cmd">${esc(c.setup)}</code> ` : ""}${c.test ? `test <code class="cmd">${esc(c.test)}</code>` : `Add <code class="cmd">"test"</code> to projects.json (or AGENT_TEAM_TEST_CMD) to gate builds and merges on your tests.`}
      ${w ? `<br>👁 Watching logs after #${w.item_id} for ${dur(w.until - Date.now() / 1000)} more` : ""}</span></div>`;
  }
  html += `<div class="crew-row"><span class="crew-name">Notifications</span><span class="badge ${S.notify.enabled ? "badge-on" : ""}">${S.notify.enabled ? "HOOKED" : "OFF"}</span><span class="crew-desc">${S.notify.enabled ? "Events: " + esc(S.notify.events.join(", ")) : "Set AGENT_TEAM_NOTIFY_CMD or AGENT_TEAM_NOTIFY_WEBHOOK to get gate, failure and regression alerts."}</span></div>`;
  if (S.ops_report) html += `<div class="crew-row"><span class="crew-name">Latest ops report</span><span class="badge">${esc(hm(S.ops_report.ts))}</span><div class="crew-desc ops-report">${esc(S.ops_report.message)}${(S.ops_report.data?.recommendations || []).map((r) => "\n→ " + esc(r)).join("")}</div></div>`;
  $("#crew").innerHTML = html;
  $("#btn-dispatch")?.addEventListener("click", () => post("/api/dispatch").then(refreshSoon).catch(alertErr));
}

function renderRouting() {
  const classes = Object.entries(S.service_classes);
  const by = Object.fromEntries((S.stats.by_model_24h || []).map((r) => [r.model, r]));
  const R = S.routing, F = R.fleet;
  const fc = Object.fromEntries((F?.enabled ? F.classes : []).map((c) => [c.class, c]));
  // Trials only run on low-risk work, so only the classes that work is routed to see challengers.
  const trialClasses = new Set(F?.enabled && F.trial_rate > 0 ? R.levels.filter((l) => F.levels.includes(l.difficulty)).map((l) => l.class) : []);
  let html = fleetStatus(F) + `<table class="fleet"><tr><th>Class</th><th>Model</th><th class="num">24h</th></tr>` + classes.map(([k, c]) => {
    const u = by[c.model];
    const cache = u && u.tin ? ` · ${Math.round(100 * (u.cached || 0) / u.tin)}% cached` : "";
    return `<tr><td>${svcBadge(k)}</td><td>${modelId(c.model)}${fleetCell(fc[k], trialClasses.has(k))}</td><td class="num">${u ? money(u.cost) + `<div class="small muted">${u.runs} runs${cache}</div>` : "—"}</td></tr>`;
  }).join("") + `</table>`;
  html += `<h3 style="margin-top:14px">Difficulty → class <span class="badge ${R.adaptive ? "badge-on" : ""}">${R.adaptive ? "ADAPTIVE" : "FIXED"}</span></h3>
    <table class="levels"><tr><th>Jev says</th><th>Runs as</th><th class="num">First-pass</th><th class="num">Avg fare</th></tr>` +
    R.levels.map((l) => {
      const h = l.history.find((x) => x.cls === l.class) || l.history[0];
      const votes = h && h.retros ? `<div class="why">${h.underpowered ? `${h.underpowered}× too weak ` : ""}${h.overpowered ? `${h.overpowered}× overkill` : ""}</div>` : "";
      const rate = h ? `${Math.round(100 * h.first_pass / h.n)}% <span class="small muted">of ${h.n}</span>${votes}` : "—";
      return `<tr><td>${esc(l.difficulty)}</td><td>${svcBadge(l.class)}${l.class !== l.default ? `<div class="why">${esc(l.why)}</div>` : ""}</td>
        <td class="num">${rate}</td><td class="num">${h ? money(h.avg_cost) : "—"}</td></tr>`;
    }).join("") + `</table>`;
  html += `<div class="stations-map">` + Object.entries(S.station_models).filter(([st]) => !st.endsWith("_retry")).map(([st, choice]) =>
    `<span class="chip"><b>${esc(st)}</b> → ${esc(choiceLabel(choice))}</span>`).join("") + `</div>`;
  html += fleetLog(F);
  html += `<p class="small muted" style="margin-top:10px">First-pass = arrived without rework. ${R.adaptive ? `With ${R.min_samples}+ samples, a class that passes first time under 50% is bumped up; one above 90% tries a class cheaper.` : "Adaptive routing is off (AGENT_TEAM_ADAPTIVE_ROUTING=0), so this history doesn't move routing."} Rework escalates a train one class. Fallback: <code>openrouter/auto</code>.</p>`;
  $("#routing").innerHTML = html;
}

/* The model fleet: sync status above the class table, each class's price and
   challengers inside it, and recent changes plus how it works below. */
const perM = (v) => v == null ? "" : `$${v < 1 ? v.toFixed(2) : v.toFixed(1)}/M`;
function fleetStatus(F) {
  if (!F) return "";
  if (!F.enabled) return `<p class="small muted" style="margin:-4px 0 8px">Model fleet off (AGENT_TEAM_MODEL_FLEET=0): classes stay on their configured models.</p>`;
  const synced = F.fetched_at ? `OpenRouter catalogue read ${ago(F.fetched_at)} ago · ${F.eligible} of ${F.total} models usable` : "OpenRouter catalogue not read yet";
  return `<p class="small muted" style="margin:-4px 0 8px">${esc(synced)}${F.error ? ` · <span class="warn-text">last read failed: ${esc(F.error)}</span>` : ""}</p>`;
}
function fleetCell(c, trials) {
  if (!c) return "";
  const pct = (v) => v == null ? "—" : Math.round(100 * v) + "%";
  const own = [perM(c.price), c.pinned ? "pinned" : "", c.stats?.n ? `${pct(c.stats.rate)} first time on ${c.stats.n} ${c.stats.n === 1 ? "train" : "trains"}` : ""].filter(Boolean).join(" · ");
  const list = trials ? c.challengers : c.challengers.filter((x) => !x.model.startsWith("typesafe/")).slice(0, 1);
  const names = list.map((x) => `${modelId(x.model)} <span class="muted">${x.n ? `${pct(x.rate)} of ${x.n}` : perM(x.price) || "trial"}</span>`).join(", ");
  const label = trials ? "trying" : "next in line";
  return (own ? `<div class="small muted">${own}</div>` : "") + (names ? `<div class="challengers"><span class="muted">${label}</span> ${names}</div>` : "");
}
function fleetLog(F) {
  if (!F?.enabled) return "";
  let html = F.history.length ? `<div class="fleet-log">` + F.history.slice(0, 4).map((h) =>
    `<div class="small"><span class="muted">${esc(ago(h.ts))} ago</span> ${esc(h.message)}</div>`).join("") + `</div>` : "";
  return html + `<p class="small muted" style="margin-top:8px">Model fleet: ${Math.round(100 * F.trial_rate)}% of ${esc(F.levels.join("/"))} trains try a challenger from their class's price band for spec and first build. After ${F.min_samples}+ trains each, one that arrives first time as often for no more per arrival becomes the class's model; one far worse sits out. Retired or repriced models are replaced from the band. Pin a class in <code>models.json</code> to keep it fixed.</p>`;
}

// "builder+1[..express]" → "one class up, at most Limited Express"
function choiceLabel(choice) {
  const m = /^builder([+-]\d+)?(?:\[(\w*)\.\.(\w*)\])?$/.exec(choice);
  if (!m) return S.service_classes[choice]?.label || choice;
  const n = Number(m[1] || 0), name = (c) => S.service_classes[c]?.label || c;
  const parts = [n === 0 ? "item's class" : `${Math.abs(n)} class${Math.abs(n) > 1 ? "es" : ""} ${n > 0 ? "up" : "down"}`];
  if (m[2] && m[3]) parts.push(`${name(m[2])}–${name(m[3])}`);
  else if (m[2]) parts.push(`at least ${name(m[2])}`);
  else if (m[3]) parts.push(`at most ${name(m[3])}`);
  return parts.join(", ");
}

/* ── retrospectives & playbook ───────────────────────── */
const FIT = { underpowered: ["fit-under", "class too weak"], overpowered: ["fit-over", "class overkill"] };
function renderLearning() {
  if (!S) return;
  const list = $("#retro-list");
  if (!S.retro_enabled) list.innerHTML = `<p class="empty">Retrospectives are off (AGENT_TEAM_RETRO=0).</p>`;
  else if (!retroData) list.innerHTML = `<p class="empty">Loading…</p>`;
  else list.innerHTML = retroData.retros.length ? retroData.retros.map(retroEntry).join("")
    : `<p class="empty">No retrospectives in this range. Every train ends its journey at JY09 Retro.</p>`;
  if (retroData) {
    const sm = retroData.summary;
    $("#retro-kpis").innerHTML = [
      tile("Retrospectives", String(sm.count), `<div class="kpi-note">${sm.arrived} arrived · ${sm.failed} failed</div>`),
      tile("Class about right", sm.count ? Math.round(100 * sm.class_fit.right / sm.count) + "%" : "—",
        `<div class="kpi-note">${sm.class_fit.underpowered} too weak · ${sm.class_fit.overpowered} overkill</div>`),
      tile("Notes added", String(sm.notes_added), `<div class="kpi-note">${sm.notes_retired} retired</div>`),
    ].join("");
  }
  let html = "";
  const projects = S.projects.filter((p) => !retroProject || p.path === retroProject);
  for (const p of projects) {
    const notes = S.playbook[p.path] || [];
    html += `<h4 class="pb-project">${esc(p.name)} <span class="badge">${notes.length}</span></h4>`;
    if (!notes.length) { html += `<p class="empty">No notes yet. Retrospectives add them when a train hits a problem the next one could avoid.</p>`; continue; }
    const by = {};
    for (const n of notes) (by[n.station] ||= []).push(n);
    for (const [st, items] of Object.entries(by)) {
      html += `<div class="pb-station">${esc(S.stations.find((x) => x.key === st)?.code || "")} ${esc(st)}</div>`;
      html += items.map((n) => `<div class="lesson"><span>${esc(n.text)} <span class="note-stats">${n.uses ? `${n.wins}/${n.uses} first-pass` : "unused yet"} · <a class="muted" data-id="${n.item_id}">#${n.item_id}</a></span></span><button title="Remove this note" data-note="${n.id}" data-project="${esc(p.path)}">×</button></div>`).join("");
    }
  }
  $("#learning").innerHTML = html || `<p class="empty">No projects.</p>`;
}

function retroEntry(r) {
  const d = r.data || {};
  const [cls, label] = r.outcome === "done" ? ["badge-on", "ARRIVED"] : ["badge-warn", "FAILED"];
  const fit = FIT[r.class_fit];
  const ul = (title, items) => items && items.length ? `<h5>${title}</h5><ul>${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
  const added = d.notes_added || [], retired = d.notes_retired || [];
  const more = ul("Went well", d.went_well) + ul("Went wrong", d.went_wrong)
    + ul("Playbook notes added", added.map((n) => `[${n.station}] ${n.text}`))
    + ul("Playbook notes retired", retired.map((n) => `[${n.station}] ${n.text} — ${n.why}`));
  const delta = [added.length ? `+${added.length} note${added.length > 1 ? "s" : ""}` : "", retired.length ? `−${retired.length} retired` : ""].filter(Boolean).join(" · ");
  const project = S.projects.find((p) => p.path === r.project)?.name;
  return `<div class="retro-entry">
    <div class="r-head"><span class="badge ${cls}">${label}</span>${svcBadge(r.first_class || r.service_class)}<span class="r-title" data-train="${r.item_id}">#${r.item_id} ${esc(r.title || "")}</span></div>
    <div class="r-meta">${esc(ago(r.ts))} ago${project && !retroProject ? ` · ${esc(project)}` : ""}${r.attempt > 1 ? ` · ${r.attempt} attempts` : ""}${r.cost_usd ? ` · $${r.cost_usd.toFixed(2)}` : ""}${fit ? ` · <span class="${fit[0]}">${fit[1]}</span>` : ""}${delta ? ` · <b>${delta}</b>` : ""}</div>
    <p>${esc(d.summary || d.error || "(no summary)")}</p>
    ${d.root_cause ? `<p class="r-cause"><b>Root cause:</b> ${esc(d.root_cause)}</p>` : ""}
    ${more ? `<details><summary>Details</summary>${more}</details>` : ""}
  </div>`;
}

let retroDays = 30, retroProject = "", retroData = null, retroTimer = null;
async function loadRetros() {
  try {
    retroData = await api(`/api/retros?days=${retroDays}${retroProject ? "&project=" + encodeURIComponent(retroProject) : ""}`);
  } catch (e) {
    if (!retroData) $("#retro-list").innerHTML = `<p class="empty">Couldn't load retrospectives: ${esc(e.message || e)}</p>`;
    return;
  }
  const sel = $("#retro-project"), cur = sel.value;
  sel.innerHTML = `<option value="">All projects</option>` + (S?.projects || []).map((pr) => `<option value="${esc(pr.path)}">${esc(pr.name)}</option>`).join("");
  sel.value = cur;
  renderLearning();
}

/* ── train diagram (ダイヤ) ──────────────────────────── */
let diagramHours = 3, diagramData = null, diagramTimer = null, diagramFetched = 0;
function diagramSoon() {
  if (Date.now() - diagramFetched < 4000) { clearTimeout(diagramTimer); diagramTimer = setTimeout(loadDiagram, 4000); return; }
  loadDiagram();
}
function diagramMessage(text) {
  const svg = $("#diagram");
  const W = svg.clientWidth || 600, H = svg.clientHeight || 300;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = `<text class="empty-note" x="${W / 2}" y="${H / 2}" text-anchor="middle">${esc(text)}</text>`;
}
async function loadDiagram() {
  diagramFetched = Date.now();
  if (!diagramData) diagramMessage("Loading the diagram…");
  try { diagramData = await api("/api/diagram?hours=" + diagramHours); renderDiagram(); }
  catch (e) { if (!diagramData) diagramMessage("Couldn't load the diagram: " + (e.message || e)); }
}
function renderDiagram() {
  const svg = $("#diagram");
  if (!diagramData || !S) return;
  // Size to the space available; on narrow screens show station codes only.
  const W = Math.max(280, svg.clientWidth || 900), H = svg.clientHeight || 300;
  const compact = W < 560;
  const L = compact ? 44 : 118, R = compact ? 8 : 16, T = 14, B = 26;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const t0 = diagramData.since, t1 = diagramData.now;
  const x = (t) => L + (Math.max(t0, Math.min(t1, t)) - t0) / (t1 - t0) * (W - L - R);
  const idx = Object.fromEntries(S.stations.map((s, i) => [s.key, i]));
  const y = (st) => T + (idx[st] ?? 0) * ((H - T - B) / (S.stations.length - 1));
  let g = "";
  S.stations.forEach((s) => {
    g += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(s.key)}" y2="${y(s.key)}"/>`;
    g += `<text class="axis station" x="4" y="${y(s.key) + 4}">${esc(s.code)}</text>` +
         (compact ? "" : `<text class="axis station-name" x="44" y="${y(s.key) + 4}">${esc(s.name)}</text>`);
  });
  // Pick the smallest "nice" tick step that keeps labels at least ~80px apart.
  const span = t1 - t0, pxPerSec = (W - L - R) / span;
  const step = [900, 1800, 3600, 7200, 3 * 3600, 6 * 3600, 12 * 3600, 86400, 2 * 86400]
    .find((st) => st * pxPerSec >= 80) || 2 * 86400;
  for (let t = Math.ceil(t0 / step) * step; t < t1; t += step) {
    g += `<line class="grid hour" x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${H - B}"/>`;
    const d = new Date(t * 1000);
    const label = step >= 86400 || (d.getHours() === 0 && d.getMinutes() === 0 && span > 86400)
      ? d.toLocaleDateString([], { month: "short", day: "numeric" }) : hm(t);
    g += `<text class="axis" x="${x(t)}" y="${H - 8}" text-anchor="middle">${esc(label)}</text>`;
  }
  g += `<line class="now" x1="${x(t1)}" x2="${x(t1)}" y1="${T}" y2="${H - B}"/>`;
  const color = { local: "var(--c-local)", rapid: "var(--c-rapid)", express: "var(--c-express)", shinkansen: "var(--c-shinkansen)" };
  const trains = diagramData.trains.filter((tr) => tr.points.length);
  for (const tr of trains) {
    const c = color[tr.cls] || "#888";
    const pts = tr.points.map(([ts, st]) => [x(ts), y(st), idx[st] ?? 0]);
    const terminal = ["done", "failed", "rejected", "cancelled"].includes(tr.status);
    if (!terminal && S.stations.some((s) => s.key === tr.station)) pts.push([x(t1), y(tr.station), idx[tr.station]]);
    let segs = "";
    for (let i = 1; i < pts.length; i++) {
      const [x0, y0, i0] = pts[i - 1], [x1, y1, i1] = pts[i];
      // forward travel: hold at the station, then move; backward (rework): dashed jump
      const back = i1 < i0;
      segs += back ? `<path class="run back" stroke="${c}" d="M${x0} ${y0} L${x1} ${y1}"/>`
                   : `<path class="run" stroke="${c}" d="M${x0} ${y0} H${x1} V${y1}"/>`;
    }
    const [ex, ey] = pts[pts.length - 1];
    let end = "";
    if (tr.status === "done") end = `<circle class="end" cx="${ex}" cy="${ey}" r="5" fill="${c}"/>`;
    else if (terminal) end = `<text x="${ex}" y="${ey + 4}" text-anchor="middle" fill="var(--bad)" font-weight="800" font-size="12">✕</text>`;
    else end = `<rect class="end" x="${ex - 3}" y="${ey - 6}" width="6" height="12" rx="2" fill="${c}"/>`;
    g += `<g class="train-line" data-train="${tr.id}"><title>#${tr.id} ${esc(tr.title || "")} — ${esc(tr.status || "")}</title>${segs}${end}</g>`;
  }
  if (!trains.length) g += `<text class="empty-note" x="${(L + W) / 2}" y="${H / 2}" text-anchor="middle">No trains ran in this window.</text>`;
  svg.innerHTML = g;
}

/* ── feed ─────────────────────────────────────────────── */
function titleFor(id) {
  const it = S && [...S.items, ...S.arrivals].find((i) => i.id === id);
  return it ? it.title : "";
}
function addFeed(evts) {
  for (const e of evts) {
    if (e.id <= lastEventId) continue;
    lastEventId = Math.max(lastEventId, e.id);
    feed.unshift(e);
  }
  feed.length = Math.min(feed.length, 150);
  renderFeed();
}
const QUIET = new Set(["classified", "context", "watching"]);
function renderFeed() {
  $("#feed").innerHTML = feed.filter((e) => !QUIET.has(e.kind)).slice(0, 80).map((e) => {
    if (e.kind === "autopilot_report") return `<li class="k-ops_report"><span class="t">${esc(hm(e.ts))}</span><div class="ops-report">🌙 ${esc(e.message)}</div></li>`;
    if (e.kind === "ops_report") return `<li class="k-ops_report"><span class="t">${esc(hm(e.ts))}</span><div class="ops-report">${esc(e.message)}</div></li>`;
    const st = e.station ? station(e.station).code : "LINE";
    const who = e.item_id ? `<a data-id="${e.item_id}">#${e.item_id}</a> ${esc(titleFor(e.item_id))} — ` : "";
    return `<li class="k-${esc(e.kind)}"><span class="t">${esc(hm(e.ts))}</span><span class="where">${esc(st)}</span><span class="msg">${who}${esc(e.message.split("\n")[0].slice(0, 260))}</span></li>`;
  }).join("") || `<li><span></span><span></span><span class="muted">Quiet on the line.</span></li>`;
}

/* ── drawer: one work item ────────────────────────────── */
async function openItem(id) {
  openId = id;
  detailTab = null;
  expandedRuns.clear();
  $("#drawer").classList.add("open");
  $("#drawer").setAttribute("aria-hidden", "false");
  $("#scrim").hidden = false;
  $("#drawer-body").innerHTML = `<p class="empty">Loading train ${id}…</p>`;
  history.replaceState(null, "", "#item-" + id);
  await loadDetail(id, true);
  renderBoards();
}
function closeDrawer() {
  openId = null; detail = null;
  $("#drawer").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
  $("#scrim").hidden = true;
  history.replaceState(null, "", location.pathname);
  if (S) renderBoards();
}
async function loadDetail(id, first) {
  try {
    const d = await api("/api/items/" + id);
    if (openId !== id) return;
    detail = d;
    if (first) {
      const running = d.runs.filter((r) => r.status === "running");
      running.forEach((r) => expandedRuns.add(r.id));
      detailTab = detailTab || (running.length ? "runs" : "timeline");
    }
    for (const rid of expandedRuns) await loadSteps(rid);
    renderDrawer();
  } catch (e) {
    $("#drawer-body").innerHTML = `<p class="error">${esc(e.message)}</p>`;
  }
}
async function loadSteps(rid) {
  const have = runSteps.get(rid) || [];
  const after = have.length ? have[have.length - 1].id : 0;
  const r = await api(`/api/runs/${rid}/steps?after=${after}`);
  runSteps.set(rid, have.concat(r.steps));
}

function routeStrip(it, events) {
  const visits = {};
  for (const e of events) if (e.kind === "arrived" && e.station) visits[e.station] = (visits[e.station] || 0) + 1;
  const idx = S.stations.findIndex((s) => s.key === it.station);
  const terminal = ["done", "failed", "rejected", "cancelled"].includes(it.status);
  const next = terminal ? null : S.stations[idx + (it.status === "running" ? 1 : 0)] || null;
  const c = S.service_classes[it.service_class];
  const head = terminal
    ? `<span class="lcd-next">${it.status === "done" ? "TERMINUS" : "OUT OF SERVICE"}<b>${esc(station(it.station).name)}</b></span>`
    : `<span class="lcd-next">${it.status === "running" ? "NOW AT" : it.status === "waiting" ? "SIGNAL AT" : "NEXT"}<b>${esc((it.status === "running" ? station(it.station) : next || station(it.station)).name)}</b></span>`;
  return `<div class="lcd"><div class="lcd-top">${head}<span class="lcd-class">${c ? svcBadge(it.service_class) + " for " + esc(base(it.project)) : ""}</span></div>
    <div class="route">${S.stations.map((s, i) => {
      let state = i < idx ? "passed" : i === idx ? (it.status === "done" ? "passed" : "current") : "future";
      if (i === idx && it.status === "waiting") state += " waiting";
      const n = visits[s.key] || 0;
      return `<div class="route-stop ${state}"><span class="visits">${n > 1 ? "×" + n : ""}</span><span class="dot"></span><span class="code">${esc(s.code)}</span><span class="name">${esc(s.name)}</span></div>`;
    }).join("")}</div></div>`;
}

function renderDrawer() {
  if (!detail) return;
  const { item: it, events, runs } = detail;
  const terminal = ["done", "failed", "rejected", "cancelled"].includes(it.status);
  const spentTime = (it.finished_at || Date.now() / 1000) - (it.started_at || it.created_at);
  let html = `<div class="d-kicker"><span>TRAIN ${esc(trainNo(it))}</span><span>·</span><span>${esc(base(it.project))}</span><span>·</span><span>${esc(it.kind)}</span><span>·</span><span>${esc(it.priority)} priority</span><span>·</span><span>from ${esc(it.source)}</span></div>
    <h2 class="d-title">${esc(it.title)}</h2>
    <div class="d-meta">${it.service_class ? svcBadge(it.service_class) : ""}
      ${it.difficulty ? `<span class="chip">difficulty: ${esc(it.difficulty)}${it.difficulty_score != null ? " (" + Number(it.difficulty_score).toFixed(1) + "/4)" : ""}</span>` : ""}
      ${it.train ? `<span class="chip">train set ${esc(it.train)}</span>` : ""}
      ${it.branch ? `<span class="chip" title="git branch"><code>${esc(it.branch)}</code></span>` : ""}
      ${it.parent_id ? `<a class="chip" data-train="${it.parent_id}" href="#item-${it.parent_id}">regression from #${it.parent_id}</a>` : ""}</div>`;
  html += routeStrip(it, events);
  if (it.status === "waiting" && it.gate === "board") {
    html += `<div class="gate-box"><h4>■ PROPOSED BY ${esc(it.source.toUpperCase())} — BOARD THIS TRAIN?</h4>
      <p class="small">The factory proposed this work itself. In Supervised mode it waits here until you board it; Autopilot boards proposals automatically.</p>
      <div class="d-actions"><button class="btn btn-primary" data-act="approve">Board</button><button class="btn btn-danger" data-act="reject">Decline</button></div></div>`;
  } else if (it.status === "waiting") {
    html += `<div class="gate-box"><h4>■ SIGNAL AT RED — ${it.gate === "spec" ? "SPEC APPROVAL" : "MERGE APPROVAL"}</h4>
      <p class="small">${it.gate === "spec" ? "Review the spec and holdout scenarios, then let the train depart to Build." : `Inspector approved and holdout satisfaction is ${pct(it.satisfaction)}. Approve to merge <code>${esc(it.branch)}</code> into trunk.`}</p>
      <div class="d-actions"><button class="btn btn-primary" data-act="approve">Approve</button><button class="btn btn-danger" data-act="reject">Reject</button></div></div>`;
  }
  if (it.outcome && (terminal || it.status === "held")) html += `<div class="outcome ${esc(it.status)}">${esc(it.outcome)}</div>`;
  html += `<div class="d-actions">${!terminal ? `<button class="btn btn-sm btn-danger" data-act="cancel">Cancel train</button>` : ""}${terminal && it.status !== "done" || it.status === "held" ? `<button class="btn btn-sm" data-act="retry">Retry</button>` : ""}</div>`;
  const scen = it.scenarios || [];
  html += `<div class="stats-grid">
    <div class="stat"><div class="k">Fare</div><div class="v">${money(it.cost_usd)}</div></div>
    <div class="stat"><div class="k">Tokens</div><div class="v">${tokens(it.tokens_in)} / ${tokens(it.tokens_out)}</div></div>
    <div class="stat"><div class="k">Attempt</div><div class="v">${it.attempt || 0}${it.conflicts ? ` · ${it.conflicts}⚠` : ""}</div></div>
    <div class="stat"><div class="k">Satisfaction</div><div class="v">${pct(it.satisfaction)}</div></div>
    <div class="stat"><div class="k">Journey</div><div class="v">${dur(spentTime)}</div></div>
    <div class="stat"><div class="k">Agent runs</div><div class="v">${runs.filter((r) => r.role !== "jev").length}</div></div></div>`;
  const tab = detailTab || "timeline";
  const retroTab = detail.retro || it.station === "retro" ? [["retro", "Retro"]] : [];
  html += `<div class="d-tabs">${[["timeline", "Timeline"], ["spec", "Spec & scenarios"], ...retroTab, ["runs", `Runs (${runs.filter((r) => r.role !== "jev").length}${runs.some((r) => r.role === "jev") ? " + Jev " + runs.filter((r) => r.role === "jev").length : ""})`]].map(([k, l]) => `<button class="d-tab${tab === k ? " active" : ""}" data-tab="${k}">${l}</button>`).join("")}</div>`;
  if (tab === "timeline") html += timelineHtml(events);
  if (tab === "spec") html += specHtml(it, scen, events);
  if (tab === "runs") html += runsHtml(runs);
  if (tab === "retro") html += retroHtml(detail.retro, it);
  const body = $("#drawer-body");
  const scroll = $("#drawer").scrollTop;
  body.innerHTML = html;
  $("#drawer").scrollTop = scroll;
}

function retroHtml(r, it) {
  if (!r) return `<p class="empty">${it.station === "retro" ? "The retrospective is running…" : "No retrospective for this train."}</p>`;
  const d = r.data || {};
  const list = (title, items) => items && items.length ? `<div class="retro-block"><h4>${title}</h4><ul>${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>` : "";
  const fit = FIT[r.class_fit];
  return `<div class="retro-block"><h4>Summary</h4><p>${esc(d.summary || d.error || "")}</p></div>
    ${d.root_cause ? `<div class="retro-block"><h4>Root cause</h4><p>${esc(d.root_cause)}</p></div>` : ""}
    ${list("Went well", d.went_well)}${list("Went wrong", d.went_wrong)}
    <div class="retro-block"><h4>Model class</h4><p>${svcBadge(it.first_class || it.service_class)} ${fit ? `<span class="${fit[0]}">${fit[1]}</span>` : "about right"}</p></div>
    ${list("Playbook notes added", (d.notes_added || []).map((n) => `[${n.station}] ${n.text}`))}
    ${list("Playbook notes retired", (d.notes_retired || []).map((n) => `[${n.station}] ${n.text} — ${n.why}`))}`;
}

function timelineHtml(events) {
  if (!events.length) return `<p class="empty">No events yet.</p>`;
  return `<ol class="timeline">${events.slice().reverse().map((e) => `<li class="k-${esc(e.kind)}"><span class="t">${esc(hms(e.ts))}</span><span class="where">${esc(e.station ? station(e.station).code : "")}</span><span class="msg"><span class="kind">${esc(e.kind)}</span>${esc(e.message)}${e.data?.results ? scenarioResults(e.data.results) : ""}${e.data?.files ? `<div class="small muted">${e.data.files.map(esc).join(", ")}</div>` : ""}</span></li>`).join("")}</ol>`;
}
function scenarioResults(results) {
  return results.map((r) => `<div class="scen"><span class="${r.passed ? "pass" : r.disputed ? "disputed" : "fail"}">${r.unrunnable ? "○ UNRUNNABLE" : r.passed ? "✓ PASS" : r.disputed ? "⚠ DISPUTED" : "✗ FAIL"}</span> ${esc(r.name)}<div class="small muted">${esc(r.evidence)}</div></div>`).join("");
}
function specHtml(it, scen, events) {
  const spec = it.spec || {};
  if (!it.spec && !scen.length) return `<p class="empty">Not specified yet — the Spec station (JY03) writes acceptance criteria, a plan and sealed holdout scenarios.</p><div class="spec-block"><h4>Request</h4><pre>${esc(it.description)}</pre></div>`;
  const lastVerify = [...events].reverse().find((e) => e.kind === "verified" && e.data?.results);
  const res = Object.fromEntries((lastVerify?.data.results || []).map((r) => [r.name, r]));
  return `<div class="spec-block"><h4>Request</h4><pre>${esc(it.description)}</pre></div>
    <div class="spec-block"><h4>Acceptance criteria</h4><ul>${(spec.acceptance_criteria || []).map((c) => `<li>${esc(c)}</li>`).join("")}</ul></div>
    ${spec.plan ? `<div class="spec-block"><h4>Plan</h4><pre>${esc(spec.plan)}</pre></div>` : ""}
    ${(spec.relevant_files || []).length ? `<div class="spec-block"><h4>Relevant files</h4>${spec.relevant_files.map((f) => `<span class="chip"><code>${esc(f)}</code></span>`).join(" ")}</div>` : ""}
    <div class="spec-block"><h4>Holdout scenarios <span class="sealed">🔒 SEALED FROM THE BUILDER</span></h4>
      ${scen.map((s) => { const r = res[s.name]; return `<div class="scen">${r ? `<span class="${r.passed ? "pass" : r.disputed ? "disputed" : "fail"}">${r.passed ? "✓" : r.disputed ? "⚠ disputed" : "✗"}</span> ` : ""}<b>${esc(s.name)}</b><div class="small"><span class="muted">Steps:</span> <code>${esc(s.steps)}</code></div><div class="small"><span class="muted">Expect:</span> ${esc(s.expected)}</div>${r ? `<div class="small muted">Evidence: ${esc(r.evidence)}</div>` : ""}</div>`; }).join("") || `<p class="empty">No scenarios.</p>`}</div>`;
}
function runsHtml(runs) {
  if (!runs.length) return `<p class="empty">No agent runs yet.</p>`;
  return runs.slice().reverse().map((r) => {
    const open = expandedRuns.has(r.id);
    const took = (r.ended_at || Date.now() / 1000) - r.started_at;
    const steps = runSteps.get(r.id) || [];
    return `<div class="run"><div class="run-head" data-run="${r.id}">
      <span><span class="role ${r.role === "ci" ? "ci" : ""}">${esc(r.role === "ci" ? "checks" : r.role)}</span><span class="run-status rs-${esc(r.status)}">${esc(r.status.toUpperCase())}</span> <span class="small muted">${esc(station(r.station).code)} · ${esc(hms(r.started_at))}</span></span>
      <span class="rstats">${money(r.cost_usd)} · ${r.steps} turns · ${dur(took)}</span>
      <span class="model">${esc(r.model || "")}${r.service_class ? " · " + esc(S.service_classes[r.service_class]?.label || r.service_class) : ""}</span>
      <span class="rstats">${tokens(r.tokens_in)} in / ${tokens(r.tokens_out)} out</span></div>
      ${open ? `<ol class="steps">${steps.map((s) => `<li class="s-${esc(s.kind)}"><span class="sk">${esc(s.kind === "model" ? "think" : s.kind === "tool" ? s.name : s.kind)}</span><span class="sd">${esc(s.detail || (s.kind === "model" ? "(tool calls)" : ""))}${s.cost_usd ? ` <span class="cost">${money(s.cost_usd)}</span>` : ""}</span></li>`).join("") || `<li><span></span><span class="muted">No steps yet.</span></li>`}${r.error ? `<li class="s-error"><span class="sk">error</span><span class="sd">${esc(r.error)}</span></li>` : ""}</ol>` : ""}</div>`;
  }).join("");
}

async function itemAction(act) {
  if (!openId) return;
  let body = {};
  if (act === "reject") {
    const reason = prompt("Reason for rejecting (optional):");
    if (reason === null) return;
    body = { reason };
  }
  if (act === "cancel" && !confirm("Cancel this train? Its worktree and branch are removed.")) return;
  try {
    await post(`/api/items/${openId}/${act}`, body);
    refreshSoon(50); detailSoon(100);
  } catch (e) { alertErr(e); }
}
function alertErr(e) { alert(e.message || String(e)); }

/* ── live stream ──────────────────────────────────────── */
function connect() {
  const es = new EventSource("/api/stream");
  es.onopen = () => { $("#conn-lost").hidden = true; refresh(); };
  es.onerror = () => { $("#conn-lost").hidden = false; };
  es.addEventListener("item", (m) => {
    const it = JSON.parse(m.data);
    refreshSoon();
    if (it && it.id === openId) detailSoon(200);
  });
  es.addEventListener("run", (m) => {
    const r = JSON.parse(m.data);
    refreshSoon();
    if (r && r.item_id === openId) {
      if (r.status === "running") expandedRuns.add(r.id);
      detailSoon(200);
    }
  });
  es.addEventListener("event", (m) => {
    const e = JSON.parse(m.data);
    addFeed([e]);
    if (e.item_id === openId) detailSoon(300);
    if (!e.item_id) refreshSoon();
  });
  es.addEventListener("step", (m) => {
    const s = JSON.parse(m.data);
    if (s.item_id === openId && expandedRuns.has(s.run_id)) {
      const have = runSteps.get(s.run_id) || [];
      if (!have.length || have[have.length - 1].id < s.id) runSteps.set(s.run_id, have.concat([s]));
      if (detailTab === "runs") renderDrawer();
      detailSoon(1500);
    }
  });
}

/* ── wiring ───────────────────────────────────────────── */
function wire() {
  document.addEventListener("click", (e) => {
    const row = e.target.closest(".board-row.item, #feed a[data-id], #away-banner a[data-id]");
    if (row) return openItem(Number(row.dataset.id));
    const chip = e.target.closest("[data-train]");
    if (chip) return openItem(Number(chip.dataset.train));
    const link = e.target.closest(".lesson a[data-id]");
    if (link) return openItem(Number(link.dataset.id));
    const note = e.target.closest("[data-note]");
    if (note) {
      if (!confirm("Remove this playbook note? Future trains will no longer see it.")) return;
      return post("/api/playbook/delete", { project: note.dataset.project, id: Number(note.dataset.note) }).then(() => refreshSoon(50)).catch(alertErr);
    }
    const hours = e.target.closest("[data-hours]");
    if (hours) {
      diagramHours = Number(hours.dataset.hours);
      $$("[data-hours]").forEach((x) => x.classList.toggle("active", x === hours));
      return loadDiagram();
    }
    const tab = e.target.closest(".d-tab");
    if (tab) { detailTab = tab.dataset.tab; return renderDrawer(); }
    const head = e.target.closest(".run-head");
    if (head) {
      const rid = Number(head.dataset.run);
      if (expandedRuns.has(rid)) { expandedRuns.delete(rid); renderDrawer(); }
      else { expandedRuns.add(rid); loadSteps(rid).then(renderDrawer); }
      return;
    }
    const act = e.target.closest("[data-act]");
    if (act) return itemAction(act.dataset.act);
  });
  $$(".tab[data-filter]").forEach((t) => t.addEventListener("click", () => {
    depFilter = t.dataset.filter;
    $$(".tab[data-filter]").forEach((x) => x.classList.toggle("active", x === t));
    renderBoards();
  }));
  $$(".tab[data-afilter]").forEach((t) => t.addEventListener("click", () => {
    arrFilter = t.dataset.afilter;
    $$(".tab[data-afilter]").forEach((x) => x.classList.toggle("active", x === t));
    renderBoards();
  }));
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#scrim").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && openId) closeDrawer(); });
  $$(".view-tab").forEach((t) => t.addEventListener("click", (e) => { e.preventDefault(); history.replaceState(null, "", t.dataset.view === "line" ? location.pathname : "#" + t.dataset.view); setView(t.dataset.view); }));
  $$("[data-days]").forEach((b) => b.addEventListener("click", () => {
    statsDays = Number(b.dataset.days); $$("[data-days]").forEach((x) => x.classList.toggle("active", x === b)); loadStats();
  }));
  $("#stats-project").addEventListener("change", (e) => { statsProject = e.target.value; loadStats(); });
  $$("[data-rdays]").forEach((b) => b.addEventListener("click", () => {
    retroDays = Number(b.dataset.rdays); $$("[data-rdays]").forEach((x) => x.classList.toggle("active", x === b)); loadRetros();
  }));
  $("#retro-project").addEventListener("change", (e) => { retroProject = e.target.value; loadRetros(); });
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-table]");
    if (!t) return;
    const id = t.dataset.table; tableMode.has(id) ? tableMode.delete(id) : tableMode.add(id); renderStats();
  });
  $("#btn-autopilot").addEventListener("click", () => {
    const on = !(S?.autopilot?.on);
    if (on) {
      const waiting = (S?.items || []).filter((i) => i.status === "waiting").length;
      if (!confirm("Turn on Autopilot?\n\nThe line runs dark: approval gates are skipped" + (waiting ? ` (${waiting} waiting train${waiting > 1 ? "s" : ""} will be released)` : "") +
        ", proposals board themselves, and post-deploy regressions are reverted automatically.")) return;
    }
    post("/api/autopilot", { on }).then(() => refreshSoon(50)).catch(alertErr);
  });
  $("#btn-settings").addEventListener("click", openSettings);
  $("#settings-cancel").addEventListener("click", () => $("#settings-dialog").close());
  $("#settings-form").addEventListener("submit", saveSettings);
  $$("[data-preset]").forEach((b) => b.addEventListener("click", () => {
    const [on, off] = b.dataset.preset.split("|"), f = $("#settings-form");
    f.on_cron.value = on; f.off_cron.value = off; f.schedule_enabled.checked = true;
  }));
  $("#btn-pause").addEventListener("click", () => post(S?.paused ? "/api/resume" : "/api/pause").then(() => refreshSoon(50)).catch(alertErr));

  const dlg = $("#new-dialog");
  $("#btn-new").addEventListener("click", () => {
    const sel = $("#project-select");
    sel.innerHTML = (S?.projects || []).map((p) => `<option value="${esc(p.path)}">${esc(p.name)}</option>`).join("");
    $("#token-row").hidden = !(S?.auth_required);
    $("#new-error").hidden = true;
    dlg.showModal();
  });
  $("#new-cancel").addEventListener("click", () => dlg.close());
  $("#new-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target));
    if (f.token) setToken(f.token);
    try {
      const r = await post("/api/items", { title: f.title, description: f.description, kind: f.kind, priority: f.priority, project: f.project });
      dlg.close(); e.target.reset();
      await refresh();
      openItem(r.item.id);
    } catch (err) {
      const el = $("#new-error"); el.textContent = err.message; el.hidden = false;
    }
  });
}

function fmtWhen(ts) {
  return ts ? new Date(ts * 1000).toLocaleString([], { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", hour12: false }) : "never";
}
function renderSettingsInfo() {
  const ap = S?.autopilot;
  if (!ap) return;
  $("#set-mode").innerHTML = ap.on
    ? `Now: <b>Autopilot</b> since ${esc(fmtWhen(ap.since))} (${esc(ap.source)}).`
    : `Now: <b>Supervised</b> since ${esc(fmtWhen(ap.since))} (${esc(ap.source)}).`;
  const p = ap.preview || {};
  $("#set-preview").textContent = p.error ? "⚠ " + p.error
    : ap.config.schedule_enabled ? `Next: autopilot on ${fmtWhen(p.next_on)} · off ${fmtWhen(p.next_off)}` : "Schedule off.";
  const all = (S.projects || []).map((p) => p.name);
  const tested = all.filter((n) => !ap.untested.includes(n));
  $("#set-untested").textContent = !all.length ? "No projects are configured yet."
    : ap.untested.length
      ? `${ap.untested.join(", ")} ${ap.untested.length > 1 ? "have" : "has"} no test command. With this off, ${ap.untested.length > 1 ? "their" : "its"} trains stop at the merge gate even in autopilot and wait for you; with it on, they merge on holdout scenarios and code review alone.`
        + (tested.length ? ` ${tested.join(", ")} ${tested.length > 1 ? "are" : "is"} tested and merge${tested.length > 1 ? "" : "s"} unattended either way.` : "")
      : `This only affects projects without a test command. Every project has one, so autopilot already merges them once their tests and holdout scenarios pass. (Currently: ${tested.join(", ")})`;
}
function openSettings() {
  const ap = S?.autopilot;
  if (!ap) return;
  const f = $("#settings-form"), c = ap.config;
  f.schedule_enabled.checked = !!c.schedule_enabled;
  f.on_cron.value = c.on_cron || "";
  f.off_cron.value = c.off_cron || "";
  f.merge_without_tests.checked = !!c.merge_without_tests;
  f.gate_spec.checked = !!c.supervised_gates?.spec;
  f.gate_merge.checked = !!c.supervised_gates?.merge;
  $("#settings-error").hidden = true;
  renderSettingsInfo();
  $("#settings-dialog").showModal();
}
async function saveSettings(e) {
  e.preventDefault();
  const f = e.target;
  try {
    await post("/api/autopilot/settings", {
      schedule_enabled: f.schedule_enabled.checked, on_cron: f.on_cron.value.trim(), off_cron: f.off_cron.value.trim(),
      merge_without_tests: f.merge_without_tests.checked,
      supervised_gates: { spec: f.gate_spec.checked, merge: f.gate_merge.checked },
    });
    await refresh();
    renderSettingsInfo();
    $("#settings-dialog").close();
  } catch (err) {
    const el = $("#settings-error"); el.textContent = err.message; el.hidden = false;
  }
}

function tickClock() {
  $("#clock").textContent = new Date().toLocaleTimeString([], { hour12: false }) + (S ? " · up " + dur(S.uptime + (Date.now() / 1000 - S.now)) : "");
}

async function boot() {
  wire();
  await refresh();
  try { addFeed((await api("/api/events")).events); } catch {}
  connect();
  setInterval(tickClock, 1000);
  setInterval(() => { if (S) { renderBoards(); if (openId && detail && detail.item.status === "running") renderDrawer(); } }, 5000);
  setInterval(refresh, 30000);
  window.addEventListener("resize", () => { renderDiagram(); if (view === "stats" && statsData) renderStats(); });
  const m = location.hash.match(/^#item-(\d+)$/);
  if (m) openItem(Number(m[1]));
  if (location.hash === "#settings") openSettings();
  if (location.hash === "#stats" || location.hash === "#retro") setView(location.hash.slice(1));
}
boot();

/* ══ Stats view ═════════════════════════════════════════
   Charts are plain SVG built here (no library): thin marks, hairline grid,
   hover tooltips, and a table view for every chart. Labels go in via
   textContent / esc() — names come from the database. */

let view = "line", statsDays = 7, statsProject = "", statsData = null, statsTimer = null;
const tableMode = new Set();
const NS = "http://www.w3.org/2000/svg";

function setView(v) {
  view = v;
  $("#line-view").hidden = v !== "line";
  $("#stats-view").hidden = v !== "stats";
  $("#retro-view").hidden = v !== "retro";
  $$(".view-tab").forEach((t) => t.classList.toggle("active", t.dataset.view === v));
  if (v === "stats") { loadStats(); clearInterval(statsTimer); statsTimer = setInterval(loadStats, 60000); }
  else clearInterval(statsTimer);
  if (v === "retro") { loadRetros(); clearInterval(retroTimer); retroTimer = setInterval(loadRetros, 30000); }
  else clearInterval(retroTimer);
}

async function loadStats() {
  $$(".chart-card").forEach((c) => c.classList.add("loading"));  // keep the frame while refetching
  try {
    statsData = await api(`/api/stats?days=${statsDays}${statsProject ? "&project=" + encodeURIComponent(statsProject) : ""}`);
    renderStats();
  } catch (e) { /* keep the previous render */ }
  $$(".chart-card").forEach((c) => c.classList.remove("loading"));
}

const pct0 = (v) => v == null ? "—" : Math.round(v * 100) + "%";
const minutes = (v) => v == null ? "—" : v === 0 ? "0" : v < 1 ? Math.round(v * 60) + "s" : (Number.isInteger(v) ? v : v.toFixed(1)) + "m";
const compact = (n) => n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e4 ? Math.round(n / 1e3) + "K" : n >= 1e3 ? (n / 1e3).toFixed(1) + "K" : String(Math.round(n || 0));
function bucketLabel(t, long) {
  const d = new Date(t * 1000);
  if (statsData.bucket === "hour") return long ? d.toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false }) : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
  return d.toLocaleDateString([], long ? { weekday: "short", month: "short", day: "numeric" } : { month: "short", day: "numeric" });
}
// Pick a round step first, then the axis max as a whole number of steps, so every
// tick is a clean value. Counts never get fractional steps.
function niceScale(v, integer) {
  const target = 4;
  if (!v || v <= 0) v = integer ? 4 : 1;
  if (integer && v < 4) v = 4;  // small counts still get a 0–4 axis
  const raw = v / target, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  let step = (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
  if (integer) step = Math.max(1, Math.ceil(step));
  const n = Math.max(1, Math.ceil(v / step - 1e-9));
  return { max: step * n, step, n };
}
const niceMax = (v) => niceScale(v).max;

/* ── tiles ── */
function delta(cur, prev, upIsGood, fmt) {
  if (cur == null || prev == null) return `<div class="kpi-delta flat">no prior data</div>`;
  const d = cur - prev;
  if (Math.abs(d) < 1e-9) return `<div class="kpi-delta flat">— same as previous period</div>`;
  const cls = upIsGood == null ? "flat" : (d > 0) === upIsGood ? "good" : "bad";
  return `<div class="kpi-delta ${cls}">${d > 0 ? "▲" : "▼"} ${esc(fmt(Math.abs(d)))} vs previous ${statsDays === 1 ? "24h" : statsDays + " days"}</div>`;
}
function tile(label, value, note) {
  return `<div class="kpi"><div class="kpi-label">${esc(label)}</div><div class="kpi-value">${esc(value)}</div>${note || ""}</div>`;
}

/* ── chart scaffolding ── */
function chartCard(id, title, sub, legend, draw, table) {
  const card = $("#" + id);
  const asTable = tableMode.has(id);
  card.setAttribute("aria-label", title);
  card.innerHTML = `<div class="chart-head"><div><h3>${esc(title)}</h3><div class="chart-sub">${esc(sub)}</div></div>
    <button class="btn btn-sm" data-table="${id}">${asTable ? "Chart" : "Table"}</button></div>
    ${legend && legend.length > 1 && !asTable ? `<div class="legend">${legend.map((l) => `<span><span class="key${l.line ? " line" : ""}" style="background:${l.color}"></span>${esc(l.label)}</span>`).join("")}</div>` : ""}
    <div class="chart"></div><div class="viz-tip" hidden></div>`;
  const el = card.querySelector(".chart");
  if (asTable) el.innerHTML = `<div class="table-wrap">${table()}</div>`;
  else draw(el, card.querySelector(".viz-tip"), card);
}
function svgEl(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function svgText(parent, x, y, text, anchor, cls) {
  const t = svgEl("text", { x, y, "text-anchor": anchor || "start", class: cls || "" }, parent);
  t.textContent = text;
  return t;
}
function showTip(tip, card, x, y, header, rows) {
  tip.replaceChildren();
  const h = document.createElement("div"); h.className = "tip-h"; h.textContent = header; tip.appendChild(h);
  for (const r of rows) {
    const row = document.createElement("div"); row.className = "tip-row";
    const k = document.createElement("span"); k.className = "tip-key" + (r.box ? " box" : ""); k.style.background = r.color;
    const v = document.createElement("b"); v.textContent = r.value;
    const l = document.createElement("span"); l.textContent = r.label;
    row.append(k, v, l); tip.appendChild(row);
  }
  tip.hidden = false;
  const cw = card.clientWidth;
  tip.style.left = Math.max(8, Math.min(cw - tip.offsetWidth - 8, x + 12)) + "px";
  tip.style.top = Math.max(8, y - tip.offsetHeight - 8) + "px";
}
const roundTop = (x, y, w, h, r) => { r = Math.max(0, Math.min(r, h, w / 2)); return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`; };
const roundRight = (x, y, w, h, r) => { r = Math.max(0, Math.min(r, w, h / 2)); return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`; };

function axes(svg, W, padL, padR, padT, plotH, yMax, fmt, n = 4) {
  for (let i = 0; i <= n; i++) {
    const v = yMax * i / n, y = padT + plotH - plotH * i / n;
    svgEl("line", { x1: padL, x2: W - padR, y1: y, y2: y, class: i === 0 ? "base" : "grid" }, svg);
    svgText(svg, padL - 6, y + 4, fmt(v), "end");
  }
}
function xLabels(svg, buckets, x0, band, y) {
  const every = Math.max(1, Math.ceil(56 / band));
  buckets.forEach((b, i) => { if (i % every === 0) svgText(svg, x0 + i * band + band / 2, y, bucketLabel(b.t), "middle"); });
}

/* Stacked (or single) columns over time. series: [{key,label,color}] */
function columns(el, tip, card, { series, fmt, ref, tipExtra, integer }) {
  const data = statsData.series, W = Math.max(280, el.clientWidth), H = 220;
  const padL = 46, padR = 10, padT = 12, padB = 24, plotH = H - padT - padB;
  const totals = data.map((b) => series.reduce((s, x) => s + (b[x.key] || 0), 0));
  const scale = niceScale(Math.max(...totals, ref ? ref.value : 0), integer), yMax = scale.max;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img" });
  axes(svg, W, padL, padR, padT, plotH, yMax, fmt, scale.n);
  const band = (W - padL - padR) / data.length, bw = Math.min(24, band * 0.62);
  const groups = [];
  data.forEach((b, i) => {
    const g = svgEl("g", {}, svg), x = padL + i * band + (band - bw) / 2;
    let y = padT + plotH;
    const parts = series.filter((s) => (b[s.key] || 0) > 0);
    parts.forEach((s, j) => {
      const h = (b[s.key] / yMax) * plotH, gap = j > 0 ? 2 : 0, top = j === parts.length - 1;
      const hh = Math.max(0, h - gap);
      svgEl("path", { d: top ? roundTop(x, y - h, bw, hh, 4) : `M${x},${y - h}h${bw}v${hh}h${-bw}Z`, fill: s.color, class: "mark" }, g);
      y -= h;
    });
    const hit = svgEl("rect", { x: padL + i * band, y: padT, width: band, height: plotH, class: "hit", tabindex: 0 }, g);
    const show = (ev) => {
      svg.classList.add("dim"); groups.forEach((gg) => gg.classList.remove("hot")); g.classList.add("hot");
      const r = card.getBoundingClientRect(), hr = hit.getBoundingClientRect();
      showTip(tip, card, (ev && ev.clientX ? ev.clientX : hr.left + hr.width / 2) - r.left, hr.top - r.top + 40, bucketLabel(b.t, true),
        [...series.map((s) => ({ color: s.color, box: true, value: fmt(b[s.key] || 0), label: s.label })), ...(tipExtra ? tipExtra(b) : [])]);
    };
    hit.addEventListener("pointermove", show); hit.addEventListener("focus", show);
    groups.push(g);
  });
  if (ref && ref.value > 0 && ref.value <= yMax) {
    const y = padT + plotH - (ref.value / yMax) * plotH;
    svgEl("line", { x1: padL, x2: W - padR, y1: y, y2: y, class: "ref" }, svg);
    svgText(svg, W - padR, y - 4, ref.label, "end");
  }
  xLabels(svg, data, padL, band, H - 6);
  svg.addEventListener("pointerleave", () => { svg.classList.remove("dim"); groups.forEach((g) => g.classList.remove("hot")); tip.hidden = true; });
  el.replaceChildren(svg);
}

/* Lines over time on one 0–100% axis, with a crosshair. */
function lines(el, tip, card, { series, fmt }) {
  const data = statsData.series, W = Math.max(280, el.clientWidth), H = 220;
  const padL = 46, padR = 54, padT = 12, padB = 24, plotH = H - padT - padB, yMax = 1;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img" });
  axes(svg, W, padL, padR, padT, plotH, yMax, fmt);
  const band = (W - padL - padR) / data.length, X = (i) => padL + i * band + band / 2, Y = (v) => padT + plotH - v / yMax * plotH;
  for (const s of series) {
    let d = "", pen = false, last = null;
    data.forEach((b, i) => { const v = b[s.key]; if (v == null) { pen = false; return; } d += (pen ? "L" : "M") + X(i) + "," + Y(v); pen = true; last = [i, v]; });
    if (d) svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    data.forEach((b, i) => { if (b[s.key] != null && (i === 0 || data[i - 1][s.key] == null) && (i === data.length - 1 || data[i + 1][s.key] == null))
      svgEl("circle", { cx: X(i), cy: Y(b[s.key]), r: 3, fill: s.color }, svg); });  // isolated points stay visible
    if (last) {
      svgEl("circle", { cx: X(last[0]), cy: Y(last[1]), r: 4, fill: s.color, stroke: "var(--viz-surface)", "stroke-width": 2 }, svg);
      s._end = { x: X(last[0]), y: Y(last[1]), v: last[1] };
    }
  }
  const ends = series.filter((s) => s._end).sort((a, b) => a._end.y - b._end.y);
  if (ends.length === 2 && Math.abs(ends[0]._end.y - ends[1]._end.y) < 14) ends.forEach((s) => (s._end.skip = true));  // converging: legend + tooltip carry it
  for (const s of ends) if (!s._end.skip) svgText(svg, s._end.x + 8, s._end.y + 4, fmt(s._end.v), "start", "val");
  const cross = svgEl("line", { y1: padT, y2: padT + plotH, class: "cross", visibility: "hidden" }, svg);
  const hit = svgEl("rect", { x: padL, y: padT, width: W - padL - padR, height: plotH, class: "hit" }, svg);
  hit.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect(), sx = (ev.clientX - r.left) * (W / r.width);
    const i = Math.max(0, Math.min(data.length - 1, Math.round((sx - padL - band / 2) / band)));
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("visibility", "visible");
    const cr = card.getBoundingClientRect();
    showTip(tip, card, ev.clientX - cr.left, r.top - cr.top + 40, bucketLabel(data[i].t, true),
      series.map((s) => ({ color: s.color, value: fmt(data[i][s.key]), label: s.label })));
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; });
  xLabels(svg, data, padL, band, H - 6);
  el.replaceChildren(svg);
}

/* Horizontal stacked bars, one per station. */
function hbars(el, tip, card, { rows, series, fmt }) {
  const W = Math.max(280, el.clientWidth), rowH = 26, padL = 92, padR = 54, padT = 4, H = padT + rows.length * rowH + 22;
  const scale = niceScale(Math.max(...rows.map((r) => series.reduce((s, x) => s + r[x.key], 0)), 0.01)), max = scale.max;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img" });
  const plotW = W - padL - padR;
  for (let i = 0; i <= scale.n; i++) {
    const x = padL + plotW * i / scale.n;
    svgEl("line", { x1: x, x2: x, y1: padT, y2: H - 20, class: i === 0 ? "base" : "grid" }, svg);
    if (W > 420 || i % 2 === 0 || i === scale.n) svgText(svg, x, H - 6, fmt(max * i / scale.n), "middle");
  }
  const groups = [];
  rows.forEach((r, i) => {
    const g = svgEl("g", {}, svg), y = padT + i * rowH + (rowH - 14) / 2;
    svgText(svg, padL - 8, y + 11, r.label, "end");
    let x = padL;
    const parts = series.filter((s) => r[s.key] > 0);
    parts.forEach((s, j) => {
      const w = r[s.key] / max * plotW, gap = j > 0 ? 2 : 0, last = j === parts.length - 1;
      svgEl("path", { d: last ? roundRight(x + gap, y, Math.max(0, w - gap), 14, 4) : `M${x + gap},${y}h${Math.max(0, w - gap)}v14h${-Math.max(0, w - gap)}Z`, fill: s.color, class: "mark" }, g);
      x += w;
    });
    const total = series.reduce((s, x2) => s + r[x2.key], 0);
    if (total > 0) svgText(g, x + 6, y + 11, fmt(total), "start", "val");
    const hit = svgEl("rect", { x: 0, y: padT + i * rowH, width: W, height: rowH, class: "hit", tabindex: 0 }, g);
    const show = (ev) => {
      svg.classList.add("dim"); groups.forEach((gg) => gg.classList.remove("hot")); g.classList.add("hot");
      const cr = card.getBoundingClientRect(), hr = hit.getBoundingClientRect();
      showTip(tip, card, (ev && ev.clientX ? ev.clientX : hr.left + 120) - cr.left, hr.top - cr.top, r.long,
        series.map((s) => ({ color: s.color, box: true, value: fmt(r[s.key]), label: s.label })));
    };
    hit.addEventListener("pointermove", show); hit.addEventListener("focus", show);
    groups.push(g);
  });
  svg.addEventListener("pointerleave", () => { svg.classList.remove("dim"); groups.forEach((g) => g.classList.remove("hot")); tip.hidden = true; });
  el.replaceChildren(svg);
}

function seriesTable(cols) {
  return `<table class="viz-table"><tr><th>${statsData.bucket === "hour" ? "Hour" : "Day"}</th>${cols.map((c) => `<th class="n">${esc(c.label)}</th>`).join("")}</tr>` +
    statsData.series.map((b) => `<tr><td>${esc(bucketLabel(b.t, true))}</td>${cols.map((c) => `<td class="n">${esc(c.fmt(b[c.key]))}</td>`).join("")}</tr>`).join("") + `</table>`;
}

function renderStats() {
  const d = statsData, k = d.kpis, p = d.previous;
  const sel = $("#stats-project"), cur = sel.value;
  sel.innerHTML = `<option value="">All projects</option>` + (S?.projects || []).map((pr) => `<option value="${esc(pr.path)}">${esc(pr.name)}</option>`).join("");
  sel.value = cur;
  const ot = S?.telemetry;
  $("#stats-otel").textContent = ot?.enabled
    ? `OpenTelemetry export on → ${ot.traces_url || ot.metrics_url} · ${ot.traces_sent} traces sent this session${ot.last_error ? " · last error: " + ot.last_error : ""}`
    : ot?.problem ? `OpenTelemetry export disabled: ${ot.problem}` : "OpenTelemetry export off (set OTEL_EXPORTER_OTLP_ENDPOINT)";

  const money2 = (v) => "$" + (v || 0).toFixed(2);
  $("#stats-kpis").innerHTML = [
    tile("Trains arrived", String(k.arrived), delta(k.arrived, p.arrived, true, (v) => String(v))),
    tile("Failure rate", pct0(k.failure_rate), delta(k.failure_rate, p.failure_rate, false, (v) => Math.round(v * 100) + " pts")),
    tile("Median journey", k.journey_median == null ? "—" : dur(k.journey_median), k.journey_p90 ? `<div class="kpi-note">p90 ${esc(dur(k.journey_p90))}</div>` : ""),
    tile("Spend", money2(k.spend), delta(k.spend, p.spend, null, money2)),
    tile("Fare per arrival", k.cost_per_arrival == null ? "—" : money2(k.cost_per_arrival), delta(k.cost_per_arrival, p.cost_per_arrival, false, money2)),
    tile("First-pass rate", pct0(k.first_pass), delta(k.first_pass, p.first_pass, true, (v) => Math.round(v * 100) + " pts")),
    tile("Merged to trunk", `+${compact(k.lines_added)} / −${compact(k.lines_removed)}`, `<div class="kpi-note">${k.merges} merges · ${k.commits} commits</div>`),
  ].join("");

  chartCard("chart-spend", "Spend", statsData.bucket === "hour" ? "OpenRouter + Jev, per hour"
    : d.project ? "OpenRouter + Jev, per day (the daily budget is line-wide)" : "OpenRouter + Jev, per day vs the daily budget", null,
    (el, tip, card) => columns(el, tip, card, { series: [{ key: "spend", label: "spend", color: "var(--viz-1)" }], fmt: (v) => "$" + (v >= 10 ? v.toFixed(0) : v.toFixed(2)),
      ref: d.bucket === "day" && !d.project ? { value: d.daily_budget, label: `daily budget $${d.daily_budget}` } : null }),
    () => seriesTable([{ key: "spend", label: "Spend", fmt: (v) => "$" + v.toFixed(2) }]));

  const trainSeries = [{ key: "done", label: "arrived", color: "var(--viz-good)" }, { key: "failed", label: "failed", color: "var(--viz-critical)" }, { key: "rejected", label: "not in service", color: "var(--viz-neutral)" }];
  chartCard("chart-trains", "Trains finished", "Journeys that ended, by outcome", trainSeries.map((s) => ({ ...s })),
    (el, tip, card) => columns(el, tip, card, { series: trainSeries, fmt: (v) => String(Math.round(v)), integer: true }),
    () => seriesTable(trainSeries.map((s) => ({ key: s.key, label: s.label, fmt: (v) => String(v) }))));

  const stSeries = [{ key: "working_min", label: "working", color: "var(--viz-1)" }, { key: "waiting_min", label: "waiting (queues, gates, retries)", color: "var(--viz-neutral)" }];
  const stRows = d.stations.filter((r) => r.working_min + r.waiting_min > 0.001).map((r) => ({ ...r, label: `${r.code} ${r.name}`, long: `${r.code} ${r.name} — per arrived train` }));
  chartCard("chart-stations", "Where the time goes", "Average minutes per arrived train at each station", stSeries,
    (el, tip, card) => stRows.length ? hbars(el, tip, card, { rows: stRows, series: stSeries, fmt: minutes }) : (el.innerHTML = `<p class="empty">No arrivals in this range.</p>`),
    () => `<table class="viz-table"><tr><th>Station</th><th class="n">Working</th><th class="n">Waiting</th></tr>${stRows.map((r) => `<tr><td>${esc(r.label)}</td><td class="n">${minutes(r.working_min)}</td><td class="n">${minutes(r.waiting_min)}</td></tr>`).join("")}</table>`);

  const qSeries = [{ key: "first_pass_rate", label: "first-pass rate", color: "var(--viz-1)", line: true }, { key: "satisfaction", label: "holdout satisfaction", color: "var(--viz-2)", line: true }];
  chartCard("chart-quality", "Quality", "Share arriving without rework, and holdout scenarios passed", qSeries,
    (el, tip, card) => lines(el, tip, card, { series: qSeries, fmt: (v) => v == null ? "—" : Math.round(v * 100) + "%" }),
    () => seriesTable(qSeries.map((s) => ({ key: s.key, label: s.label, fmt: (v) => v == null ? "—" : Math.round(v * 100) + "%" }))));

  chartCard("chart-merges", "Merges", "Trains merged to trunk; lines changed in the tooltip", null,
    (el, tip, card) => columns(el, tip, card, { series: [{ key: "merges", label: "merges", color: "var(--viz-1)" }], fmt: (v) => String(Math.round(v)), integer: true,
      tipExtra: (b) => [{ color: "transparent", value: `+${b.lines_added} / −${b.lines_removed}`, label: "lines" }] }),
    () => seriesTable([{ key: "merges", label: "Merges", fmt: (v) => String(v) }, { key: "lines_added", label: "Lines added", fmt: (v) => String(v) }, { key: "lines_removed", label: "Lines removed", fmt: (v) => String(v) }]));

  const e = d.efficiency;
  $("#chart-efficiency").innerHTML = `<div class="chart-head"><div><h3>Efficiency</h3><div class="chart-sub">Cheap decisions and learning in this range</div></div></div>
    <div class="eff-list">
      ${tile("Prompt cache hit rate", pct0(k.cache_rate), `<div class="kpi-note">of input tokens on ${k.llm_runs} model runs</div>`)}
      ${tile("Jev decisions", String(k.jev_decisions), `<div class="kpi-note">$${k.jev_cost.toFixed(3)} total</div>`)}
      ${tile("LLM triage avoided", String(e.jev_fast_pass + e.jev_rejects + e.jev_holds), `<div class="kpi-note">${e.jev_fast_pass} fast-pass · ${e.jev_rejects} rejected · ${e.jev_holds} held by Jev</div>`)}
      ${tile("Playbook notes added", String(e.playbook_notes), `<div class="kpi-note">by retrospectives</div>`)}
      ${tile("Scenarios disputed", String(e.disputes), `<div class="kpi-note">self-contradictory holdouts</div>`)}
    </div>`;

  const ms = d.models;
  $("#stats-models").innerHTML = `<h3>Models</h3><div class="chart-sub" style="margin:-6px 0 10px">Every model and tool that did work in this range, by spend</div>
    <div class="table-wrap"><table class="viz-table"><tr><th>Model</th><th>Used by</th><th class="n">Runs</th><th class="n">Tokens in</th><th class="n">Tokens out</th><th class="n">Cached</th><th class="n">Cost</th><th>Share of spend</th><th class="n">Avg / run</th><th class="n">Avg time</th><th class="n">Errors</th></tr>
    ${ms.map((m) => `<tr><td>${modelId(m.model)}</td><td>${esc(m.roles.join(", "))}</td><td class="n">${m.runs}</td><td class="n">${compact(m.tokens_in)}</td><td class="n">${compact(m.tokens_out)}</td>
      <td class="n">${pct0(m.cache_rate)}</td><td class="n">$${m.cost.toFixed(2)}</td><td><div class="meter" title="${Math.round(m.share * 100)}%"><div style="width:${(m.share * 100).toFixed(1)}%"></div></div></td>
      <td class="n">$${m.avg_cost.toFixed(3)}</td><td class="n">${dur(m.avg_seconds)}</td><td class="n">${m.errors}</td></tr>`).join("") || `<tr><td colspan="11" class="muted">No model runs in this range.</td></tr>`}
    </table></div>`;
}
