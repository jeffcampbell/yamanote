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
  if (S.paused) { pill.className = "pill pill-warn"; pill.textContent = "■ Paused · 運転見合わせ"; }
  else if (S.suspended) { pill.className = "pill pill-bad"; pill.textContent = "■ Suspended"; }
  else { pill.className = "pill pill-ok"; pill.textContent = "● In service · 運転中"; }
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
  let html = `<table><tr><th>Class</th><th>Model</th><th class="num">24h</th></tr>` + classes.map(([k, c]) => {
    const u = by[c.model];
    const cache = u && u.tin ? ` · ${Math.round(100 * (u.cached || 0) / u.tin)}% cached` : "";
    return `<tr><td>${svcBadge(k)}</td><td>${modelId(c.model)}</td><td class="num">${u ? money(u.cost) + `<div class="small muted">${u.runs} runs${cache}</div>` : "—"}</td></tr>`;
  }).join("") + `</table>`;
  const R = S.routing;
  html += `<h3 style="margin-top:14px">Difficulty → class <span class="badge ${R.adaptive ? "badge-on" : ""}">${R.adaptive ? "ADAPTIVE" : "FIXED"}</span></h3>
    <table class="levels"><tr><th>Jev says</th><th>Runs as</th><th class="num">First-pass</th><th class="num">Avg fare</th></tr>` +
    R.levels.map((l) => {
      const h = l.history.find((x) => x.cls === l.class) || l.history[0];
      const votes = h && h.retros ? `<div class="why">${h.underpowered ? `${h.underpowered}× too weak ` : ""}${h.overpowered ? `${h.overpowered}× overkill` : ""}</div>` : "";
      const rate = h ? `${Math.round(100 * h.first_pass / h.n)}% <span class="small muted">of ${h.n}</span>${votes}` : "—";
      return `<tr><td>${esc(l.difficulty)}</td><td>${svcBadge(l.class)}${l.class !== l.default ? `<div class="why">${esc(l.why)}</div>` : ""}</td>
        <td class="num">${rate}</td><td class="num">${h ? money(h.avg_cost) : "—"}</td></tr>`;
    }).join("") + `</table>`;
  const labels = { builder: "item's class", "builder+1": "one class up" };
  html += `<div class="stations-map">` + Object.entries(S.station_models).filter(([st]) => !st.endsWith("_retry")).map(([st, choice]) =>
    `<span class="chip"><b>${esc(st)}</b> → ${esc(labels[choice] || S.service_classes[choice]?.label || choice)}</span>`).join("") + `</div>`;
  html += `<p class="small muted" style="margin-top:10px">First-pass = arrived without rework. ${R.adaptive ? `With ${R.min_samples}+ samples, a class that passes first time under 50% is bumped up; one above 90% tries a class cheaper.` : "Set AGENT_TEAM_ADAPTIVE_ROUTING=1 to let this history move routing."} Rework escalates a train one class. Fallback: <code>openrouter/auto</code>.</p>`;
  $("#routing").innerHTML = html;
}

/* ── retrospectives & playbook ───────────────────────── */
const FIT = { underpowered: ["fit-under", "class too weak"], overpowered: ["fit-over", "class overkill"] };
function renderLearning() {
  if (!S.retro_enabled) {
    $("#learning").innerHTML = `<p class="empty">Retrospectives are off (AGENT_TEAM_RETRO=0).</p>`;
    return;
  }
  let html = S.retros.length ? S.retros.map((r) => {
    const [cls, label] = r.outcome === "done" ? ["badge-on", "ARRIVED"] : ["badge-warn", "FAILED"];
    const fit = FIT[r.class_fit];
    const delta = [r.added ? `+${r.added} note${r.added > 1 ? "s" : ""}` : "", r.retired ? `−${r.retired} retired` : ""].filter(Boolean).join(" · ");
    return `<div class="retro-row" data-train="${r.item_id}"><span class="badge ${cls}">${label}</span>
      <span class="retro-title">#${r.item_id} ${esc(r.title || "")}</span>
      <span class="retro-sum">${esc(r.summary || "(no summary)")}${fit ? ` <span class="${fit[0]}">· ${fit[1]}</span>` : ""}${delta ? ` · <b>${delta}</b>` : ""}</span></div>`;
  }).join("") : `<p class="empty">Every train ends its journey at JY09 Retro. Retrospectives will appear here.</p>`;
  for (const p of S.projects) {
    const notes = S.playbook[p.path] || [];
    html += `<h3 style="margin-top:14px">${esc(p.name)} playbook <span class="badge">${notes.length}</span></h3>`;
    if (!notes.length) { html += `<p class="empty">No notes yet. Retrospectives add them when a train hits a problem the next one could avoid.</p>`; continue; }
    const by = {};
    for (const n of notes) (by[n.station] ||= []).push(n);
    for (const [station, list] of Object.entries(by)) {
      html += `<div class="pb-station">${esc(S.stations.find((x) => x.key === station)?.code || "")} ${esc(station)}</div>`;
      html += list.map((n) => `<div class="lesson"><span>${esc(n.text)} <span class="note-stats">${n.uses ? `${n.wins}/${n.uses} first-pass` : "unused yet"} · <a class="muted" data-id="${n.item_id}">#${n.item_id}</a></span></span><button title="Remove this note" data-note="${n.id}" data-project="${esc(p.path)}">×</button></div>`).join("");
    }
  }
  $("#learning").innerHTML = html;
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
    <div class="d-meta">${svcBadge(it.service_class)}
      ${it.difficulty ? `<span class="chip">difficulty: ${esc(it.difficulty)}${it.difficulty_score != null ? " (" + Number(it.difficulty_score).toFixed(1) + "/4)" : ""}</span>` : ""}
      ${it.train ? `<span class="chip">train set ${esc(it.train)}</span>` : ""}
      ${it.branch ? `<span class="chip" title="git branch"><code>${esc(it.branch)}</code></span>` : ""}
      ${it.parent_id ? `<a class="chip" data-train="${it.parent_id}" href="#item-${it.parent_id}">regression from #${it.parent_id}</a>` : ""}</div>`;
  html += routeStrip(it, events);
  if (it.status === "waiting") {
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
    const row = e.target.closest(".board-row.item, #feed a[data-id]");
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
  window.addEventListener("resize", () => renderDiagram());
  const m = location.hash.match(/^#item-(\d+)$/);
  if (m) openItem(Number(m[1]));
}
boot();
