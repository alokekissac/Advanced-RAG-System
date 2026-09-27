"use strict";

const $ = (id) => document.getElementById(id);
const state = { kb: "demo", mode: "hybrid", k: 5, myDocs: [], busy: false, lastSources: [] };
const STOP = new Set("a an and are as at be by can do does for from has have how i if in is it of on or the to was what when where which who why will with you your".split(" "));

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = j.detail;
    throw new Error(Array.isArray(d) ? d.map((x) => x.msg).join("; ") : d || `HTTP ${r.status}`);
  }
  return j;
}

// ------------------------------------------------------------------ boot
async function boot() {
  try {
    const h = await api("/api/health");
    $("genChip").textContent = h.generator === "gemini" ? "LLM: Gemini" : "Answers: extractive (no LLM key)";
    $("genChip").className = "chip " + (h.generator === "gemini" ? "ok" : "info");
    $("embChip").textContent = `Embeddings: ${h.embeddings} · ${h.chunks} chunks`;
  } catch { $("genChip").textContent = "API offline"; }

  try {
    const docs = await api("/api/documents");
    $("docList").innerHTML = docs.map((d) => `
      <li><button data-doc="${esc(d.doc_id)}">
        <span class="t">${esc(d.title)}</span>
        ${d.doc_id.includes("vendor") ? '<span class="flag">INJECTION</span>' : ""}
        <span class="m">${d.words} words · ${d.sections.length} sections</span>
      </button></li>`).join("");
  } catch (e) { $("docList").innerHTML = `<li class="hint">${esc(e.message)}</li>`; }
}

// ------------------------------------------------------------------ UI wiring
document.querySelectorAll("[data-kb]").forEach((b) => b.addEventListener("click", () => {
  state.kb = b.dataset.kb;
  document.querySelectorAll("[data-kb]").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-selected", x === b); });
  $("kbDemo").hidden = state.kb !== "demo";
  $("kbMine").hidden = state.kb !== "mine";
  $("q").placeholder = state.kb === "mine" ? "Ask a question about your documents…" : "Ask a question about the handbook…";
}));
$("mode").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  state.mode = b.dataset.mode;
  $("mode").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
});
$("k").addEventListener("input", (e) => { state.k = +e.target.value; $("kOut").textContent = e.target.value; });
$("suggest").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  if (state.kb !== "demo") document.querySelector('[data-kb="demo"]').click();
  ask(b.childNodes[0].textContent.trim());
});
$("form").addEventListener("submit", (e) => { e.preventDefault(); ask($("q").value); });
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask($("q").value); } });
$("q").addEventListener("input", (e) => { e.target.style.height = "auto"; e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px"; });

$("docList").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-doc]"); if (!b) return;
  const d = await api(`/api/documents/${encodeURIComponent(b.dataset.doc)}`);
  $("vTitle").textContent = d.title; $("vBody").textContent = d.text; $("viewer").showModal();
});
$("vClose").addEventListener("click", () => $("viewer").close());
$("viewer").addEventListener("click", (e) => { if (e.target === $("viewer")) $("viewer").close(); });

// ------------------------------------------------------------------ uploads
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
$("file").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });

async function addFiles(files) {
  for (const f of [...files]) {
    if (state.myDocs.length >= 5) { alertBot("You can add up to 5 documents."); break; }
    const fd = new FormData(); fd.append("file", f);
    const li = document.createElement("li");
    li.innerHTML = `<div class="mine"><span class="t">${esc(f.name)}</span><span class="m">Reading…</span></div>`;
    $("myDocs").appendChild(li);
    try {
      const d = await api("/api/extract", { method: "POST", body: fd });
      state.myDocs.push({ name: d.name, text: d.text });
      li.remove();
    } catch (err) {
      li.querySelector(".m").textContent = err.message;
      setTimeout(() => li.remove(), 4000);
    }
    renderMyDocs();
  }
}
function renderMyDocs() {
  $("myDocs").innerHTML = state.myDocs.map((d, i) => `
    <li><div class="mine"><span class="t">${esc(d.name)}</span>
      <button class="x" data-rm="${i}" aria-label="Remove ${esc(d.name)}">✕</button>
      <span class="m">${d.text.split(/\s+/).length.toLocaleString()} words</span></div></li>`).join("");
}
$("myDocs").addEventListener("click", (e) => {
  const b = e.target.closest("[data-rm]"); if (!b) return;
  state.myDocs.splice(+b.dataset.rm, 1); renderMyDocs();
});

// ------------------------------------------------------------------ asking
function addMsg(role, html) {
  $("empty")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="bubble">${html}</div>`;
  $("thread").appendChild(el);
  el.scrollIntoView({ behavior: "smooth", block: "end" });
  return el;
}
function alertBot(text) { addMsg("bot", `<div class="answer">${esc(text)}</div>`); }

async function ask(question) {
  question = question.trim();
  if (!question || state.busy) return;
  if (state.kb === "mine" && !state.myDocs.length) { alertBot("Add at least one document first, or switch back to the demo handbook."); return; }
  state.busy = true; $("send").disabled = true; $("q").value = ""; $("q").style.height = "auto";
  addMsg("user", esc(question));
  const pending = addMsg("bot", '<span class="typing"><i></i><i></i><i></i></span>');
  try {
    const body = { question, k: state.k, mode: state.mode };
    if (state.kb === "mine") body.documents = state.myDocs;
    const a = await api("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    pending.querySelector(".bubble").innerHTML = renderAnswer(a);
    renderEvidence(a);
  } catch (e) {
    pending.querySelector(".bubble").innerHTML = `<div class="answer">Something went wrong: ${esc(e.message)}</div>`;
  } finally {
    state.busy = false; $("send").disabled = false; $("q").focus();
  }
}

function renderAnswer(a) {
  const text = esc(a.answer).replace(/\[(\d+)\]/g, (_, n) => `<button class="cite" data-cite="${n}" title="Show source ${n}">${n}</button>`);
  const pct = Math.round(a.confidence * 100);
  const color = a.confidence >= 0.8 ? "var(--green)" : a.confidence >= 0.65 ? "var(--accent)" : "var(--violet)";
  const v = a.verification || {};
  const genLabel = { gemini: "Gemini", extractive: "Extractive", guardrail: "Abstained" }[a.generator] || a.generator;
  let meta = `<span class="badge ${esc(a.abstained ? "guardrail" : a.generator)}">${esc(genLabel)}</span>
    <span class="conf" title="Evidence coverage of the question">Confidence <i><b style="width:${pct}%;background:${color}"></b></i> ${pct}%</span>`;
  if (v.checked) meta += `<span class="badge ${v.supported === v.checked ? "ok" : "bad"}">${v.supported}/${v.checked} citations verified</span>`;
  if (a.timings_ms?.total != null) meta += `<span>${Math.round(a.timings_ms.total)} ms</span>`;
  const warn = (a.warnings || []).map((w) => `<div class="warn">⚠ ${esc(w)}</div>`).join("");
  const why = a.abstained ? `<div class="muted small" style="margin-top:8px">The retrieved passages don't cover this question well enough (confidence below the threshold), so the system declines rather than guessing.</div>` : "";
  return `<div class="answer ${a.abstained ? "abstain" : ""}">${text}</div>${why}${warn}<div class="meta">${meta}</div>`;
}

function highlight(text, question) {
  const terms = question.toLowerCase().match(/[a-z0-9]+/g)?.filter((t) => t.length > 2 && !STOP.has(t)) || [];
  let html = esc(text);
  if (!terms.length) return html;
  const stems = [...new Set(terms.map((t) => t.replace(/(ies|es|s|ing|ed)$/, "")))].filter((t) => t.length > 2);
  const re = new RegExp(`\\b(${stems.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})[a-z]*`, "gi");
  return html.replace(re, "<mark>$&</mark>");
}

function renderEvidence(a) {
  state.lastSources = a.sources;
  const t = a.timings_ms || {};
  const steps = [];
  if (a.queries?.length > 1) steps.push(["Rewrite", t.rewrite]);
  steps.push(["Retrieve", t.retrieve], ["Guardrails", null]);
  if (!a.abstained || a.generator !== "guardrail") steps.push(["Generate", t.generate]);
  steps.push(["Verify", null]);
  $("pipeline").hidden = false;
  $("pipeline").innerHTML = steps.map(([n, ms]) => `<span class="step">${n}${ms != null ? `<b>${Math.round(ms)}ms</b>` : ""}</span>`).join('<span class="muted">→</span>');
  $("evMeta").textContent = `${a.sources.length} passages`;
  if (!a.sources.length) { $("evList").innerHTML = '<p class="hint">No passages retrieved.</p>'; return; }
  $("evList").innerHTML = a.sources.map((s, i) => {
    const bars = [["BM25", Math.min(s.bm25 / 12, 1), s.bm25.toFixed(2), "#f5a524"],
                  ["Dense", Math.max(s.dense, 0), s.dense.toFixed(2), "#a78bfa"],
                  ["Coverage", s.coverage, Math.round(s.coverage * 100) + "%", "#22d38a"]];
    return `<div class="src" id="src${i + 1}">
      <div class="src-head"><span class="num">${i + 1}</span>
        <div class="src-title">${esc(s.title)}${s.section ? ` <span>› ${esc(s.section)}</span>` : ""}</div></div>
      <div class="src-text">${highlight(s.text, a.question)}</div>
      ${s.text.length > 420 ? '<button class="more">Show more</button>' : ""}
      <div class="bars">${bars.map(([n, w, v, c]) => `<span>${n}</span><i><b style="width:${Math.round(w * 100)}%;background:${c}"></b></i><code>${v}</code>`).join("")}</div>
    </div>`;
  }).join("");
}

document.addEventListener("click", (e) => {
  const c = e.target.closest("[data-cite]");
  if (c) {
    const el = $("src" + c.dataset.cite);
    if (el) { el.scrollIntoView({ behavior: "smooth", block: "center" }); el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1400); }
  }
  const m = e.target.closest(".more");
  if (m) { const t = m.previousElementSibling; t.classList.toggle("open"); m.textContent = t.classList.contains("open") ? "Show less" : "Show more"; }
});

boot();
