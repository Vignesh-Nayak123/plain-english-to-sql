// SQLForge demo — frontend logic
const $ = (id) => document.getElementById(id);
const dbSelect = $("dbSelect"), questionEl = $("question"), examplesEl = $("examples"),
      schemaView = $("schemaView"), runBtn = $("runBtn"),
      placeholder = $("placeholder"), result = $("result"),
      sqlOut = $("sqlOut"), badges = $("badges"),
      traceBlock = $("traceBlock"), traceList = $("traceList"),
      resultsBlock = $("resultsBlock"), resultsHead = $("resultsHead"),
      resultsTable = $("resultsTable"), errorBlock = $("errorBlock"), errorOut = $("errorOut");

let EXAMPLES = [], modelReady = false;

// ---------- model status ----------
async function pollHealth() {
  const el = $("status"), label = $("statusLabel");
  try {
    const h = await (await fetch("/api/health")).json();
    if (h.status === "online") { el.dataset.state = "online"; label.textContent = "model online"; modelReady = true; }
    else if (h.status === "loading") { el.dataset.state = "loading"; label.textContent = "loading model…"; modelReady = false; }
    else { el.dataset.state = "error"; label.textContent = "model error"; modelReady = false; }
  } catch { el.dataset.state = "offline"; label.textContent = "offline"; modelReady = false; }
  updateRunBtn();
}
function updateRunBtn() {
  runBtn.disabled = !modelReady || runBtn.classList.contains("loading");
}

// ---------- examples + schema ----------
async function loadExamples() {
  EXAMPLES = await (await fetch("/api/examples")).json();
  dbSelect.innerHTML = EXAMPLES.map(e => `<option value="${e.db_id}">${e.label}</option>`).join("");
  if (EXAMPLES.length) { await onDbChange(); }
}
async function onDbChange() {
  const ex = EXAMPLES.find(e => e.db_id === dbSelect.value);
  examplesEl.innerHTML = (ex?.questions || [])
    .map(q => `<span class="chip" title="${esc(q)}">${esc(q)}</span>`).join("");
  examplesEl.querySelectorAll(".chip").forEach((c, i) => {
    c.onclick = () => { questionEl.value = ex.questions[i]; questionEl.focus(); };
  });
  try {
    const s = await (await fetch(`/api/schema?db_id=${encodeURIComponent(dbSelect.value)}`)).json();
    schemaView.textContent = s.schema;
  } catch { schemaView.textContent = "—"; }
}

// ---------- generate ----------
async function run() {
  const question = questionEl.value.trim();
  if (!question || !modelReady) return;
  runBtn.classList.add("loading"); updateRunBtn();
  runBtn.querySelector(".btn-label").textContent = "Generating…";
  try {
    const resp = await fetch("/api/generate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, db_id: dbSelect.value, self_correct: true }),
    });
    if (!resp.ok) { const e = await resp.json().catch(() => ({})); throw new Error(e.detail || resp.statusText); }
    render(await resp.json());
  } catch (err) {
    placeholder.classList.add("hidden"); result.classList.remove("hidden");
    showError("Request failed", err.message);
  } finally {
    runBtn.classList.remove("loading"); updateRunBtn();
    runBtn.querySelector(".btn-label").textContent = "Generate SQL";
  }
}

function render(r) {
  placeholder.classList.add("hidden"); result.classList.remove("hidden");
  sqlOut.innerHTML = highlight(r.sql);

  // badges
  const b = [];
  b.push(`<span class="badge">${r.elapsed_s}s</span>`);
  if (r.self_corrected) b.push(`<span class="badge fix">🛠️ self-corrected (${r.attempts} tries)</span>`);
  if (r.executed && !r.error) b.push(`<span class="badge good">✓ ran · ${r.row_count} row${r.row_count === 1 ? "" : "s"}</span>`);
  if (r.executed && r.error) b.push(`<span class="badge warn">✕ execution error</span>`);
  badges.innerHTML = b.join("");

  // self-correction trace (only show if there was more than one attempt)
  if (r.trace && r.trace.length > 1) {
    traceList.innerHTML = r.trace.map(step => {
      const ok = !step.error;
      return `<div class="trace-step">
        <div class="tlabel">Attempt ${step.attempt}
          <span class="tag ${ok ? "ok" : "fail"}">${ok ? "ran clean" : "crashed"}</span></div>
        <pre class="code sql mono">${highlight(step.sql)}</pre>
        ${step.error ? `<div class="terr">↳ ${esc(step.error)}</div>` : ""}
      </div>`;
    }).join("");
    traceBlock.classList.remove("hidden");
  } else { traceBlock.classList.add("hidden"); }

  // results / error
  errorBlock.classList.add("hidden"); resultsBlock.classList.add("hidden");
  if (r.executed && r.error) { showError("Execution error", r.error); }
  else if (r.executed) { renderTable(r.columns, r.rows, r.row_count); }
}

function renderTable(cols, rows, n) {
  if (!cols || !cols.length) { resultsBlock.classList.add("hidden"); return; }
  resultsHead.textContent = `Query results — ${n} row${n === 1 ? "" : "s"}`;
  const head = `<thead><tr>${cols.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead>`;
  const body = `<tbody>${rows.map(row =>
    `<tr>${row.map(c => `<td>${esc(c === null ? "NULL" : String(c))}</td>`).join("")}</tr>`).join("")}</tbody>`;
  resultsTable.innerHTML = head + (rows.length ? body : `<tbody><tr><td colspan="${cols.length}" class="muted">(no rows returned)</td></tr></tbody>`);
  resultsBlock.classList.remove("hidden");
}

function showError(title, msg) {
  document.querySelector("#errorBlock .block-head span").textContent = title;
  errorOut.textContent = msg; errorBlock.classList.remove("hidden");
}

// ---------- tiny SQL highlighter ----------
const KW = /\b(SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|ON|AS|AND|OR|NOT|IN|IS|NULL|GROUP|BY|ORDER|HAVING|LIMIT|OFFSET|DISTINCT|UNION|ALL|INTERSECT|EXCEPT|INSERT|INTO|VALUES|UPDATE|SET|DELETE|CREATE|TABLE|LIKE|BETWEEN|ASC|DESC|CASE|WHEN|THEN|ELSE|END|EXISTS)\b/gi;
const FN = /\b(COUNT|SUM|AVG|MIN|MAX|ROUND|ABS|LENGTH|LOWER|UPPER|SUBSTR|COALESCE|CAST)\b/gi;
function highlight(sql) {
  let h = esc(sql || "");
  h = h.replace(/'([^']*)'/g, '<span class="str">\'$1\'</span>');
  h = h.replace(/\b(\d+(\.\d+)?)\b/g, '<span class="num">$1</span>');
  h = h.replace(FN, (m) => `<span class="fn">${m}</span>`);
  h = h.replace(KW, (m) => `<span class="kw">${m}</span>`);
  return h;
}
function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

// ---------- wire up ----------
dbSelect.onchange = onDbChange;
runBtn.onclick = run;
questionEl.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); });
$("copyBtn").onclick = () => {
  navigator.clipboard.writeText(sqlOut.textContent).then(() => {
    const b = $("copyBtn"); b.textContent = "copied!"; setTimeout(() => b.textContent = "copy", 1200);
  });
};

loadExamples();
pollHealth();
setInterval(pollHealth, 4000);
