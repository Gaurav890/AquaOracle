(function () {
  let summaryScope = "mine";
  let logsScope = "mine";
  let logsOffset = 0;
  const LOGS_LIMIT = 25;
  let lastLogsCount = 0;

  const summaryTbody = document.getElementById("summary-tbody");
  const logsTbody = document.getElementById("logs-tbody");
  const summaryToggle = document.getElementById("summary-scope-toggle");
  const logsToggle = document.getElementById("logs-scope-toggle");
  const logsPrevBtn = document.getElementById("logs-prev");
  const logsNextBtn = document.getElementById("logs-next");
  const logsPageInfo = document.getElementById("logs-page-info");
  const detailModal = document.getElementById("log-detail-modal");
  const detailBody = document.getElementById("log-detail-body");

  function esc(value) {
    if (value === null || value === undefined) return "";
    const div = document.createElement("div");
    div.textContent = String(value);
    return div.innerHTML;
  }

  function fmtMs(ms) {
    if (ms === null || ms === undefined) return "—";
    return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
  }

  function fmtNum(n) {
    return n === null || n === undefined ? "—" : Math.round(n);
  }

  function fmtCost(usd) {
    if (usd === null || usd === undefined) return "—";
    return usd === 0 ? "$0" : `$${usd.toFixed(4)}`;
  }

  function fmtPct(fraction) {
    return fraction === null || fraction === undefined ? "—" : `${Math.round(fraction * 100)}%`;
  }

  function fmtDate(iso) {
    const d = new Date(iso);
    return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  }

  function groundednessBadge(rate) {
    if (rate === null || rate === undefined) return '<span class="eval-badge">—</span>';
    const cls = rate === 1 ? "eval-badge-pass" : "eval-badge-flag";
    return `<span class="eval-badge ${cls}">${fmtPct(rate)}</span>`;
  }

  function setupScopeToggle(toggleEl, onChange) {
    if (!toggleEl) return;
    toggleEl.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        toggleEl.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        onChange(btn.dataset.scope);
      });
    });
  }

  async function loadSummary() {
    let rows;
    try {
      rows = await API.get(`/api/eval/summary?scope=${summaryScope}`);
    } catch (e) {
      summaryTbody.innerHTML = `<tr><td colspan="10" class="eval-empty">Couldn't load summary: ${e.message}</td></tr>`;
      return;
    }
    if (rows.length === 0) {
      summaryTbody.innerHTML = '<tr><td colspan="10" class="eval-empty">No queries yet.</td></tr>';
      return;
    }
    summaryTbody.innerHTML = rows
      .map(
        (r) => `
      <tr>
        <td>${esc(r.provider.toUpperCase())}</td>
        <td>${esc(r.model || "default")}</td>
        <td>${r.count}</td>
        <td>${fmtNum(r.avg_prompt_tokens)}</td>
        <td>${fmtNum(r.avg_completion_tokens)}</td>
        <td>${fmtMs(r.avg_total_ms)}</td>
        <td>${r.total_estimated_cost_usd === null ? "—" : fmtCost(r.total_estimated_cost_usd)}</td>
        <td>${groundednessBadge(r.groundedness_pass_rate)}</td>
        <td>&#128077;${r.thumbs_up} &#128078;${r.thumbs_down} (${r.no_feedback} unrated)</td>
        <td>${r.containment_events || "—"}</td>
      </tr>`
      )
      .join("");
  }

  async function loadLogs() {
    let rows;
    try {
      rows = await API.get(`/api/eval/logs?scope=${logsScope}&limit=${LOGS_LIMIT}&offset=${logsOffset}`);
    } catch (e) {
      logsTbody.innerHTML = `<tr><td colspan="8" class="eval-empty">Couldn't load logs: ${e.message}</td></tr>`;
      return;
    }
    lastLogsCount = rows.length;
    logsPageInfo.textContent = `Rows ${logsOffset + 1}–${logsOffset + rows.length}`;
    logsPrevBtn.disabled = logsOffset === 0;
    logsNextBtn.disabled = rows.length < LOGS_LIMIT;

    if (rows.length === 0) {
      logsTbody.innerHTML = '<tr><td colspan="8" class="eval-empty">No queries yet.</td></tr>';
      return;
    }

    logsTbody.innerHTML = "";
    for (const log of rows) {
      const tr = document.createElement("tr");
      tr.className = "eval-row-clickable";
      const feedback = log.user_feedback === "up" ? "\u{1F44D}" : log.user_feedback === "down" ? "\u{1F44E}" : "—";
      const ungrounded = log.verification.ungrounded_citation_indices;
      const groundedRate = ungrounded === null || ungrounded === undefined ? null : ungrounded.length === 0 ? 1 : 0;
      tr.innerHTML = `
        <td>${fmtDate(log.created_at)}</td>
        <td class="eval-cell-text">${esc(log.chat_title || "(untitled)")}</td>
        <td class="eval-cell-text">${esc(log.question)}</td>
        <td>${esc(log.provider.toUpperCase())}${log.model ? " / " + esc(log.model) : ""}</td>
        <td>${fmtMs(log.budget.total_ms)}</td>
        <td>${groundednessBadge(groundedRate)}</td>
        <td>${log.containment ? `<span class="eval-badge eval-badge-flag">${esc(log.containment.event)}</span>` : "—"}</td>
        <td>${feedback}</td>
      `;
      tr.addEventListener("click", () => openDetail(log));
      logsTbody.appendChild(tr);
    }
  }

  function detailGrid(obj) {
    return `<dl class="eval-detail-grid">${Object.entries(obj)
      .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v === null || v === undefined ? "—" : esc(JSON.stringify(v))}</dd>`)
      .join("")}</dl>`;
  }

  function openDetail(log) {
    detailBody.innerHTML = `
      <div class="eval-detail-section">
        <div class="eval-detail-label">Question</div>
        <div>${esc(log.question)}</div>
      </div>
      <div class="eval-detail-section">
        <div class="eval-detail-label">Scope — what did retrieval touch</div>
        ${detailGrid(log.scope)}
      </div>
      <div class="eval-detail-section">
        <div class="eval-detail-label">Authority — did anything leave the machine / get persisted</div>
        ${detailGrid(log.authority)}
      </div>
      <div class="eval-detail-section">
        <div class="eval-detail-label">Budget — tokens, latency, cost</div>
        ${detailGrid(log.budget)}
      </div>
      <div class="eval-detail-section">
        <div class="eval-detail-label">Verification — how do we know this was grounded</div>
        ${detailGrid(log.verification)}
        <div style="margin-top: var(--space-2)">User feedback: ${esc(log.user_feedback || "not rated")}${log.user_feedback_comment ? ` — "${esc(log.user_feedback_comment)}"` : ""}</div>
      </div>
      <div class="eval-detail-section">
        <div class="eval-detail-label">Containment — what happened if something deviated</div>
        ${log.containment ? detailGrid(log.containment) : "<div>Nothing deviated — normal completion.</div>"}
      </div>
    `;
    detailModal.classList.add("open");
  }

  document.getElementById("log-detail-close").addEventListener("click", () => detailModal.classList.remove("open"));
  detailModal.addEventListener("click", (e) => {
    if (e.target === detailModal) detailModal.classList.remove("open");
  });

  logsPrevBtn.addEventListener("click", () => {
    logsOffset = Math.max(0, logsOffset - LOGS_LIMIT);
    loadLogs();
  });
  logsNextBtn.addEventListener("click", () => {
    if (lastLogsCount < LOGS_LIMIT) return;
    logsOffset += LOGS_LIMIT;
    loadLogs();
  });

  setupScopeToggle(summaryToggle, (scope) => {
    summaryScope = scope;
    loadSummary();
  });
  setupScopeToggle(logsToggle, (scope) => {
    logsScope = scope;
    logsOffset = 0;
    loadLogs();
  });

  (async function init() {
    let user;
    try {
      user = await API.get("/api/auth/me");
    } catch (e) {
      window.location.href = "/auth.html";
      return;
    }
    document.getElementById("user-email").textContent = user.email;
    document.getElementById("logout-btn").addEventListener("click", async () => {
      await API.post("/api/auth/logout").catch(() => {});
      window.location.href = "/auth.html";
    });

    if (user.is_admin) {
      summaryToggle.style.display = "flex";
      logsToggle.style.display = "flex";
    }

    loadSummary();
    loadLogs();
  })();
})();
