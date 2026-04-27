(function () {
  const config = window.__DOCUMENT_STUDIO__ || {};
  const initialStatus = config.initialStatus || {};
  const defaultQaStrategy = config.initialStatus?.semantic_ready
    ? (config.defaultQaStrategy || "semantic")
    : "rule_based";

  const state = {
    documents: [],
    selectedDocumentId: "",
    checkedDocumentIds: [],
    starterDocumentId: "",
    chatHistory: [],
    summaryCache: {},
    summaryInflight: {},
    sourceContentCache: {},
    sourceContentInflight: {},
    originalPrefetchInflight: {},
    originalPrefetchDone: {},
    sourceViewerDocumentId: "",
    sourceSummaryCollapsed: false,
    summaryCollapsed: false,
    modalOpen: false,
    uploadJobs: {},
    uploadNotice: null,
    sourceHighlight: null,
  };

  function isLlmSummaryPayload(summary) {
    if (!summary || typeof summary !== "object") return false;
    const source = String(summary.summary_source || "");
    if (["on_demand_llm", "cached_ui_summary", "pipeline_llm_summary"].includes(source)) return true;
    const usedModel = String(summary.used_model || "");
    return Boolean(usedModel && !["cached_basic_summary", "cached_llm_summary"].includes(usedModel));
  }

  function hasLlmPageSummaries(items) {
    return Array.isArray(items) && items.length > 0 && items.every((item) => String(item?.summary_source || "") === "on_demand_page_llm");
  }

  const body = document.body;
  const appShell = document.querySelector(".app-shell");
  const qaCaption = document.getElementById("qa-caption");
  const documentList = document.getElementById("document-list");
  const selectAllRow = document.getElementById("select-all-row");
  const selectAllCheck = document.getElementById("select-all-check");
  const deleteSelectedButton = document.getElementById("delete-selected-button");

  const sourceListView = document.getElementById("source-list-view");
  const sourceDetailView = document.getElementById("source-detail-view");
  const closeSourceViewerButton = document.getElementById("close-source-viewer-button");
  const sourceDetailTitle = document.getElementById("source-detail-title");
  const sourceDetailName = document.getElementById("source-detail-name");
  const sourceDetailSummary = document.getElementById("source-detail-summary");
  const sourceSummaryCard = document.getElementById("source-summary-card");
  const sourceSummaryToggle = document.getElementById("source-summary-toggle");
  const sourceDetailMeta = document.getElementById("source-detail-meta");
  const sourceDetailOriginal = document.getElementById("source-detail-original");
  const sourceDetailContent = document.getElementById("source-detail-content");
  const sourcePageSummaryList = document.getElementById("source-page-summary-list");

  const summaryContent = document.getElementById("summary-content");
  const downloadMdButton = document.getElementById("download-md-button");
  const summaryToggle = document.getElementById("summary-toggle");
  const queryForm = document.getElementById("query-form");
  const queryInput = document.getElementById("query-input");
  const queryButton = document.getElementById("query-button");
  const chatHistoryEl = document.getElementById("chat-history");

  const uploadModal = document.getElementById("upload-modal");
  const openUploadModalButton = document.getElementById("open-upload-modal-button");
  const closeUploadModalButton = document.getElementById("close-upload-modal-button");
  const workspaceUploadForm = document.getElementById("workspace-upload-form");
  const workspaceFileInput = document.getElementById("workspace-documents");
  const workspacePickButton = document.getElementById("workspace-pick-button");
  const workspaceSelectedFiles = document.getElementById("workspace-selected-files");
  const uploadDropzone = document.getElementById("upload-dropzone");
  const uploadSpinner = document.getElementById("upload-spinner");
  const uploadNoticeEl = document.getElementById("upload-notice");
  const backgroundWarmQueue = [];
  const backgroundWarmQueued = new Set();
  let backgroundWarmActive = 0;
  const BACKGROUND_WARM_CONCURRENCY = 1;

  if (qaCaption) {
    qaCaption.textContent = initialStatus.qa_caption || "";
  }

  summaryContent.classList.toggle("guide-collapsed", state.summaryCollapsed);

  function esc(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function renderMarkdown(text) {
    if (!text) return "";
    if (typeof marked === "undefined" || typeof DOMPurify === "undefined") {
      return `<p>${esc(text)}</p>`;
    }
    const html = marked.parse(String(text), { breaks: true, gfm: true });
    return DOMPurify.sanitize(html);
  }

  function dedupeDocumentIds(ids) {
    return [...new Set((ids || []).filter(Boolean))];
  }

  function buildCitationSummary(result) {
    const citations = Array.isArray(result?.citations) ? result.citations : [];
    if (!citations.length) return "";

    const seen = new Set();
    const labels = [];
    citations.forEach((citation) => {
      const sourceName = String(citation?.source_name || citation?.document_id || "").trim();
      const rawPageNumber = Number(citation?.page_number);
      const pageNumber = Number.isFinite(rawPageNumber) && rawPageNumber > 0 ? rawPageNumber : null;
      if (!sourceName) return;
      const label = pageNumber ? `${sourceName} p.${pageNumber}` : sourceName;
      if (seen.has(label)) return;
      seen.add(label);
      labels.push(label);
    });
    return labels.length ? `참조: ${labels.join(", ")}` : "";
  }

  function buildCitationMeta(citation) {
    const parts = [];
    const sourceName = String(citation?.source_name || citation?.document_id || "").trim();
    const sectionHint = String(citation?.section_hint || "").trim();
    const pageNumber = Number(citation?.page_number);
    if (sourceName) parts.push(sourceName);
    if (Number.isFinite(pageNumber) && pageNumber > 0) parts.push(`p.${pageNumber}`);
    if (sectionHint && !["none", "null", "undefined"].includes(sectionHint.toLowerCase())) parts.push(sectionHint);
    return parts.join(" · ");
  }

  function buildSupportingAssetMeta(asset) {
    const parts = [];
    const pageNumber = Number(asset?.page_number);
    if (Number.isFinite(pageNumber) && pageNumber > 0) {
      parts.push(`p.${pageNumber}`);
    }
    if (String(asset?.type || "") === "table") {
      const rowCount = Number(asset?.row_count);
      const columnCount = Number(asset?.column_count);
      if (Number.isFinite(rowCount) && rowCount > 0 && Number.isFinite(columnCount) && columnCount > 0) {
        parts.push(`${rowCount}행 x ${columnCount}열`);
      }
    }
    return parts.join(" · ");
  }

  function normalizeAssetMarkdown(markdown) {
    const lines = String(markdown || "")
      .split(/\r?\n/)
      .map((line) => line.trimEnd())
      .filter(Boolean);
    return lines.slice(0, 6).join("\n");
  }

  function renderSupportingAssetCard(asset, turnIndex, citationIndex, assetIndex) {
    const assetType = String(asset?.type || "");
    const assetLabel = assetType === "table"
      ? "표 근거"
      : assetType === "page_region"
        ? "페이지 근거"
        : "이미지 근거";
    const title = String(asset?.title || assetLabel).trim();
    const meta = buildSupportingAssetMeta(asset);
    const previewUrl = String(asset?.preview_url || "").trim();
    const previewMarkdown = assetType === "table" ? "" : "";
    const previewText = assetType === "table" ? "" : String(asset?.caption_text || asset?.text || "").trim();

    return `
      <div
        class="supporting-asset-card"
        role="button"
        tabindex="0"
        data-turn-index="${turnIndex}"
        data-citation-index="${citationIndex}"
        data-asset-index="${assetIndex}"
      >
        <div class="supporting-asset-head">
          <div class="supporting-asset-title-wrap">
            <span class="supporting-asset-kicker">${esc(assetLabel)}</span>
            <strong>${esc(title)}</strong>
          </div>
          ${meta ? `<span class="supporting-asset-meta">${esc(meta)}</span>` : ""}
        </div>
        ${previewUrl ? `
          <div class="supporting-asset-preview">
            <img src="${esc(previewUrl)}" alt="${esc(title)}" loading="lazy">
          </div>
        ` : ""}
        ${previewMarkdown
          ? `<div class="supporting-asset-body markdown-body">${renderMarkdown(previewMarkdown)}</div>`
          : previewText
            ? `<div class="supporting-asset-body">${esc(previewText)}</div>`
            : ""}
      </div>
    `;
  }

  function buildCitationPreviewText(citation) {
    const candidates = [
      citation?.highlight_text,
      citation?.quote,
      citation?.chunk_text,
    ].map((item) => String(item || "").trim()).filter(Boolean);
    const preview = candidates[0] || "근거 미리보기가 없습니다.";
    return preview.length > 320 ? `${preview.slice(0, 317).trimEnd()}...` : preview;
  }

  function renderCitationCard(citation, turnIndex, citationIndex) {
    const meta = buildCitationMeta(citation);
    const previewText = buildCitationPreviewText(citation);
    const supportingAssets = Array.isArray(citation?.supporting_assets) ? citation.supporting_assets : [];
    const hasCitationText = Boolean(previewText && previewText !== "근거 미리보기가 없습니다.");
    if (!supportingAssets.length && !hasCitationText) {
      return "";
    }

    return `
      <section class="answer-evidence-block">
        ${hasCitationText
          ? `<div class="answer-evidence-tools">
              <button
                type="button"
                class="citation-text-toggle"
                aria-expanded="false"
                data-turn-index="${turnIndex}"
                data-citation-toggle-index="${citationIndex}"
              >텍스트 근거 보기</button>
            </div>`
          : ""}
        ${hasCitationText
          ? `<div class="citation-text-panel" hidden>
              ${meta ? `<div class="citation-text-meta">${esc(meta)}</div>` : ""}
              <div class="citation-text-body">${esc(previewText)}</div>
            </div>`
          : ""}
        <div class="supporting-asset-list">
          ${supportingAssets.map((asset, assetIndex) => renderSupportingAssetCard(asset, turnIndex, citationIndex, assetIndex)).join("")}
        </div>
      </section>
    `;
  }

  function bindInteractiveCard(node, onActivate) {
    node.addEventListener("click", onActivate);
    node.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      onActivate();
    });
  }

  function getSelectedDocument() {
    return state.documents.find((item) => item.document_id === state.selectedDocumentId) || null;
  }

  function isCheckedDocument(documentId) {
    return state.checkedDocumentIds.includes(documentId);
  }

  function getScopedDocumentIds() {
    if (state.checkedDocumentIds.length) {
      return dedupeDocumentIds(state.checkedDocumentIds);
    }
    return state.selectedDocumentId ? [state.selectedDocumentId] : [];
  }

  function getActiveChatHistory() {
    return state.chatHistory;
  }

  function normalizeSummaryPayload(selected, summary) {
    return {
      document_id: selected.document_id,
      source_name: selected.source_name,
      document_type: selected.document_type || selected.extension || "document",
      summary_text: summary?.summary_text || "",
      key_points: summary?.key_points || summary?.highlights || [],
      page_summaries: hasLlmPageSummaries(summary?.page_summaries) ? summary.page_summaries : [],
      summary_source: summary?.summary_source || "basic_summary",
      used_model: summary?.used_model || "",
      warning: summary?.warning || "",
    };
  }

  function getSummaryPayload(documentId) {
    if (!documentId) return null;
    const selected = state.documents.find((item) => item.document_id === documentId);
    if (!selected) return null;

    const cached = state.summaryCache[documentId];
    if (isLlmSummaryPayload(cached)) return cached;

    const summary = [selected.ui_summary, selected.llm_summary].find((item) => isLlmSummaryPayload(item)) || null;
    if (!summary) return null;

    return normalizeSummaryPayload(selected, summary);
  }

  function getSummaryStatusMessage(documentId) {
    const ready = getSummaryPayload(documentId);
    if (ready?.summary_text) return ready.summary_text;
    if (state.summaryInflight[documentId]) {
      return "LLM 요약을 생성 중입니다. 문서 길이에 따라 1~2분 정도 걸릴 수 있습니다.";
    }
    const attempted = state.summaryCache[documentId];
    if (attempted?.warning) {
      return "LLM 요약을 아직 완료하지 못했습니다. 잠시 후 다시 시도해주세요.";
    }
    return "LLM 요약을 준비 중입니다.";
  }

  function buildStarterSummaryHtml() {
    const selected = state.documents.find((item) => item.document_id === state.starterDocumentId) || getSelectedDocument();
    if (!selected) return "";

    const title = selected.source_name || selected.document_id || "문서";
    const summaryText = getSummaryStatusMessage(selected.document_id);

    return `
      <section class="chat-starter-card chat-empty-state">
        <div class="chat-empty-summary">
          <h3 class="chat-empty-summary-title">${esc(title)}</h3>
          <p class="chat-empty-summary-text">${esc(summaryText)}</p>
        </div>
      </section>
    `;
  }

  function getPendingFilenames() {
    return Object.values(state.uploadJobs)
      .flatMap((job) => job.filenames || [])
      .filter(Boolean);
  }

  function reconcileUploadJobsWithDocuments() {
    const existingNames = new Set(state.documents.map((doc) => doc.source_name).filter(Boolean));
    let changed = false;

    Object.entries(state.uploadJobs).forEach(([jobId, job]) => {
      const remaining = (job.filenames || []).filter((name) => !existingNames.has(name));
      if (remaining.length !== (job.filenames || []).length) {
        changed = true;
      }
      if (remaining.length) {
        state.uploadJobs[jobId] = { ...job, filenames: remaining };
        return;
      }
      delete state.uploadJobs[jobId];
      changed = true;
    });

    if (changed) {
      refreshUploadUiState();
    }
  }

  function refreshUploadUiState() {
    uploadSpinner.classList.toggle("active", Object.keys(state.uploadJobs).length > 0);
    renderDocumentList();
  }

  function addUploadJob(jobId, filenames) {
    state.uploadJobs[jobId] = { filenames: [...new Set(filenames.filter(Boolean))] };
    refreshUploadUiState();
  }

  function removeUploadJob(jobId) {
    if (!jobId || !state.uploadJobs[jobId]) return;
    delete state.uploadJobs[jobId];
    refreshUploadUiState();
  }

  function renderSelectedFiles(files) {
    if (!workspaceSelectedFiles) return;
    if (!files.length) {
      workspaceSelectedFiles.innerHTML = '<span class="file-chip muted">선택된 문서가 없습니다.</span>';
      return;
    }
    workspaceSelectedFiles.innerHTML = files
      .map((file) => `<span class="file-chip">${esc(file.name)} · ${Math.max(1, Math.round(file.size / 1024))}KB</span>`)
      .join("");
  }

  function openUploadModal() {
    state.modalOpen = true;
    uploadModal.hidden = false;
    body.classList.add("modal-open");
  }

  function closeUploadModal() {
    state.modalOpen = false;
    uploadModal.hidden = true;
    body.classList.remove("modal-open");
  }

  function resetUploadUiState() {
    workspaceFileInput.value = "";
    renderSelectedFiles([]);
  }

  function updateSelectAllCheckbox() {
    if (!state.documents.length || state.sourceViewerDocumentId) {
      selectAllRow.hidden = true;
      return;
    }

    selectAllRow.hidden = false;
    const allChecked = state.documents.every((doc) => state.checkedDocumentIds.includes(doc.document_id));
    selectAllCheck.classList.toggle("checked", allChecked);
    selectAllCheck.textContent = allChecked ? "✓" : "";
    deleteSelectedButton.disabled = state.checkedDocumentIds.length === 0;
  }

  function renderDocumentPlaceholder() {
    summaryContent.innerHTML = '';
    summaryContent.hidden = true;
    const selected = getSelectedDocument();
    if (!selected) return;

    const readySummary = getSummaryPayload(selected.document_id);
    const attempted = state.summaryCache[selected.document_id];
    if (!readySummary && !attempted?.warning) {
      void autoSummarizeDocument(selected.document_id);
    }
  }

  function buildDocumentOriginalUrl(doc) {
    if (!doc?.document_id) return "";
    return `/api/document-original?document_id=${encodeURIComponent(doc.document_id)}&source_name=${encodeURIComponent(doc.source_name || "")}`;
  }

  async function autoSummarizeDocument(documentId, options = {}) {
    const force = Boolean(options.force);
    if (state.summaryInflight[documentId]) return;
    if (!force) {
      const cached = state.summaryCache[documentId];
      if (isLlmSummaryPayload(cached) || cached?.warning) return;
    }

    const doc = state.documents.find((item) => item.document_id === documentId);
    if (!doc) return;

    state.summaryInflight[documentId] = true;
    try {
      const response = await fetch("/api/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: doc.document_id,
          source_name: doc.source_name,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (response.ok && isLlmSummaryPayload(payload)) {
        state.summaryCache[documentId] = normalizeSummaryPayload(doc, payload || {});
        doc.ui_summary = payload;
      } else {
        state.summaryCache[documentId] = normalizeSummaryPayload(doc, {
          summary_source: payload?.summary_source || "llm_summary_unavailable",
          warning: String(payload?.warning || payload?.error || "LLM 요약을 완료하지 못했습니다."),
        });
      }
    } finally {
      delete state.summaryInflight[documentId];
      if (state.selectedDocumentId === documentId) {
        renderChat();
      }
      if (state.sourceViewerDocumentId === documentId) {
        renderSourceViewer();
      }
    }
  }

  async function fetchDocumentContent(documentId) {
    if (!documentId) return null;
    if (state.sourceContentCache[documentId]) return state.sourceContentCache[documentId];
    if (state.sourceContentInflight[documentId]) return state.sourceContentInflight[documentId];

    const selected = state.documents.find((doc) => doc.document_id === documentId);
    if (!selected) return null;

    const request = (async () => {
      try {
        const url = `/api/document-content?document_id=${encodeURIComponent(selected.document_id)}&source_name=${encodeURIComponent(selected.source_name || "")}`;
        const response = await fetch(url, { cache: "no-store" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "document_content_failed");
        state.sourceContentCache[documentId] = payload;
      } catch (error) {
        state.sourceContentCache[documentId] = {
          summary_text: "문서 정보를 불러오지 못했습니다.",
          markdown: String(error),
          original_url: "",
          document_type: selected.document_type || selected.extension || "",
        };
      } finally {
        delete state.sourceContentInflight[documentId];
      }
      return state.sourceContentCache[documentId];
    })();

    state.sourceContentInflight[documentId] = request;
    return request;
  }

  async function prefetchOriginalDocument(documentId) {
    if (!documentId || state.originalPrefetchDone[documentId]) return;
    if (state.originalPrefetchInflight[documentId]) return state.originalPrefetchInflight[documentId];

    const doc = state.documents.find((item) => item.document_id === documentId);
    const url = buildDocumentOriginalUrl(doc);
    if (!url) return;

    const request = (async () => {
      try {
        const response = await fetch(url, { cache: "force-cache" });
        if (response.ok) {
          await response.blob();
          state.originalPrefetchDone[documentId] = true;
        }
      } catch (_) {
        // Ignore prefetch failures and let the viewer retry on demand.
      } finally {
        delete state.originalPrefetchInflight[documentId];
      }
    })();

    state.originalPrefetchInflight[documentId] = request;
    return request;
  }

  async function warmDocumentInBackground(documentId) {
    await Promise.allSettled([
      autoSummarizeDocument(documentId),
      fetchDocumentContent(documentId),
    ]);
  }

  function pumpBackgroundWarmQueue() {
    while (backgroundWarmActive < BACKGROUND_WARM_CONCURRENCY && backgroundWarmQueue.length) {
      const documentId = backgroundWarmQueue.shift();
      backgroundWarmQueued.delete(documentId);
      backgroundWarmActive += 1;
      warmDocumentInBackground(documentId)
        .catch(() => {})
        .finally(() => {
          backgroundWarmActive -= 1;
          pumpBackgroundWarmQueue();
        });
    }
  }

  function scheduleBackgroundWarmup(documentIds) {
    (documentIds || []).forEach((documentId) => {
      if (!documentId || backgroundWarmQueued.has(documentId)) return;
      if (state.summaryCache[documentId] && state.sourceContentCache[documentId] && state.originalPrefetchDone[documentId]) {
        return;
      }
      backgroundWarmQueued.add(documentId);
      backgroundWarmQueue.push(documentId);
    });
    pumpBackgroundWarmQueue();
  }

  function escapeRegExp(value) {
    return String(value ?? "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function normalizeHighlightText(value) {
    return String(value ?? "")
      .replace(/<br\s*\/?>/gi, "\n")
      .replace(/\r\n/g, "\n")
      .replace(/\r/g, "\n")
      .replace(/\s+/g, " ")
      .trim();
  }

  function scoreTextSimilarity(sourceText, targetText) {
    const source = normalizeHighlightText(sourceText);
    const target = normalizeHighlightText(targetText);
    if (!source || !target) return -1;
    if (source.includes(target)) return 1000 + target.length;
    if (target.includes(source)) return 800 + source.length;

    const sourceTokens = new Set(source.toLowerCase().match(/[0-9a-zA-Z가-힣]+/g) || []);
    const targetTokens = new Set(target.toLowerCase().match(/[0-9a-zA-Z가-힣]+/g) || []);
    if (!sourceTokens.size || !targetTokens.size) return -1;

    let overlap = 0;
    targetTokens.forEach((token) => {
      if (sourceTokens.has(token)) overlap += token.length > 1 ? 2 : 1;
    });
    return overlap;
  }

  function applyInlineHighlightHtml(html, highlightText) {
    const needle = normalizeHighlightText(highlightText);
    if (!needle) return html;
    const escapedNeedle = escapeRegExp(needle);
    const regex = new RegExp(escapedNeedle, "i");
    if (!regex.test(html)) return html;
    return html.replace(regex, (match) => `<mark class="source-inline-highlight">${match}</mark>`);
  }

  function formatMarkdownForViewer(markdown, highlight = null) {
    if (!markdown) {
      return '<div class="muted">파싱 결과가 없습니다.</div>';
    }

    const highlightText = highlight?.highlight_text || "";
    const blockText = highlight?.block_text || highlightText;
    const blocks = esc(markdown)
      .split(/\n{2,}/)
      .map((block) => block.trim())
      .filter(Boolean);

    let bestIndex = -1;
    let bestScore = -1;
    blocks.forEach((block, index) => {
      const score = scoreTextSimilarity(block, blockText);
      if (score > bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    });

    return blocks.map((block, index) => {
      let blockHtml = block.replace(/\n/g, "<br>");
      if (index === bestIndex && blockText) {
        blockHtml = `<span class="source-block-highlight">${blockHtml}</span>`;
      }
      if (index === bestIndex && highlightText) {
        blockHtml = applyInlineHighlightHtml(blockHtml, highlightText);
      }
      return `<p>${blockHtml}</p>`;
    }).join("");
  }

  function renderOriginalPreview(content) {
    const originalUrl = content.original_url || "";
    if (!originalUrl) {
      return '<div class="source-original-fallback">문서 원본을 불러올 수 없습니다.</div>';
    }
    if (content.original_preview_kind === "image") {
      return `<img class="source-original-image" src="${esc(originalUrl)}" alt="${esc(content.source_name || "원본 문서 미리보기")}">`;
    }
    const pageNumber = state.sourceHighlight?.page_number;
    const pageHash = pageNumber ? `#page=${pageNumber}&toolbar=0&navpanes=0` : "#toolbar=0&navpanes=0";
    return `<iframe class="source-original-embed" src="${esc(originalUrl)}${pageHash}"></iframe>`;
  }

  function renderPageSummaryList(content, summary, documentId = "") {
    const pageSummaries = Array.isArray(summary?.page_summaries) ? summary.page_summaries : [];

    if (!sourcePageSummaryList) return;
    if (!pageSummaries.length) {
      sourcePageSummaryList.innerHTML = state.summaryInflight[documentId]
        ? '<div class="muted">페이지별 LLM 요약을 생성 중입니다.</div>'
        : '<div class="muted">페이지별 LLM 요약이 아직 준비되지 않았습니다.</div>';
      return;
    }

    sourcePageSummaryList.innerHTML = pageSummaries.map((item) => {
      const pageNumber = Number(item?.page_number) || 0;
      const keyPoints = Array.isArray(item?.key_points) ? item.key_points.filter(Boolean).slice(0, 3) : [];
      return `
        <button class="source-page-summary-item" type="button" data-page-number="${esc(pageNumber)}">
          <div class="source-page-summary-head">
            <span class="source-page-summary-number">p.${esc(pageNumber)}</span>
            <span class="source-page-summary-source">${esc(item?.summary_source || "")}</span>
          </div>
          <p class="source-page-summary-text">${esc(item?.summary_text || "")}</p>
          ${keyPoints.length
            ? `<div class="source-page-summary-points">${keyPoints.map((point) => `<span class="source-page-summary-point">${esc(point)}</span>`).join("")}</div>`
            : ""}
        </button>
      `;
    }).join("");

    sourcePageSummaryList.querySelectorAll("[data-page-number]").forEach((button) => {
      button.addEventListener("click", () => {
        const pageNumber = Number(button.getAttribute("data-page-number") || 0);
        if (!pageNumber) return;
        state.sourceHighlight = {
          page_number: pageNumber,
          highlight_text: "",
          block_text: "",
        };
        renderSourceViewer();
      });
    });
  }

  function renderSourceViewer() {
    const documentId = state.sourceViewerDocumentId;
    if (!documentId) {
      appShell.classList.remove("source-open");
      sourceListView.hidden = false;
      sourceDetailView.hidden = true;
      closeSourceViewerButton.hidden = true;
      updateSelectAllCheckbox();
      return;
    }

    appShell.classList.add("source-open");
    sourceListView.hidden = true;
    sourceDetailView.hidden = false;
    closeSourceViewerButton.hidden = false;

    const selected = state.documents.find((doc) => doc.document_id === documentId) || {};
    const content = state.sourceContentCache[documentId] || {};
    const summary = getSummaryPayload(documentId) || {};

    sourceDetailTitle.textContent = selected.source_name || selected.document_id || "문서";
    sourceDetailName.textContent = selected.source_name || selected.document_id || "문서";
    sourceDetailSummary.textContent = getSummaryStatusMessage(documentId);
    sourceDetailMeta.textContent = [
      content.document_type || selected.document_type,
      selected.origin,
      state.sourceHighlight?.page_number ? `p.${state.sourceHighlight.page_number}` : "",
    ].filter(Boolean).join(" · ");
    sourceSummaryCard.classList.toggle("collapsed", state.sourceSummaryCollapsed);
    sourceSummaryToggle.setAttribute("aria-expanded", String(!state.sourceSummaryCollapsed));

    if (!content.original_url && !content.markdown) {
      sourceDetailOriginal.innerHTML = '<div class="source-original-fallback">문서를 불러오는 중입니다.</div>';
      sourceDetailContent.innerHTML = '<div class="muted">파싱 결과를 불러오는 중입니다.</div>';
      renderPageSummaryList(content, summary, documentId);
      return;
    }

    renderPageSummaryList(content, summary, documentId);
    sourceDetailOriginal.innerHTML = renderOriginalPreview(content);
    sourceDetailContent.innerHTML = formatMarkdownForViewer(
      content.markdown || "",
      state.sourceHighlight,
    );
    const marker = sourceDetailContent.querySelector(".source-inline-highlight, .source-block-highlight");
    if (marker) {
      marker.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  function collapseSummary() {
    state.summaryCollapsed = true;
    summaryContent.classList.add("guide-collapsed");
  }

  async function openSourceViewer(documentId, options = {}) {
    const syncSelection = options.syncSelection !== false;
    state.sourceViewerDocumentId = documentId;
    if (syncSelection) {
      state.selectedDocumentId = documentId;
    }
    collapseSummary();
    renderSourceViewer();
    renderDocumentList();
    renderDocumentPlaceholder();
    renderChat();
    void autoSummarizeDocument(documentId);
    await fetchDocumentContent(documentId);
    renderSourceViewer();
  }

  function closeSourceViewer() {
    state.sourceViewerDocumentId = "";
    state.sourceHighlight = null;
    renderSourceViewer();
    renderDocumentList();
  }

  async function focusCitation(citation) {
    const documentId = citation.document_id || state.selectedDocumentId;
    if (!documentId) return;
    state.sourceHighlight = {
      page_number: citation.page_number || null,
      highlight_text: citation.highlight_text || citation.quote || "",
      block_text: citation.chunk_text || citation.quote || citation.highlight_text || "",
    };
    await openSourceViewer(documentId, { syncSelection: false });
  }

  async function focusSupportingAsset(citation, asset) {
    const documentId = citation?.document_id || state.selectedDocumentId;
    if (!documentId) return;
    state.sourceHighlight = {
      page_number: asset?.page_number || citation?.page_number || null,
      highlight_text: asset?.caption_text || asset?.text || citation?.highlight_text || citation?.quote || "",
      block_text: asset?.markdown || asset?.text || asset?.caption_text || citation?.chunk_text || citation?.quote || "",
    };
    await openSourceViewer(documentId, { syncSelection: false });
  }

  function renderUploadNotice() {
    if (!uploadNoticeEl) return;
    const n = state.uploadNotice;
    if (!n) {
      uploadNoticeEl.hidden = true;
      uploadNoticeEl.textContent = "";
      uploadNoticeEl.className = "upload-notice";
      return;
    }
    uploadNoticeEl.hidden = false;
    uploadNoticeEl.className = "upload-notice" + (n.isDuplicate ? " is-duplicate" : "");
    uploadNoticeEl.textContent = n.message;
  }

  function getDocumentBadge(doc) {
    const extension = String(doc.extension || "").replace(".", "").toUpperCase();
    return extension || "DOC";
  }

  function renderDocumentList() {
    const existingNames = new Set(state.documents.map((d) => d.source_name));
    const pendingHtml = [...new Set(getPendingFilenames())]
      .filter((name) => !existingNames.has(name))
      .map((name) => {
        const ext = (name.split(".").pop() || "DOC").toUpperCase().slice(0, 5);
        return `
          <button class="doc-card" type="button" disabled title="${esc(name)}">
            <span class="doc-icon">${esc(ext)}</span>
            <span class="doc-copy">
              <strong>${esc(name)}</strong>
              <span class="muted" style="font-size:12px">처리 중...</span>
            </span>
            <span class="doc-spinner"></span>
          </button>
        `;
      }).join("");

    if (!state.documents.length) {
      documentList.innerHTML = pendingHtml || '<div class="placeholder-card">표시할 문서가 없습니다.</div>';
      updateSelectAllCheckbox();
      return;
    }

    const pendingNameSet = new Set(getPendingFilenames());
    const isPending = (doc) => pendingNameSet.has(doc.source_name);

    documentList.innerHTML = pendingHtml + state.documents.map((doc) => {
      const activeClass = doc.document_id === state.selectedDocumentId ? " active" : "";
      const checked = isCheckedDocument(doc.document_id);
      const checkEl = isPending(doc)
        ? `<span class="doc-spinner"></span>`
        : `<span class="doc-check${checked ? " checked" : ""}" data-check-id="${esc(doc.document_id)}">${checked ? "✓" : ""}</span>`;
      return `
        <button class="doc-card${activeClass}" type="button" data-id="${esc(doc.document_id)}" title="${esc(doc.source_name || doc.document_id)}">
          <span class="doc-icon">${esc(getDocumentBadge(doc))}</span>
          <span class="doc-copy">
            <strong>${esc(doc.source_name || doc.document_id)}</strong>
          </span>
          ${checkEl}
        </button>
      `;
    }).join("");

    documentList.querySelectorAll("[data-id]").forEach((button) => {
      button.addEventListener("click", () => {
        const docId = button.getAttribute("data-id") || "";
        if (!docId) return;
        if (!isCheckedDocument(docId)) {
          state.checkedDocumentIds = dedupeDocumentIds([...state.checkedDocumentIds, docId]);
        }
        void openSourceViewer(docId);
      });
    });

    documentList.querySelectorAll("[data-check-id]").forEach((checkEl) => {
      checkEl.addEventListener("click", (event) => {
        event.stopPropagation();
        const documentId = checkEl.getAttribute("data-check-id") || "";
        if (!documentId) return;
        if (isCheckedDocument(documentId)) {
          state.checkedDocumentIds = state.checkedDocumentIds.filter((item) => item !== documentId);
        } else {
          state.checkedDocumentIds = dedupeDocumentIds([...state.checkedDocumentIds, documentId]);
        }
        renderDocumentList();
      });
    });

    updateSelectAllCheckbox();
  }

  function renderAnswerBubble(result, turnIndex) {
    const isError = Boolean(result.error);
    const citations = Array.isArray(result?.citations) ? result.citations : [];
    const evidenceMarkup = citations
      .map((citation, citationIndex) => renderCitationCard(citation, turnIndex, citationIndex))
      .filter(Boolean)
      .join("");
    return `
      <div class="chat-answer-card">
        <div class="chat-answer-head">
          <h3>답변</h3>
        </div>
        <div class="chat-answer-text markdown-body">${renderMarkdown(result.answer || (isError ? `오류: ${String(result.error || "unknown_error")}` : "응답이 없습니다."))}</div>
        ${evidenceMarkup
          ? `<div class="answer-evidence-list">
              ${evidenceMarkup}
            </div>`
          : ""}
      </div>
    `;
  }

  function renderChat(scrollToBottom = false) {
    const history = getActiveChatHistory();
    const starterHtml = buildStarterSummaryHtml();
    if (!history.length) {
      chatHistoryEl.innerHTML = starterHtml;
      return;
    }

    chatHistoryEl.innerHTML = starterHtml + history.map((entry, turnIndex) => {
      const aiHtml = entry.loading
        ? '<div class="chat-loading">답변을 생성하는 중입니다...</div>'
        : renderAnswerBubble(entry.result || {}, turnIndex);

      return `
        <div class="chat-turn">
          <div class="chat-bubble-user">${esc(entry.question)}</div>
          <div class="chat-bubble-ai">${aiHtml}</div>
        </div>
      `;
    }).join("");

    chatHistoryEl.querySelectorAll("[data-citation-index]:not([data-asset-index])").forEach((card) => {
      const turnIndex = Number(card.getAttribute("data-turn-index") || -1);
      const citationIndex = Number(card.getAttribute("data-citation-index") || -1);
      bindInteractiveCard(card, () => {
        const citation = history[turnIndex]?.result?.citations?.[citationIndex];
        if (citation) {
          void focusCitation(citation);
        }
      });
    });

    chatHistoryEl.querySelectorAll("[data-asset-index]").forEach((card) => {
      const turnIndex = Number(card.getAttribute("data-turn-index") || -1);
      const citationIndex = Number(card.getAttribute("data-citation-index") || -1);
      const assetIndex = Number(card.getAttribute("data-asset-index") || -1);
      bindInteractiveCard(card, () => {
        const citation = history[turnIndex]?.result?.citations?.[citationIndex];
        const asset = citation?.supporting_assets?.[assetIndex];
        if (citation && asset) {
          void focusSupportingAsset(citation, asset);
        }
      });
    });

    chatHistoryEl.querySelectorAll("[data-citation-toggle-index]").forEach((button) => {
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        const panel = button.closest(".answer-evidence-block")?.querySelector(".citation-text-panel");
        if (!panel) return;
        const expanded = button.getAttribute("aria-expanded") === "true";
        button.setAttribute("aria-expanded", expanded ? "false" : "true");
        button.textContent = expanded ? "텍스트 근거 보기" : "텍스트 근거 숨기기";
        panel.hidden = expanded;
      });
    });

    if (scrollToBottom) {
      requestAnimationFrame(() => {
        chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
      });
    }
  }

  async function loadDocuments(preferredDocumentId = "", options = {}) {
    const skipSummary = Boolean(options.skipSummary);
    const response = await fetch("/api/documents", { cache: "no-store" });
    const payload = await response.json();
    state.documents = payload.documents || [];
    reconcileUploadJobsWithDocuments();

    state.documents.forEach((doc) => {
      if (!doc.document_id) return;
      const cachedSummary = [doc.ui_summary, doc.llm_summary].find((item) => isLlmSummaryPayload(item)) || null;
      if (!cachedSummary) return;
      state.summaryCache[doc.document_id] = normalizeSummaryPayload(doc, cachedSummary);
    });

    if (!state.documents.length) {
      state.selectedDocumentId = "";
      state.starterDocumentId = "";
      state.checkedDocumentIds = [];
      state.sourceViewerDocumentId = "";
      state.chatHistory = [];
      renderSourceViewer();
      renderDocumentList();
      renderDocumentPlaceholder();
      renderChat();
      return;
    }

    if (preferredDocumentId && state.documents.some((item) => item.document_id === preferredDocumentId)) {
      state.selectedDocumentId = preferredDocumentId;
    } else if (!state.selectedDocumentId || !state.documents.some((item) => item.document_id === state.selectedDocumentId)) {
      state.selectedDocumentId = state.documents[0].document_id;
    }
    if (!state.starterDocumentId || !state.documents.some((item) => item.document_id === state.starterDocumentId)) {
      state.starterDocumentId = state.selectedDocumentId;
    }

    const availableIds = new Set(state.documents.map((item) => item.document_id));
    state.checkedDocumentIds = dedupeDocumentIds(state.checkedDocumentIds.filter((item) => availableIds.has(item)));
    if (!state.checkedDocumentIds.length && state.selectedDocumentId) {
      state.checkedDocumentIds = [state.selectedDocumentId];
    }
    if (state.sourceViewerDocumentId && !availableIds.has(state.sourceViewerDocumentId)) {
      state.sourceViewerDocumentId = "";
    }

    renderSourceViewer();
    renderDocumentList();
    if (!skipSummary) {
      renderDocumentPlaceholder();
    }
    renderChat();
    scheduleBackgroundWarmup(
      [state.selectedDocumentId].filter(Boolean)
    );
  }

  async function pollJob(jobId) {
    while (true) {
      const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
      const payload = await response.json();

      if (payload.status === "completed") {
        removeUploadJob(jobId);
        const result = payload.result || {};
        const vectorSummary = result.vector_index || {};
        const duplicateUploads = result.duplicate_uploads || [];
        const contentDuplicates = result.content_duplicate_uploads || [];
        const indexedDocs = (vectorSummary.documents || []).filter((item) => item.status === "indexed");
        const batchDocIds = [...new Set([
          ...duplicateUploads.map((item) => item.existing_document_id).filter(Boolean),
          ...contentDuplicates.map((item) => item.document_id).filter(Boolean),
          ...indexedDocs.map((item) => item.document_id).filter(Boolean),
        ])];

        const dupeNames = [
          ...duplicateUploads.map((item) => item.original_name || item.stored_name || ""),
          ...contentDuplicates.map((item) => item.source_name || ""),
        ].filter(Boolean);
        const processingNames = duplicateUploads
          .filter((item) => item.reason === "already_processing")
          .map((item) => item.original_name || item.stored_name || "")
          .filter(Boolean);
        state.uploadNotice = dupeNames.length
          ? {
              isDuplicate: true,
              message: processingNames.length
                ? (processingNames.length === 1
                    ? `"${processingNames[0]}"은(는) 현재 업로드 처리 중입니다.`
                    : `${processingNames.length}개 문서가 현재 업로드 처리 중입니다.`)
                : (dupeNames.length === 1
                    ? `"${dupeNames[0]}"은(는) 이미 존재하는 문서입니다.`
                    : `${dupeNames.length}개 문서가 이미 존재합니다.`),
            }
          : null;
        await loadDocuments(batchDocIds[0] || state.selectedDocumentId || "");
        if (batchDocIds.length) {
          scheduleBackgroundWarmup(batchDocIds);
        }
        renderUploadNotice();
        return;
      }

      if (payload.status === "failed") {
        removeUploadJob(jobId);
        window.alert(String(payload.error || payload.message || "업로드 처리에 실패했습니다."));
        return;
      }

      await new Promise((resolve) => setTimeout(resolve, 1200));
    }
  }

  async function submitUploads(files) {
    if (!files.length) {
      return;
    }

    const formData = new FormData();
    files.forEach((file) => formData.append("documents", file, file.name));

    try {
      const response = await fetch("/api/run", { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "업로드 실행에 실패했습니다.");
      }
      addUploadJob(payload.job_id, files.map((f) => f.name));
      state.uploadNotice = null;
      closeUploadModal();
      resetUploadUiState();
      renderUploadNotice();
      void pollJob(payload.job_id);
    } catch (error) {
      resetUploadUiState();
      window.alert(String(error));
    }
  }

  function downloadSummary(format) {
    const selected = getSelectedDocument();
    if (!selected) return;
    window.location.href = `/api/download-summary?document_id=${encodeURIComponent(selected.document_id)}&source_name=${encodeURIComponent(selected.source_name || "")}&format=${encodeURIComponent(format)}`;
  }

  async function deleteSelectedDocuments() {
    const ids = [...state.checkedDocumentIds];
    if (!ids.length) return;
    if (!window.confirm(`선택한 문서 ${ids.length}개를 삭제할까요?`)) return;

    deleteSelectedButton.disabled = true;
    try {
      const response = await fetch("/api/delete-documents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_ids: ids }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "삭제에 실패했습니다.");
      state.checkedDocumentIds = [];
      state.selectedDocumentId = "";
      state.sourceViewerDocumentId = "";
      await loadDocuments();
    } catch (error) {
      window.alert(String(error));
    } finally {
      updateSelectAllCheckbox();
    }
  }

  openUploadModalButton.addEventListener("click", openUploadModal);
  closeUploadModalButton.addEventListener("click", closeUploadModal);
  closeSourceViewerButton.addEventListener("click", closeSourceViewer);
  sourceSummaryToggle.addEventListener("click", () => {
    state.sourceSummaryCollapsed = !state.sourceSummaryCollapsed;
    renderSourceViewer();
  });

  uploadModal.addEventListener("click", (event) => {
    const target = event.target;
    if (target instanceof HTMLElement && target.dataset.closeModal === "true") {
      closeUploadModal();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.modalOpen) {
      closeUploadModal();
    }
  });

  workspacePickButton.addEventListener("click", () => workspaceFileInput.click());
  workspaceFileInput.addEventListener("change", () => {
    const files = Array.from(workspaceFileInput.files || []);
    renderSelectedFiles(files);
    if (files.length) {
      void submitUploads(files);
    }
  });

  workspaceUploadForm.addEventListener("submit", (event) => {
    event.preventDefault();
  });

  uploadDropzone.addEventListener("click", () => workspaceFileInput.click());
  uploadDropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      workspaceFileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    uploadDropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      uploadDropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    uploadDropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      if (eventName === "drop") {
        const files = Array.from(event.dataTransfer.files || []);
        uploadDropzone.classList.remove("drag-over");
        renderSelectedFiles(files);
        if (files.length) {
          void submitUploads(files);
        }
        return;
      }
      if (!uploadDropzone.contains(event.relatedTarget)) {
        uploadDropzone.classList.remove("drag-over");
      }
    });
  });

  document.addEventListener("dragover", (event) => event.preventDefault());
  document.addEventListener("drop", (event) => event.preventDefault());

  if (summaryToggle) {
    summaryToggle.addEventListener("click", () => {
      state.summaryCollapsed = !state.summaryCollapsed;
      summaryContent.classList.toggle("guide-collapsed", state.summaryCollapsed);
    });
  }

  downloadMdButton.addEventListener("click", () => downloadSummary("md"));
  selectAllRow.addEventListener("click", () => {
    const allChecked = state.documents.every((doc) => state.checkedDocumentIds.includes(doc.document_id));
    state.checkedDocumentIds = allChecked ? [] : state.documents.map((doc) => doc.document_id);
    renderDocumentList();
  });
  deleteSelectedButton.addEventListener("click", (event) => {
    event.stopPropagation();
    void deleteSelectedDocuments();
  });

  queryInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      queryForm.requestSubmit();
    }
  });

  queryForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const selected = getSelectedDocument();
    const query = queryInput.value.trim();
    const scopedDocumentIds = getScopedDocumentIds();

    if (!selected) {
      chatHistoryEl.innerHTML = '<div class="chat-empty">먼저 문서를 선택해주세요.</div>';
      return;
    }
    if (!scopedDocumentIds.length) {
      chatHistoryEl.innerHTML = '<div class="chat-empty">검색할 문서를 체크해주세요.</div>';
      return;
    }
    if (!query) {
      queryInput.focus();
      return;
    }

    queryInput.value = "";
    queryButton.disabled = true;
    collapseSummary();

    const entry = { question: query, result: null, loading: true };
    state.chatHistory.push(entry);
    renderChat(true);  // 질문 제출 시에만 스크롤

    try {
      const useScopedOnly = scopedDocumentIds.length > 0;
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          strategy: defaultQaStrategy,
          document_id: useScopedOnly ? "" : selected.document_id,
          source_name: useScopedOnly ? "" : selected.source_name,
          selected_document_ids: scopedDocumentIds,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "질의응답에 실패했습니다.");
      entry.result = payload || {};
    } catch (error) {
      entry.result = { answer: String(error), citations: [], matches: [], error: String(error) };
    } finally {
      entry.loading = false;
      queryButton.disabled = false;
      renderChat(false);  // 답변 완료 후 스크롤 이동 없음
    }
  });

  renderSelectedFiles([]);
  renderSourceViewer();

  loadDocuments()
    .then(() => {
      renderChat();
    })
    .catch(() => {
      documentList.innerHTML = '<div class="placeholder-card">문서 목록을 불러오지 못했습니다.</div>';
    });

  const evtSource = new EventSource("/api/events");
  evtSource.addEventListener("documents_updated", async () => {
    const previousIds = new Set(state.documents.map((doc) => doc.document_id));
    await loadDocuments(state.selectedDocumentId || "", { skipSummary: false });
    const newDocumentIds = state.documents
      .filter((doc) => !previousIds.has(doc.document_id))
      .map((doc) => doc.document_id);
    if (newDocumentIds.length) {
      scheduleBackgroundWarmup(newDocumentIds);
    }
  });

  window.documentStudio = {
    refreshDocuments: loadDocuments,
    openSourceViewer,
    closeSourceViewer,
  };
})();
