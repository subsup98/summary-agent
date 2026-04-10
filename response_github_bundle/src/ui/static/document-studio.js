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
    chatHistory: {},
    summaryCache: {},
    summaryInflight: {},
    sourceContentCache: {},
    sourceViewerDocumentId: "",
    sourceSummaryCollapsed: false,
    summaryCollapsed: false,
    modalOpen: false,
    uploadPending: false,
    pendingFilenames: [],
    uploadNotice: null,
  };

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

  const summaryContent = document.getElementById("summary-content");
  const downloadMdButton = document.getElementById("download-md-button");
  const downloadTxtButton = document.getElementById("download-txt-button");
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

  function setUploadPending(pending) {
    state.uploadPending = pending;
    workspacePickButton.disabled = pending;
    uploadSpinner.classList.toggle("active", pending);
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
    setUploadPending(false);
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

  function renderSummaryHtml(payload) {
    const sourceLabel = payload.source_name || payload.document_id || "document";
    const keyPoints = payload.key_points || [];
    const isFallback = payload.summary_source === "basic_summary_fallback";
    const fallbackBadge = isFallback
      ? `<span class="summary-fallback-badge" title="OpenAI 미연결 — 기본 요약 사용 중">⚠️ 기본 요약</span>`
      : "";
    summaryContent.innerHTML = `
      <div class="summary-banner-kicker">${esc(sourceLabel)}${fallbackBadge}</div>
      <p class="summary-banner-text">${esc(payload.summary_text || "요약이 아직 준비되지 않았습니다.")}</p>
      ${
        keyPoints.length
          ? `<div class="tag-row">${keyPoints.slice(0, 4).map((item) => `<span class="tag">${esc(item)}</span>`).join("")}</div>`
          : ""
      }
    `;
  }

  function renderDocumentPlaceholder() {
    const selected = getSelectedDocument();
    if (!selected) {
      summaryContent.innerHTML = '<div class="summary-banner-kicker">선택한 문서 안내</div><p class="summary-banner-text muted">문서를 선택하면 요약이 여기에 표시됩니다.</p>';
      return;
    }

    const cached = state.summaryCache[selected.document_id];
    if (cached && cached.summary_source !== "basic_summary_fallback") {
      renderSummaryHtml(cached);
      return;
    }

    if (cached && cached.summary_source === "basic_summary_fallback") {
      renderSummaryHtml(cached);
      void autoSummarizeDocument(selected.document_id);
      return;
    }

    summaryContent.innerHTML = `
      <div class="summary-banner-kicker">${esc(selected.source_name || selected.document_id)}</div>
      <p class="summary-banner-text muted">요약을 불러오는 중입니다...</p>
    `;
    void autoSummarizeDocument(selected.document_id);
  }

  async function autoSummarizeDocument(documentId, options = {}) {
    const force = Boolean(options.force);
    if (state.summaryInflight[documentId]) return;
    if (!force && state.summaryCache[documentId]) return;

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
      const payload = await response.json();
      if (response.ok) {
        state.summaryCache[documentId] = payload;
        if (state.selectedDocumentId === documentId) {
          renderSummaryHtml(payload);
        }
        if (state.sourceViewerDocumentId === documentId) {
          renderSourceViewer();
        }
      }
    } finally {
      delete state.summaryInflight[documentId];
    }
  }

  function formatMarkdownForViewer(markdown) {
    if (!markdown) {
      return '<div class="muted">파싱 결과가 없습니다.</div>';
    }
    return esc(markdown)
      .split(/\n{2,}/)
      .map((block) => `<p>${block.replace(/\n/g, "<br>")}</p>`)
      .join("");
  }

  function renderOriginalPreview(content) {
    const originalUrl = content.original_url || "";
    if (!originalUrl) {
      return '<div class="source-original-fallback">문서 원본을 불러올 수 없습니다.</div>';
    }
    return `<iframe class="source-original-embed" src="${esc(originalUrl)}#toolbar=0&navpanes=0"></iframe>`;
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
    const summary = state.summaryCache[documentId] || {};

    sourceDetailTitle.textContent = selected.source_name || selected.document_id || "문서";
    sourceDetailName.textContent = selected.source_name || selected.document_id || "문서";
    sourceDetailSummary.textContent = summary.summary_text || content.summary_text || "요약이 아직 준비되지 않았습니다.";
    sourceDetailMeta.textContent = [content.document_type || selected.document_type, selected.origin].filter(Boolean).join(" · ");
    sourceSummaryCard.classList.toggle("collapsed", state.sourceSummaryCollapsed);
    sourceSummaryToggle.setAttribute("aria-expanded", String(!state.sourceSummaryCollapsed));

    if (!content.original_url && !content.markdown) {
      sourceDetailOriginal.innerHTML = '<div class="source-original-fallback">문서를 불러오는 중입니다.</div>';
      sourceDetailContent.innerHTML = '<div class="muted">파싱 결과를 불러오는 중입니다.</div>';
      return;
    }

    sourceDetailOriginal.innerHTML = renderOriginalPreview(content);
    sourceDetailContent.innerHTML = formatMarkdownForViewer(content.markdown || "");
  }

  function collapseSummary() {
    state.summaryCollapsed = true;
    summaryContent.classList.add("guide-collapsed");
  }

  async function openSourceViewer(documentId) {
    state.sourceViewerDocumentId = documentId;
    state.selectedDocumentId = documentId;
    collapseSummary();
    renderSourceViewer();
    renderDocumentList();
    renderDocumentPlaceholder();
    renderChat();

    if (state.summaryCache[documentId]) {
      renderSourceViewer();
    } else {
      void autoSummarizeDocument(documentId);
    }

    if (state.sourceContentCache[documentId]) {
      renderSourceViewer();
      return;
    }

    const selected = state.documents.find((doc) => doc.document_id === documentId);
    if (!selected) return;

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
    }

    renderSourceViewer();
  }

  function closeSourceViewer() {
    state.sourceViewerDocumentId = "";
    renderSourceViewer();
    renderDocumentList();
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
    const pendingHtml = state.pendingFilenames
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

    const isPending = (doc) => state.pendingFilenames.includes(doc.source_name);

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

  function renderAnswerBubble(result) {
    const isError = Boolean(result.error);
    const citations = result.citations || [];
    const matches = result.matches || result.source_documents || [];

    const evidenceHtml = isError
      ? `<div class="evidence-item">질문 처리 중 오류가 발생했습니다. ${esc(String(result.error || "unknown_error"))}</div>`
      : citations.length
        ? citations.map((citation) => `
            <div class="evidence-item">
              <strong>${esc(citation.source_name || "document")}</strong>
              <div class="muted">${esc(citation.section_hint || "section")}</div>
              <div>${esc(citation.quote || "")}</div>
            </div>
          `).join("")
        : matches.slice(0, 3).map((match) => {
            const metadata = match.metadata || {};
            const excerpt = match.document || match.page_content || "";
            return `
              <div class="evidence-item">
                <strong>${esc(metadata.source_name || metadata.document_id || "document")}</strong>
                <div class="muted">${esc(metadata.section_hint || "section")}</div>
                <div>${esc(String(excerpt).slice(0, 220))}</div>
              </div>
            `;
          }).join("") || '<div class="evidence-item">근거가 없습니다.</div>';

    return `
      <div class="chat-answer-card">
        <div class="chat-answer-head">
          <h3>답변</h3>
          ${result.used_model ? `<span class="answer-badge">${esc(result.used_model)}</span>` : ""}
        </div>
        <div class="chat-answer-text markdown-body">${renderMarkdown(result.answer || "응답이 없습니다.")}</div>
        <div class="evidence-list">${evidenceHtml}</div>
      </div>
    `;
  }

  function renderChat(scrollToBottom = false) {
    const history = state.chatHistory[state.selectedDocumentId] || [];
    if (!history.length) {
      chatHistoryEl.innerHTML = '';
      return;
    }

    chatHistoryEl.innerHTML = history.map((entry) => {
      const aiHtml = entry.loading
        ? '<div class="chat-loading">답변을 생성하는 중입니다...</div>'
        : renderAnswerBubble(entry.result || {});

      return `
        <div class="chat-turn">
          <div class="chat-bubble-user">${esc(entry.question)}</div>
          <div class="chat-bubble-ai">${aiHtml}</div>
        </div>
      `;
    }).join("");

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

    state.documents.forEach((doc) => {
      const llmCached = doc.ui_summary || doc.llm_summary;
      if (!llmCached || !doc.document_id) return;
      state.summaryCache[doc.document_id] = {
        document_id: doc.document_id,
        source_name: doc.source_name,
        document_type: doc.document_type || doc.extension || "document",
        summary_text: llmCached.summary_text || "",
        key_points: llmCached.key_points || llmCached.highlights || [],
        summary_source: llmCached.summary_source || "",
      };
    });

    if (!state.documents.length) {
      state.selectedDocumentId = "";
      state.checkedDocumentIds = [];
      state.sourceViewerDocumentId = "";
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
  }

  async function pollJob(jobId) {
    while (true) {
      const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
      const payload = await response.json();

      if (payload.status === "completed") {
        resetUploadUiState();
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
        state.uploadNotice = dupeNames.length
          ? {
              isDuplicate: true,
              message: dupeNames.length === 1
                ? `"${dupeNames[0]}"은(는) 이미 존재하는 문서입니다.`
                : `${dupeNames.length}개 문서가 이미 존재합니다.`,
            }
          : null;
        state.pendingFilenames = [];
        await loadDocuments(batchDocIds[0] || state.selectedDocumentId || "");
        renderUploadNotice();
        return;
      }

      if (payload.status === "failed") {
        state.pendingFilenames = [];
        resetUploadUiState();
        renderDocumentList();
        window.alert(String(payload.error || payload.message || "업로드 처리에 실패했습니다."));
        return;
      }

      await new Promise((resolve) => setTimeout(resolve, 1200));
    }
  }

  async function submitUploads(files) {
    if (!files.length || state.uploadPending) {
      return;
    }

    setUploadPending(true);
    const formData = new FormData();
    files.forEach((file) => formData.append("documents", file, file.name));

    try {
      const response = await fetch("/api/run", { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "업로드 실행에 실패했습니다.");
      }
      state.pendingFilenames = files.map((f) => f.name);
      state.uploadNotice = null;
      closeUploadModal();
      resetUploadUiState();
      renderDocumentList();
      renderUploadNotice();
      void pollJob(payload.job_id);
    } catch (error) {
      state.pendingFilenames = [];
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
    if (target instanceof HTMLElement && target.dataset.closeModal === "true" && !state.uploadPending) {
      closeUploadModal();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.modalOpen && !state.uploadPending) {
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

  summaryToggle.addEventListener("click", () => {
    state.summaryCollapsed = !state.summaryCollapsed;
    summaryContent.classList.toggle("guide-collapsed", state.summaryCollapsed);
  });

  downloadMdButton.addEventListener("click", () => downloadSummary("md"));
  downloadTxtButton.addEventListener("click", () => downloadSummary("txt"));
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

    if (!state.chatHistory[state.selectedDocumentId]) {
      state.chatHistory[state.selectedDocumentId] = [];
    }

    const entry = { question: query, result: null, loading: true };
    state.chatHistory[state.selectedDocumentId].push(entry);
    renderChat(true);  // 질문 제출 시에만 스크롤

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          strategy: defaultQaStrategy,
          document_id: selected.document_id,
          source_name: selected.source_name,
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
    await loadDocuments(state.selectedDocumentId || "", { skipSummary: false });
  });

  window.documentStudio = {
    refreshDocuments: loadDocuments,
    openSourceViewer,
    closeSourceViewer,
  };
})();
