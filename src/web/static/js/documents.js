const Documents = (function () {
  const drawer = document.getElementById("doc-drawer");
  const listEl = document.getElementById("doc-drawer-list");
  const toggleBtn = document.getElementById("docs-toggle-btn");
  const closeBtn = document.getElementById("doc-drawer-close");
  const uploadZone = document.getElementById("upload-zone");
  const uploadInput = document.getElementById("upload-input");
  const uploadLabel = document.getElementById("upload-label");
  const shareCheckbox = document.getElementById("upload-share-checkbox");

  let currentChatId = null;

  function render(docs) {
    listEl.innerHTML = "";
    if (docs.length === 0) {
      listEl.innerHTML = '<div class="doc-drawer-empty">No documents in this chat yet.</div>';
      return;
    }
    for (const doc of docs) {
      const item = document.createElement("div");
      item.className = "doc-item";

      const title = document.createElement("div");
      title.className = "doc-item-title";
      title.textContent = doc.title || doc.file_name;
      item.appendChild(title);

      const meta = document.createElement("div");
      meta.className = "doc-item-meta";
      meta.textContent = `${doc.page_count} pages · ${doc.chunk_count} chunks`;
      item.appendChild(meta);

      const actions = document.createElement("div");
      actions.className = "doc-item-actions";

      const shareLabel = document.createElement("label");
      const shareInput = document.createElement("input");
      shareInput.type = "checkbox";
      shareInput.checked = doc.is_shared;
      shareInput.addEventListener("change", async () => {
        try {
          await API.post(`/api/documents/${doc.doc_id}/share?share=${shareInput.checked}`);
        } catch (e) {
          shareInput.checked = !shareInput.checked;
          alert(`Couldn't update sharing: ${e.message}`);
        }
      });
      shareLabel.appendChild(shareInput);
      shareLabel.appendChild(document.createTextNode("Shared"));
      actions.appendChild(shareLabel);

      const removeBtn = document.createElement("span");
      removeBtn.className = "doc-item-remove";
      removeBtn.textContent = "Remove from chat";
      removeBtn.style.cursor = "pointer";
      removeBtn.addEventListener("click", async () => {
        await API.del(`/api/chats/${currentChatId}/documents/${doc.doc_id}`);
        load(currentChatId);
      });
      actions.appendChild(removeBtn);

      item.appendChild(actions);
      listEl.appendChild(item);
    }
  }

  async function load(chatId) {
    currentChatId = chatId;
    toggleBtn.disabled = !chatId;
    if (!chatId) {
      listEl.innerHTML = "";
      drawer.classList.remove("open");
      return;
    }
    const docs = await API.get(`/api/chats/${chatId}/documents`);
    render(docs);
  }

  async function uploadFile(file) {
    if (!currentChatId || !file) return;
    uploadLabel.textContent = `Uploading ${file.name}...`;
    try {
      const share = shareCheckbox.checked ? "true" : "false";
      await API.uploadFile(`/api/chats/${currentChatId}/documents?share=${share}`, file);
      uploadLabel.textContent = "Drop a PDF here or click to upload";
      await load(currentChatId);
    } catch (e) {
      uploadLabel.textContent = `Failed: ${e.message}`;
      setTimeout(() => {
        uploadLabel.textContent = "Drop a PDF here or click to upload";
      }, 3000);
    }
  }

  toggleBtn.addEventListener("click", () => {
    drawer.classList.toggle("open");
  });
  closeBtn.addEventListener("click", () => drawer.classList.remove("open"));

  uploadInput.addEventListener("change", () => uploadFile(uploadInput.files[0]));
  uploadZone.addEventListener("dragover", (e) => e.preventDefault());
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  return { load };
})();
