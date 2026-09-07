const Settings = (function () {
  const overlay = document.getElementById("settings-modal");
  const closeBtn = document.getElementById("settings-modal-close");
  const CLOUD_PROVIDERS = ["openai", "anthropic"];

  function rowFor(provider) {
    return document.querySelector(`.provider-row[data-provider="${provider}"]`);
  }

  function renderConfigured(provider, maskedKey) {
    const row = rowFor(provider);
    row.querySelector("[data-status]").textContent = `Configured (${maskedKey})`;
    row.querySelector("[data-status]").classList.add("configured");

    const actions = row.querySelector("[data-actions]");
    actions.innerHTML = "";
    const removeBtn = document.createElement("button");
    removeBtn.className = "btn btn-secondary";
    removeBtn.textContent = "Remove";
    removeBtn.addEventListener("click", async () => {
      await API.del(`/api/settings/api-keys/${provider}`);
      renderNotConfigured(provider);
    });
    actions.appendChild(removeBtn);
  }

  function renderNotConfigured(provider) {
    const row = rowFor(provider);
    row.querySelector("[data-status]").textContent = "Not configured";
    row.querySelector("[data-status]").classList.remove("configured");

    const actions = row.querySelector("[data-actions]");
    actions.innerHTML = "";

    const input = document.createElement("input");
    input.type = "password";
    input.placeholder = `${provider} API key`;
    actions.appendChild(input);

    const saveBtn = document.createElement("button");
    saveBtn.className = "btn btn-primary";
    saveBtn.textContent = "Save";
    saveBtn.addEventListener("click", async () => {
      const key = input.value.trim();
      if (!key) return;
      try {
        await API.put("/api/settings/api-keys", { provider, api_key: key });
        await load();
      } catch (e) {
        alert(`Couldn't save key: ${e.message}`);
      }
    });
    actions.appendChild(saveBtn);
  }

  async function load() {
    const keys = await API.get("/api/settings/api-keys");
    const configured = new Map(keys.map((k) => [k.provider, k.masked_key]));
    for (const provider of CLOUD_PROVIDERS) {
      if (configured.has(provider)) {
        renderConfigured(provider, configured.get(provider));
      } else {
        renderNotConfigured(provider);
      }
    }
  }

  function open() {
    load();
    overlay.classList.add("open");
  }

  function close() {
    overlay.classList.remove("open");
  }

  document.getElementById("settings-btn").addEventListener("click", open);
  closeBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });

  return { open, close };
})();

const ChatProvider = (function () {
  const overlay = document.getElementById("provider-modal");
  const select = document.getElementById("provider-select");
  const modelInput = document.getElementById("provider-model-input");
  const errorEl = document.getElementById("provider-modal-error");
  const saveBtn = document.getElementById("provider-modal-save");
  const cancelBtn = document.getElementById("provider-modal-cancel");
  const badge = document.getElementById("provider-badge");

  let currentChatId = null;

  function open(chatId, currentProvider, currentModel) {
    currentChatId = chatId;
    select.value = currentProvider;
    modelInput.value = currentModel || "";
    errorEl.style.display = "none";
    overlay.classList.add("open");
  }

  function close() {
    overlay.classList.remove("open");
  }

  async function save() {
    try {
      const chat = await API.patch(`/api/chats/${currentChatId}/provider`, {
        provider: select.value,
        model: modelInput.value.trim() || null,
      });
      badge.textContent = `${chat.provider.toUpperCase()}${chat.model ? " · " + chat.model : ""}`;
      close();
    } catch (e) {
      errorEl.textContent = e.message;
      errorEl.style.display = "block";
    }
  }

  badge.addEventListener("click", () => {
    const chatId = Chat.getCurrentChatId();
    if (!chatId) return;
    API.get(`/api/chats/${chatId}`).then((chat) => open(chatId, chat.provider, chat.model));
  });
  saveBtn.addEventListener("click", save);
  cancelBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });

  return {};
})();
