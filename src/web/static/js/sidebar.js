const Sidebar = (function () {
  let chats = [];
  let activeChatId = null;
  let onSelectCallback = null;

  const listEl = document.getElementById("chat-list");

  function render() {
    listEl.innerHTML = "";
    for (const chat of chats) {
      const item = document.createElement("div");
      item.className = "chat-item" + (chat.id === activeChatId ? " active" : "");
      item.dataset.chatId = chat.id;

      const title = document.createElement("span");
      title.className = "chat-item-title";
      title.textContent = chat.title;
      item.appendChild(title);

      const del = document.createElement("span");
      del.className = "chat-item-delete";
      del.textContent = "×";
      del.title = "Delete chat";
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteChat(chat.id);
      });
      item.appendChild(del);

      item.addEventListener("click", () => selectChat(chat.id));
      listEl.appendChild(item);
    }
  }

  async function refresh() {
    chats = await API.get("/api/chats");
    render();
  }

  function selectChat(chatId) {
    activeChatId = chatId;
    render();
    if (onSelectCallback) onSelectCallback(chatId);
  }

  async function createChat() {
    const chat = await API.post("/api/chats", {});
    chats.unshift(chat);
    selectChat(chat.id);
  }

  async function deleteChat(chatId) {
    await API.del(`/api/chats/${chatId}`);
    chats = chats.filter((c) => c.id !== chatId);
    render();
    if (activeChatId === chatId) {
      activeChatId = null;
      if (onSelectCallback) onSelectCallback(null);
    }
  }

  // Move a chat to the top and update its title after a message is sent
  // (matches the backend bumping updated_at / setting the title from the
  // first message) — avoids a full refetch after every send.
  function touchChat(chatId, updates) {
    const idx = chats.findIndex((c) => c.id === chatId);
    if (idx === -1) return;
    const chat = { ...chats[idx], ...updates };
    chats.splice(idx, 1);
    chats.unshift(chat);
    render();
  }

  return {
    async init(onSelect) {
      onSelectCallback = onSelect;
      document.getElementById("new-chat-btn").addEventListener("click", createChat);
      await refresh();
      if (chats.length > 0) {
        selectChat(chats[0].id);
      }
    },
    getActiveChatId: () => activeChatId,
    touchChat,
  };
})();
