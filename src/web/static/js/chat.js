const Chat = (function () {
  let currentChatId = null;

  const messagesEl = document.getElementById("messages");
  const emptyStateEl = document.getElementById("empty-state");
  const titleInput = document.getElementById("chat-title-input");
  const providerBadge = document.getElementById("provider-badge");
  const composerInput = document.getElementById("composer-input");
  const sendBtn = document.getElementById("send-btn");

  function setComposerEnabled(enabled) {
    composerInput.disabled = !enabled;
    sendBtn.disabled = !enabled;
    titleInput.disabled = !enabled;
  }

  function formatPageAndLines(pageNumbers, lineRanges) {
    if (!pageNumbers || pageNumbers.length === 0) return "";
    if (!lineRanges || Object.keys(lineRanges).length === 0) {
      return pageNumbers.length === 1
        ? `p.${pageNumbers[0]}`
        : `p.${pageNumbers[0]}-${pageNumbers[pageNumbers.length - 1]}`;
    }
    return pageNumbers
      .map((page) => {
        const range = lineRanges[page] ?? lineRanges[String(page)];
        if (!range) return `p.${page}`;
        const [start, end] = range;
        return start === end ? `p.${page} L${start}` : `p.${page} L${start}-${end}`;
      })
      .join("; ");
  }

  function renderSources(sources) {
    if (!sources || sources.length === 0) return null;
    const wrap = document.createElement("div");
    wrap.className = "message-sources";
    for (const s of sources) {
      const chip = document.createElement("a");
      chip.className = "source-chip";
      const locator = formatPageAndLines(s.page_numbers, s.line_ranges);
      chip.textContent = `[${s.index}] ${s.doc_id}${locator ? " " + locator : ""}`;

      const firstPage = s.page_numbers && s.page_numbers.length ? s.page_numbers[0] : null;
      chip.href = `/api/documents/${encodeURIComponent(s.doc_id)}/file${firstPage ? `#page=${firstPage}` : ""}`;
      chip.target = "_blank";
      chip.rel = "noopener";

      wrap.appendChild(chip);
    }
    return wrap;
  }

  function typingIndicator() {
    const wrap = document.createElement("span");
    wrap.className = "typing-indicator";
    wrap.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
    return wrap;
  }

  function renderFeedbackControls(messageId, currentRating) {
    const wrap = document.createElement("div");
    wrap.className = "feedback-controls";

    function makeButton(rating, glyph) {
      const btn = document.createElement("button");
      btn.className = "feedback-btn" + (currentRating === rating ? " active" : "");
      btn.textContent = glyph;
      btn.title = rating === "up" ? "Good answer" : "Bad answer";
      btn.addEventListener("click", async () => {
        try {
          await API.post(`/api/messages/${messageId}/feedback`, { rating });
          wrap.querySelectorAll(".feedback-btn").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
        } catch (e) {
          // No QueryLog row (e.g. a message from before this feature shipped) — nothing to rate.
        }
      });
      return btn;
    }

    wrap.appendChild(makeButton("up", "\u{1F44D}"));
    wrap.appendChild(makeButton("down", "\u{1F44E}"));
    return wrap;
  }

  function appendMessage({ role, content, sources, isError, pending, messageId, userFeedback }) {
    const msgEl = document.createElement("div");
    msgEl.className = `message message-${role}` + (isError ? " message-error" : "");

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    if (pending) {
      bubble.appendChild(typingIndicator());
    } else {
      bubble.textContent = content;
    }
    msgEl.appendChild(bubble);

    const sourcesEl = renderSources(sources);
    if (sourcesEl) msgEl.appendChild(sourcesEl);

    if (role === "assistant" && !isError && messageId) {
      msgEl.appendChild(renderFeedbackControls(messageId, userFeedback));
    }

    messagesEl.querySelector(".messages-inner")?.appendChild(msgEl) || messagesEl.appendChild(msgEl);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  function clearMessages() {
    messagesEl.innerHTML = '<div class="messages-inner"></div>';
  }

  async function load(chatId) {
    if (!chatId) {
      currentChatId = null;
      clearMessages();
      messagesEl.appendChild(emptyStateEl);
      titleInput.value = "";
      setComposerEnabled(false);
      return;
    }

    currentChatId = chatId;
    const [chat, messages] = await Promise.all([
      API.get(`/api/chats/${chatId}`),
      API.get(`/api/chats/${chatId}/messages`),
    ]);

    titleInput.value = chat.title;
    providerBadge.textContent = `${chat.provider.toUpperCase()}${chat.model ? " · " + chat.model : ""}`;
    clearMessages();
    for (const m of messages) {
      appendMessage({
        role: m.role,
        content: m.content,
        sources: m.sources,
        isError: m.content.startsWith("Error:"),
        messageId: m.id,
        userFeedback: m.user_feedback,
      });
    }
    setComposerEnabled(true);
    composerInput.focus();
  }

  function send() {
    const text = composerInput.value.trim();
    if (!text || !currentChatId) return;

    composerInput.value = "";
    composerInput.style.height = "auto";
    setComposerEnabled(false);

    appendMessage({ role: "user", content: text });
    const assistantBubble = appendMessage({ role: "assistant", content: "", pending: true });
    let accumulated = "";

    API.streamPost(
      `/api/chats/${currentChatId}/messages`,
      { message: text },
      (eventName, data) => {
        if (eventName === "token") {
          accumulated += data.text;
          assistantBubble.textContent = accumulated;
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (eventName === "done") {
          assistantBubble.textContent = data.answer;
          const sourcesEl = renderSources(data.sources);
          if (sourcesEl) assistantBubble.parentElement.appendChild(sourcesEl);
          if (data.message_id) {
            assistantBubble.parentElement.appendChild(renderFeedbackControls(data.message_id, null));
          }
          setComposerEnabled(true);
          composerInput.focus();
          Sidebar.touchChat(currentChatId, { updated_at: new Date().toISOString() });
          // Title may have been set server-side from the first message.
          API.get(`/api/chats/${currentChatId}`).then((chat) => {
            titleInput.value = chat.title;
            Sidebar.touchChat(currentChatId, { title: chat.title });
          });
        } else if (eventName === "error") {
          assistantBubble.parentElement.classList.add("message-error");
          assistantBubble.textContent = `Error: ${data.message}`;
          setComposerEnabled(true);
        }
      },
      (err) => {
        assistantBubble.parentElement.classList.add("message-error");
        assistantBubble.textContent = `Error: ${err.message}`;
        setComposerEnabled(true);
      }
    );
  }

  async function renameCurrentChat(title) {
    if (!currentChatId || !title.trim()) return;
    const chat = await API.patch(`/api/chats/${currentChatId}`, { title: title.trim() });
    Sidebar.touchChat(currentChatId, { title: chat.title });
  }

  return { load, send, renameCurrentChat, getCurrentChatId: () => currentChatId };
})();
