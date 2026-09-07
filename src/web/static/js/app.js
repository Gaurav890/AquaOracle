(async function () {
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

  await Sidebar.init((chatId) => {
    Chat.load(chatId);
    Documents.load(chatId);
  });

  const composerInput = document.getElementById("composer-input");
  const sendBtn = document.getElementById("send-btn");
  const titleInput = document.getElementById("chat-title-input");

  sendBtn.addEventListener("click", () => Chat.send());
  composerInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      Chat.send();
    }
  });
  composerInput.addEventListener("input", () => {
    composerInput.style.height = "auto";
    composerInput.style.height = Math.min(composerInput.scrollHeight, 200) + "px";
  });

  titleInput.addEventListener("blur", () => Chat.renameCurrentChat(titleInput.value));
  titleInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      titleInput.blur();
    }
  });
})();
