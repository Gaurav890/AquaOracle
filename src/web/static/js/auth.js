(function () {
  const tabs = document.querySelectorAll(".auth-tab");
  const loginForm = document.getElementById("login-form");
  const signupForm = document.getElementById("signup-form");
  const errorEl = document.getElementById("auth-error");

  // If already logged in, skip straight to the app.
  API.get("/api/auth/me")
    .then(() => { window.location.href = "/app.html"; })
    .catch(() => {});

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      errorEl.textContent = "";
      const isLogin = tab.dataset.tab === "login";
      loginForm.style.display = isLogin ? "block" : "none";
      signupForm.style.display = isLogin ? "none" : "block";
    });
  });

  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorEl.textContent = "";
    try {
      await API.post("/api/auth/login", {
        email: document.getElementById("login-email").value,
        password: document.getElementById("login-password").value,
      });
      window.location.href = "/app.html";
    } catch (err) {
      errorEl.textContent = err.message || "Login failed";
    }
  });

  signupForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorEl.textContent = "";
    try {
      await API.post("/api/auth/signup", {
        email: document.getElementById("signup-email").value,
        password: document.getElementById("signup-password").value,
        display_name: document.getElementById("signup-name").value || null,
      });
      window.location.href = "/app.html";
    } catch (err) {
      errorEl.textContent = err.message || "Sign up failed";
    }
  });
})();
