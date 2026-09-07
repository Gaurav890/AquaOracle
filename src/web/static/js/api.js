// Thin fetch() wrapper. Session auth is a cookie (set by the server on
// login/signup), so every call just needs `credentials: "same-origin"` —
// no token to attach manually.

const API = {
  async _request(method, path, body) {
    const opts = {
      method,
      credentials: "same-origin",
      headers: {},
    };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const data = await res.json();
        detail = data.detail || detail;
      } catch (_) {
        // no JSON body
      }
      const err = new Error(detail);
      err.status = res.status;
      throw err;
    }
    if (res.status === 204) return null;
    const contentType = res.headers.get("content-type") || "";
    return contentType.includes("application/json") ? res.json() : res.text();
  },

  get(path) {
    return this._request("GET", path);
  },
  post(path, body) {
    return this._request("POST", path, body ?? {});
  },
  patch(path, body) {
    return this._request("PATCH", path, body ?? {});
  },
  put(path, body) {
    return this._request("PUT", path, body ?? {});
  },
  del(path) {
    return this._request("DELETE", path);
  },

  async uploadFile(path, file) {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(path, { method: "POST", credentials: "same-origin", body: form });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const data = await res.json();
        detail = data.detail || detail;
      } catch (_) {}
      const err = new Error(detail);
      err.status = res.status;
      throw err;
    }
    return res.json();
  },

  // Server-Sent Events helper for streaming chat responses. `onEvent`
  // receives (eventName, parsedJsonData) for each `event:`/`data:` pair.
  streamPost(path, body, onEvent, onError) {
    fetch(path, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    })
      .then(async (res) => {
        if (!res.ok || !res.body) {
          throw new Error(`Request failed: ${res.status}`);
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let boundary;
          while ((boundary = buffer.indexOf("\n\n")) !== -1) {
            const rawEvent = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + 2);
            const lines = rawEvent.split("\n");
            let eventName = "message";
            let dataLine = "";
            for (const line of lines) {
              if (line.startsWith("event:")) eventName = line.slice(6).trim();
              else if (line.startsWith("data:")) dataLine += line.slice(5).trim();
            }
            if (dataLine) {
              try {
                onEvent(eventName, JSON.parse(dataLine));
              } catch (e) {
                onEvent(eventName, dataLine);
              }
            }
          }
        }
      })
      .catch((err) => {
        if (onError) onError(err);
      });
  },
};
