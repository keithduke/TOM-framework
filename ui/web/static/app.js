const state = {
  session: null,
  loading: false,
};

const els = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  bindEvents();
  refreshHealth();
  setInterval(refreshHealth, 15000);
});

function cacheElements() {
  els.statusDot = document.getElementById("status-dot");
  els.statusText = document.getElementById("status-text");
  els.newSessionBtn = document.getElementById("new-session");
  els.sessionId = document.getElementById("session-id");
  els.sessionContext = document.getElementById("session-context");
  els.conversation = document.getElementById("conversation");
  els.chatForm = document.getElementById("chat-form");
  els.messageInput = document.getElementById("message-input");
  els.runTools = document.getElementById("run-tools");
  els.errorBanner = document.getElementById("error-banner");
}

function bindEvents() {
  els.newSessionBtn.addEventListener("click", () => {
    state.session = null;
    updateSessionMeta();
    renderConversation();
  });

  els.chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await handleSend();
  });
}

async function refreshHealth() {
  setStatus("Checking backend…", false);
  try {
    const data = await apiFetch("/health");
    const text = data.model_path
      ? `Model ready – ${data.sessions} session(s)`
      : "Online";
    setStatus(text, true);
  } catch (err) {
    setStatus("Offline – start python main.py", false);
  }
}

async function ensureSession() {
  if (state.session) {
    return state.session;
  }
  const session = await apiFetch("/sessions", {
    method: "POST",
    body: JSON.stringify({}),
  });
  state.session = session;
  updateSessionMeta(session);
  renderConversation(session.messages);
  return session;
}

async function handleSend() {
  const message = (els.messageInput.value || "").trim();
  if (!message || state.loading) {
    return;
  }
  showError("");
  setLoading(true);

  try {
    const session = await ensureSession();
    const payload = {
      content: message,
      run_tools: els.runTools.checked,
    };
    const response = await apiFetch(
      `/sessions/${session.session_id}/chat`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
    );
    state.session = response.session;
    updateSessionMeta(response.session);
    els.messageInput.value = "";
    renderConversation(response.session.messages, response);
  } catch (err) {
    showError(err.message || "Request failed");
  } finally {
    setLoading(false);
  }
}

function updateSessionMeta(session = null) {
  if (!session) {
    els.sessionId.textContent = "not created";
    els.sessionContext.textContent = "—";
    return;
  }
  els.sessionId.textContent = session.session_id;
  els.sessionContext.textContent = `${session.max_context_tokens} tokens`;
}

function renderConversation(messages = [], extra = null) {
  els.conversation.innerHTML = "";

  if (!messages.length) {
    const empty = document.createElement("div");
    empty.className = "message system";
    empty.textContent = "No messages yet. Ask something to begin.";
    els.conversation.appendChild(empty);
  } else {
    messages.forEach((msg) => {
      els.conversation.appendChild(buildMessageNode(msg.role, msg.content));
    });
  }

  if (extra?.response) {
    const last = messages[messages.length - 1];
    const lastContent = last?.content || "";
    const lastRole = (last?.role || "").toLowerCase();
    if (lastContent !== extra.response || lastRole !== "assistant") {
      els.conversation.appendChild(buildMessageNode("assistant", extra.response));
    }
  }

  if (extra?.thinking || (extra?.tool_calls?.length ?? 0) > 0) {
    const meta = document.createElement("div");
    meta.className = "message assistant";

    if (extra.thinking) {
      const thinkHeading = document.createElement("div");
      thinkHeading.className = "message-heading";
      thinkHeading.textContent = "Thinking";
      const thinkBlock = document.createElement("div");
      thinkBlock.className = "thinking-block";
      thinkBlock.textContent = extra.thinking.trim();
      meta.appendChild(thinkHeading);
      meta.appendChild(thinkBlock);
    }

    if (extra.tool_calls?.length) {
      const toolsHeading = document.createElement("div");
      toolsHeading.className = "message-heading";
      toolsHeading.textContent = "Tool calls";
      const list = document.createElement("ul");
      list.className = "tool-call-list";
      extra.tool_calls.forEach((call) => {
        const item = document.createElement("li");
        const args =
          call.arguments && Object.keys(call.arguments).length
            ? JSON.stringify(call.arguments)
            : "{}";
        item.textContent = `${call.name} ${args}\n→ ${call.output.substring(
          0,
          200,
        )}${call.output.length > 200 ? "…" : ""}`;
        list.appendChild(item);
      });
      meta.appendChild(toolsHeading);
      meta.appendChild(list);
    }

    els.conversation.appendChild(meta);
  }

  els.conversation.scrollTop = els.conversation.scrollHeight;
}

function buildMessageNode(role, content) {
  const div = document.createElement("div");
  const normalizedRole = (role || "assistant").toLowerCase();
  const roleClass = ["user", "assistant", "system", "tool"].includes(
    normalizedRole,
  )
    ? normalizedRole
    : "assistant";
  div.className = `message ${roleClass}`;

  const heading = document.createElement("div");
  heading.className = "message-heading";
  heading.textContent = normalizedRole;

  const body = document.createElement("div");
  body.textContent = content;

  div.appendChild(heading);
  div.appendChild(body);
  return div;
}

function setStatus(text, online) {
  els.statusText.textContent = text;
  els.statusDot.classList.toggle("online", online);
  els.statusDot.classList.toggle("offline", !online);
}

function showError(message) {
  els.errorBanner.textContent = message;
}

function setLoading(loading) {
  state.loading = loading;
  const button = els.chatForm.querySelector("button[type=submit]");
  button.disabled = loading;
}

async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    let detail = "";
    try {
      const payload = await response.json();
      detail = payload.detail || JSON.stringify(payload);
    } catch {
      detail = await response.text();
    }
    throw new Error(
      detail || `Request failed with status ${response.status}`,
    );
  }
  return response.json();
}
