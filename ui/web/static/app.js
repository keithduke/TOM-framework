const state = {
  session: null,
  loading: false,
};

const els = {};
const streamNodes = {};

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
  clearStreamingMarkers();

  try {
    const session = await ensureSession();
    els.messageInput.value = "";
    await streamChat(session.session_id, message, els.runTools.checked);
  } catch (err) {
    showError(err.message || "Request failed");
  } finally {
    setLoading(false);
  }
}

async function streamChat(sessionId, content, runTools) {
  const payload = {
    content,
    run_tools: runTools,
  };

  const response = await fetch(`/sessions/${sessionId}/chat/stream`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });

  if (response.status === 404) {
    const fallback = await apiFetch(
      `/sessions/${sessionId}/chat`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
    );
    applyChatResponse(fallback);
    return;
  }

  if (!response.ok || !response.body) {
    throw new Error("Streaming request failed");
  }

  await readSseStream(response.body, handleStreamEvent);
}

function applyChatResponse(data) {
  state.session = data.session;
  updateSessionMeta(data.session);
  renderConversation(data.session.messages, data);
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
  body.className = "message-body";
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

async function readSseStream(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      buffer += decoder.decode();
      break;
    }
    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (!chunk.trim()) {
        continue;
      }

      const lines = chunk.split("\n");
      let event = "message";
      const dataLines = [];

      for (const line of lines) {
        if (line.startsWith("event:")) {
          event = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trim());
        }
      }

      if (dataLines.length) {
        const raw = dataLines.join("\n");
        let payload;
        try {
          payload = JSON.parse(raw);
        } catch {
          payload = { content: raw };
        }
        onEvent(event, payload);
      }
    }
  }

  if (buffer.trim()) {
    const lines = buffer.split("\n");
    let event = "message";
    const dataLines = [];
    for (const line of lines) {
      if (line.startsWith("event:")) {
        event = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }
    if (dataLines.length) {
      const raw = dataLines.join("\n");
      let payload;
      try {
        payload = JSON.parse(raw);
      } catch {
        payload = { content: raw };
      }
      onEvent(event, payload);
    }
  }
}

function handleStreamEvent(event, payload) {
  switch (event) {
    case "thinking": {
      const text = payload.delta || payload.content || "";
      if (text) {
        appendStreamingText("thinking", text, "thinking", { prefix: "💭 " });
      }
      break;
    }
    case "tool_call": {
      const name = payload.name || "tool";
      appendStreamingBlock("tool", `Running ${name}…`, "tool");
      break;
    }
    case "tool_result": {
      const name = payload.name || "tool";
      const output = payload.output || "";
      appendStreamingBlock("tool", `${name}\n${output}`, "tool");
      break;
    }
    case "assistant": {
      const text = payload.delta || payload.content || "";
      if (text) {
        const cls = payload.final ? "assistant-final" : "assistant-draft";
        appendStreamingText("assistant", text, cls, { prefix: "T.O.M.: " });
      }
      if (payload.final) {
        delete streamNodes.assistant;
      }
      break;
    }
    case "final": {
      if (payload.session) {
        state.session = payload.session;
        updateSessionMeta(payload.session);
        clearStreamingMarkers();
        renderConversation(payload.session.messages, {
          thinking: payload.thinking,
          response: payload.response,
          tool_calls: payload.tool_calls,
        });
      }
      break;
    }
    case "error":
      showError(payload.message || "Server error");
      break;
    default:
      break;
  }
}

function appendStreamingBlock(role, content, extraClass) {
  if (!content) {
    return;
  }
  const node = buildMessageNode(role, content);
  node.classList.add("streaming-temp");
  if (extraClass) {
    node.classList.add(extraClass);
  }
  els.conversation.appendChild(node);
  els.conversation.scrollTop = els.conversation.scrollHeight;
}

function appendStreamingText(role, text, extraClass, options = {}) {
  if (!text) {
    return;
  }
  let node = streamNodes[role];
  if (!node) {
    node = buildMessageNode(role, "");
    node.classList.add("streaming-temp");
    if (extraClass) {
      node.classList.add(extraClass);
    }
    const bodyEl = node.querySelector(".message-body") || node;
    if (options.prefix) {
      bodyEl.textContent = options.prefix;
    }
    els.conversation.appendChild(node);
    streamNodes[role] = node;
  }
  const body = node.querySelector(".message-body") || node;
  body.textContent += text;
  els.conversation.scrollTop = els.conversation.scrollHeight;
}

function clearStreamingMarkers() {
  const temps = els.conversation.querySelectorAll(".streaming-temp");
  temps.forEach((node) => node.remove());
  for (const key of Object.keys(streamNodes)) {
    delete streamNodes[key];
  }
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
