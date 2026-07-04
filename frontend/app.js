const form = document.querySelector("#ask-form");
const queryInput = document.querySelector("#query");
const modeInput = document.querySelector("#mode");
const topKInput = document.querySelector("#top-k");
const submitButton = document.querySelector("#submit-button");
const answerTitle = document.querySelector("#answer-title");
const answerBox = document.querySelector("#answer");
const sourcesBox = document.querySelector("#sources");
const statusPill = document.querySelector("#status-pill");

function setStatus(text, state = "idle") {
  statusPill.textContent = text;
  statusPill.className = `pill ${state === "idle" ? "" : state}`.trim();
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => {
    const escapes = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    };
    return escapes[char];
  });
}

function renderSources(sources) {
  if (!sources.length) {
    sourcesBox.className = "sources empty";
    sourcesBox.textContent = "No sources returned.";
    return;
  }

  sourcesBox.className = "sources";
  sourcesBox.innerHTML = sources
    .map((source, index) => {
      const page = source.page === null ? "" : `, page ${source.page + 1}`;
      return `
        <article class="source">
          <strong>${index + 1}. ${escapeHtml(source.source)}${page}</strong>
          <p>${escapeHtml(source.preview)}...</p>
        </article>
      `;
    })
    .join("");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const query = queryInput.value.trim();
  if (!query) {
    queryInput.focus();
    return;
  }

  submitButton.disabled = true;
  answerTitle.textContent = "Thinking through the retrieval...";
  answerBox.className = "answer";
  answerBox.textContent =
    "Loading relevant chunks and asking the fine-tuned model. The first request can take a while because the model loads into memory.";
  sourcesBox.className = "sources empty";
  sourcesBox.textContent = "Waiting for retrieval results...";
  setStatus("Working", "loading");

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        mode: modeInput.value,
        top_k: Number(topKInput.value),
        max_new_tokens: modeInput.value === "summary" ? 280 : 520,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "The tutor could not answer right now.");
    }

    answerTitle.textContent =
      modeInput.value === "summary" ? "Summary answer" : "Raw answer";
    answerBox.textContent = payload.answer;
    renderSources(payload.sources || []);
    setStatus("Complete");
  } catch (error) {
    answerTitle.textContent = "Something blocked the answer.";
    answerBox.textContent = error.message;
    sourcesBox.className = "sources empty";
    sourcesBox.textContent = "No sources available because the request failed.";
    setStatus("Error", "error");
  } finally {
    submitButton.disabled = false;
  }
});
