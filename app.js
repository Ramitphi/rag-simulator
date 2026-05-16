const sampleDoc = `RAG means retrieval augmented generation. It is a pattern for answering with help from external documents instead of relying only on what a language model already knows.

First, a document is ingested. The system may remove repeated spaces, normalize punctuation, remove irrelevant boilerplate, and split long text into smaller chunks. Chunks should be large enough to preserve meaning but small enough to retrieve precisely.

Each chunk is converted into a vector embedding. An embedding is a list of numbers that roughly captures the meaning of the text. Similar ideas should land near each other in vector space, even if they do not use the exact same words.

The embeddings are stored in a vector database along with the original chunk text and metadata such as document name, page number, date, or section. The vector database is used for similarity search.

When a user asks a question, the query is also embedded. The system compares the query vector with the stored chunk vectors and retrieves the most relevant chunks. Some systems rerank the results, filter by metadata, or rewrite the query before searching.

The language model receives the original user question and the retrieved text chunks. The embedding itself is usually not sent to the model. The model uses the retrieved context to generate an answer that is grounded in the source material.

Good RAG systems also handle weak retrieval. If no chunk is relevant, they should say that the context is missing instead of inventing an answer. They may show citations so a human can inspect the evidence.`;

const stages = [
  { id: "clean", name: "Clean", code: "CLN", detail: "text normalized", accent: "#207a7d" },
  { id: "chunk", name: "Chunk", code: "CHK", detail: "document split", accent: "#2e7b55" },
  { id: "embed", name: "Embed", code: "EMB", detail: "meaning as vectors", accent: "#345f9a" },
  { id: "store", name: "Store", code: "VDB", detail: "vector DB loaded", accent: "#d7a335" },
  { id: "retrieve", name: "Retrieve", code: "TOP", detail: "best chunks found", accent: "#b64a3a" },
  { id: "answer", name: "Answer", code: "LLM", detail: "grounded response", accent: "#7256a8" }
];

const spinSymbols = [
  { name: "Token", code: "TOK", detail: "raw words", accent: "#207a7d" },
  { name: "Noise", code: "NOI", detail: "messy input", accent: "#6f6b63" },
  { name: "Chunk", code: "CHK", detail: "text slice", accent: "#2e7b55" },
  { name: "Vector", code: "VEC", detail: "number map", accent: "#345f9a" },
  { name: "Index", code: "IDX", detail: "stored memory", accent: "#d7a335" },
  { name: "Match", code: "MAT", detail: "similarity hit", accent: "#b64a3a" },
  { name: "Answer", code: "ANS", detail: "final payout", accent: "#7256a8" }
];

const vocabulary = [
  ["rag", "retrieval", "retrieve", "retrieved", "search", "similarity"],
  ["document", "text", "source", "page", "section", "metadata"],
  ["clean", "normalize", "remove", "punctuation", "boilerplate", "spaces"],
  ["chunk", "chunks", "split", "smaller", "overlap", "meaning"],
  ["embedding", "embeddings", "vector", "vectors", "numbers", "space"],
  ["database", "stored", "store", "db", "index"],
  ["query", "question", "asks", "user", "rewrite"],
  ["model", "llm", "answer", "generate", "grounded", "citations", "context"]
];

const state = {
  score: 0,
  cleaned: "",
  chunks: [],
  vectors: [],
  storedRecords: [],
  queryVector: [],
  ranked: [],
  answer: "",
  activeStage: "",
  completed: new Set(),
  isAnimating: false
};

const el = {
  reels: document.getElementById("reels"),
  machineStatus: document.getElementById("machineStatus"),
  winLineText: document.getElementById("winLineText"),
  sampleBtn: document.getElementById("sampleBtn"),
  resetBtn: document.getElementById("resetBtn"),
  chunkBtn: document.getElementById("chunkBtn"),
  embedBtn: document.getElementById("embedBtn"),
  storeBtn: document.getElementById("storeBtn"),
  retrieveBtn: document.getElementById("retrieveBtn"),
  answerBtn: document.getElementById("answerBtn"),
  documentInput: document.getElementById("documentInput"),
  queryInput: document.getElementById("queryInput"),
  chunkSize: document.getElementById("chunkSize"),
  chunkOverlap: document.getElementById("chunkOverlap"),
  topK: document.getElementById("topK"),
  chunkSizeValue: document.getElementById("chunkSizeValue"),
  chunkOverlapValue: document.getElementById("chunkOverlapValue"),
  topKValue: document.getElementById("topKValue"),
  wordCount: document.getElementById("wordCount"),
  contextScore: document.getElementById("contextScore"),
  recordCount: document.getElementById("recordCount"),
  queryStatus: document.getElementById("queryStatus"),
  answerBox: document.getElementById("answerBox"),
  retrievalList: document.getElementById("retrievalList"),
  steps: document.getElementById("steps")
};

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[char]);
}

function words(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

function cleanText(text) {
  return text
    .replace(/\r/g, "\n")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .trim();
}

function chunkText(text, size, overlap) {
  const allWords = text.split(/\s+/).filter(Boolean);
  const chunks = [];
  const step = Math.max(1, size - overlap);
  let start = 0;

  while (start < allWords.length) {
    const slice = allWords.slice(start, start + size);
    chunks.push({
      id: `chunk-${chunks.length + 1}`,
      start,
      end: Math.min(start + size, allWords.length),
      text: slice.join(" ")
    });
    start += step;
  }

  return chunks;
}

function vectorize(text) {
  const list = words(text);
  const vector = vocabulary.map(group => {
    const count = list.reduce((sum, token) => sum + (group.includes(token) ? 1 : 0), 0);
    return count / Math.max(1, list.length);
  });
  const length = Math.hypot(...vector) || 1;
  return vector.map(value => value / length);
}

function cosine(a, b) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function importantTerms(text) {
  const stop = new Set("the a an and or but to of in is are with for from into then than it as on what how does use uses user query question".split(" "));
  const counts = {};
  words(text).forEach(word => {
    if (!stop.has(word) && word.length > 2) counts[word] = (counts[word] || 0) + 1;
  });
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([word]) => word);
}

function makeAnswer(query, ranked) {
  if (!ranked.length || ranked[0].score < 0.05) {
    return "No context jackpot this round. The retrieved chunks are too weak, so a careful RAG system should say the source material is missing instead of making up an answer.";
  }

  const terms = importantTerms(query);
  const sentences = ranked
    .flatMap(item => item.text.match(/[^.!?]+[.!?]?/g) || [item.text])
    .map(sentence => sentence.trim())
    .filter(Boolean);
  const picked = [];

  sentences.forEach(sentence => {
    const lower = sentence.toLowerCase();
    const matchesQuery = terms.some(term => lower.includes(term));
    if ((matchesQuery || picked.length < 2) && !picked.includes(sentence)) {
      picked.push(sentence);
    }
  });

  return `${picked.slice(0, 4).join(" ")}\n\nGrounding payout: ${ranked.map(item => item.id).join(", ")}.`;
}

function symbolTemplate(symbol) {
  return `
    <div class="reel-symbol">
      <span class="symbol-icon" style="--accent:${symbol.accent}">${symbol.code}</span>
      <span class="symbol-name">${symbol.name}</span>
      <span class="symbol-detail">${symbol.detail}</span>
    </div>
  `;
}

function randomSymbol() {
  return spinSymbols[Math.floor(Math.random() * spinSymbols.length)];
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function vectorBars(vector) {
  return `
    <div class="mini-vector">
      ${vector.map(value => `<span style="height:${Math.max(3, value * 28)}px"></span>`).join("")}
    </div>
  `;
}

function resetAllProgress() {
  state.score = 0;
  state.cleaned = "";
  state.chunks = [];
  state.vectors = [];
  state.storedRecords = [];
  state.queryVector = [];
  state.ranked = [];
  state.answer = "";
  state.activeStage = "";
  state.completed = new Set();
}

function resetAfterChunks() {
  state.vectors = [];
  state.storedRecords = [];
  state.queryVector = [];
  state.ranked = [];
  state.answer = "";
  state.score = 0;
  state.completed.delete("embed");
  state.completed.delete("store");
  state.completed.delete("retrieve");
  state.completed.delete("answer");
}

function resetAfterEmbeddings() {
  state.storedRecords = [];
  state.queryVector = [];
  state.ranked = [];
  state.answer = "";
  state.score = 0;
  state.completed.delete("store");
  state.completed.delete("retrieve");
  state.completed.delete("answer");
}

function resetAfterStore() {
  state.queryVector = [];
  state.ranked = [];
  state.answer = "";
  state.score = 0;
  state.completed.delete("retrieve");
  state.completed.delete("answer");
}

function resetAfterQuery() {
  state.queryVector = [];
  state.ranked = [];
  state.answer = "";
  state.score = 0;
  state.completed.delete("retrieve");
  state.completed.delete("answer");
}

function renderReels() {
  el.reels.innerHTML = stages.map((stage, index) => {
    const isDone = state.completed.has(stage.id);
    const isActive = state.activeStage === stage.id;
    const symbol = isDone || isActive ? stage : { name: "Locked", code: "???", detail: "waiting", accent: "#6f6b63" };
    return `
      <div class="reel ${isDone ? "done" : ""} ${isActive ? "active" : ""}" id="reel-${index}">
        <div class="reel-window">${symbolTemplate(symbol)}</div>
        <div class="reel-label">${stage.name}</div>
      </div>
    `;
  }).join("");
}

async function animateStage(stageIds) {
  const ids = Array.isArray(stageIds) ? stageIds : [stageIds];
  state.isAnimating = true;
  renderControls();

  const indexes = ids.map(id => stages.findIndex(stage => stage.id === id)).filter(index => index >= 0);
  const intervals = indexes.map(index => {
    const reel = document.getElementById(`reel-${index}`);
    reel.classList.add("spinning");
    return window.setInterval(() => {
      reel.querySelector(".reel-window").innerHTML = symbolTemplate(randomSymbol());
    }, 80 + index * 15);
  });

  await sleep(520);

  indexes.forEach((index, offset) => {
    window.clearInterval(intervals[offset]);
    const stage = stages[index];
    const reel = document.getElementById(`reel-${index}`);
    reel.classList.remove("spinning");
    reel.querySelector(".reel-window").innerHTML = symbolTemplate(stage);
  });

  state.isAnimating = false;
  renderAll();
}

async function generateChunks() {
  state.cleaned = cleanText(el.documentInput.value);
  state.chunks = chunkText(state.cleaned, Number(el.chunkSize.value), Number(el.chunkOverlap.value));
  resetAfterChunks();
  state.completed.add("clean");
  state.completed.add("chunk");
  state.activeStage = "chunk";
  el.machineStatus.textContent = "Chunks generated";
  await animateStage(["clean", "chunk"]);
}

async function generateEmbeddings() {
  if (!state.chunks.length) return;
  state.vectors = state.chunks.map(chunk => ({ ...chunk, vector: vectorize(chunk.text) }));
  resetAfterEmbeddings();
  state.completed.add("embed");
  state.activeStage = "embed";
  el.machineStatus.textContent = "Embeddings generated";
  await animateStage("embed");
}

async function storeVectorDb() {
  if (!state.vectors.length) return;
  state.storedRecords = state.vectors.map(chunk => ({
    id: chunk.id,
    metadata: `words ${chunk.start}-${chunk.end}`,
    text: chunk.text,
    vector: chunk.vector
  }));
  resetAfterStore();
  state.completed.add("store");
  state.activeStage = "store";
  el.machineStatus.textContent = "Vector DB loaded";
  await animateStage("store");
}

async function retrieveChunks() {
  if (!state.storedRecords.length) return;
  state.queryVector = vectorize(el.queryInput.value);
  state.ranked = state.storedRecords
    .map(record => ({ ...record, score: cosine(state.queryVector, record.vector) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, Number(el.topK.value));
  state.answer = "";
  state.score = Math.round((state.ranked[0]?.score || 0) * 100);
  state.completed.add("retrieve");
  state.completed.delete("answer");
  state.activeStage = "retrieve";
  el.machineStatus.textContent = "Relevant chunks retrieved";
  await animateStage("retrieve");
}

async function generateAnswer() {
  if (!state.ranked.length) return;
  state.answer = makeAnswer(el.queryInput.value, state.ranked);
  state.completed.add("answer");
  state.activeStage = "answer";
  el.machineStatus.textContent = "Answer generated";
  await animateStage("answer");
}

function renderRetrieval() {
  if (!state.ranked.length) {
    el.retrievalList.innerHTML = `<div class="empty">After storing vectors, enter a query and tap Retrieve Relevant Chunks.</div>`;
    return;
  }

  el.retrievalList.innerHTML = state.ranked.map(item => `
    <div class="retrieval-card">
      <div class="retrieval-top">
        <span>${item.id}</span>
        <span>${(item.score * 100).toFixed(1)}%</span>
      </div>
      <div class="meter"><span style="--w:${Math.max(2, item.score * 100)}%"></span></div>
      <div class="chunk-text">${escapeHtml(item.text)}</div>
    </div>
  `).join("");
}

function renderSteps() {
  const cleanPreview = state.cleaned
    ? `${words(state.cleaned).length} clean words. Preview: ${escapeHtml(state.cleaned.slice(0, 180))}${state.cleaned.length > 180 ? "..." : ""}`
    : "Tap Generate Chunks to clean the document first.";
  const chunkPreview = state.chunks.length
    ? state.chunks.slice(0, 3).map(chunk => `${chunk.id}: ${escapeHtml(chunk.text.slice(0, 95))}${chunk.text.length > 95 ? "..." : ""}`).join("<br>")
    : "No chunks yet.";
  const vectorPreview = state.vectors.length
    ? state.vectors.slice(0, 3).map(chunk => `${chunk.id} ${vectorBars(chunk.vector)}`).join("")
    : "No embeddings yet.";
  const storePreview = state.storedRecords.length
    ? `${state.storedRecords.length} vector records are stored with chunk text, metadata, and embeddings.`
    : "Vector DB is empty.";
  const retrievePreview = state.ranked.length
    ? `Query embedded and compared with stored vectors. Best match: ${state.ranked[0].id} at ${(state.ranked[0].score * 100).toFixed(1)}%.`
    : "No chunks retrieved yet.";
  const answerPreview = state.answer
    ? "The LLM receives the original query and retrieved chunks, then writes the grounded answer shown in the payout card."
    : "No answer generated yet.";

  const stepBodies = {
    clean: cleanPreview,
    chunk: chunkPreview,
    embed: vectorPreview,
    store: storePreview,
    retrieve: retrievePreview,
    answer: answerPreview
  };

  el.steps.innerHTML = stages.map((stage, index) => {
    const done = state.completed.has(stage.id);
    return `
      <div class="step-card ${done ? "done" : ""}">
        <div class="step-top">
          <span class="step-number" style="--accent:${stage.accent}">${index + 1}</span>
          <span class="step-name">${stage.name}</span>
          <span class="symbol-detail">${done ? "complete" : stage.detail}</span>
        </div>
        <div class="step-body">${stepBodies[stage.id]}</div>
      </div>
    `;
  }).join("");
}

function renderControls() {
  el.chunkBtn.disabled = state.isAnimating;
  el.embedBtn.disabled = state.isAnimating || !state.chunks.length;
  el.storeBtn.disabled = state.isAnimating || !state.vectors.length;
  el.retrieveBtn.disabled = state.isAnimating || !state.storedRecords.length;
  el.answerBtn.disabled = state.isAnimating || !state.ranked.length;

  el.chunkBtn.classList.toggle("done", state.completed.has("chunk"));
  el.embedBtn.classList.toggle("done", state.completed.has("embed"));
  el.storeBtn.classList.toggle("done", state.completed.has("store"));
  el.retrieveBtn.classList.toggle("done", state.completed.has("retrieve"));
  el.answerBtn.classList.toggle("done", state.completed.has("answer"));
}

function renderStats() {
  el.wordCount.textContent = `${words(el.documentInput.value).length} words`;
  el.chunkSizeValue.textContent = el.chunkSize.value;
  el.chunkOverlapValue.textContent = el.chunkOverlap.value;
  el.topKValue.textContent = el.topK.value;
  el.contextScore.textContent = state.score;
  el.recordCount.textContent = state.storedRecords.length;
  el.queryStatus.textContent = state.completed.has("answer")
    ? "answered"
    : state.completed.has("retrieve")
      ? "retrieved"
      : state.storedRecords.length
        ? "ready"
        : "waiting";
}

function renderWinLine() {
  if (state.completed.has("answer")) {
    el.winLineText.textContent = state.score >= 70 ? "Context jackpot: strong grounded answer" : "Answer payout: inspect the retrieved chunks";
  } else if (state.completed.has("retrieve")) {
    el.winLineText.textContent = "Chunks retrieved. Generate the answer next.";
  } else if (state.completed.has("store")) {
    el.winLineText.textContent = "Vector DB ready. Enter a query and retrieve.";
  } else if (state.completed.has("embed")) {
    el.winLineText.textContent = "Embeddings ready. Store them in the vector DB.";
  } else if (state.completed.has("chunk")) {
    el.winLineText.textContent = "Chunks ready. Generate embeddings next.";
  } else {
    el.winLineText.textContent = "Tap Generate Chunks to start the RAG run.";
  }
}

function renderAll() {
  renderStats();
  renderControls();
  renderReels();
  renderRetrieval();
  renderSteps();
  renderWinLine();
  el.answerBox.textContent = state.answer || "After retrieving chunks, tap Generate Answer to create the final grounded response.";
}

function resetGame() {
  resetAllProgress();
  el.machineStatus.textContent = "Ready to generate chunks";
  renderAll();
}

function boot() {
  el.documentInput.value = sampleDoc;
  renderAll();

  el.chunkBtn.addEventListener("click", generateChunks);
  el.embedBtn.addEventListener("click", generateEmbeddings);
  el.storeBtn.addEventListener("click", storeVectorDb);
  el.retrieveBtn.addEventListener("click", retrieveChunks);
  el.answerBtn.addEventListener("click", generateAnswer);
  el.resetBtn.addEventListener("click", resetGame);
  el.sampleBtn.addEventListener("click", () => {
    el.documentInput.value = sampleDoc;
    resetGame();
  });

  el.documentInput.addEventListener("input", resetGame);
  el.chunkSize.addEventListener("input", () => {
    resetAllProgress();
    renderAll();
  });
  el.chunkOverlap.addEventListener("input", () => {
    resetAllProgress();
    renderAll();
  });
  el.topK.addEventListener("input", () => {
    resetAfterQuery();
    renderAll();
  });
  el.queryInput.addEventListener("input", () => {
    resetAfterQuery();
    renderAll();
  });
}

boot();
