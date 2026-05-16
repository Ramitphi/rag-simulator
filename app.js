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
  spins: 0,
  score: 0,
  cleaned: "",
  chunks: [],
  vectors: [],
  queryVector: [],
  ranked: [],
  answer: "",
  isSpinning: false
};

const el = {
  reels: document.getElementById("reels"),
  machineStatus: document.getElementById("machineStatus"),
  winLineText: document.getElementById("winLineText"),
  spinBtn: document.getElementById("spinBtn"),
  quickSpinBtn: document.getElementById("quickSpinBtn"),
  sampleBtn: document.getElementById("sampleBtn"),
  resetBtn: document.getElementById("resetBtn"),
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
  spinCount: document.getElementById("spinCount"),
  answerBox: document.getElementById("answerBox"),
  retrievalList: document.getElementById("retrievalList"),
  steps: document.getElementById("steps")
};

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, char => ({
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
    return "No context jackpot this spin. The retrieved chunks are too weak, so a careful RAG system should say the source material is missing instead of making up an answer.";
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

function runPipeline() {
  state.cleaned = cleanText(el.documentInput.value);
  state.chunks = chunkText(state.cleaned, Number(el.chunkSize.value), Number(el.chunkOverlap.value));
  state.vectors = state.chunks.map(chunk => ({ ...chunk, vector: vectorize(chunk.text) }));
  state.queryVector = vectorize(el.queryInput.value);
  state.ranked = state.vectors
    .map(chunk => ({ ...chunk, score: cosine(state.queryVector, chunk.vector) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, Number(el.topK.value));
  state.answer = makeAnswer(el.queryInput.value, state.ranked);
  state.score = Math.round((state.ranked[0]?.score || 0) * 100);
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

function renderReels(symbols = stages) {
  el.reels.innerHTML = stages.map((stage, index) => {
    const symbol = symbols[index] || stage;
    return `
      <div class="reel" id="reel-${index}">
        <div class="reel-window">${symbolTemplate(symbol)}</div>
        <div class="reel-label">${stage.name}</div>
      </div>
    `;
  }).join("");
}

function vectorBars(vector) {
  return `
    <div class="mini-vector">
      ${vector.map(value => `<span style="height:${Math.max(3, value * 28)}px"></span>`).join("")}
    </div>
  `;
}

function renderRetrieval() {
  if (!state.ranked.length) {
    el.retrievalList.innerHTML = `<div class="empty">The payline will show the chunks with the strongest similarity scores.</div>`;
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
  const chunkPreview = state.chunks.slice(0, 3).map(chunk => `${chunk.id}: ${chunk.text.slice(0, 95)}${chunk.text.length > 95 ? "..." : ""}`).join("<br>");
  const vectorPreview = state.vectors.slice(0, 3).map(chunk => `${chunk.id} ${vectorBars(chunk.vector)}`).join("");
  const promptPreview = state.ranked.map(item => `[${item.id}] ${item.text.slice(0, 130)}${item.text.length > 130 ? "..." : ""}`).join("<br><br>");
  const stepBodies = {
    clean: `${words(state.cleaned).length} clean words. Extra spaces and noisy line breaks are normalized before retrieval starts.`,
    chunk: chunkPreview || "No chunks yet.",
    embed: vectorPreview || "No embeddings yet.",
    store: `${state.vectors.length} vector records stored with chunk text and simple metadata.`,
    retrieve: state.ranked.length ? `The query vector is compared against stored vectors. Best match: ${state.ranked[0].id} at ${(state.ranked[0].score * 100).toFixed(1)}%.` : "No retrieval yet.",
    answer: promptPreview ? `The LLM receives the original question plus retrieved text, not the raw embedding numbers.<br><br>${promptPreview}` : "No prompt yet."
  };

  el.steps.innerHTML = stages.map((stage, index) => `
    <div class="step-card">
      <div class="step-top">
        <span class="step-number" style="--accent:${stage.accent}">${index + 1}</span>
        <span class="step-name">${stage.name}</span>
        <span class="symbol-detail">${stage.detail}</span>
      </div>
      <div class="step-body">${stepBodies[stage.id]}</div>
    </div>
  `).join("");
}

function renderStats() {
  const wordTotal = words(el.documentInput.value).length;
  el.wordCount.textContent = `${wordTotal} words`;
  el.chunkSizeValue.textContent = el.chunkSize.value;
  el.chunkOverlapValue.textContent = el.chunkOverlap.value;
  el.topKValue.textContent = el.topK.value;
  el.contextScore.textContent = state.score;
  el.spinCount.textContent = state.spins;
}

function renderResults() {
  renderStats();
  el.answerBox.textContent = state.answer || "Pull the lever to generate a grounded answer.";
  renderRetrieval();
  renderSteps();

  if (!state.spins) {
    el.winLineText.textContent = "Match the pipeline to win a grounded answer";
  } else if (state.score >= 70) {
    el.winLineText.textContent = "Context jackpot: strong retrieval";
  } else if (state.score >= 35) {
    el.winLineText.textContent = "Partial match: useful context, check citations";
  } else {
    el.winLineText.textContent = "Low match: answer should be cautious";
  }
}

function randomSymbol() {
  return spinSymbols[Math.floor(Math.random() * spinSymbols.length)];
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function spin() {
  if (state.isSpinning) return;
  state.isSpinning = true;
  state.spins += 1;
  el.machineStatus.textContent = "Reels spinning";
  el.spinBtn.disabled = true;
  el.quickSpinBtn.disabled = true;

  const intervals = stages.map((_, index) => {
    const reel = document.getElementById(`reel-${index}`);
    reel.classList.add("spinning");
    return window.setInterval(() => {
      reel.querySelector(".reel-window").innerHTML = symbolTemplate(randomSymbol());
    }, 85 + index * 12);
  });

  runPipeline();

  for (let index = 0; index < stages.length; index += 1) {
    await sleep(360 + index * 90);
    window.clearInterval(intervals[index]);
    const reel = document.getElementById(`reel-${index}`);
    reel.classList.remove("spinning");
    reel.querySelector(".reel-window").innerHTML = symbolTemplate(stages[index]);
  }

  el.machineStatus.textContent = state.score >= 70 ? "Context jackpot" : "Spin complete";
  state.isSpinning = false;
  el.spinBtn.disabled = false;
  el.quickSpinBtn.disabled = false;
  renderResults();
}

function resetGame() {
  state.spins = 0;
  state.score = 0;
  state.cleaned = "";
  state.chunks = [];
  state.vectors = [];
  state.queryVector = [];
  state.ranked = [];
  state.answer = "";
  el.machineStatus.textContent = "Ready for retrieval";
  renderReels();
  renderResults();
}

function boot() {
  el.documentInput.value = sampleDoc;
  renderReels();
  runPipeline();
  renderResults();

  el.spinBtn.addEventListener("click", spin);
  el.quickSpinBtn.addEventListener("click", spin);
  el.sampleBtn.addEventListener("click", () => {
    el.documentInput.value = sampleDoc;
    spin();
  });
  el.resetBtn.addEventListener("click", resetGame);

  [el.documentInput, el.queryInput, el.chunkSize, el.chunkOverlap, el.topK].forEach(input => {
    input.addEventListener("input", () => {
      runPipeline();
      renderResults();
    });
  });
}

boot();
