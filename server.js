import { GoogleGenerativeAI } from "@google/generative-ai";
import fs from "fs";
import path from "path";
import { fileURLToPath, pathToFileURL } from "url";
import dotenv from "dotenv";
dotenv.config();
import express from "express";
import cors from "cors";
import rateLimit from "express-rate-limit";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const standardFontDataUrl =
  pathToFileURL(path.join(__dirname, "node_modules", "pdfjs-dist", "standard_fonts")).href + "/";

const app = express();

app.use((req, res, next) => { req.setTimeout(180000); res.setTimeout(180000); next(); });
app.use(rateLimit({ windowMs: 60 * 1000, max: 80 }));
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// ── DOCS DIR ────────────────────────────────────────────────────
const docsDir = path.join(__dirname, "docs");
if (!fs.existsSync(docsDir)) fs.mkdirSync(docsDir, { recursive: true });

// ── ADMIN SECRET (set ADMIN_SECRET=... in .env) ─────────────────
const ADMIN_SECRET = process.env.ADMIN_SECRET || null;

function isAdmin(req) {
  if (!ADMIN_SECRET) return false; // no secret set → admin routes disabled
  const h = req.headers["x-admin-secret"] || req.body?.adminSecret || req.query?.secret;
  return h === ADMIN_SECRET;
}

// ── GEMINI ──────────────────────────────────────────────────────
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// ── SESSION STORE ────────────────────────────────────────────────
// Each session: { history: [{role,content}], lastActive: timestamp }
// Sessions expire after 24h of inactivity and are cleaned every hour.
const SESSION_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours
const MAX_HISTORY    = 20;
const sessions       = {}; // sessionId → { history, lastActive }

function touchSession(id) {
  if (!sessions[id]) sessions[id] = { history: [], lastActive: Date.now() };
  else sessions[id].lastActive = Date.now();
  return sessions[id];
}

// Hourly cleanup of expired sessions
setInterval(() => {
  const now    = Date.now();
  let cleaned  = 0;
  for (const id of Object.keys(sessions)) {
    if (now - sessions[id].lastActive > SESSION_TTL_MS) {
      delete sessions[id];
      cleaned++;
    }
  }
  if (cleaned) console.log(`🧹 Cleaned ${cleaned} expired session(s). Active: ${Object.keys(sessions).length}`);
}, 60 * 60 * 1000);

// Hard cap — evict oldest if too many sessions accumulate
function evictIfNeeded() {
  const ids = Object.keys(sessions);
  if (ids.length <= 500) return;
  const sorted = ids.sort((a, b) => sessions[a].lastActive - sessions[b].lastActive);
  sorted.slice(0, ids.length - 400).forEach(id => delete sessions[id]);
}

// ── EMBEDDING ───────────────────────────────────────────────────
async function getEmbedding(text) {
  try {
    if (!text || typeof text !== "string") return null;
    const cleanText = text.replace(/\s+/g, " ").slice(0, 8000);
    if (!cleanText) return null;
    const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-001" });
    const result = await embeddingModel.embedContent(cleanText);
    return result.embedding.values || null;
  } catch (err) {
    console.log("❌ Embedding error:", err.message);
    return null;
  }
}

// ── TEXT SPLITTER ────────────────────────────────────────────────
function splitText(text, source, size = 500, overlap = 100) {
  const words  = text.split(/\s+/);
  const result = [];
  const step   = size - overlap;
  for (let i = 0; i < words.length; i += step) {
    const chunk = words.slice(i, i + size).join(" ");
    if (chunk.trim()) result.push({ text: chunk, source });
    if (i + size >= words.length) break;
  }
  return result;
}

// ── EXTRACT TEXT FROM ONE PDF ────────────────────────────────────
async function extractPdfText(pdfPath) {
  const data = new Uint8Array(fs.readFileSync(pdfPath));
  const pdf  = await pdfjsLib.getDocument({ data, standardFontDataUrl }).promise;
  let text    = "";
  for (let i = 1; i <= pdf.numPages; i++) {
    const page    = await pdf.getPage(i);
    const content = await page.getTextContent();
    text += content.items.map(item => item.str).filter(s => s?.trim()).join(" ") + "\n";
  }
  return text;
}

// ── LOAD PDFs ────────────────────────────────────────────────────
let chunks   = [];
let vectorDB = [];

async function loadPDFs() {
  try {
    console.log("📁 Loading PDFs from:", docsDir);
    if (!fs.existsSync(docsDir)) return;
    const files = fs.readdirSync(docsDir);
    chunks = [];
    for (const file of files) {
      if (!file.toLowerCase().endsWith(".pdf")) continue;
      const pdfPath = path.join(docsDir, file);
      console.log("📄 Loading:", file);
      try {
        const fileText = await extractPdfText(pdfPath);
        if (!fileText.trim()) { console.log(`⚠️ No text: ${file}`); continue; }
        const fileChunks = splitText(fileText, file.replace(/\.pdf$/i, ""), 500, 100);
        chunks.push(...fileChunks);
        console.log(`✅ ${file}: ${fileChunks.length} chunks`);
      } catch (e) {
        console.log(`❌ Failed ${file}:`, e.message);
      }
    }
    console.log(`✅ Total chunks: ${chunks.length}`);
  } catch (err) {
    console.log("❌ PDF ERROR:", err.message);
  }
}

// ── VECTOR DB CACHE ──────────────────────────────────────────────
const VECTOR_CACHE_PATH = path.join(__dirname, "vectordb_cache.json");

function computeDocFingerprint() {
  if (!fs.existsSync(docsDir)) return "empty";
  return fs.readdirSync(docsDir)
    .filter(f => f.toLowerCase().endsWith(".pdf"))
    .sort()
    .map(f => `${f}:${fs.statSync(path.join(docsDir, f)).size}`)
    .join("|");
}

async function buildVectorDB() {
  const currentKey = computeDocFingerprint();
  if (fs.existsSync(VECTOR_CACHE_PATH)) {
    try {
      const cached = JSON.parse(fs.readFileSync(VECTOR_CACHE_PATH, "utf-8"));
      if (cached.cacheKey === currentKey && Array.isArray(cached.vectorDB) && cached.vectorDB.length > 0) {
        vectorDB = cached.vectorDB;
        console.log(`✅ Vector DB from cache: ${vectorDB.length} entries`);
        return;
      }
      console.log("⚠️ Cache mismatch — rebuilding...");
    } catch (e) {
      console.log("⚠️ Cache read failed:", e.message);
    }
  }
  console.log("🔄 Building vector DB...");
  vectorDB = [];
  for (const chunk of chunks) {
    const embedding = await getEmbedding(chunk.text);
    if (!embedding) continue;
    vectorDB.push({ text: chunk.text, source: chunk.source, embedding });
    await new Promise(r => setTimeout(r, 100));
  }
  console.log(`✅ Vector DB ready: ${vectorDB.length} entries`);
  try {
    fs.writeFileSync(VECTOR_CACHE_PATH, JSON.stringify({ cacheKey: currentKey, chunkCount: chunks.length, vectorDB }));
    console.log("💾 Vector DB cached");
  } catch (e) {
    console.log("⚠️ Cache save failed:", e.message);
  }
}

// ── SIMILARITY SEARCH ────────────────────────────────────────────
function cosineSimilarity(a, b) {
  const dot   = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (normA * normB);
}

function applyMMR(candidates, topK, lambda = 0.6) {
  if (candidates.length <= topK) return candidates;
  const selected  = [candidates[0]];
  const remaining = candidates.slice(1);
  while (selected.length < topK && remaining.length > 0) {
    let bestIdx = 0, bestScore = -Infinity;
    for (let i = 0; i < remaining.length; i++) {
      const relevance = remaining[i].score;
      const maxSim    = Math.max(...selected.map(s => cosineSimilarity(remaining[i].embedding, s.embedding)));
      const mmrScore  = lambda * relevance - (1 - lambda) * maxSim;
      if (mmrScore > bestScore) { bestScore = mmrScore; bestIdx = i; }
    }
    selected.push(remaining.splice(bestIdx, 1)[0]);
  }
  return selected;
}

async function searchRelevantChunks(query, topK = 16) {
  if (vectorDB.length === 0) return chunks.slice(0, topK);
  const queryEmbedding = await getEmbedding(query);
  if (!queryEmbedding) return chunks.slice(0, topK);
  const scored = vectorDB
    .map(item => ({ text: item.text, source: item.source || "Document", score: cosineSimilarity(queryEmbedding, item.embedding), embedding: item.embedding }))
    .sort((a, b) => b.score - a.score);
  console.log("📊 Top 5:", scored.slice(0, 5).map(s => `${s.source}: ${s.score.toFixed(3)}`).join(", "));
  const candidatePool = scored.filter(i => i.score > 0.15).slice(0, topK * 2);
  const pool          = candidatePool.length >= 3 ? candidatePool : scored.slice(0, topK * 2);
  const diverse       = applyMMR(pool, topK, 0.6);
  return diverse.map(({ text, source, score }) => ({ text, source, score }));
}

function smartTruncateForEmbedding(text, maxWords = 500) {
  const words = text.trim().split(/\s+/);
  if (words.length <= maxWords) return text;
  return `${words.slice(0, 300).join(" ")} ... ${words.slice(-200).join(" ")}`;
}

// ── STARTUP ──────────────────────────────────────────────────────
await loadPDFs();
await buildVectorDB();

// ── SYSTEM PROMPT ────────────────────────────────────────────────
const systemPrompt = `You are Sayar Nyan Lin Aung's personal AI consultant — someone who genuinely knows partnership business, law, and strategy inside out. Think of yourself as a sharp, experienced friend who actually cares about helping people navigate this stuff. You are not a chatbot reading a script. You are not a legal textbook. You are a real consultant with real opinions and real knowledge.

HOW YOU TALK
Natural, direct, warm — like a real conversation. Not a formal report.

Good tone:
  "So here's the thing with that..."
  "Honestly, this comes up a lot — here's how I'd handle it..."
  "Yeah, that's a tricky one. Let me walk you through it..."
  "Short answer is X. Here's why that matters..."
  "Good news — this is actually solvable..."
  "Look, the honest truth is..."

Never sound like this:
  "According to established legal principles..." -- NO
  "It is imperative that you ensure compliance..." -- NO
  "One must consider the following ramifications..." -- NO
  "Great question! Certainly!" -- NO

When someone is dealing with a stressful situation — a bad partner, a dispute, a shaky agreement — acknowledge that first. Then help. People need to feel heard before they need advice.

LANGUAGE RULE — STRICT
Detect the user's language and reply ENTIRELY in that language. No mixing.
  English message → English reply only
  Burmese message → Burmese reply only
  Mixed → match whichever language dominates

YOUR EXPERTISE
Deep knowledge in:
  Partnership law — General Partnerships, LP, LLP, Myanmar Partnership Act, UPA, RUPA
  Partnership agreements — drafting, enforcing, renegotiating, disputing
  Fiduciary duties, partner liability, dissolution, winding up
  Capital contributions, profit and loss allocation, valuations, buy-sell agreements
  Sweat equity, vesting schedules, partner draws, tax treatment
  Governance, voting rights, deadlock mechanisms, non-competes, exit strategies
  SME best practices in Myanmar and Southeast Asia
  General business advice, strategy, and operations

You have live Google Search — use it for current information like tax rates, recent law changes, specific company data, 2025 news. Use it silently without announcing it.

ANSWER STYLE
  Get to the point — no padding, no filler intros
  Use numbered steps or bullets only when it genuinely helps
  Real-world examples beat abstract theory every time
  Match length to what the question needs — simple question gets direct answer
  Always finish your answer completely — never cut off
  End every response with one genuinely useful follow-up question

ABSOLUTE RULES — NEVER BREAK THESE:
  Never mention documents, files, uploaded content, reference materials, or any source
  Never say "According to...", "Based on the materials...", "The document says...", "I found..."
  Never use stiff formal legal-brief language
  Never be condescending or lecture the user
  Never start with filler: "Certainly!", "Of course!", "Absolutely!", "Great question!"`;

// ── ROUTES ────────────────────────────────────────────────────────
app.get("/", (req, res) => res.sendFile(path.join(__dirname, "index.html")));

// ── INIT SESSION — server issues a session ID ────────────────────
// Client calls this once on load; the returned ID is stored in sessionStorage
// (not localStorage) so it dies when the browser tab closes — no cross-user leakage.
app.post("/init-session", (req, res) => {
  evictIfNeeded();
  const id  = crypto.randomUUID();
  const sess = touchSession(id);
  console.log(`🆕 Session created: ${id.slice(0,8)}… | Active sessions: ${Object.keys(sessions).length}`);
  res.json({ sessionId: id });
});

// ── ADMIN: RELOAD DOCS ────────────────────────────────────────────
// Protected by ADMIN_SECRET header: x-admin-secret: <your_secret>
// Add ADMIN_SECRET=your_password to .env to enable.
app.post("/reload-docs", async (req, res) => {
  if (!isAdmin(req)) {
    return res.status(403).json({ success: false, error: "Forbidden. Valid admin secret required." });
  }
  try {
    console.log("🔄 Admin: reload-docs triggered.");
    await loadPDFs();
    if (fs.existsSync(VECTOR_CACHE_PATH)) fs.unlinkSync(VECTOR_CACHE_PATH);
    await buildVectorDB();
    res.json({ success: true, chunks: chunks.length, vectors: vectorDB.length });
  } catch (e) {
    res.status(500).json({ success: false, error: e.message });
  }
});

// ── CHAT API — STREAMING ─────────────────────────────────────────
app.post("/chat", async (req, res) => {
  try {
    const userMessage = req.body.message?.trim();
    const sessionId   = req.body.sessionId;

    if (!userMessage) {
      res.setHeader("Content-Type", "text/event-stream");
      res.write(`data: ${JSON.stringify({ text: "Please type a message. | စာတစ်ခုခု ရိုက်ထည့်ပါ။" })}\n\n`);
      res.write(`data: [DONE]\n\n`);
      return res.end();
    }

    // Validate session — must be a server-issued ID
    if (!sessionId || !sessions[sessionId]) {
      res.setHeader("Content-Type", "text/event-stream");
      res.write(`data: ${JSON.stringify({ text: "Session expired or invalid. Please refresh the page. | Session သက်တမ်းကုန်သွားပါပြီ။ Page ကို refresh လုပ်ပါ။" })}\n\n`);
      res.write(`data: [DONE]\n\n`);
      return res.end();
    }

    const sess = touchSession(sessionId);

    const truncated      = smartTruncateForEmbedding(userMessage);
    const relevantChunks = await searchRelevantChunks(truncated, 16);

    const DOC_THRESHOLD = 0.45;
    const topScore = (
      relevantChunks.length > 0 &&
      typeof relevantChunks[0] === "object" &&
      typeof relevantChunks[0].score === "number"
    ) ? relevantChunks[0].score : 0;

    const useDocContext = topScore >= DOC_THRESHOLD;
    let fullSystemInstruction;

    if (useDocContext) {
      console.log(`📚 Doc match (${(topScore * 100).toFixed(0)}%) [${sessionId.slice(0,8)}]`);
      const pdfContext = relevantChunks.filter(c => c.text).map(c => c.text).join("\n\n---\n\n");
      fullSystemInstruction = `${systemPrompt}

[BACKGROUND — treat this as knowledge you already have. NEVER mention it, quote it, or refer to it.]
${pdfContext}
[END BACKGROUND]
Answer naturally from the above background knowledge. Never tell the user you have files, documents, or any reference material. Just answer like you know it. Be thorough — cover every relevant angle from the background. Complete your answer fully, then end with one useful follow-up question.`;

    } else {
      console.log(`🌐 General mode (${(topScore * 100).toFixed(0)}%) [${sessionId.slice(0,8)}]`);
      fullSystemInstruction = `${systemPrompt}

This question goes beyond the specific documents in your knowledge base. Answer it using your deep general expertise. Go broad and thorough:
  - Pull in everything you know about the topic from your training
  - Use Google Search to find current, specific, or recent information (tax rates, law changes, live data, company-specific facts, 2025 developments)
  - Cross-reference multiple angles — legal, practical, strategic, financial
  - Give the most complete, useful answer you can
  - Real examples and numbers always beat abstract advice
Complete your answer fully, then end with one genuinely useful follow-up question.`;
    }

    // Convert history (stored as {role, content}) → Gemini format
    const geminiHistory = sess.history.map(msg => ({
      role:  msg.role === "assistant" ? "model" : "user",
      parts: [{ text: msg.content }]
    }));

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders();

    const heartbeat = setInterval(() => res.write(`: heartbeat\n\n`), 15000);
    let fullReply = "";

    try {
      const model = genAI.getGenerativeModel({
        model: "gemini-2.5-flash",
        systemInstruction: fullSystemInstruction,
        tools: [{ googleSearch: {} }],
        generationConfig: { temperature: 0.6, maxOutputTokens: 8000 }
      });

      const chat   = model.startChat({ history: geminiHistory });
      const result = await chat.sendMessageStream(userMessage);

      const iter = result.stream[Symbol.asyncIterator]();
      while (true) {
        let next;
        try { next = await iter.next(); } catch (_) { continue; }
        if (next.done) break;
        try {
          const text = next.value.text();
          if (text) { fullReply += text; res.write(`data: ${JSON.stringify({ text })}\n\n`); }
        } catch (_) { /* thinking-only chunk */ }
      }

      sess.history.push({ role: "user",      content: userMessage });
      sess.history.push({ role: "assistant", content: fullReply   });
      sess.history = sess.history.slice(-MAX_HISTORY);
      sess.lastActive = Date.now();

      res.write(`data: [DONE]\n\n`);

    } catch (streamErr) {
      console.error("❌ Stream error:", streamErr.message);
      if (fullReply.length > 0) {
        res.write(`data: [DONE]\n\n`);
      } else {
        const hint =
          streamErr.message?.includes("429") ? "I'm receiving too many requests right now. Please wait a moment and try again."
        : streamErr.message?.includes("401") ? "There's an authentication issue. Please check the API configuration."
        : streamErr.message?.includes("503") ? "The AI service is temporarily unavailable. Please try again in a moment."
        : "Something went wrong on my end. Please try sending your message again.";
        res.write(`data: ${JSON.stringify({ text: hint })}\n\n`);
        res.write(`data: [DONE]\n\n`);
      }
    } finally {
      clearInterval(heartbeat);
      res.end();
    }

  } catch (err) {
    console.error("❌ ERROR:", err.message);
    if (!res.headersSent) {
      res.setHeader("Content-Type", "text/event-stream");
      res.write(`data: ${JSON.stringify({ text: "Server error occurred. Please try again. | Server error ဖြစ်ပါတယ်။" })}\n\n`);
      res.write(`data: [DONE]\n\n`);
      res.end();
    }
  }
});

// ── CLEAR SESSION ────────────────────────────────────────────────
app.post("/clear-session", (req, res) => {
  const { sessionId } = req.body;
  if (sessionId && sessions[sessionId]) {
    sessions[sessionId].history = [];
    sessions[sessionId].lastActive = Date.now();
  }
  res.json({ success: true });
});

// ── START ─────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Server on port ${PORT}`);
  console.log(`🔐 Admin routes: ${ADMIN_SECRET ? "ENABLED" : "DISABLED (set ADMIN_SECRET in .env)"}`);
});
