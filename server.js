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
const __dirname = path.dirname(__filename);

const standardFontDataUrl =
  pathToFileURL(path.join(__dirname, "node_modules", "pdfjs-dist", "standard_fonts")).href + "/";

const app = express();

app.use((req, res, next) => {
  req.setTimeout(180000);
  res.setTimeout(180000);
  next();
});

app.use(rateLimit({ windowMs: 60 * 1000, max: 80 }));
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// ── GEMINI ──────────────────────────────────────────────────────
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// ── MEMORY ──────────────────────────────────────────────────────
let chunks = [];
let vectorDB = [];
const sessionHistories = {};
const MAX_HISTORY = 20;

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
  const words = text.split(/\s+/);
  const result = [];
  const step = size - overlap;
  for (let i = 0; i < words.length; i += step) {
    const chunk = words.slice(i, i + size).join(" ");
    if (chunk.trim()) result.push({ text: chunk, source });
    if (i + size >= words.length) break;
  }
  return result;
}

// ── LOAD PDFs ────────────────────────────────────────────────────
async function loadPDFs() {
  try {
    const docsDir = path.join(__dirname, "docs");
    console.log("📁 Loading PDFs from:", docsDir);
    if (!fs.existsSync(docsDir)) return;
    const files = fs.readdirSync(docsDir);
    chunks = [];
    for (const file of files) {
      if (!file.toLowerCase().endsWith(".pdf")) continue;
      const pdfPath = path.join(docsDir, file);
      console.log("📄 Loading:", file);
      try {
        const data = new Uint8Array(fs.readFileSync(pdfPath));
        const pdf = await pdfjsLib.getDocument({ data, standardFontDataUrl }).promise;
        let fileText = "";
        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const content = await page.getTextContent();
          fileText += content.items.map(item => item.str).filter(s => s?.trim()).join(" ") + "\n";
        }
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
  const docsDir = path.join(__dirname, "docs");
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
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (normA * normB);
}

function applyMMR(candidates, topK, lambda = 0.6) {
  if (candidates.length <= topK) return candidates;
  const selected = [candidates[0]];
  const remaining = candidates.slice(1);
  while (selected.length < topK && remaining.length > 0) {
    let bestIdx = 0, bestScore = -Infinity;
    for (let i = 0; i < remaining.length; i++) {
      const relevance = remaining[i].score;
      const maxSim = Math.max(...selected.map(s => cosineSimilarity(remaining[i].embedding, s.embedding)));
      const mmrScore = lambda * relevance - (1 - lambda) * maxSim;
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
  const pool = candidatePool.length >= 3 ? candidatePool : scored.slice(0, topK * 2);
  const diverse = applyMMR(pool, topK, 0.6);
  return diverse.map(({ text, source, score }) => ({ text, source, score }));
}

// ── LANGUAGE DETECTION ──────────────────────────────────────────
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

// ── CHAT API — STREAMING ─────────────────────────────────────────
app.post("/chat", async (req, res) => {
  try {
    const userMessage = req.body.message?.trim();
    const sessionId = req.body.sessionId || "default";

    if (!userMessage) {
      res.setHeader("Content-Type", "text/event-stream");
      res.write(`data: ${JSON.stringify({ text: "Please type a message. | စာတစ်ခုခု ရိုက်ထည့်ပါ။" })}\n\n`);
      res.write(`data: [DONE]\n\n`);
      return res.end();
    }

    if (!sessionHistories[sessionId]) sessionHistories[sessionId] = [];

    const truncated = smartTruncateForEmbedding(userMessage);
    const relevantChunks = await searchRelevantChunks(truncated, 16);

    // Decide whether the question is actually related to the uploaded documents.
    // Similarity >= 0.62 = genuinely doc-specific. Below that → general knowledge mode.
    const DOC_THRESHOLD = 0.62;
    const topScore = (
      relevantChunks.length > 0 &&
      typeof relevantChunks[0] === "object" &&
      typeof relevantChunks[0].score === "number"
    ) ? relevantChunks[0].score : 0;

    const useDocContext = topScore >= DOC_THRESHOLD;

    let fullSystemInstruction;

    if (useDocContext) {
      // Question is relevant to uploaded docs — silently bake them into the answer
      console.log(`📚 Doc match (${(topScore * 100).toFixed(0)}%) — injecting context`);
      const pdfContext = relevantChunks
        .filter(c => c.text)
        .map(c => c.text)
        .join("\n\n---\n\n");

      fullSystemInstruction = `${systemPrompt}

[BACKGROUND — treat this as knowledge you already have. NEVER mention it, quote it, or refer to it.]
${pdfContext}
[END BACKGROUND]
Answer naturally from the above. Never tell the user you have files, documents, or any reference material. Just answer like you know it. Complete your answer fully, then end with one useful follow-up question.`;

    } else {
      // Question is off-topic for the docs — use general knowledge and/or Google Search
      console.log(`📚 No doc match (${(topScore * 100).toFixed(0)}%) — general knowledge mode`);
      fullSystemInstruction = `${systemPrompt}

Answer this from your general expertise in partnership business, law, and strategy. Use Google Search if you need current or specific information. Complete your answer fully, then end with one useful follow-up question.`;
    }

    // Convert session history to Gemini format
    // Gemini uses "user" and "model" roles (not "assistant")
    const geminiHistory = sessionHistories[sessionId].map(msg => ({
      role: msg.role === "assistant" ? "model" : "user",
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
        generationConfig: {
          temperature: 0.6,
          maxOutputTokens: 8000
          // thinkingBudget intentionally NOT set — Google Search needs thinking enabled
        }
      });

      const chat = model.startChat({ history: geminiHistory });
      const result = await chat.sendMessageStream(userMessage);

      // Manual iteration so a single bad chunk (thinking token, search metadata, etc.)
      // is silently skipped instead of crashing the entire stream.
      const iter = result.stream[Symbol.asyncIterator]();
      while (true) {
        let next;
        try {
          next = await iter.next();
        } catch (_) {
          // SDK failed to parse one raw HTTP chunk — skip it and keep going
          continue;
        }
        if (next.done) break;
        try {
          const text = next.value.text();
          if (text) {
            fullReply += text;
            res.write(`data: ${JSON.stringify({ text })}\n\n`);
          }
        } catch (_) {
          // thinking-only chunk — no text to extract, skip silently
        }
      }

      sessionHistories[sessionId].push({ role: "user", content: userMessage });
      sessionHistories[sessionId].push({ role: "assistant", content: fullReply });
      sessionHistories[sessionId] = sessionHistories[sessionId].slice(-MAX_HISTORY);

      const keys = Object.keys(sessionHistories);
      if (keys.length > 100) delete sessionHistories[keys[0]];
      res.write(`data: [DONE]\n\n`);

    } catch (streamErr) {
      console.error("❌ Stream error:", streamErr.message);
      if (fullReply.length > 0) {
        // Partial content was already streamed — close cleanly without an error message
        res.write(`data: [DONE]\n\n`);
      } else {
        // Nothing was sent yet — give a friendly retry message
        const hint =
          streamErr.message?.includes("429")
            ? "I'm receiving too many requests right now. Please wait a moment and try again."
          : streamErr.message?.includes("401")
            ? "There's an authentication issue. Please check the API configuration."
          : streamErr.message?.includes("503")
            ? "The AI service is temporarily unavailable. Please try again in a moment."
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
  const sessionId = req.body.sessionId || "default";
  delete sessionHistories[sessionId];
  res.json({ success: true, message: "Session cleared. | Session ရှင်းလင်းပြီးပါပြီ။" });
});

// ── START ─────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Server on port ${PORT}`));
