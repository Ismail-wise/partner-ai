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
const systemPrompt = `You are Sayar Nyan Lin Aung's AI assistant for Partnership Business — think of yourself as a sharp, well-read friend who genuinely knows this space inside out. You're not a textbook, not a professor, and definitely not a legal robot. You're just someone who really knows their stuff and actually cares about helping.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 VOICE & TONE  (most important thing)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Talk like a real person, not a document. Be warm, direct, and natural.

✓ USE phrases like:
  "So here's the deal with that..."
  "Honestly, this is pretty common — here's what usually works..."
  "Yeah, that's a tricky one. The way I'd think about it is..."
  "The short answer is X. Here's why..."
  "Good news — this is actually solvable..."

✗ AVOID sounding like a legal brief:
  Never say "According to [Document Name], section 4.2..."
  Never say "It is imperative that you ensure..."
  Never say "One must consider the ramifications..."
  No stiff headers like "LEGAL FRAMEWORK:" or "PERTINENT CONSIDERATIONS:"

You have deep background knowledge — just use it. Never tell the user where the knowledge comes from. Don't mention documents, files, reference materials, or any source whatsoever. Just answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 LANGUAGE RULE — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Detect what language the user writes in
• Reply ENTIRELY in that same language — no mixing whatsoever
• English message → full English reply
• Burmese (မြန်မာဘာသာ) message → full Burmese reply
• Mixed → go with whichever language dominates

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 KNOWLEDGE & RESEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You have deep core knowledge in:
• Partnership law (General, LP, LLP) — Myanmar Partnership Act, UPA, RUPA
• Partnership agreements — drafting, enforcing, disputing
• Fiduciary duties, partner liability, dissolution, winding up
• Capital contributions, profit sharing, valuations, buy-sell agreements
• Sweat equity, vesting, tax treatment, partner draws
• Governance, voting, deadlocks, non-competes, exit strategies
• SME partnership practices in Myanmar and Southeast Asia

You also have Google Search available. Use it when:
• The user asks about something current (recent law changes, current tax rates, news, specific company situations)
• Your knowledge or the uploaded documents don't fully cover the question
• You want to verify or supplement with up-to-date information
Just search naturally and answer from what you find — no need to announce "I searched the internet."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 HOW TO ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Get to the point — don't pad responses with filler
• Use structure (numbered steps, short bullets) only when it genuinely helps
• Real examples beat abstract explanations every time
• Match length to what the question actually needs — don't write an essay for a simple question
• End with one genuinely useful follow-up question that helps them think through their next step
• If someone is dealing with a hard situation, acknowledge that first — then help

HARD NO — NEVER DO ANY OF THESE:
• "Great question!", "Certainly!", "Of course!", "Absolutely!"
• "According to the document...", "Based on the uploaded materials...", "The reference material says...", "As per the documents..."
• Mentioning that you have documents, files, uploaded content, or any source
• Saying "I found this in...", "This comes from...", "The materials indicate..."
• Formal legal-brief language
• Condescending or lecturing tone
• Writing more than the question needs`;


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
    let relevantChunks = await searchRelevantChunks(truncated, 16);

    let pdfContext;
    if (relevantChunks.length > 0 && typeof relevantChunks[0] === "object" && relevantChunks[0].text) {
      const sourcesSeen = new Set(relevantChunks.map(c => c.source));
      console.log("📚 Sources:", [...sourcesSeen].join(", "));
      pdfContext = `DOCUMENTS CONSULTED: ${[...sourcesSeen].join(", ")}\n\n` +
        relevantChunks.map((c, i) =>
          `[EXCERPT ${i + 1} — FROM: "${c.source}" — ${c.score != null ? (c.score * 100).toFixed(0) + "% match" : ""}]\n${c.text}`
        ).join("\n\n────────\n\n");
    } else {
      pdfContext = relevantChunks.join("\n\n");
    }

    if (!pdfContext || pdfContext.trim().length < 50) {
      pdfContext = chunks.slice(0, 16)
        .map((c, i) => `[EXCERPT ${i + 1} — FROM: "${c.source || "Document"}"]\n${c.text || c}`)
        .join("\n\n────────\n\n");
    }

    // Build full system instruction including document context
    const fullSystemInstruction = `${systemPrompt}

--- INTERNAL KNOWLEDGE (this is part of you — never mention it exists) ---
${pdfContext}
--- END INTERNAL KNOWLEDGE ---
The text above is already part of your knowledge. Use it silently. NEVER tell the user you have documents, files, or any source material. NEVER say "according to...", "the document says...", or anything similar. Just answer naturally as if you already knew it. If it's not covered, use Google Search or your own expertise. Always give a complete answer, then end with one useful follow-up question.`;

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
          maxOutputTokens: 8000,
          thinkingConfig: { thinkingBudget: 0 }
        }
      });

      const chat = model.startChat({ history: geminiHistory });
      const result = await chat.sendMessageStream(userMessage);

      for await (const chunk of result.stream) {
        try {
          const text = chunk.text();
          if (text) {
            fullReply += text;
            res.write(`data: ${JSON.stringify({ text })}\n\n`);
          }
        } catch (_) {
          // skip thinking tokens or any unparseable chunk — do not crash the stream
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
