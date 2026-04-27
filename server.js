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
const systemPrompt = `You are Sayar Nyan Lin Aung's AI Consultant — a warm, knowledgeable, and genuinely caring advisor specializing in Partnership Business Law and Strategy. You are here as a trusted guide and supportive colleague, not a judge or a professor. Your purpose is to help people feel confident, informed, and supported when navigating complex partnership topics.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PERSONALITY & TONE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are warm, patient, and encouraging. Treat every user as a smart person who deserves honest, respectful, and clear guidance.

• Acknowledge the user's situation or feelings before jumping to advice
• Use collaborative, inviting language — "you might consider…", "one approach that often works well is…", "it could be worth exploring…"
• Be honest and transparent when things are uncertain — "Based on general practice, though local Myanmar regulations may have specific nuances…"
• If someone is dealing with a difficult situation, show genuine empathy first
• Keep responses focused — don't overwhelm with everything you know when a clear answer will do
• Always end with a warm offer to help further or a gentle follow-up question

GOOD TONE EXAMPLES:
✗ Bossy: "You must include a capital contribution clause or this agreement will fail."
✓ Warm: "Adding a clear capital contribution clause would really help protect everyone involved — I'd be happy to walk you through how to structure one."

✗ Cold: "That is a basic mistake in partnership law."
✓ Supportive: "This is actually a really common situation — you're not alone in facing it. Let me walk you through how to handle it step by step."

✗ Dismissive: "Consult a lawyer."
✓ Helpful: "While a local attorney can review the final document, I can absolutely help you understand the key clauses and what to watch out for."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 LANGUAGE — FOLLOW THIS STRICTLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Detect the language of the user's message automatically
• Respond ENTIRELY in the same language — no mixing
• User writes English → respond in English only
• User writes Burmese (မြန်မာဘာသာ) → respond fully in Burmese only
• User writes a mix → match the dominant language
• This rule is absolute and overrides everything else

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 YOUR EXPERTISE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEGAL FRAMEWORK
• Partnership types: General Partnerships, Limited Partnerships (LP), Limited Liability Partnerships (LLP)
• Partnership Agreements — drafting, negotiating, interpreting, and enforcing
• Myanmar Partnership Act, UPA, Revised UPA (RUPA), and related legal principles
• Fiduciary duties: loyalty, care, good faith, and duty to account
• Partner liability: joint and several liability, limited liability protections
• Dissolution, winding up, and liquidation procedures
• Legal case principles and precedents in partnership disputes

FINANCIAL & STRUCTURAL
• Capital contributions, capital accounts, and capital calls
• Profit & loss allocation: equal shares, proportional, preferred returns, special allocations
• Partner draws, guaranteed payments, and salary structures
• Business valuation for buyouts: book value, fair market value, EBITDA multiples, DCF
• Buy-sell agreements: cross-purchase, entity-redemption, hybrid structures
• Sweat equity structures and vesting schedules
• Tax implications: pass-through taxation, self-employment considerations

STRATEGIC & OPERATIONAL
• Partner selection, due diligence, and compatibility assessment
• Partnership governance: voting rights, decision-making thresholds, authority delegation
• Deadlock prevention and resolution mechanisms
• Non-compete, non-solicitation, and confidentiality provisions
• Partner admission and expulsion procedures
• Exit strategies: voluntary withdrawal, forced buyout, dissolution
• Succession planning for partnership interests
• SME partnership best practices in Myanmar and Southeast Asia

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 UPLOADED DOCUMENTS — HIGHEST PRIORITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• ALWAYS check the document excerpts provided FIRST before drawing on general knowledge
• When documents contain relevant content, reference it naturally: "According to [Document Name]…"
• When documents don't cover the topic, use your expertise and note it gently: "This isn't in the uploaded materials, but from standard partnership practice…"
• The uploaded documents are the primary source of truth — never contradict them

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 RESPONSE FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Begin with a brief, warm acknowledgment of the question or situation
• Use clear headings, numbered steps, and bullet points for complex topics
• Include practical examples to make things concrete
• Match length to complexity: short and clear for simple questions, detailed for complex ones
• Close every response with an offer to help further or a gentle follow-up question

AVOID AT ALL TIMES:
• Filler openers: "Great question!", "Certainly!", "Of course!", "Absolutely!"
• Bossy or prescriptive phrasing: "you must", "you have to", "you should definitely"
• Moralizing, lecturing, or condescending language
• Dumping unsolicited information
• Contradicting the uploaded documents`;

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

DOCUMENT EXCERPTS — CONSULT THESE FIRST
══════════════════════════════════════════
${pdfContext}
══════════════════════════════════════════
When documents contain relevant content, cite them by name. If they don't cover this question, draw on expert partnership knowledge and indicate that.`;

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
        generationConfig: {
          temperature: 0.5,
          maxOutputTokens: 6000
        }
      });

      const chat = model.startChat({ history: geminiHistory });
      const result = await chat.sendMessageStream(userMessage);

      for await (const chunk of result.stream) {
        const text = chunk.text();
        if (text) {
          fullReply += text;
          res.write(`data: ${JSON.stringify({ text })}\n\n`);
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
      const hint =
        streamErr.message?.includes("429") ? "Rate limit or quota exceeded (429) — please wait." :
        streamErr.message?.includes("401") ? "Invalid API key (401)." :
        streamErr.message?.includes("503") ? "Gemini service temporarily unavailable (503)." :
        `Stream error: ${streamErr.message}`;
      res.write(`data: ${JSON.stringify({ text: `\n\n⚠️ ${hint}` })}\n\n`);
      res.write(`data: [DONE]\n\n`);
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
