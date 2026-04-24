import OpenAI from "openai";
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

const standardFontDataUrl = pathToFileURL(
  path.join(__dirname, "node_modules", "pdfjs-dist", "standard_fonts")
).href + "/";

const app = express();

app.use((req, res, next) => {
  req.setTimeout(180000);
  res.setTimeout(180000);
  next();
});

app.use(rateLimit({ windowMs: 60 * 1000, max: 50 }));
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// ==========================
// OPENAI CLIENT
// ==========================

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ==========================
// MEMORY STORAGE
// ==========================

// chunks: { text: string, source: string }[]
// vectorDB: { text: string, source: string, embedding: number[] }[]
let chunks = [];
let vectorDB = [];
const sessionHistories = {};
const MAX_HISTORY = 20;

// ==========================
// WEB SEARCH
// ==========================

async function webSearch(query) {
  try {
    const result = await openai.chat.completions.create({
      model: "gpt-4o-search-preview",
      messages: [
        {
          role: "user",
          content: `Search for real-world case studies, specific data, and named examples about: "${query}".
Focus on:
- Actual partnership business disputes and how they were resolved (named cases or companies)
- Real share structures, buyout clauses, equity arrangements used by real companies
- Myanmar or Southeast Asian SME partnership examples if available
- Legal precedents, court cases, expert opinions with specific data (percentages, timelines, figures)
Return concrete facts and named examples only. No generic advice.`
        }
      ]
    });

    let content = result.choices[0]?.message?.content || null;
    // Truncate to avoid token overload
    if (content && content.length > 5000) content = content.slice(0, 5000) + "\n[Truncated]";
    return content;
  } catch (err) {
    console.log("❌ Web search error:", err.message);
    return null;
  }
}

// Only triggers on genuine requests for external real-world data
function needsWebSearch(message) {
  const triggers = [
    "case study", "case studies", "real example", "real-world example",
    "what do successful companies", "how do other companies",
    "industry standard", "market rate", "average percentage",
    "actual lawsuit", "court case", "legal precedent",
    "statistics show", "data shows", "research says",
    "famous company", "well known example", "real company"
  ];
  const lower = message.toLowerCase();
  return triggers.some(k => lower.includes(k));
}

// ==========================
// QUERY EXPANSION
// ==========================

function buildExpandedQuery(message) {
  return message;
}

// ==========================
// EMBEDDING
// ==========================

async function getEmbedding(text) {
  try {
    if (!text || typeof text !== "string") return null;
    const cleanText = text.replace(/\s+/g, " ").slice(0, 8000);
    if (!cleanText) return null;

    const response = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: cleanText
    });

    return response.data[0]?.embedding || null;
  } catch (err) {
    console.log("❌ Embedding error:", err.message);
    return null;
  }
}

// ==========================
// TEXT SPLITTER
// ==========================

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

// ==========================
// LOAD PDFs
// ==========================

async function loadPDFs() {
  try {
    const folderPath = path.join(__dirname, "docs");
    console.log("📁 Loading PDFs from:", folderPath);

    if (!fs.existsSync(folderPath)) {
      console.log("⚠️ docs folder not found");
      return;
    }

    const files = fs.readdirSync(folderPath);
    chunks = [];

    for (const file of files) {
      if (!file.toLowerCase().endsWith(".pdf")) continue;
      const pdfPath = path.join(folderPath, file);
      console.log("📄 Loading:", file);

      try {
        const data = new Uint8Array(fs.readFileSync(pdfPath));
        const pdf = await pdfjsLib.getDocument({ data, standardFontDataUrl }).promise;

        let fileText = "";
        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const content = await page.getTextContent();
          const strings = content.items
            .map(item => item.str)
            .filter(str => str && str.trim().length > 0);
          fileText += strings.join(" ") + "\n";
        }

        if (!fileText.trim()) {
          console.log(`⚠️ No text extracted from ${file}`);
          continue;
        }

        // Tag every chunk with this file's name as source
        const fileChunks = splitText(fileText, file.replace(/\.pdf$/i, ""), 500, 100);
        chunks.push(...fileChunks);
        console.log(`✅ ${file}: ${fileChunks.length} chunks`);

      } catch (fileErr) {
        console.log(`❌ Failed to read ${file}:`, fileErr.message);
      }
    }

    if (chunks.length === 0) {
      console.log("⚠️ No chunks created from any PDF");
      return;
    }

    console.log(`✅ Total chunks: ${chunks.length}`);
    console.log("🔥 Sample:", chunks[0]?.text?.slice(0, 200));
  } catch (err) {
    console.log("❌ PDF ERROR:", err.message);
  }
}

const VECTOR_CACHE_PATH = path.join(__dirname, "vectordb_cache.json");

// Stable fingerprint of docs folder: filename + size for each PDF (sorted)
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
        console.log(`✅ Loaded vector DB from cache: ${vectorDB.length} entries`);
        return;
      }
      console.log("⚠️ Cache fingerprint mismatch — rebuilding vector DB...");
    } catch (e) {
      console.log("⚠️ Cache read failed — rebuilding:", e.message);
    }
  }

  console.log("🔄 Building vector DB from scratch...");
  vectorDB = [];
  for (const chunk of chunks) {
    const embedding = await getEmbedding(chunk.text);
    if (!embedding) continue;
    vectorDB.push({ text: chunk.text, source: chunk.source, embedding });
    await new Promise(resolve => setTimeout(resolve, 80));
  }
  console.log("✅ Vector DB ready:", vectorDB.length, "entries from", new Set(chunks.map(c => c.source)).size, "documents");

  try {
    fs.writeFileSync(VECTOR_CACHE_PATH, JSON.stringify({ cacheKey: currentKey, chunkCount: chunks.length, vectorDB }));
    console.log("💾 Vector DB cached to disk");
  } catch (e) {
    console.log("⚠️ Could not save vector DB cache:", e.message);
  }
}

// ==========================
// COSINE SIMILARITY SEARCH
// ==========================

function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (normA * normB);
}

// Maximal Marginal Relevance: balances relevance vs. diversity to avoid duplicate chunks
function applyMMR(candidates, topK, lambda = 0.6) {
  if (candidates.length <= topK) return candidates;
  const selected = [candidates[0]];
  const remaining = candidates.slice(1);
  while (selected.length < topK && remaining.length > 0) {
    let bestIdx = 0, bestScore = -Infinity;
    for (let i = 0; i < remaining.length; i++) {
      const relevance = remaining[i].score;
      const maxSim = Math.max(...selected.map(s =>
        cosineSimilarity(remaining[i].embedding, s.embedding)
      ));
      const mmrScore = lambda * relevance - (1 - lambda) * maxSim;
      if (mmrScore > bestScore) { bestScore = mmrScore; bestIdx = i; }
    }
    selected.push(remaining.splice(bestIdx, 1)[0]);
  }
  return selected;
}

async function searchRelevantChunks(query, topK = 16) {
  // Returns: { text, source, score }[]
  if (vectorDB.length === 0) return chunks.slice(0, topK);

  const queryEmbedding = await getEmbedding(query);
  if (!queryEmbedding) return chunks.slice(0, topK);

  // Score all, keep embedding for MMR diversity calculation
  const scored = vectorDB
    .map(item => ({
      text: item.text,
      source: item.source || "Document",
      score: cosineSimilarity(queryEmbedding, item.embedding),
      embedding: item.embedding   // kept for MMR, stripped before returning
    }))
    .sort((a, b) => b.score - a.score);

  // Log top scores for debugging
  console.log("📊 Top 5 scores:", scored.slice(0, 5).map(s => `${s.source}: ${s.score.toFixed(3)}`).join(", "));

  // Candidate pool = 2x topK, then apply MMR for diversity
  const candidatePool = scored.filter(i => i.score > 0.15).slice(0, topK * 2);
  const pool = candidatePool.length >= 3 ? candidatePool : scored.slice(0, topK * 2);
  const diverse = applyMMR(pool, topK, 0.6);

  // Strip embeddings before returning (not needed downstream)
  return diverse.map(({ text, source, score }) => ({ text, source, score }));
}

// ==========================
// LANGUAGE DETECTION
// ==========================

function detectLanguage(text) {
  const burmeseRegex = /[\u1000-\u109F\uAA60-\uAA7F]/;
  if (burmeseRegex.test(text)) return "burmese";
  const lower = text.toLowerCase();
  if (lower.includes("burmese") || lower.includes("myanmar") || lower.includes("မြန်မာ")) return "burmese";
  if (lower.includes("english")) return "english";
  return "english";
}


// ==========================
// SMART MESSAGE TRUNCATION
// ==========================

function smartTruncateForEmbedding(text, maxWords = 500) {
  const words = text.trim().split(/\s+/);
  if (words.length <= maxWords) return text;
  const start = words.slice(0, 300).join(" ");
  const end = words.slice(-200).join(" ");
  return `${start} ... ${end}`;
}

// ==========================
// STARTUP
// ==========================

await loadPDFs();
await buildVectorDB();

// ==========================
// SYSTEM PROMPT
// ==========================

const systemPrompt = `
You are a document-based AI assistant. You answer questions strictly based on the content of the uploaded documents provided to you.

== CORE RULES ==
- Answer ONLY using facts, data, and information found in the document excerpts provided.
- If the answer is not in the documents, say clearly: "I don't have information on that in the uploaded documents."
- Never make up facts, statistics, or details not present in the documents.
- When a fact or figure comes from a document, cite it: (Source: [document name])
- Quote or closely paraphrase the relevant document text whenever possible.
- If numbers or data appear in the documents, use them exactly as written.

== TONE ==
- Be clear, direct, and concise.
- Answer in the same language the user writes in.
- Do not add filler phrases like "Great question!" — just answer.

== MEMORY ==
- Remember everything the user told you in this conversation.
- Never ask for information they already provided.
`;

// ==========================
// ROUTES
// ==========================

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

// ==========================
// CHAT API — STREAMING
// ==========================

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

    if (!sessionHistories[sessionId]) {
      sessionHistories[sessionId] = [];
    }

    const truncatedForEmbedding = smartTruncateForEmbedding(userMessage);
    const enhancedQuery = buildExpandedQuery(truncatedForEmbedding);

    // Retrieve diverse, source-tagged chunks via MMR
    let relevantChunks = await searchRelevantChunks(enhancedQuery, 16);

    // Build source-labelled context with relevance scores
    let pdfContext;
    if (relevantChunks.length > 0 && typeof relevantChunks[0] === "object" && relevantChunks[0].text) {
      const sourcesSeen = new Set(relevantChunks.map(c => c.source));
      console.log("📚 Sources used:", [...sourcesSeen].join(", "));

      pdfContext = `DOCUMENTS CONSULTED: ${[...sourcesSeen].join(", ")}\n\n` +
        relevantChunks
          .map((c, i) => `[EXCERPT ${i + 1} — FROM: "${c.source}" — ${c.score != null ? (c.score * 100).toFixed(0) + "% match" : ""}]\n${c.text}`)
          .join("\n\n────────\n\n");
    } else {
      pdfContext = relevantChunks.join("\n\n");
    }

    if (!pdfContext || pdfContext.trim().length < 50) {
      pdfContext = chunks.slice(0, 16)
        .map((c, i) => `[EXCERPT ${i + 1} — FROM: "${c.source || "Document"}"]\n${c.text || c}`)
        .join("\n\n────────\n\n");
    }

    const webContext = null;

    sessionHistories[sessionId].push({ role: "user", content: userMessage });
    let hist = sessionHistories[sessionId].slice(-MAX_HISTORY);
    sessionHistories[sessionId] = hist;

    // Build the document context block — placed LAST (right before user messages)
    // so it has strongest attention from the model
    const docContextMessage = {
      role: "system",
      content: `DOCUMENT EXCERPTS — YOUR ONLY SOURCE OF TRUTH
The excerpts below come directly from the uploaded documents.
You MUST:
1. Answer only from what the documents say — not from general knowledge
2. Quote or closely paraphrase the document text when relevant content is present
3. Cite the source document by name: e.g. "According to [document name]..."
4. Use any numbers, data, or facts exactly as they appear in the text
5. If the documents do NOT cover the question, say so clearly: "I don't have information on that in the uploaded documents."

══════════════════════════════════════════
${pdfContext}
══════════════════════════════════════════`
    };

    const webContextMessage = webContext ? {
      role: "system",
      content: `REAL-WORLD DATA (web search):
${webContext}
Use this to add a "Real-World Example:" section. Connect back to the document rule. If specific numbers, named cases, or timelines are present — use them. No emojis.`
    } : null;

    const messages = [
      { role: "system", content: systemPrompt },
      ...(webContextMessage ? [webContextMessage] : []),
      ...sessionHistories[sessionId].slice(0, -1),  // history except last user msg
      docContextMessage,                             // doc context right before user's message
      sessionHistories[sessionId].at(-1)             // the actual user message last
    ];

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders();

    const heartbeat = setInterval(() => {
      res.write(`: heartbeat\n\n`);
    }, 15000);

    let fullReply = "";

    try {
      const stream = await openai.chat.completions.create({
        model: "gpt-4o",
        messages,
        stream: true,
        temperature: 0.55,
        max_tokens: 5000
      });

      for await (const chunk of stream) {
        const text = chunk.choices[0]?.delta?.content || "";
        if (text) {
          fullReply += text;
          res.write(`data: ${JSON.stringify({ text })}\n\n`);
        }
      }

      sessionHistories[sessionId].push({ role: "assistant", content: fullReply });

      const sessionKeys = Object.keys(sessionHistories);
      if (sessionKeys.length > 100) delete sessionHistories[sessionKeys[0]];

      res.write(`data: [DONE]\n\n`);

    } catch (streamErr) {
      console.error("❌ Stream error:", streamErr.message);
      res.write(`data: ${JSON.stringify({ text: "\n\nStream interrupted. Please try again. | ချိတ်ဆက်မှု ပြတ်တောက်သွားသည်။ ထပ်စမ်းပါ။" })}\n\n`);
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

// ==========================
// CLEAR SESSION
// ==========================

app.post("/clear-session", (req, res) => {
  const sessionId = req.body.sessionId || "default";
  if (sessionHistories[sessionId]) {
    delete sessionHistories[sessionId];
  }
  res.json({ success: true, message: "Session cleared. | Session ရှင်းလင်းပြီးပါပြီ။" });
});

// ==========================
// START SERVER
// ==========================

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
