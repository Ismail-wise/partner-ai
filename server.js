import { GoogleGenerativeAI } from "@google/generative-ai";
import fs from "fs";
import path from "path";
import { fileURLToPath, pathToFileURL } from "url";
import dotenv from "dotenv";
dotenv.config();
import express from "express";
import cors from "cors";
import rateLimit from "express-rate-limit";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const standardFontDataUrl =
  pathToFileURL(path.join(__dirname, "node_modules", "pdfjs-dist", "standard_fonts")).href + "/";

const app = express();

app.use((req, res, next) => { req.setTimeout(180000); res.setTimeout(180000); next(); });
app.use(rateLimit({ windowMs: 60 * 1000, max: 120 }));
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// ── CONFIG ───────────────────────────────────────────────────────
const JWT_SECRET     = process.env.JWT_SECRET     || "change-this-secret-in-production";
const ADMIN_SECRET   = process.env.ADMIN_SECRET   || null;
const FREE_DAILY_CAP = parseInt(process.env.FREE_DAILY_CAP || "10", 10);

// Premium codes: set PREMIUM_CODES=code1,code2,code3 in .env
const PREMIUM_CODES  = (process.env.PREMIUM_CODES || "")
  .split(",").map(c => c.trim()).filter(Boolean);

if (JWT_SECRET === "change-this-secret-in-production") {
  console.warn("⚠️  WARNING: Using default JWT_SECRET. Set JWT_SECRET in .env for production!");
}

// ── DOCS DIR ────────────────────────────────────────────────────
const docsDir = path.join(__dirname, "docs");
if (!fs.existsSync(docsDir)) fs.mkdirSync(docsDir, { recursive: true });

// ── USER STORE (users.json) ──────────────────────────────────────
// Schema: { users: [{ id, email, passwordHash, role, createdAt, daily: { date, count } }] }
const USERS_PATH = path.join(__dirname, "users.json");

function loadUsers() {
  try {
    if (!fs.existsSync(USERS_PATH)) return { users: [] };
    return JSON.parse(fs.readFileSync(USERS_PATH, "utf-8"));
  } catch (_) { return { users: [] }; }
}

function saveUsers(store) {
  fs.writeFileSync(USERS_PATH, JSON.stringify(store, null, 2));
}

function findUserByEmail(email) {
  return loadUsers().users.find(u => u.email.toLowerCase() === email.toLowerCase());
}

function findUserById(id) {
  return loadUsers().users.find(u => u.id === id);
}

function todayStr() {
  return new Date().toISOString().slice(0, 10); // YYYY-MM-DD
}

function getDailyCount(user) {
  if (!user.daily || user.daily.date !== todayStr()) return 0;
  return user.daily.count || 0;
}

function incrementDailyCount(userId) {
  const store = loadUsers();
  const user  = store.users.find(u => u.id === userId);
  if (!user) return;
  const today = todayStr();
  if (!user.daily || user.daily.date !== today) {
    user.daily = { date: today, count: 1 };
  } else {
    user.daily.count = (user.daily.count || 0) + 1;
  }
  saveUsers(store);
}

// ── ADMIN CHECK ──────────────────────────────────────────────────
function isAdmin(req) {
  if (!ADMIN_SECRET) return false;
  const h = req.headers["x-admin-secret"] || req.body?.adminSecret || req.query?.secret;
  return h === ADMIN_SECRET;
}

// ── JWT MIDDLEWARE ───────────────────────────────────────────────
function verifyToken(req, res, next) {
  const auth  = req.headers.authorization;
  const token = auth?.startsWith("Bearer ") ? auth.slice(7) : null;
  if (!token) return res.status(401).json({ success: false, error: "Not authenticated." });
  try {
    req.user = jwt.verify(token, JWT_SECRET);
    next();
  } catch (_) {
    return res.status(401).json({ success: false, error: "Session expired. Please log in again." });
  }
}

// ── GEMINI ──────────────────────────────────────────────────────
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// ── SESSION STORE ────────────────────────────────────────────────
const SESSION_TTL_MS = 24 * 60 * 60 * 1000;
const MAX_HISTORY    = 20;
const sessions       = {};

function touchSession(id) {
  if (!sessions[id]) sessions[id] = { history: [], lastActive: Date.now() };
  else sessions[id].lastActive = Date.now();
  return sessions[id];
}

setInterval(() => {
  const now = Date.now();
  let cleaned = 0;
  for (const id of Object.keys(sessions)) {
    if (now - sessions[id].lastActive > SESSION_TTL_MS) { delete sessions[id]; cleaned++; }
  }
  if (cleaned) console.log(`🧹 Cleaned ${cleaned} sessions. Active: ${Object.keys(sessions).length}`);
}, 60 * 60 * 1000);

function evictIfNeeded() {
  const ids = Object.keys(sessions);
  if (ids.length <= 500) return;
  ids.sort((a, b) => sessions[a].lastActive - sessions[b].lastActive)
     .slice(0, ids.length - 400).forEach(id => delete sessions[id]);
}

// ── PDF PIPELINE ─────────────────────────────────────────────────
let chunks   = [];
let vectorDB = [];

async function getEmbedding(text) {
  try {
    if (!text || typeof text !== "string") return null;
    const cleanText = text.replace(/\s+/g, " ").slice(0, 8000);
    if (!cleanText) return null;
    const m      = genAI.getGenerativeModel({ model: "gemini-embedding-001" });
    const result = await m.embedContent(cleanText);
    return result.embedding.values || null;
  } catch (err) {
    console.log("❌ Embedding error:", err.message);
    return null;
  }
}

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

async function loadPDFs() {
  try {
    console.log("📁 Loading PDFs from:", docsDir);
    if (!fs.existsSync(docsDir)) return;
    chunks = [];
    for (const file of fs.readdirSync(docsDir)) {
      if (!file.toLowerCase().endsWith(".pdf")) continue;
      try {
        const txt = await extractPdfText(path.join(docsDir, file));
        if (!txt.trim()) continue;
        const fc = splitText(txt, file.replace(/\.pdf$/i, ""), 500, 100);
        chunks.push(...fc);
        console.log(`✅ ${file}: ${fc.length} chunks`);
      } catch (e) { console.log(`❌ Failed ${file}:`, e.message); }
    }
    console.log(`✅ Total chunks: ${chunks.length}`);
  } catch (err) { console.log("❌ PDF ERROR:", err.message); }
}

const VECTOR_CACHE_PATH = path.join(__dirname, "vectordb_cache.json");

function computeDocFingerprint() {
  if (!fs.existsSync(docsDir)) return "empty";
  return fs.readdirSync(docsDir).filter(f => f.toLowerCase().endsWith(".pdf")).sort()
    .map(f => `${f}:${fs.statSync(path.join(docsDir, f)).size}`).join("|");
}

async function buildVectorDB() {
  const key = computeDocFingerprint();
  if (fs.existsSync(VECTOR_CACHE_PATH)) {
    try {
      const c = JSON.parse(fs.readFileSync(VECTOR_CACHE_PATH, "utf-8"));
      if (c.cacheKey === key && Array.isArray(c.vectorDB) && c.vectorDB.length > 0) {
        vectorDB = c.vectorDB;
        console.log(`✅ Vector DB from cache: ${vectorDB.length} entries`);
        return;
      }
    } catch (_) {}
  }
  console.log("🔄 Building vector DB...");
  vectorDB = [];
  for (const chunk of chunks) {
    const emb = await getEmbedding(chunk.text);
    if (!emb) continue;
    vectorDB.push({ text: chunk.text, source: chunk.source, embedding: emb });
    await new Promise(r => setTimeout(r, 100));
  }
  console.log(`✅ Vector DB ready: ${vectorDB.length} entries`);
  try { fs.writeFileSync(VECTOR_CACHE_PATH, JSON.stringify({ cacheKey: key, vectorDB })); }
  catch (_) {}
}

function cosineSim(a, b) {
  const dot = a.reduce((s, v, i) => s + v * b[i], 0);
  return dot / (Math.sqrt(a.reduce((s, v) => s + v * v, 0)) * Math.sqrt(b.reduce((s, v) => s + v * v, 0)));
}

function applyMMR(candidates, topK, lambda = 0.6) {
  if (candidates.length <= topK) return candidates;
  const sel = [candidates[0]], rem = candidates.slice(1);
  while (sel.length < topK && rem.length > 0) {
    let bi = 0, bs = -Infinity;
    for (let i = 0; i < rem.length; i++) {
      const score = lambda * rem[i].score - (1 - lambda) * Math.max(...sel.map(s => cosineSim(rem[i].embedding, s.embedding)));
      if (score > bs) { bs = score; bi = i; }
    }
    sel.push(rem.splice(bi, 1)[0]);
  }
  return sel;
}

async function searchRelevantChunks(query, topK = 16) {
  if (vectorDB.length === 0) return chunks.slice(0, topK);
  const qEmb = await getEmbedding(query);
  if (!qEmb) return chunks.slice(0, topK);
  const scored = vectorDB.map(item => ({ ...item, score: cosineSim(qEmb, item.embedding) }))
    .sort((a, b) => b.score - a.score);
  const pool = scored.filter(i => i.score > 0.15).slice(0, topK * 2);
  return applyMMR(pool.length >= 3 ? pool : scored.slice(0, topK * 2), topK)
    .map(({ text, source, score }) => ({ text, source, score }));
}

function smartTruncate(text, maxWords = 500) {
  const w = text.trim().split(/\s+/);
  if (w.length <= maxWords) return text;
  return `${w.slice(0, 300).join(" ")} ... ${w.slice(-200).join(" ")}`;
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

When someone is dealing with a stressful situation — a bad partner, a dispute, a shaky agreement — acknowledge that first. Then help.

LANGUAGE RULE — STRICT
Detect the user's language and reply ENTIRELY in that language. No mixing.
  English message → English reply only
  Burmese message → Burmese reply only
  Mixed → match whichever language dominates

YOUR EXPERTISE
Partnership law — General Partnerships, LP, LLP, Myanmar Partnership Act, UPA, RUPA
Partnership agreements — drafting, enforcing, renegotiating, disputing
Fiduciary duties, partner liability, dissolution, winding up
Capital contributions, profit and loss allocation, valuations, buy-sell agreements
Sweat equity, vesting schedules, partner draws, tax treatment
Governance, voting rights, deadlock mechanisms, non-competes, exit strategies
SME best practices in Myanmar and Southeast Asia
General business advice, strategy, and operations

You have live Google Search — use it for current information. Use it silently without announcing it.

ANSWER STYLE
Get to the point. Use bullets/numbers only when it helps. Real examples beat theory.
Always finish completely — never cut off.
End every response with one genuinely useful follow-up question.

NEVER: mention documents/files/sources, use formal legal-brief language, start with filler words.`;

// ═══════════════════════════════════════════════════════════════════
// ROUTES
// ═══════════════════════════════════════════════════════════════════
app.get("/", (req, res) => res.sendFile(path.join(__dirname, "index.html")));
app.get("/admin", (req, res) => res.sendFile(path.join(__dirname, "admin.html")));

// ── AUTH: REGISTER ───────────────────────────────────────────────
app.post("/auth/register", async (req, res) => {
  const { email, password } = req.body;
  if (!email || !password)
    return res.status(400).json({ success: false, error: "Email and password are required." });
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email))
    return res.status(400).json({ success: false, error: "Invalid email address." });
  if (password.length < 6)
    return res.status(400).json({ success: false, error: "Password must be at least 6 characters." });

  const store = loadUsers();
  if (store.users.find(u => u.email.toLowerCase() === email.toLowerCase()))
    return res.status(409).json({ success: false, error: "An account with this email already exists." });

  const passwordHash = await bcrypt.hash(password, 12);
  const user = {
    id: crypto.randomUUID(),
    email: email.toLowerCase().trim(),
    passwordHash,
    role: "free",
    createdAt: new Date().toISOString(),
    daily: { date: todayStr(), count: 0 }
  };
  store.users.push(user);
  saveUsers(store);

  const token = jwt.sign({ id: user.id, email: user.email, role: user.role }, JWT_SECRET, { expiresIn: "30d" });
  console.log(`✅ New user: ${user.email}`);
  res.json({ success: true, token, user: { id: user.id, email: user.email, role: user.role, dailyCount: 0, dailyCap: FREE_DAILY_CAP } });
});

// ── AUTH: LOGIN ──────────────────────────────────────────────────
app.post("/auth/login", async (req, res) => {
  const { email, password } = req.body;
  if (!email || !password)
    return res.status(400).json({ success: false, error: "Email and password are required." });

  const user = findUserByEmail(email);
  if (!user || !(await bcrypt.compare(password, user.passwordHash)))
    return res.status(401).json({ success: false, error: "Incorrect email or password." });

  const expiresIn = req.body.rememberMe ? "30d" : "1d";
  const token = jwt.sign({ id: user.id, email: user.email, role: user.role }, JWT_SECRET, { expiresIn });
  const count = getDailyCount(user);
  console.log(`🔑 Login: ${user.email} (${user.role})`);
  res.json({ success: true, token, user: { id: user.id, email: user.email, role: user.role, dailyCount: count, dailyCap: FREE_DAILY_CAP } });
});

// ── AUTH: ME ─────────────────────────────────────────────────────
app.get("/auth/me", verifyToken, (req, res) => {
  const user = findUserById(req.user.id);
  if (!user) return res.status(404).json({ success: false, error: "User not found." });
  const count = getDailyCount(user);
  res.json({ success: true, user: { id: user.id, email: user.email, role: user.role, dailyCount: count, dailyCap: FREE_DAILY_CAP } });
});

// ── AUTH: UPGRADE WITH CODE ──────────────────────────────────────
app.post("/auth/upgrade", verifyToken, (req, res) => {
  const { code } = req.body;
  if (!code) return res.status(400).json({ success: false, error: "Please enter a premium code." });
  if (!PREMIUM_CODES.includes(code.trim()))
    return res.status(400).json({ success: false, error: "Invalid premium code." });

  const store = loadUsers();
  const user  = store.users.find(u => u.id === req.user.id);
  if (!user) return res.status(404).json({ success: false, error: "User not found." });
  if (user.role === "premium")
    return res.json({ success: true, alreadyPremium: true, message: "You're already a Premium member!" });

  user.role = "premium";
  saveUsers(store);

  // Issue fresh token with updated role
  const token = jwt.sign({ id: user.id, email: user.email, role: "premium" }, JWT_SECRET, { expiresIn: "30d" });
  console.log(`⭐ Upgraded to premium: ${user.email}`);
  res.json({ success: true, token, user: { id: user.id, email: user.email, role: "premium", dailyCount: 0, dailyCap: FREE_DAILY_CAP } });
});

// ── INIT SESSION ─────────────────────────────────────────────────
app.post("/init-session", verifyToken, (req, res) => {
  evictIfNeeded();
  const id = crypto.randomUUID();
  touchSession(id);
  console.log(`🆕 Session: ${id.slice(0,8)} for ${req.user.email}`);
  res.json({ sessionId: id });
});

// ── ADMIN: RELOAD DOCS ────────────────────────────────────────────
app.post("/reload-docs", async (req, res) => {
  if (!isAdmin(req)) return res.status(403).json({ success: false, error: "Forbidden." });
  await loadPDFs();
  if (fs.existsSync(VECTOR_CACHE_PATH)) fs.unlinkSync(VECTOR_CACHE_PATH);
  await buildVectorDB();
  res.json({ success: true, chunks: chunks.length, vectors: vectorDB.length });
});

// ── ADMIN: LIST USERS ─────────────────────────────────────────────
app.get("/admin/users", (req, res) => {
  if (!isAdmin(req)) return res.status(403).json({ success: false, error: "Forbidden." });
  const store = loadUsers();
  const safe  = store.users.map(u => ({
    id: u.id, email: u.email, role: u.role, createdAt: u.createdAt,
    dailyCount: getDailyCount(u), dailyDate: u.daily?.date || "—"
  }));
  res.json({ success: true, users: safe, total: safe.length,
    free: safe.filter(u => u.role === "free").length,
    premium: safe.filter(u => u.role === "premium").length });
});

// ── ADMIN: SET ROLE ───────────────────────────────────────────────
app.post("/admin/users/:id/role", (req, res) => {
  if (!isAdmin(req)) return res.status(403).json({ success: false, error: "Forbidden." });
  const { role } = req.body;
  if (!["free", "premium"].includes(role))
    return res.status(400).json({ success: false, error: "Role must be 'free' or 'premium'." });
  const store = loadUsers();
  const user  = store.users.find(u => u.id === req.params.id);
  if (!user) return res.status(404).json({ success: false, error: "User not found." });
  user.role = role;
  saveUsers(store);
  console.log(`👤 Admin set ${user.email} → ${role}`);
  res.json({ success: true, user: { id: user.id, email: user.email, role } });
});

// ── ADMIN: DELETE USER ────────────────────────────────────────────
app.delete("/admin/users/:id", (req, res) => {
  if (!isAdmin(req)) return res.status(403).json({ success: false, error: "Forbidden." });
  const store = loadUsers();
  const idx   = store.users.findIndex(u => u.id === req.params.id);
  if (idx === -1) return res.status(404).json({ success: false, error: "User not found." });
  const [removed] = store.users.splice(idx, 1);
  saveUsers(store);
  console.log(`🗑️  Admin deleted: ${removed.email}`);
  res.json({ success: true, deleted: removed.email });
});

// ── ADMIN: RESET PASSWORD ─────────────────────────────────────────
app.post("/admin/users/:id/reset-password", async (req, res) => {
  if (!isAdmin(req)) return res.status(403).json({ success: false, error: "Forbidden." });
  const { newPassword } = req.body;
  if (!newPassword || newPassword.length < 6)
    return res.status(400).json({ success: false, error: "Password must be at least 6 characters." });
  const store = loadUsers();
  const user  = store.users.find(u => u.id === req.params.id);
  if (!user) return res.status(404).json({ success: false, error: "User not found." });
  user.passwordHash = await bcrypt.hash(newPassword, 12);
  saveUsers(store);
  res.json({ success: true, message: `Password reset for ${user.email}` });
});

// ── CHAT API — STREAMING ─────────────────────────────────────────
app.post("/chat", verifyToken, async (req, res) => {
  try {
    const userMessage = req.body.message?.trim();
    const sessionId   = req.body.sessionId;
    const userId      = req.user.id;

    if (!userMessage) {
      res.setHeader("Content-Type", "text/event-stream");
      res.write(`data: ${JSON.stringify({ text: "Please type a message." })}\n\n`);
      res.write(`data: [DONE]\n\n`);
      return res.end();
    }

    // Validate session
    if (!sessionId || !sessions[sessionId]) {
      res.setHeader("Content-Type", "text/event-stream");
      res.write(`data: ${JSON.stringify({ text: "Session expired. Please refresh the page." })}\n\n`);
      res.write(`data: [DONE]\n\n`);
      return res.end();
    }

    // ── RATE LIMIT: free users only ─────────────────────────────
    const userRecord = findUserById(userId);
    if (!userRecord) {
      res.setHeader("Content-Type", "text/event-stream");
      res.write(`data: ${JSON.stringify({ text: "Account not found. Please log in again." })}\n\n`);
      res.write(`data: [DONE]\n\n`);
      return res.end();
    }

    if (userRecord.role === "free") {
      const count = getDailyCount(userRecord);
      if (count >= FREE_DAILY_CAP) {
        res.setHeader("Content-Type", "text/event-stream");
        res.write(`data: ${JSON.stringify({ limitReached: true, count, cap: FREE_DAILY_CAP })}\n\n`);
        res.write(`data: [DONE]\n\n`);
        return res.end();
      }
    }

    const sess = touchSession(sessionId);

    const truncated      = smartTruncate(userMessage);
    const relevantChunks = await searchRelevantChunks(truncated, 16);

    const DOC_THRESHOLD = 0.45;
    const topScore = relevantChunks[0]?.score ?? 0;
    const useDocContext = topScore >= DOC_THRESHOLD;
    let fullSystemInstruction;

    if (useDocContext) {
      console.log(`📚 Doc match (${(topScore*100).toFixed(0)}%) [${req.user.email}]`);
      const pdfContext = relevantChunks.filter(c => c.text).map(c => c.text).join("\n\n---\n\n");
      fullSystemInstruction = `${systemPrompt}\n\n[BACKGROUND — treat as your own knowledge. Never cite it.]\n${pdfContext}\n[END BACKGROUND]\nAnswer naturally. Never mention files or documents. Complete answer fully, then end with one useful follow-up question.`;
    } else {
      console.log(`🌐 General mode (${(topScore*100).toFixed(0)}%) [${req.user.email}]`);
      fullSystemInstruction = `${systemPrompt}\n\nUse your deep expertise and Google Search for current info. Go thorough: legal, practical, strategic, financial angles. Real examples and numbers beat abstract advice. Complete answer fully, then end with one useful follow-up question.`;
    }

    const geminiHistory = sess.history.map(m => ({
      role:  m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content }]
    }));

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders();

    const heartbeat = setInterval(() => res.write(`: heartbeat\n\n`), 15000);
    let fullReply = "";

    try {
      const model  = genAI.getGenerativeModel({
        model: "gemini-2.5-flash",
        systemInstruction: fullSystemInstruction,
        tools: [{ googleSearch: {} }],
        generationConfig: { temperature: 0.6, maxOutputTokens: 8000 }
      });
      const chat   = model.startChat({ history: geminiHistory });
      const result = await chat.sendMessageStream(userMessage);
      const iter   = result.stream[Symbol.asyncIterator]();

      while (true) {
        let next;
        try { next = await iter.next(); } catch (_) { continue; }
        if (next.done) break;
        try {
          const text = next.value.text();
          if (text) { fullReply += text; res.write(`data: ${JSON.stringify({ text })}\n\n`); }
        } catch (_) {}
      }

      sess.history.push({ role: "user",      content: userMessage });
      sess.history.push({ role: "assistant", content: fullReply   });
      sess.history = sess.history.slice(-MAX_HISTORY);
      sess.lastActive = Date.now();

      // Increment daily count AFTER successful response (free users only)
      if (userRecord.role === "free") incrementDailyCount(userId);

      // Send updated count to client
      const newCount = userRecord.role === "free" ? getDailyCount(findUserById(userId)) : null;
      res.write(`data: ${JSON.stringify({ dailyCount: newCount, dailyCap: userRecord.role === "free" ? FREE_DAILY_CAP : null })}\n\n`);
      res.write(`data: [DONE]\n\n`);

    } catch (streamErr) {
      console.error("❌ Stream error:", streamErr.message);
      if (fullReply.length > 0) {
        res.write(`data: [DONE]\n\n`);
      } else {
        const hint =
          streamErr.message?.includes("429") ? "Too many requests. Please wait a moment."
        : streamErr.message?.includes("503") ? "AI service temporarily unavailable."
        : "Something went wrong. Please try again.";
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
      res.write(`data: ${JSON.stringify({ text: "Server error. Please try again." })}\n\n`);
      res.write(`data: [DONE]\n\n`);
      res.end();
    }
  }
});

// ── CLEAR SESSION ────────────────────────────────────────────────
app.post("/clear-session", verifyToken, (req, res) => {
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
  console.log(`🔐 Admin: ${ADMIN_SECRET ? "ENABLED" : "DISABLED — set ADMIN_SECRET in .env"}`);
  console.log(`💎 Premium codes: ${PREMIUM_CODES.length} loaded`);
  console.log(`📊 Free daily cap: ${FREE_DAILY_CAP} messages`);
});
