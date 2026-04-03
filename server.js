import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
dotenv.config();
import express from "express";
import OpenAI from "openai";
import cors from "cors";
import rateLimit from "express-rate-limit";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

app.use(rateLimit({ windowMs: 60 * 1000, max: 50 }));
app.use(cors());
app.use(express.json());

// ==========================
// OPENAI
// ==========================

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ==========================
// WEB SEARCH
// ==========================

async function webSearch(query) {
  try {
    const response = await openai.responses.create({
      model: "gpt-4o",
      tools: [{ type: "web_search_preview" }],
      input: `Search for real-world case studies, legal precedents, and expert advice about: ${query}. Focus on partnership business disputes, share structures, investment rules, exit strategies, and business conflict resolution.`
    });

    const text = response.output
      .filter(block => block.type === "message")
      .flatMap(block => block.content)
      .filter(c => c.type === "output_text")
      .map(c => c.text)
      .join("\n");

    return text || null;
  } catch (err) {
    console.log("❌ Web search error:", err.message);
    return null;
  }
}

function needsWebSearch(message) {
  const triggers = [
    "case study", "case studies", "real world", "example",
    "scenario", "situation", "what should i", "should we",
    "advice", "help me decide", "recommend", "suggestion",
    "my partner", "our company", "our business", "we have",
    "problem with", "issue with", "dispute", "conflict",
    "disagreement", "argument", "fighting", "not contributing",
    "exit", "leaving", "want to leave", "selling shares",
    "new partner", "investor", "how do other", "what do successful",
    "industry standard", "best practice", "common mistake",
    "failed", "success story", "what happens when", "risk",
    "legal", "law", "contract", "agreement", "penalty"
  ];
  const lower = message.toLowerCase();
  return triggers.some(k => lower.includes(k));
}

// ==========================
// EMBEDDING
// ==========================

async function getEmbedding(text) {
  try {
    if (!text || typeof text !== "string") return null;
    const cleanText = text.replace(/\s+/g, " ").slice(0, 2000);
    if (!cleanText) return null;
    const res = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: cleanText
    });
    return res.data[0].embedding;
  } catch (err) {
    console.log("❌ Embedding error:", err.message);
    return null;
  }
}

// ==========================
// MEMORY STORAGE
// ==========================

let chunks = [];
let vectorDB = [];

// Per-session chat history stored by sessionId
// Format: { sessionId: [ {role, content}, ... ] }
const sessionHistories = {};

const MAX_HISTORY = 20; // keep last 20 messages per session for strong memory

// ==========================
// TEXT SPLITTER
// ==========================

function splitText(text, size = 300) {
  const words = text.split(/\s+/);
  let result = [];
  for (let i = 0; i < words.length; i += size) {
    result.push(words.slice(i, i + size).join(" "));
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
    let allText = "";

    for (const file of files) {
      if (!file.toLowerCase().endsWith(".pdf")) continue;
      const pdfPath = path.join(folderPath, file);
      console.log("📄 Loading:", file);

      try {
        const data = new Uint8Array(fs.readFileSync(pdfPath));
        const pdf = await pdfjsLib.getDocument({ data }).promise;

        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const content = await page.getTextContent();
          const strings = content.items
            .map(item => item.str)
            .filter(str => str && str.trim().length > 0);
          allText += strings.join(" ") + "\n";
        }
      } catch (fileErr) {
        console.log(`❌ Failed to read ${file}:`, fileErr.message);
      }
    }

    if (!allText.trim()) {
      console.log("⚠️ No text extracted from PDFs");
      return;
    }

    chunks = splitText(allText, 300);
    console.log(`✅ Created ${chunks.length} chunks`);
    console.log("🔥 Sample:", chunks[0]?.slice(0, 200));
  } catch (err) {
    console.log("❌ PDF ERROR:", err.message);
  }
}

async function buildVectorDB() {
  console.log("🔄 Building vector DB...");
  for (const chunk of chunks) {
    const embedding = await getEmbedding(chunk);
    if (!embedding) continue;
    vectorDB.push({ text: chunk, embedding });
  }
  console.log("✅ Vector DB ready:", vectorDB.length);
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

async function searchRelevantChunks(query, topK = 8) {
  const queryEmbedding = await getEmbedding(query);
  if (!queryEmbedding) return chunks.slice(0, 8);

  return vectorDB
    .map(item => ({
      text: item.text,
      score: cosineSimilarity(queryEmbedding, item.embedding)
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(i => i.text);
}

// ==========================
// LANGUAGE DETECTION
// ==========================

function detectLanguage(text) {
  // Detect if message contains Burmese script
  const burmeseRegex = /[\u1000-\u109F\uAA60-\uAA7F]/;
  if (burmeseRegex.test(text)) return "burmese";

  const lower = text.toLowerCase();
  if (lower.includes("burmese") || lower.includes("myanmar") || lower.includes("မြန်မာ")) return "burmese";
  if (lower.includes("english")) return "english";

  return "english"; // default
}

// ==========================
// SCENARIO DIAGNOSIS ENGINE
// ==========================

function diagnoseScenario(message) {
  const lower = message.toLowerCase();

  const categories = {
    capital: ["capital", "contribution", "invest", "money", "fund", "paid", "deposit", "deadline", "late"],
    shares: ["share", "equity", "ownership", "percent", "stake", "dilute", "transfer"],
    labor: ["work", "effort", "contribute", "labor", "service", "salary", "sweat", "doing nothing"],
    profit: ["profit", "dividend", "loss", "income", "earning", "distribution", "payout"],
    financial: ["account", "bank", "audit", "financial", "expense", "spending", "budget"],
    leadership: ["decision", "leader", "ceo", "director", "manage", "authority", "power", "control", "vote"],
    exit: ["exit", "leave", "quit", "sell", "buyout", "leaving", "withdraw", "out"],
    death: ["death", "die", "inheritance", "heir", "spouse", "will", "estate"],
    transfer: ["transfer", "sell shares", "new owner", "third party", "outside"],
    dispute: ["dispute", "conflict", "fight", "argue", "disagree", "problem", "issue", "unfair", "cheating"]
  };

  const matched = [];
  for (const [category, keywords] of Object.entries(categories)) {
    if (keywords.some(k => lower.includes(k))) {
      matched.push(category);
    }
  }

  return matched.length > 0 ? matched : ["general"];
}

// ==========================
// LOAD PDFs ON START
// ==========================

await loadPDFs();
await buildVectorDB();

// ==========================
// SYSTEM PROMPT
// ==========================

const systemPrompt = `
You are "Sayar Nyan Lin Aung's AI Chat" — an expert Partnership Business Rules (PBR) Consultant trained on the complete PBR course by Nyan Lin Aung, Business Coach & Trainer, "Unlock Your Future".

Your motto: "Without Rules, we all go back to the jungle."

## CRITICAL IDENTITY RULE
- ALWAYS start EVERY response with: "I am Sayar Nyan Lin Aung's AI Chat 🤝"
- Never forget this opening line in any response.

## LANGUAGE RULE — BILINGUAL EVERY RESPONSE
- ALWAYS respond in BOTH English AND Burmese in every single response.
- Format: Give the full answer in English first, then provide the full answer again in Burmese below it.
- Separate the two sections with a divider like: ──────────────────
- Label sections clearly: "🇬🇧 English:" and "🇲🇲 Burmese (မြန်မာဘာသာ):"

## YOUR CONSULTING APPROACH (Follow this order every time)

### Step 1 — DIAGNOSE FIRST
- Start by identifying which PBR chapter(s) the user's situation falls under
- State clearly: "This situation involves: [Chapter Name(s)]"
- Identify the root problem, not just the surface issue

### Step 2 — ASK ONE CLARIFYING QUESTION (if critical info is missing)
- If you need ONE key piece of information to give better advice, ask it first
- Keep the question short and focused
- Example: "Before I advise — do you have a written partnership agreement?"

### Step 3 — GIVE DIRECT ADVICE IMMEDIATELY
- Do NOT say "it depends" without also giving a clear direction
- Be confident and direct like a senior consultant
- If you have enough info, give advice right away

### Step 4 — SHOW OPTIONS (Option A vs Option B)
- Always present at least 2 options with clear pros and cons
- Format:
  ✅ Option A: [Name] — [What to do]
  • Pro: ...
  • Con: ...
  
  ✅ Option B: [Name] — [What to do]
  • Pro: ...
  • Con: ...

  💡 My Recommendation: [Clear recommendation with reason]

### Step 5 — ALWAYS END WITH NEXT STEPS
- End every response with a "📋 Next Steps" section
- List 3–5 concrete actions the user should take immediately
- Include what to document, who to talk to, and what to watch out for

## DEEPER SCENARIO DIAGNOSIS
When a user shares a scenario, always analyze:
1. What PBR chapter(s) apply?
2. What is the ROOT cause vs the surface symptom?
3. What RISKS exist if they do nothing?
4. What have other businesses done in similar situations?
5. What does the PBR framework specifically recommend?

## PBR FRAMEWORK — YOUR CORE KNOWLEDGE (10 Chapters)
1. **Capital** — contribution amounts, deadlines, penalties; options: dilution / forfeiture / convert to loan / eject
2. **Shares** — par value, share formula (Total Capital ÷ Par Value), face/book/market/intrinsic value
3. **Labor Value** — how to value service contributions; compensation: profit margin, equity, or salary
4. **Profit & Loss** — profit sharing based on EAT; BOD approves dividends; profit ≠ cash; retained earnings policy
5. **Financial Management** — GAAP, 2-signatory bank account, no mixing funds, CapEx needs unanimous BOD, annual audit
6. **Leadership** — McKinsey 7S, major vs minor decisions, non-compete, misconduct rules, asset usage rules
7. **Exit Rules** — offer to internals first at Book Value −10–20%; 7-day window; lock-up period
8. **Death & Inheritance** — spouse consent at purchase; define if heir gets shares only or shares + leadership
9. **Share Transfer** — written notice, 7-day response, all transfers via company bank, money released after name transfer
10. **Dispute Resolution** — 6 methods in order: mediation → committee → majority vote → third-party binding → shareholder weighted vote → buyout

## KEY FORMULAS
- Shares: Total Capital ÷ Par Value
- GPM: (Revenue − COGS) / Revenue × 100
- BEP: Fixed Costs ÷ (Price − Variable Cost per Unit)
- ROI: Net Profit / Investment × 100
- Profit sharing always uses EAT, not gross profit
- Start-Up Capital: Fixed Costs + Working Capital + Contingency Fund

## MEMORY & CONTEXT
- You have memory of this conversation. Always refer back to what the user has already told you.
- Never ask for information the user already provided earlier in the conversation.
- Build on previous answers to give increasingly specific advice.

## TONE & STYLE
- Warm, direct, and confident — like a trusted senior business consultant
- Use emojis for section headers to improve readability
- Validate emotions first in sensitive situations (partner disputes, someone leaving)
- Never be vague — always give a clear direction

## FIRST MESSAGE GREETING
For the very first message in a conversation, greet as:
"I am Sayar Nyan Lin Aung's AI Chat 🤝 — Your Partnership Business Rules Expert Consultant. Tell me about your partnership situation and I will give you clear, direct advice based on the full PBR framework."
`;

// ==========================
// ROUTES
// ==========================

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

// ==========================
// CHAT API
// ==========================

app.post("/chat", async (req, res) => {
  try {
    const userMessage = req.body.message?.trim();
    const sessionId = req.body.sessionId || "default";

    if (!userMessage) {
      return res.json({ reply: "Please type a message. | စာတစ်ခုခု ရိုက်ထည့်ပါ။" });
    }

    // Initialize session history if new session
    if (!sessionHistories[sessionId]) {
      sessionHistories[sessionId] = [];
    }

    // Diagnose scenario categories for better PDF search
    const diagnosedCategories = diagnoseScenario(userMessage);
    console.log("🔍 Diagnosed categories:", diagnosedCategories);

    // Enhanced search query combining user message + diagnosed categories
    const enhancedQuery = `${userMessage} ${diagnosedCategories.join(" ")}`;

    // Search PDF knowledge base with enhanced query
    let relevantChunks = await searchRelevantChunks(enhancedQuery, 8);
    let pdfContext = relevantChunks.join("\n\n");

    if (!pdfContext || pdfContext.trim().length < 50) {
      pdfContext = chunks.slice(0, 10).join("\n\n");
    }

    // Web search for real-world case studies when needed
    let webContext = null;
    if (needsWebSearch(userMessage)) {
      console.log("🌐 Running web search for:", userMessage.slice(0, 80));
      webContext = await webSearch(userMessage);
      if (webContext) {
        console.log("✅ Web search results received");
      }
    }

    // Add user message to session history
    sessionHistories[sessionId].push({ role: "user", content: userMessage });

    // Keep last MAX_HISTORY messages for strong memory
    sessionHistories[sessionId] = sessionHistories[sessionId].slice(-MAX_HISTORY);

    // Build context blocks
    const contextBlocks = [
      {
        role: "system",
        content: `DIAGNOSED PBR CHAPTERS FOR THIS QUERY: ${diagnosedCategories.join(", ").toUpperCase()}\nFocus your answer on these chapters first.`
      },
      {
        role: "system",
        content: `PBR Course Knowledge Base (from uploaded PDFs):\n${pdfContext}`
      }
    ];

    if (webContext) {
      contextBlocks.push({
        role: "system",
        content: `Real-World Case Studies & Industry Data (from web search):\n${webContext}`
      });
    }

    // Build full messages array with memory
    const messages = [
      { role: "system", content: systemPrompt },
      ...contextBlocks,
      ...sessionHistories[sessionId]
    ];

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages,
      temperature: 0.4, // slightly lower for more consistent, precise consulting advice
      max_tokens: 3000  // increased for bilingual responses
    });

    const reply = response.choices?.[0]?.message?.content || "⚠️ No response received.";

    // Save assistant reply to session history
    sessionHistories[sessionId].push({ role: "assistant", content: reply });

    // Clean up old sessions (keep max 100 sessions in memory)
    const sessionKeys = Object.keys(sessionHistories);
    if (sessionKeys.length > 100) {
      delete sessionHistories[sessionKeys[0]];
    }

    res.json({ reply, diagnosedCategories });

  } catch (err) {
    console.error("❌ ERROR:", err.message);
    res.json({
      reply: "⚠️ Server error occurred. Please try again. | Server error ဖြစ်ပါတယ်။ ထပ်စမ်းကြည့်ပါ။"
    });
  }
});

// ==========================
// CLEAR SESSION (optional endpoint to reset memory)
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