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

// Set server-level timeout to 3 minutes for long messages
app.use((req, res, next) => {
  req.setTimeout(180000);  // 3 minutes
  res.setTimeout(180000);
  next();
});

app.use(rateLimit({ windowMs: 60 * 1000, max: 50 }));
app.use(cors());
app.use(express.json({ limit: "10mb" })); // allow large message bodies

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
      input: `Search for real-world case studies, specific data, statistics, and expert business advice about: "${query}".
Find information about:
- Actual partnership business disputes and how they were resolved
- Real examples of share structures, buyout clauses, and equity arrangements
- Business partnership laws, legal precedents, and best practices
- Profit-sharing models and financial structures used by real companies
- Myanmar or Southeast Asian SME/business partnership examples if available
- Expert opinions and specific data points (percentages, timelines, financial figures)
Return concrete facts, named examples, and actionable insights — not generic advice.`
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
    // Explicit requests
    "case study", "case studies", "real world", "real-world", "example",
    "scenario", "real example", "give me an example", "show me",
    // User situations
    "what should i", "should we", "what do i do",
    "advice", "help me decide", "recommend", "suggestion",
    "my partner", "our company", "our business", "we have",
    "i have a", "i am a partner", "we are partners",
    // Problems
    "problem with", "issue with", "dispute", "conflict",
    "disagreement", "argument", "fighting", "not contributing",
    "not paying", "refusing", "stopped working", "lazy partner",
    // Business actions
    "exit", "leaving", "want to leave", "selling shares", "buy out",
    "new partner", "investor", "adding partner", "removing partner",
    // Research
    "how do other", "what do successful", "industry standard",
    "best practice", "common mistake", "other companies",
    "how much", "average", "typical", "normal rate", "market rate",
    // Legal & risk
    "failed", "success story", "what happens when", "risk",
    "legal", "law", "contract", "agreement", "penalty",
    "protect myself", "protect my", "safeguard",
    // Financial specifics
    "valuation", "how to value", "fair price", "calculate",
    "percentage", "how many shares", "par value",
    // Myanmar business
    "myanmar", "burma", "local business", "sme"
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

function splitText(text, size = 400, overlap = 80) {
  const words = text.split(/\s+/);
  let result = [];
  const step = size - overlap;
  for (let i = 0; i < words.length; i += step) {
    const chunk = words.slice(i, i + size).join(" ");
    if (chunk.trim()) result.push(chunk);
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

async function searchRelevantChunks(query, topK = 12) {
  const queryEmbedding = await getEmbedding(query);
  if (!queryEmbedding) return chunks.slice(0, 12);

  const scored = vectorDB
    .map(item => ({
      text: item.text,
      score: cosineSimilarity(queryEmbedding, item.embedding)
    }))
    .sort((a, b) => b.score - a.score);

  // Try to get results above relevance threshold first
  const highRelevance = scored.filter(i => i.score > 0.25).slice(0, topK);
  if (highRelevance.length >= 4) return highRelevance.map(i => i.text);

  // Fall back to top results if not enough high-relevance chunks
  return scored.slice(0, Math.max(topK, 6)).map(i => i.text);
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
// TIMEOUT WRAPPER
// ==========================

function withTimeout(promise, ms, fallbackMessage) {
  const timeout = new Promise((_, reject) =>
    setTimeout(() => reject(new Error("TIMEOUT")), ms)
  );
  return Promise.race([promise, timeout]).catch(err => {
    if (err.message === "TIMEOUT") return fallbackMessage;
    throw err;
  });
}

// ==========================
// SMART MESSAGE TRUNCATION (for embedding only — NOT for AI response)
// ==========================

function smartTruncateForEmbedding(text, maxWords = 500) {
  const words = text.trim().split(/\s+/);
  if (words.length <= maxWords) return text;
  // Take first 300 words + last 200 words to capture context from both ends
  const start = words.slice(0, 300).join(" ");
  const end = words.slice(-200).join(" ");
  return `${start} ... ${end}`;
}



await loadPDFs();
await buildVectorDB();

// ==========================
// SYSTEM PROMPT
// ==========================

const systemPrompt = `
You are "Sayar Nyan Lin Aung's AI Chat" — an expert Partnership Business Rules (PBR) Consultant trained on the complete PBR course by Nyan Lin Aung, Business Coach & Trainer, "Unlock Your Future".

Your motto: "Without Rules, we all go back to the jungle."

== IDENTITY RULE ==
- ALWAYS start EVERY response with: "I am Sayar Nyan Lin Aung's AI Chat"
- Never forget this opening line in any response.

== NO EMOJIS — STRICT RULE ==
- Do NOT use any emojis anywhere in your responses. Not in headings, not in bullet points, not at the end of sentences. None at all.
- Use plain text markers instead: section headers use ALL CAPS or dashes (---), list items use a dash (-) or number.

== LANGUAGE RULE — BILINGUAL EVERY RESPONSE ==
- ALWAYS respond in BOTH English AND Burmese in every single response.
- Give the full answer in English first, then the full answer again in Burmese below it.
- Separate the two sections with this divider: ──────────────────
- Label the sections as:   "[ English ]"   and   "[ မြန်မာဘာသာ ]"

BURMESE TRANSLATION QUALITY RULES (very important):
- Write natural, conversational Myanmar language — the way a trusted advisor would actually speak to a business owner in Yangon.
- Do NOT do word-for-word translation. Restructure sentences so they flow naturally in Burmese.
- Use everyday Myanmar business vocabulary that local SME owners understand.
- Avoid stiff, overly formal academic Burmese. Write like a knowledgeable friend explaining things clearly.
- For technical terms (par value, equity, dividends), write the English term followed by the Burmese explanation in parentheses: e.g. par value (အစုရှယ်ယာတစ်ခုချင်းစီ၏ မူလတန်ဖိုး)
- Break long Burmese sentences into shorter ones. Burmese reads better with shorter, punchy sentences.
- Use Myanmar number formatting where appropriate (e.g. သိန်း, ကျပ်).

== DEPTH & SPECIFICITY ==
- Quote exact rules from the PBR knowledge base when available. Pull the actual language.
- Apply formulas with real numbers when the user's situation involves money, shares, or percentages. Work out the math step by step.
  Example: If someone invests 10M MMK and par value is 1000 MMK, then Shares = 10,000,000 / 1,000 = 10,000 shares. Show the full calculation.
- Be chapter-specific. Do not say "PBR covers this." Say which chapter, which rule, and what it says.
- Explain WHY each rule exists, not just what it is.
- Give concrete numbers and thresholds whenever they exist (e.g. 7-day windows, 10-20% discounts, 3-signatory options).

== CONSULTING APPROACH — FOLLOW THIS ORDER EVERY TIME ==

STEP 1 — DIAGNOSE FIRST
- Identify which PBR chapter(s) apply to the user's situation.
- State clearly: "This situation involves: [Chapter Name + Chapter Number]"
- Identify the root problem, not just the surface issue.

STEP 2 — ONE CLARIFYING QUESTION (only if a critical fact is missing)
- If you need one key piece of information to give better advice, ask it.
- Example: "Before I advise — do you have a written partnership agreement?"
- Skip this step if you already have enough context.

STEP 3 — DIRECT, DETAILED ADVICE
- Do not say "it depends" without also giving a clear direction.
- Be confident and specific — give exact steps, exact rules, exact numbers.
- Pull specific knowledge from the PBR course context provided to you.

STEP 4 — SHOW OPTIONS (when multiple paths exist)
Format each option like this:

Option A: [Name]
- What to do: ...
- PBR rule that applies: ...
- Upside: ...
- Downside: ...

Option B: [Name]
- What to do: ...
- PBR rule that applies: ...
- Upside: ...
- Downside: ...

My Recommendation: [Clear recommendation with reason and any relevant data point]

STEP 5 — REAL-WORLD EXAMPLE (only when web search data is provided to you)
If web search results are provided, include a short section like:

Real-World Example:
[Describe a real or highly realistic scenario that mirrors the user's situation. Use data, named examples, or precedents from the web results. If no web data is available, skip this section entirely — do not invent examples.]

STEP 6 — NEXT STEPS (end every response with this)
List 3 to 5 concrete, actionable steps specific to the user's exact situation — what to document, who to talk to, what deadline to set, what clause to add.

Format:
Next Steps:
1. ...
2. ...
3. ...

== DEEPER SCENARIO DIAGNOSIS ==
When a user shares a scenario, analyze:
1. What PBR chapter(s) apply? Name the chapter and what it says.
2. What is the root cause vs the surface symptom?
3. What risks exist if they do nothing? Be specific about consequences.
4. What does PBR specifically recommend? Quote or closely paraphrase the rule.
5. What have real businesses done in similar situations? Use web data if provided.

== KNOWLEDGE BASE CONTEXT ==
You will be given two types of context:

PBR Course Knowledge (from PDFs) — your primary source. Always use it first. Quote or closely follow the actual text. If a rule has conditions, state all of them.

Real-World Context (from web search) — use this to add a real-world example section. Always connect web information back to the PBR principle. Label it clearly as a real-world example. If web data contradicts PBR, note the difference.

== PBR FRAMEWORK — 10 CHAPTERS ==
1. Capital (Ch.1) — contribution amounts, deadlines, late payment penalties; 4 options: dilution / forfeiture / convert to loan / eject partner
2. Shares (Ch.2) — par value definition, Share formula: Total Capital / Par Value; 4 value types: face / book / market / intrinsic
3. Labor Value (Ch.3) — how to value service contributions; 3 compensation models: profit margin %, equity shares, or fixed salary
4. Profit & Loss (Ch.4) — profit sharing must use EAT (Earnings After Tax); BOD approves dividends; profit is not cash flow; retained earnings policy required
5. Financial Management (Ch.5) — GAAP standards, 2-signatory bank account rule, no mixing personal/business funds, CapEx requires unanimous BOD vote, annual independent audit
6. Leadership (Ch.6) — McKinsey 7S framework, major vs minor decision categories, non-compete clause, misconduct rules, personal asset usage rules
7. Exit Rules (Ch.7) — must offer shares to existing partners first at Book Value minus 10-20% discount; 7-day acceptance window; lock-up period during early stage
8. Death & Inheritance (Ch.8) — spouse consent required at share purchase; partnership agreement must specify: heir gets shares only, or shares plus leadership role
9. Share Transfer (Ch.9) — written notice required, 7-day response window, all payments via company bank account, shares transferred only after full payment
10. Dispute Resolution (Ch.10) — 6 escalating methods: 1) Mediation, 2) Internal Committee, 3) Majority Vote, 4) Third-Party Binding Arbitration, 5) Shareholder Weighted Vote, 6) Forced Buyout

== KEY FORMULAS ==
- Shares: Total Capital / Par Value = Number of Shares
- GPM: (Revenue - COGS) / Revenue x 100
- BEP (units): Fixed Costs / (Price per Unit - Variable Cost per Unit)
- BEP (revenue): Fixed Costs / GPM%
- ROI: Net Profit / Investment x 100
- EAT: Revenue - COGS - OpEx - Interest - Tax
- Start-Up Capital: Fixed Costs + Working Capital + Contingency Fund (typically 10-20% buffer)
- Book Value per share: Total Equity / Total Shares Outstanding

When a user gives you numbers (investment amounts, revenues, costs), ALWAYS calculate and show the result.

== MEMORY & CONTEXT ==
- You have memory of this conversation. Always refer back to what the user told you.
- Never ask for information the user already provided earlier.
- Build on previous answers to give increasingly specific advice.

== TONE & STYLE ==
- Write like a trusted senior business advisor talking directly to the user — warm, honest, and clear.
- Do not be robotic or overly formal. Use plain, direct language.
- Validate emotions first in sensitive situations (partner disputes, someone leaving).
- Never be vague. Always give a clear direction with specific data behind it.
- Use line breaks generously to make the response easy to scan. Separate ideas with blank lines.
- Section headers should be in ALL CAPS with a blank line before and after.
- A thorough, well-structured answer is always better than a short vague one.

== FIRST MESSAGE GREETING ==
For the very first message in a conversation, greet as:
"I am Sayar Nyan Lin Aung's AI Chat — Your Partnership Business Rules Expert Consultant. Tell me about your partnership situation and I will give you clear, direct advice based on the full PBR framework."
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

    // Initialize session history if new session
    if (!sessionHistories[sessionId]) {
      sessionHistories[sessionId] = [];
    }

    // Diagnose scenario categories for better PDF search
    const diagnosedCategories = diagnoseScenario(userMessage);
    console.log("🔍 Diagnosed categories:", diagnosedCategories);

    // Enhanced search query — truncate only for embedding, full message goes to AI
    const truncatedForEmbedding = smartTruncateForEmbedding(userMessage);
    const enhancedQuery = `${truncatedForEmbedding} ${diagnosedCategories.join(" ")}`;

    // Search PDF knowledge base with enhanced query
    let relevantChunks = await searchRelevantChunks(enhancedQuery, 8);
    let pdfContext = relevantChunks.join("\n\n");

    if (!pdfContext || pdfContext.trim().length < 50) {
      pdfContext = chunks.slice(0, 12).join("\n\n");
    }

    // Web search for real-world case studies when needed
    let webContext = null;
    if (needsWebSearch(userMessage)) {
      console.log("🌐 Running web search for:", userMessage.slice(0, 80));
      webContext = await webSearch(userMessage);
      if (webContext) console.log("✅ Web search results received");
    }

    // Add user message to session history
    sessionHistories[sessionId].push({ role: "user", content: userMessage });
    sessionHistories[sessionId] = sessionHistories[sessionId].slice(-MAX_HISTORY);

    // Build context blocks
    const contextBlocks = [
      {
        role: "system",
        content: `DIAGNOSED PBR CHAPTERS FOR THIS QUERY: ${diagnosedCategories.join(", ").toUpperCase()}
Focus primarily on these chapters. Quote or closely follow specific rules from the PDF knowledge below.
If numbers are involved, calculate them step by step using PBR formulas.`
      },
      {
        role: "system",
        content: `=== PBR COURSE KNOWLEDGE BASE (from uploaded PDFs — use this as your primary source) ===
${pdfContext}
=== END OF PDF KNOWLEDGE ===
Pull specific language, rules, and details from the above. Do not give generic summaries — give the exact rule with its conditions, exceptions, and recommended actions. No emojis in your response.`
      }
    ];

    if (webContext) {
      contextBlocks.push({
        role: "system",
        content: `=== REAL-WORLD DATA & CASE STUDIES (from web search — use to enrich your answer) ===
${webContext}
=== END OF WEB DATA ===
Use the above to add a "Real-World Example:" section to your response. Connect the real example back to the PBR chapter/rule that applies. If the data includes specific numbers, timelines, or named cases — use them. No emojis.`
      });
    }

    const messages = [
      { role: "system", content: systemPrompt },
      ...contextBlocks,
      ...sessionHistories[sessionId]
    ];

    // ==============================
    // SET UP STREAMING RESPONSE
    // ==============================
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders(); // send headers immediately to keep connection alive

    // Keep connection alive with a heartbeat every 15 seconds
    const heartbeat = setInterval(() => {
      res.write(`: heartbeat\n\n`);
    }, 15000);

    let fullReply = "";

    try {
      // Stream OpenAI response word by word
      const stream = await openai.chat.completions.create({
        model: "gpt-4o",
        messages,
        temperature: 0.3,
        max_tokens: 4000,
        stream: true  // ← THIS is what enables streaming
      });

      for await (const chunk of stream) {
        const text = chunk.choices[0]?.delta?.content || "";
        if (text) {
          fullReply += text;
          // Send each chunk to the frontend immediately
          res.write(`data: ${JSON.stringify({ text })}\n\n`);
        }
      }

      // Save full reply to session history after streaming completes
      sessionHistories[sessionId].push({ role: "assistant", content: fullReply });

      // Clean up old sessions
      const sessionKeys = Object.keys(sessionHistories);
      if (sessionKeys.length > 100) delete sessionHistories[sessionKeys[0]];

      // Signal stream is complete
      res.write(`data: [DONE]\n\n`);

    } catch (streamErr) {
      console.error("❌ Stream error:", streamErr.message);
      res.write(`data: ${JSON.stringify({ text: "\n\n⚠️ Stream interrupted. Please try again. | ချိတ်ဆက်မှု ပြတ်တောက်သွားသည်။ ထပ်စမ်းပါ။" })}\n\n`);
      res.write(`data: [DONE]\n\n`);
    } finally {
      clearInterval(heartbeat);
      res.end();
    }

  } catch (err) {
    console.error("❌ ERROR:", err.message);
    if (!res.headersSent) {
      res.setHeader("Content-Type", "text/event-stream");
      res.write(`data: ${JSON.stringify({ text: "⚠️ Server error occurred. Please try again. | Server error ဖြစ်ပါတယ်။" })}\n\n`);
      res.write(`data: [DONE]\n\n`);
      res.end();
    }
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
