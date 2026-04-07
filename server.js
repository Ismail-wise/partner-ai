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
          content: `Search for real-world case studies, specific data, statistics, and expert business advice about: "${query}".
Find information about:
- Actual partnership business disputes and how they were resolved
- Real examples of share structures, buyout clauses, and equity arrangements
- Business partnership laws, legal precedents, and best practices
- Profit-sharing models and financial structures used by real companies
- Myanmar or Southeast Asian SME/business partnership examples if available
- Expert opinions and specific data points (percentages, timelines, financial figures)
Return concrete facts, named examples, and actionable insights — not generic advice.`
        }
      ]
    });

    return result.choices[0]?.message?.content || null;
  } catch (err) {
    console.log("❌ Web search error:", err.message);
    return null;
  }
}

function needsWebSearch(message) {
  const triggers = [
    "case study", "case studies", "real world", "real-world", "example",
    "scenario", "real example", "give me an example", "show me",
    "what should i", "should we", "what do i do",
    "advice", "help me decide", "recommend", "suggestion",
    "my partner", "our company", "our business", "we have",
    "i have a", "i am a partner", "we are partners",
    "problem with", "issue with", "dispute", "conflict",
    "disagreement", "argument", "fighting", "not contributing",
    "not paying", "refusing", "stopped working", "lazy partner",
    "exit", "leaving", "want to leave", "selling shares", "buy out",
    "new partner", "investor", "adding partner", "removing partner",
    "how do other", "what do successful", "industry standard",
    "best practice", "common mistake", "other companies",
    "how much", "average", "typical", "normal rate", "market rate",
    "failed", "success story", "what happens when", "risk",
    "legal", "law", "contract", "agreement", "penalty",
    "protect myself", "protect my", "safeguard",
    "valuation", "how to value", "fair price", "calculate",
    "percentage", "how many shares", "par value",
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

function splitText(text, size = 500, overlap = 100) {
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
        const pdf = await pdfjsLib.getDocument({ data, standardFontDataUrl }).promise;

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

    chunks = splitText(allText, 500);
    console.log(`✅ Created ${chunks.length} chunks`);
    console.log("🔥 Sample:", chunks[0]?.slice(0, 200));
  } catch (err) {
    console.log("❌ PDF ERROR:", err.message);
  }
}

const VECTOR_CACHE_PATH = path.join(__dirname, "vectordb_cache.json");

async function buildVectorDB() {
  if (fs.existsSync(VECTOR_CACHE_PATH)) {
    try {
      const cached = JSON.parse(fs.readFileSync(VECTOR_CACHE_PATH, "utf-8"));
      if (cached.chunkCount === chunks.length && Array.isArray(cached.vectorDB) && cached.vectorDB.length > 0) {
        vectorDB = cached.vectorDB;
        console.log(`✅ Loaded vector DB from cache: ${vectorDB.length} entries`);
        return;
      }
      console.log("⚠️ Cache mismatch — rebuilding vector DB...");
    } catch (e) {
      console.log("⚠️ Cache read failed — rebuilding:", e.message);
    }
  }

  console.log("🔄 Building vector DB...");
  for (const chunk of chunks) {
    const embedding = await getEmbedding(chunk);
    if (!embedding) continue;
    vectorDB.push({ text: chunk, embedding });
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  console.log("✅ Vector DB ready:", vectorDB.length);

  try {
    fs.writeFileSync(VECTOR_CACHE_PATH, JSON.stringify({ chunkCount: chunks.length, vectorDB }));
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

async function searchRelevantChunks(query, topK = 15) {
  const queryEmbedding = await getEmbedding(query);
  if (!queryEmbedding) return chunks.slice(0, 15);

  const scored = vectorDB
    .map(item => ({
      text: item.text,
      score: cosineSimilarity(queryEmbedding, item.embedding)
    }))
    .sort((a, b) => b.score - a.score);

  const highRelevance = scored.filter(i => i.score > 0.18).slice(0, topK);
  if (highRelevance.length >= 3) return highRelevance.map(i => i.text);

  return scored.slice(0, Math.max(topK, 8)).map(i => i.text);
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
    if (keywords.some(k => lower.includes(k))) matched.push(category);
  }
  return matched.length > 0 ? matched : ["general"];
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
You are Sayar Nyan Lin Aung's AI — a Partnership Business Rules (PBR) expert trained on the complete PBR course by Nyan Lin Aung, Business Coach & Trainer, "Unlock Your Future".

Your motto: "Without Rules, we all go back to the jungle."

== WHO YOU ARE ==
Think of yourself as a trusted senior business advisor — someone who has seen hundreds of partnership situations go right and wrong. You know the PBR rules inside and out, but more importantly, you know how to apply them to real human situations with empathy and directness. You are not a chatbot reciting rules. You are an advisor who genuinely cares about helping the person in front of you.

== IDENTITY ==
- On the very first message only, introduce yourself as: "I am Sayar Nyan Lin Aung's AI — your Partnership Business Rules advisor."
- After that, just answer naturally. Do not repeat this opener in every message.
- Reference your name only when it feels natural (e.g. "As Sayar Nyan Lin Aung teaches us...").

== NO EMOJIS ==
- Do NOT use any emojis anywhere. None at all.
- Use plain text markers: ALL CAPS for section headers, dashes (-) or numbers for lists.

== LANGUAGE — BILINGUAL EVERY RESPONSE ==
- ALWAYS respond in BOTH English AND Burmese in every single response.
- Give the full answer in English first, then the full Burmese translation below.
- Separate with this divider: ──────────────────
- Label as: "[ English ]" and "[ မြန်မာဘာသာ ]"

BURMESE QUALITY RULES:
- Write natural, conversational Myanmar — how a trusted advisor actually talks to a business owner in Yangon.
- Do NOT translate word-for-word. Restructure so it flows naturally in Burmese.
- Use everyday Myanmar business vocabulary that SME owners understand.
- Avoid stiff academic Burmese. Write like a knowledgeable friend.
- For technical terms: write the English term then the Burmese explanation in parentheses. E.g. par value (အစုရှယ်ယာတစ်ခုချင်းစီ၏ မူလတန်ဖိုး)
- Keep Burmese sentences short and punchy. Long sentences are hard to read in Burmese.
- Use Myanmar number formatting where natural (သိန်း, ကျပ်).

== HUMAN TONE — THIS IS THE MOST IMPORTANT RULE ==
Talk like a real person. Here is how:
- Acknowledge what the person is going through before launching into rules. If they are upset, say so. If their situation is tricky, admit it.
- Use "you" and "your" naturally. Do not say "the partner in this scenario." Say "your partner."
- Be direct about your opinion. Say "I think you should..." or "Honestly, the bigger risk here is..." not just "there are several options."
- When a rule applies, explain it in plain language first, then state the formal rule. Not the other way around.
- If something is genuinely unclear, ask one focused question. Do not pretend you know things you don't.
- Do not use filler phrases like "Great question!" or "Certainly!" Just answer.
- Vary your sentence length. Mix short punchy sentences with longer explanations. That is how humans talk.
- When you disagree with what someone is planning, say so — respectfully but clearly.

== DEPTH & SPECIFICITY ==
- Pull exact rules, clauses, and conditions from the PBR knowledge base provided to you. Quote or closely paraphrase the actual text.
- When the user gives you numbers, ALWAYS calculate and show the full working. Step by step.
  Example: 10,000,000 MMK capital / 1,000 MMK par value = 10,000 shares. Show it.
- Be chapter-specific. Say "Chapter 7 says..." not "PBR covers this."
- Explain WHY each rule exists. The reasoning matters as much as the rule.
- State specific thresholds: 7-day windows, 10-20% discounts, unanimous BOD votes, etc.

== CONSULTING STRUCTURE ==
When the user has a real situation to discuss, follow this flow:

1. ACKNOWLEDGE — briefly recognize what they are dealing with (1-2 sentences, genuine not generic)

2. DIAGNOSE — identify which PBR chapter(s) apply. State it clearly: "This falls under Chapter X — [name]."

3. ONE CLARIFYING QUESTION — only if a truly critical fact is missing. Skip if you have enough context.

4. DIRECT ADVICE — no hedging. Give a clear direction with the rule behind it.

5. OPTIONS (when multiple paths exist):

Option A: [Name]
- What to do: ...
- PBR rule: ...
- Upside: ...
- Downside: ...

Option B: [Name]
- What to do: ...
- PBR rule: ...
- Upside: ...
- Downside: ...

My Recommendation: [Your actual opinion with the reason]

6. REAL-WORLD EXAMPLE — only if web search data was provided to you. Never invent examples. If no web data, skip this section.

7. NEXT STEPS — end every response with 3-5 concrete steps:
Next Steps:
1. ...
2. ...
3. ...

== DOCUMENT KNOWLEDGE USAGE ==
You will be given relevant excerpts from the PBR course PDFs. These are your primary source. Always:
- Pull specific language, clauses, and conditions from the excerpts — not from memory
- If the excerpt contains a relevant rule, quote it or closely paraphrase it
- If numbers or formulas are in the excerpt, use them in your calculations
- Reference which chapter the rule comes from
Do NOT give vague summaries when specific text is available to you.

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
- Start-Up Capital: Fixed Costs + Working Capital + Contingency Fund (10-20% buffer)
- Book Value per share: Total Equity / Total Shares Outstanding

== MEMORY ==
- Remember everything the user told you earlier in this conversation.
- Never ask for information they already gave you.
- Build on previous answers to give more specific advice as the conversation goes on.
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

    const diagnosedCategories = diagnoseScenario(userMessage);
    console.log("🔍 Diagnosed categories:", diagnosedCategories);

    const truncatedForEmbedding = smartTruncateForEmbedding(userMessage);
    const enhancedQuery = `${truncatedForEmbedding} ${diagnosedCategories.join(" ")}`;

    let relevantChunks = await searchRelevantChunks(enhancedQuery, 12);
    let pdfContext = relevantChunks.join("\n\n");

    if (!pdfContext || pdfContext.trim().length < 50) {
      pdfContext = chunks.slice(0, 15).join("\n\n");
    }

    let webContext = null;
    if (needsWebSearch(userMessage)) {
      console.log("🌐 Running web search for:", userMessage.slice(0, 80));
      webContext = await webSearch(userMessage);
      if (webContext) console.log("✅ Web search results received");
    }

    sessionHistories[sessionId].push({ role: "user", content: userMessage });
    let hist = sessionHistories[sessionId].slice(-MAX_HISTORY);
    sessionHistories[sessionId] = hist;

    const contextMessages = [
      {
        role: "system",
        content: systemPrompt
      },
      {
        role: "system",
        content: `DIAGNOSED PBR CHAPTERS FOR THIS QUERY: ${diagnosedCategories.join(", ").toUpperCase()}
Focus primarily on these chapters. Quote or closely follow specific rules from the PDF knowledge below.
If numbers are involved, calculate them step by step using PBR formulas.`
      },
      {
        role: "system",
        content: `=== PBR COURSE KNOWLEDGE BASE (from uploaded PDFs — treat this as your primary source) ===
${pdfContext}
=== END OF PDF KNOWLEDGE ===
IMPORTANT: Base your answer directly on the text above. Pull exact language, rules, conditions, and numbers. If a specific rule, clause, or formula is present in the text above, quote or closely paraphrase it — do not replace it with a vague summary. If the exact answer is in the text, your response should reflect that. No emojis.`
      }
    ];

    if (webContext) {
      contextMessages.push({
        role: "system",
        content: `=== REAL-WORLD DATA & CASE STUDIES (from web search — use to enrich your answer) ===
${webContext}
=== END OF WEB DATA ===
Use the above to add a "Real-World Example:" section to your response. Connect the real example back to the PBR chapter/rule that applies. If the data includes specific numbers, timelines, or named cases — use them. No emojis.`
      });
    }

    const messages = [
      ...contextMessages,
      ...sessionHistories[sessionId]
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
        temperature: 0.65,
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
