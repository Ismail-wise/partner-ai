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

const CATEGORY_EXPANSION = {
  capital: "capital contribution deadline late payment penalty dilution forfeiture convert loan eject partner Chapter 1",
  shares: "par value share formula total capital equity ownership book value face value market value intrinsic value Chapter 2",
  labor: "labor value service contribution sweat equity salary profit margin compensation Chapter 3",
  profit: "profit sharing EAT earnings after tax dividend BOD board of directors retained earnings payout Chapter 4",
  financial: "GAAP bank account two signatory audit CapEx personal funds mixing financial management Chapter 5",
  leadership: "McKinsey 7S framework major decision minor decision non-compete misconduct personal asset Chapter 6",
  exit: "exit rules book value discount 10 20 percent 7 day acceptance window lock-up period Chapter 7",
  death: "death inheritance heir spouse consent shares leadership role Chapter 8",
  transfer: "share transfer written notice 7 day response payment company bank account Chapter 9",
  dispute: "dispute resolution mediation internal committee majority vote third party arbitration shareholder forced buyout Chapter 10",
  general: "partnership business rules PBR Myanmar SME agreement Nyan Lin Aung Unlock Your Future"
};

function buildExpandedQuery(message, categories) {
  const expansions = categories.map(c => CATEGORY_EXPANSION[c] || "").join(" ");
  return `${message} ${expansions}`;
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

== CONSULTING FLOW (ADAPT AS NEEDED) ==
Do not force all steps into a simple factual question. Short direct questions get focused answers. For real scenarios, follow this flow:

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
You will be given relevant excerpts from the PBR course PDFs. These are your ground truth. Always:
- Base your answer on what the document text actually says
- Quote or closely paraphrase the excerpt when a relevant rule is present
- Cite inline after using a rule: (Source: Content - 7) — use the exact document name shown in the excerpt header
- If numbers, formulas, or thresholds appear in the excerpt, use them exactly
- If the excerpts do NOT cover the specific question, say so clearly: "The current excerpts don't have the exact rule on this." Then use your PBR framework backup, clearly labeled: [PBR Framework knowledge]
Do NOT give vague summaries when specific text is available. Never say "according to the documents" without naming the document.

== MYANMAR BUSINESS CONTEXT ==
You understand the Myanmar business reality:
- Many partnerships start on trust and verbal agreements between family or friends
- Business disputes in Myanmar are often avoided to save face — people suffer in silence until it's too late
- The formal company registry (MyCO) and Companies Law 2017 exist but enforcement is weak for SMEs
- Most Yangon and Mandalay SME owners have not written a proper partnership agreement
- When you apply PBR rules, acknowledge this reality: "I know many Myanmar partnerships skip this step — but here is exactly why that leads to problems later."

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
    const enhancedQuery = buildExpandedQuery(truncatedForEmbedding, diagnosedCategories);

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

    let webContext = null;
    if (needsWebSearch(userMessage)) {
      console.log("🌐 Running web search for:", userMessage.slice(0, 80));
      webContext = await webSearch(userMessage);
      if (webContext) console.log("✅ Web search results received");
    }

    sessionHistories[sessionId].push({ role: "user", content: userMessage });
    let hist = sessionHistories[sessionId].slice(-MAX_HISTORY);
    sessionHistories[sessionId] = hist;

    // Build the document context block — placed LAST (right before user messages)
    // so it has strongest attention from the model
    const docContextMessage = {
      role: "system",
      content: `DOCUMENT KNOWLEDGE — YOUR PRIMARY SOURCE OF TRUTH
The excerpts below come directly from Sayar Nyan Lin Aung's uploaded course documents.
When you answer, you MUST:
1. Base your answer on what the documents actually say — not on general knowledge
2. Quote or closely paraphrase the document text when a relevant rule is present
3. Cite the source document by name: e.g. "According to [document name]..."
4. If a rule, number, formula, or clause appears in the text below, use it exactly
5. If the document does NOT cover something, say so clearly, then use your PBR framework knowledge as backup

══════════════════════════════════════════
${pdfContext}
══════════════════════════════════════════

CHAPTER FOCUS FOR THIS QUERY: ${diagnosedCategories.join(", ").toUpperCase()}
If numbers are involved, calculate step by step. No emojis.`
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
