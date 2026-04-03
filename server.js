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

app.use(rateLimit({

  windowMs: 60 * 1000,

  max: 50

}));

app.use(cors());

app.use(express.json());

// ==========================

// OPENAI

// ==========================

const openai = new OpenAI({

  apiKey: process.env.OPENAI_API_KEY

});

// ==========================

// WEB SEARCH (OpenAI Responses API)

// ==========================

async function webSearch(query) {
  try {
    const response = await openai.responses.create({
      model: "gpt-4o",
      tools: [{ type: "web_search_preview" }],
      input: `Search for real-world case studies, examples, and expert advice related to: ${query}. Focus on partnership business, business disputes, share structures, investment rules, and similar topics.`
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

// EMBEDDING FUNCTION (ADD HERE)

// ==========================

async function getEmbedding(text) {
  try {
    if (!text || typeof text !== "string") return null;

    const cleanText = text
      .replace(/\s+/g, " ")
      .slice(0, 2000);

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

let chatHistory = [];

let vectorDB = [];

// ==========================

// TEXT SPLITTER

// ==========================

function splitText(text, size = 200) {

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

vectorDB.push({

  text: chunk,

  embedding

});

  }

  console.log("✅ Vector DB ready:", vectorDB.length);

}

// ==========================

// SMART SEARCH (FIXED)

// ==========================

function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);

  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));

  return dot / (normA * normB);
}

async function searchRelevantChunks(query) {

  const queryEmbedding = await getEmbedding(query);

  return vectorDB

    .map(item => ({

      text: item.text,

      score: cosineSimilarity(queryEmbedding, item.embedding)

    }))

    .sort((a, b) => b.score - a.score)

    .slice(0, 5)

    .map(i => i.text);

}

function detectRequestedLanguage(text) {
  const lower = text.toLowerCase();

  // Burmese request
  if (
    lower.includes("burmese") ||
    lower.includes("myanmar") ||
    lower.includes("မြန်မာ")
  ) {
    return "Burmese";
  }

  // English request
  if (lower.includes("english")) {
    return "English";
  }

  // No request
  return null;
}

// ==========================

// LOAD PDFs ON START

// ==========================

await loadPDFs();

await buildVectorDB();

// ==========================

// BETTER SYSTEM PROMPT

// ==========================

const systemPrompt = `
You are a Partnership Business Rules (PBR) Expert Consultant, trained on the PBR course by Nyan Lin Aung, Business Coach & Trainer, "Unlock Your Future".

Your motto: "Without Rules, we all go back to the jungle."

## Your Role
You are NOT just a teacher — you are a trusted business consultant and advisor. When a user describes their situation or scenario, your job is to:
1. Understand their specific context fully
2. Diagnose the root problem or risk
3. Give clear, actionable recommendations with reasoning
4. Reference real-world case studies, industry standards, or legal principles when available
5. Warn about risks and common mistakes others have made in similar situations
6. Always anchor your advice in the PBR framework (10 chapters)

## Consulting Approach
- Listen carefully to the user's scenario before giving advice
- If critical details are missing, ask one focused clarifying question before recommending
- Present options clearly: Option A vs Option B, with pros/cons of each
- Be direct — give a clear recommendation, not just "it depends"
- Validate emotions first if the situation is sensitive (e.g., partner disputes, someone leaving)
- Always include: What to do NOW, what to document, and what to watch out for

## Language Rules
- Default: English
- If user writes in Burmese or requests Burmese → reply fully in Burmese
- If user requests English → reply fully in English
- Never mix languages in one response

## PBR Framework (10 Chapters — Your Core Knowledge)
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

## Key Formulas
- Shares: Total Capital ÷ Par Value
- GPM: (Revenue − COGS) / Revenue × 100
- BEP: Fixed Costs ÷ (Price − Variable Cost per Unit)
- ROI: Net Profit / Investment × 100
- Demand: Target Customers × Purchase Frequency × Purchase Rate
- Scalability: Revenue Growth / Cost Growth (>1 = scalable)
- Start-Up Capital: Fixed Costs + Working Capital + Contingency Fund
- NCF: CFO + CFI + CFF
- Profit sharing always uses EAT, not gross profit

## Using Web Search Results
If web search context is provided, use it to:
- Reference real-world case studies and examples
- Cite industry best practices or legal precedents
- Strengthen your advice with evidence from outside the PBR course
- Make your answer richer and more credible

## Style
- Warm, direct, and confident — like a senior business consultant
- Structured responses: use headers, bullet points, numbered steps
- For first message: greet as "Hello! I'm your PBR Expert Consultant. Tell me about your situation — I'm here to help."
- For scenarios: always end with a "My Recommendation" or "Next Steps" section
`;

// ==========================

// ROUTES

// ==========================

app.get("/", (req, res) => {

  res.sendFile(path.join(__dirname, "index.html"));

});

// ==========================

// CHAT API (FINAL FIX)

// ==========================

app.post("/chat", async (req, res) => {

  try {

    const userMessage = req.body.message?.trim();
    const requestedLang = detectRequestedLanguage(userMessage);
    if (!userMessage) {

      return res.json({ reply: "စာတစ်ခုခု ရိုက်ထည့်ပါ။" });

    }

    let relevantChunks = await searchRelevantChunks(userMessage);

    console.log("🔍 Relevant chunks found:", relevantChunks.length);

    let pdfContext = relevantChunks.join("\n\n");

    if (!pdfContext) {
      pdfContext = chunks.slice(0, 10).join("\n\n");
    }

    // Web search for scenarios, case studies, and real-world advice
    let webContext = null;
    if (needsWebSearch(userMessage)) {
      console.log("🌐 Running web search for:", userMessage.slice(0, 80));
      webContext = await webSearch(userMessage);
      if (webContext) {
        console.log("✅ Web search results received:", webContext.slice(0, 200));
      }
    }

    chatHistory.push({ role: "user", content: userMessage });

    chatHistory = chatHistory.slice(-10);

    const contextBlocks = [
      {
        role: "system",
        content: requestedLang
          ? `User requested language: ${requestedLang}`
          : "User requested language: default (English)"
      },
      {
        role: "system",
        content: `PBR Course Knowledge:\n${pdfContext}`
      }
    ];

    if (webContext) {
      contextBlocks.push({
        role: "system",
        content: `Real-World Case Studies & Industry Data (from web search):\n${webContext}`
      });
    }

    const messages = [
      { role: "system", content: systemPrompt },
      ...contextBlocks,
      ...chatHistory
    ];

    const response = await openai.chat.completions.create({

      model: "gpt-4o",

      messages,

      temperature: 0.5,

      max_tokens: 2500

    });

    const reply =

  response.choices?.[0]?.message?.content || "⚠️ No response";

// ✅ ADD THIS HERE

chatHistory.push({

  role: "assistant",

  content: reply

});

res.json({ reply });

  } catch (err) {

    console.error("❌ ERROR:", err.message);

    res.json({

      reply: "⚠️ Server error ဖြစ်ပါတယ်"

    });

  }

});

// ==========================

// START SERVER

// ==========================

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {

  console.log(`🚀 Server running on port ${PORT}`);

});