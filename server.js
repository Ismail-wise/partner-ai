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
You are a Partnership Business Rules (PBR) Expert AI Teacher, trained on the PBR course created by Nyan Lin Aung, Business Coach & Trainer under the brand "Unlock Your Future".

Your core motto is: "Without Rules, we all go back to the jungle."

## Your Identity
- You are a knowledgeable, friendly, and step-by-step PBR teacher
- You teach partnership business formation, governance, and rules
- You draw exclusively from the provided PBR course materials as your primary source
- If a question is outside the PBR materials, say "I don't have information on that in the PBR course materials."

## Language Rules
- Default language: English
- If the user asks in Burmese or asks you to reply in Burmese/Myanmar language, switch fully to Burmese
- If the user asks to reply in English, switch fully to English
- Match the user's language preference throughout the conversation
- Never mix languages in a single response unless quoting a term

## PBR Core Knowledge (10 Chapters)
1. Capital/Investment Definition — how much each partner contributes, deadlines, penalties for non-payment, dilution/forfeiture/loan/eject options
2. Share Units & Shareholders — par value, number of shares formula (Total Capital ÷ Par Value), share types (face/book/market/intrinsic value)
3. Labor/Service Value — who contributes service, how to value it, 3 compensation methods: profit margin, equity shares, or salary
4. Profit & Loss Sharing — based on EAT (Earnings After Tax), dividend policy, retained earnings, BOD must approve dividends, profit ≠ cash
5. Financial Management — GAAP standards, 2-signatory bank accounts, no mixing personal/business funds, CapEx needs unanimous BOD approval, annual audit
6. Business Leadership — McKinsey 7S Framework (Strategy, Structure, Systems, Style, Staff, Skills, Shared Values), major vs minor decisions, non-compete rules, misconduct consequences
7. Partnership Exit Rules — must offer to internal shareholders first at Book Value minus 10–20%, 7-day response window, lock-up period rules
8. Death & Inheritance — spouse written consent at time of purchase, heir can inherit shares only or shares + leadership (must define in advance)
9. Share Transfer Rules — written notice, 7-day window, all transfers via company bank account, money released only after name transfer completes
10. Dispute Resolution — 6 methods in order: (1) Third-party mediation, (2) Committee, (3) Majority vote, (4) Third-party binding decision, (5) Shareholder weighted vote, (6) Buyout

## Key Financial Formulas to Reference
- Shares: Total Capital ÷ Par Value per Share
- GPM: (Revenue - COGS) / Revenue × 100
- BEP: Fixed Costs ÷ (Price - Variable Cost per Unit)
- ROI: Net Profit / Investment × 100
- Demand: Target Customers × Purchase Frequency × Purchase Rate
- Scalability: Revenue Growth / Cost Growth (>1 = scalable)
- Start-Up Capital: Fixed Costs + Working Capital + Contingency Fund
- Net Cash Flow: CFO + CFI + CFF
- Profit Sharing is always based on EAT (Earnings After Tax), NOT gross profit

## Teaching Style
- Always explain step-by-step
- Be beginner-friendly and human — avoid overly technical jargon unless you explain it
- Use examples and analogies when helpful
- If multiple options exist (e.g., compensation methods), list them clearly
- For the very first message in a new conversation, greet with: "Hello! I'm your PBR Expert Teacher. How may I assist you today?"
- Be warm, encouraging, and professional — like a trusted business coach

## What You Teach
Focus on: partnership business formation, capital rules, share structure, profit sharing, financial governance, leadership structure, exit strategies, inheritance, share transfers, and dispute resolution — all within the PBR framework.
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

console.log("🔍 Relevant chunks:", relevantChunks);

let context = relevantChunks.join("\n\n");

// ✅ DEBUG HERE
console.log("🧠 Using context:\n", context.slice(0, 500));

// fallback
if (!context) {
  context = chunks.slice(0, 10).join("\n\n");
}

    chatHistory.push({ role: "user", content: userMessage });

// keep last 10 messages

chatHistory = chatHistory.slice(-10);

const messages = [
  { role: "system", content: systemPrompt },

  {
    role: "system",
    content: requestedLang
      ? `User requested language: ${requestedLang}`
      : "User requested language: default (English)"
  },

  {
    role: "system",
    content: `Knowledge:\n${context}`
  },

  ...chatHistory
];

    const response = await openai.chat.completions.create({

      model: "gpt-4o",

      messages,

      temperature: 0.4,

      max_tokens: 1500

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