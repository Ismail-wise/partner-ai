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

    // ✅ ensure valid string

    if (!text || typeof text !== "string") return null;

    // ✅ clean + limit text

    const cleanText = text

      .replace(/\s+/g, " ")

      .replace(/[^\x00-\x7F]/g, "") // remove broken chars

      .slice(0, 1000);

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

          const strings = content.items.map(item => item.str);

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

  return a.reduce((sum, val, i) => sum + val * b[i], 0);

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

// ==========================

// LOAD PDFs ON START

// ==========================

await loadPDFs();

await buildVectorDB();

// ==========================

// BETTER SYSTEM PROMPT

// ==========================

const systemPrompt = `

သင်သည် မြန်မာဘာသာဖြင့် သင်ကြားပေးသော AI ဆရာဖြစ်သည်။

စည်းကမ်းများ:

- မြန်မာဘာသာဖြင့်သာ ဖြေပါ

- Step-by-step ရှင်းပြပါ

- Beginner-friendly ဖြစ်ရမည်

- ဥပမာများ ထည့်ပါ

- PDF ထဲမှ အချက်အလက်ကို အဓိကအသုံးပြုပါ

- မသိပါက "မသိပါ" ဟုသာ ပြန်ဖြေပါ

User ကို သင်ကြားရန် အဓိကထားပါ။

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

    if (!userMessage) {

      return res.json({ reply: "စာတစ်ခုခု ရိုက်ထည့်ပါ။" });

    }

    let relevantChunks = await searchRelevantChunks(userMessage);

console.log("🔍 Relevant chunks:", relevantChunks); // MUST be here

let context = relevantChunks.join("\n\n");

    // ✅ fallback (always have context)

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