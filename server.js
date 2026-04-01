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
// MEMORY STORAGE
// ==========================
let chunks = [];

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

// ==========================
// SMART SEARCH (FIXED)
// ==========================
function searchRelevantChunks(query) {
  const cleanQuery = query.toLowerCase();

  return chunks
    .map(chunk => {
      const text = chunk.toLowerCase();
      let score = 0;

      // strong match
      if (text.includes(cleanQuery)) score += 10;

      // partial match
      const words = cleanQuery.split(/\s+/);
      for (const w of words) {
        if (w.length > 2 && text.includes(w)) {
          score += 2;
        }
      }

      return { text: chunk, score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
    .map(c => c.text);
}

// ==========================
// LOAD PDFs ON START
// ==========================
await loadPDFs();

// ==========================
// BETTER SYSTEM PROMPT
// ==========================
const systemPrompt = `
သင်သည် မြန်မာဘာသာဖြင့် သင်ကြားပေးသော AI ဆရာဖြစ်သည်။

စည်းကမ်းများ:
- မြန်မာဘာသာဖြင့်သာ ပြန်ဖြေပါ
- အလွန်ရှင်းလင်းပြီး Step-by-step ရှင်းပြပါ
- Beginner-friendly ဖြစ်ရမည်
- ဥပမာများ ထည့်ပါ
- အကြောင်းအရာကို တစ်ဆင့်ချင်းရှင်းပြပါ
- သက်ဆိုင်သော အချက်အလက်ရှိပါက အဲဒါကို အခြေခံပြီး ဖြေပါ
- မရှိပါကသာ "မသိပါ" ဟု ပြန်ဖြေပါ
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

    let relevantChunks = searchRelevantChunks(userMessage);
    let context = relevantChunks.join("\n\n");

    // ✅ fallback (always have context)
    if (!context) {
      context = chunks.slice(0, 5).join("\n\n");
    }

    const messages = [
      { role: "system", content: systemPrompt },

      {
        role: "system",
        content: `
Use the knowledge below to answer the question.

If relevant information exists, explain clearly from it.
If partially relevant, still answer using best match.
Only say "မသိပါ" if truly no information exists.

===== KNOWLEDGE =====
${context}
`
      },

      { role: "user", content: userMessage }
    ];

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages,
      temperature: 0.4,
      max_tokens: 1500
    });

    const reply =
      response.choices?.[0]?.message?.content || "⚠️ No response";

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