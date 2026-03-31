import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import dotenv from "dotenv";
dotenv.config();

import express from "express";
import OpenAI from "openai";
import cors from "cors";
import rateLimit from "express-rate-limit";

// PDF LIB
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";

// ==========================
// PATH SETUP
// ==========================
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ==========================
// APP INIT
// ==========================
const app = express();

// ==========================
// MIDDLEWARE
// ==========================
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
// SMART MEMORY SYSTEM
// ==========================
let chunks = [];

// 🔹 Split text into chunks
function splitText(text, size = 300) {
  const words = text.split(" ");
  let result = [];

  for (let i = 0; i < words.length; i += size) {
    result.push(words.slice(i, i + size).join(" "));
  }

  return result;
}

// 🔹 Load PDFs and create chunks
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
        console.log(`⚠️ Failed to read ${file}:`, fileErr.message);
      }
    }

    if (!allText) {
      console.log("⚠️ No text extracted from PDFs");
      return;
    }

    chunks = splitText(allText, 300);

    console.log(`✅ Created ${chunks.length} chunks`);

  } catch (err) {
    console.log("❌ PDF SYSTEM ERROR:", err.message);
  }
}

// 🔹 Search relevant chunks
function searchRelevantChunks(query) {
  const words = query.toLowerCase().split(" ");

  return chunks
    .map(chunk => {
      const text = chunk.toLowerCase();

      let score = 0;
      for (const w of words) {
        if (text.includes(w)) score++;
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
// SYSTEM PROMPT (UPGRADED)
// ==========================
const systemPrompt = `
သင်သည် မြန်မာဘာသာဖြင့် သင်ကြားပေးသော အတွေ့အကြုံရှိ AI ဆရာဖြစ်သည်။

စည်းကမ်းများ:
- မြန်မာဘာသာဖြင့်သာ ပြန်ဖြေပါ
- အလွန်အသေးစိတ်ပြီး နားလည်လွယ်အောင် ရှင်းပြပါ
- အကြောင်းအရာကို အဆင့်လိုက် (Step-by-step) ခွဲခြားပြီး ရှင်းပြပါ
- ဥပမာများကို လိုအပ်သလို ထည့်ပါ
- Bullet points ဖြင့် ရှင်းပြနိုင်ပါက အသုံးပြုပါ
- အခြေခံမှစ၍ ရှင်းပြပါ (Beginner friendly)
- အဖြေကို အပြည့်အစုံ ရှင်းပြပါ (Short မဖြစ်စေရ)
- မသိပါက မသိကြောင်းရှင်းပြပါ
`;

// ==========================
// ROUTES
// ==========================
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

// ==========================
// CHAT API (SMART)
// ==========================
app.post("/chat", async (req, res) => {
  try {
    const userMessage = req.body.message;

    if (!userMessage || !userMessage.trim()) {
      return res.json({ reply: "စာတစ်ခုခု ရိုက်ထည့်ပါ။" });
    }

    // 🔥 Get relevant content
    const relevantChunks = searchRelevantChunks(userMessage);
    const context = relevantChunks.join("\n\n");

    const messages = [
      { role: "system", content: systemPrompt },

      {
        role: "system",
        content: `
အောက်ပါအချက်အလက်များကို အခြေခံပြီး အလွန်အသေးစိတ်၊ အဆင့်လိုက်ရှင်းပြပါ:

${context || "ဆိုင်ရာ အချက်အလက် မရှိပါ"}
`
      },

      {
        role: "system",
        content: `
အဖြေကို အပြည့်အစုံ ရှင်းပြပါ။
Step-by-step format အသုံးပြုပါ။
ဥပမာများပါ ထည့်ပါ။
ရှင်းလင်းမှုကို အထူးအလေးထားပါ။
`
      },

      { role: "user", content: userMessage }
    ];

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages,
      temperature: 0.7,
      max_tokens: 2000
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