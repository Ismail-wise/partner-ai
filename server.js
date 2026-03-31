import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import dotenv from "dotenv";
dotenv.config();

import express from "express";
import OpenAI from "openai";
import cors from "cors";
import rateLimit from "express-rate-limit";

// ✅ NEW PDF LIB
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";

// ==========================
// 📁 PATH SETUP
// ==========================
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ==========================
// 🚀 APP INIT
// ==========================
const app = express();

// ==========================
// ⚙️ MIDDLEWARE
// ==========================
app.use(rateLimit({
  windowMs: 60 * 1000,
  max: 50
}));

app.use(cors());
app.use(express.json());

// ==========================
// 🤖 OPENAI
// ==========================
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// ==========================
// 📄 LOAD PDF MEMORY
// ==========================
let courseContent = "";

async function loadPDF() {
  try {
    const pdfPath = path.resolve("docs/course.pdf");

    console.log("📄 Loading PDF from:", pdfPath);

    if (!fs.existsSync(pdfPath)) {
      console.log("⚠️ course.pdf not found");
      return;
    }

    const data = new Uint8Array(fs.readFileSync(pdfPath));

    const pdf = await pdfjsLib.getDocument({ data }).promise;

    let text = "";

    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();

      const strings = content.items.map(item => item.str);
      text += strings.join(" ") + "\n";
    }

    courseContent = text.slice(0, 15000);

    console.log("✅ PDF Loaded Successfully");

  } catch (err) {
    console.log("❌ PDF Error:", err.message);
  }
}

// Load PDF
await loadPDF();

// ==========================
// 🧠 SYSTEM PROMPT
// ==========================
const systemPrompt = `
သင်သည် မြန်မာဘာသာဖြင့် သင်ကြားပေးသော AI ဆရာဖြစ်သည်။

စည်းကမ်းများ:
- မြန်မာဘာသာဖြင့်သာ ပြန်ဖြေပါ
- ရိုးရှင်းပြီး နားလည်လွယ်အောင်ရှင်းပြပါ
- ဥပမာများထည့်ပါ
- မသိပါက မသိကြောင်းပြောပါ
- PDF content ကို အဓိကအသုံးပြုပါ
`;

// ==========================
// 🌐 ROUTES
// ==========================
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

// ==========================
// 💬 CHAT API
// ==========================
app.post("/chat", async (req, res) => {
  try {
    const userMessage = req.body.message;

    if (!userMessage || !userMessage.trim()) {
      return res.json({ reply: "စာတစ်ခုခု ရိုက်ထည့်ပါ။" });
    }

    const messages = [
      { role: "system", content: systemPrompt },
      {
        role: "system",
        content: courseContent
          ? `ဒီဟာ PDF course content ဖြစ်ပါတယ်:\n${courseContent}`
          : "PDF content မရှိသေးပါ"
      },
      { role: "user", content: userMessage }
    ];

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages,
      temperature: 0.6
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
// 🚀 START SERVER
// ==========================
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});