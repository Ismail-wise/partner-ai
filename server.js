import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import dotenv from "dotenv";
dotenv.config();

import express from "express";
import OpenAI from "openai";
import cors from "cors";
import rateLimit from "express-rate-limit";

// ✅ PDF LIBRARY
import * as pdfjsLib from "pdfjs-dist";
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.js",
  import.meta.url
).toString();

// Fix __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// ✅ Rate limit
app.use(rateLimit({
  windowMs: 60 * 1000,
  max: 30
}));

// ✅ Middleware
app.use(cors());
app.use(express.json());

// ✅ OpenAI (🔥 upgraded model support)
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// ==========================
// 🧠 MEMORY STORAGE
// ==========================
const userConversations = {};
let courseMemory = "";

// ==========================
// 📄 LOAD PDF (IMPROVED)
// ==========================
async function loadPDF() {
  try {
    const filePath = path.join(__dirname, "docs", "course.pdf");

    if (!fs.existsSync(filePath)) {
      console.log("⚠️ No PDF found in /docs folder");
      return;
    }

    const data = new Uint8Array(fs.readFileSync(filePath));
    const pdf = await pdfjsLib.getDocument({ data }).promise;

    let text = "";

    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();

      const strings = content.items
        .map(item => item.str.trim())
        .filter(Boolean);

      text += strings.join(" ") + "\n\n";
    }

    // ✅ Limit but keep structure
    courseMemory = text.substring(0, 8000);

    console.log("✅ PDF Loaded into AI memory");
  } catch (err) {
    console.error("❌ PDF Load Error:", err.message);
  }
}

await loadPDF();

// ==========================
// 🧠 BURMESE SYSTEM PROMPT (IMPROVED)
// ==========================
const systemPrompt = {
  role: "system",
  content: `
သင်သည် မြန်မာဘာသာကို နားလည်မှုမြင့်မားပြီး သင်ကြားနိုင်သော AI ဆရာတစ်ဦးဖြစ်သည်။

စည်းကမ်းများ:
- အမြဲ မြန်မာဘာသာဖြင့်သာ ပြန်ဖြေပါ
- သဘာဝဆန်ပြီး လူတစ်ယောက်လို ပြောပါ
- ရိုးရှင်းပြီး နားလည်လွယ်အောင်ရှင်းပြပါ
- လိုအပ်လျှင် ဥပမာများထည့်ပါ
- မသေချာပါက "မသေချာပါ" ဟုပြောပါ

အရေးကြီး:
- အသုံးပြုသူမေးခွန်းကို အဓိကဦးစားပေးပါ
- Course content ကို လိုအပ်မှသာ အသုံးပြုပါ
`
};

// ==========================
// 🌐 ROUTES
// ==========================

// Serve frontend
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

// ==========================
// 💬 CHAT API (SMART VERSION)
// ==========================
app.post("/chat", async (req, res) => {
  try {
    const userMessage = req.body.message;

    if (!userMessage || userMessage.trim() === "") {
      return res.status(400).json({
        reply: "စာတစ်ခုခု ရိုက်ထည့်ပါ။"
      });
    }

    if (userMessage.length > 500) {
      return res.status(400).json({
        reply: "စာရှည်လွန်းပါတယ် (500 characters အတွင်းသာ)"
      });
    }

    const userId = req.ip;

    if (!userConversations[userId]) {
      userConversations[userId] = [];
    }

    const conversation = userConversations[userId];

    // ✅ Add user message
    conversation.push({
      role: "user",
      content: userMessage
    });

    // ✅ Keep only last 8 messages (clean memory)
    const recentMessages = conversation.slice(-8);

    // ==========================
    // 🧠 BUILD SMART MESSAGES
    // ==========================
    const messages = [
      systemPrompt,
      ...recentMessages
    ];

    // ✅ Inject PDF ONLY when needed
    if (courseMemory) {
      messages.push({
        role: "system",
        content: `
အောက်ပါ course content ကို လိုအပ်ပါက အသုံးပြုပါ:

${courseMemory}
`
      });
    }

    // ==========================
    // 🔥 OPENAI CALL (UPGRADED)
    // ==========================
    const response = await openai.chat.completions.create({
      model: "gpt-4o", // ✅ MUCH BETTER FOR BURMESE
      messages: messages,
      temperature: 0.7
    });

    const aiReply = response.choices[0].message.content;

    // Save AI reply
    conversation.push({
      role: "assistant",
      content: aiReply
    });

    res.json({
      reply: aiReply
    });

  } catch (error) {
    console.error("❌ ERROR:", error.message);

    res.status(500).json({
      reply: "⚠️ Server error ဖြစ်ပါတယ်။ နောက်တစ်ကြိမ် ထပ်ကြိုးစားပါ။"
    });
  }
});

// ==========================
// 🚀 START SERVER
// ==========================
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
});