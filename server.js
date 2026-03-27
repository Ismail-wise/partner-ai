import fs from "fs";
import pdf from "pdf-parse";
import rateLimit from "express-rate-limit";
import path from "path";
import { fileURLToPath } from "url";

import dotenv from "dotenv";
dotenv.config();

import express from "express";
import OpenAI from "openai";
import cors from "cors";

// Fix __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// ✅ Rate limit
app.use(rateLimit({
  windowMs: 60 * 1000,
  max: 20
}));

// ✅ Middleware
app.use(cors());
app.use(express.json());

// ✅ OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// ✅ User memory
const userConversations = {};

// ✅ PDF Memory (GLOBAL)
let courseMemory = "";

// 🔥 LOAD PDF ON SERVER START (BEST WAY)
async function loadPDF() {
  try {
    const filePath = path.join(__dirname, "docs", "course.pdf");

    if (!fs.existsSync(filePath)) {
      console.log("⚠️ No PDF found in /docs folder");
      return;
    }

    const dataBuffer = fs.readFileSync(filePath);
    const data = await pdf(dataBuffer);

    // Limit size (important)
    courseMemory = data.text.substring(0, 3000);

    console.log("✅ PDF Loaded into AI memory");
  } catch (err) {
    console.error("❌ PDF Load Error:", err.message);
  }
}

// 🚀 Run once at startup
await loadPDF();

// ✅ Serve frontend
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

// 💬 Chat API
app.post("/chat", async (req, res) => {
  try {
    const userMessage = req.body.message;

    // ✅ Validation
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

    // ✅ Create memory
    if (!userConversations[userId]) {
      userConversations[userId] = [];
    }

    const conversation = userConversations[userId];

    // ✅ Add user message
    conversation.push({
      role: "user",
      content: userMessage
    });

    // ✅ Limit memory
    if (conversation.length > 20) {
      conversation.splice(0, 2);
    }

    // 🔥 SYSTEM PROMPT WITH PDF MEMORY
    const systemMessage = {
      role: "system",
      content: `
သင်သည် မြန်မာဘာသာဖြင့် အလွန်ကျွမ်းကျင်သော AI ဆရာတစ်ဦး ဖြစ်သည်။

📚 သင့်မှာရှိသော သင်ခန်းစာအချက်အလက်များ:
${courseMemory}

လိုက်နာရန်:
- အမြဲ မြန်မာဘာသာဖြင့် ပြန်ဖြေပါ
- သဘာဝဆန်ပြီး လူတစ်ယောက်လို ပြောပါ
- ရိုးရှင်းစွာရှင်းပြပြီး လိုအပ်ရင် နက်ရှိုင်းစွာဆက်ရှင်းပါ
- ဥပမာများထည့်ပါ
- သင်ခန်းစာအချက်အလက်များကို အသုံးပြုပါ
`
    };

    // ✅ OpenAI call
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        systemMessage,
        ...conversation
      ]
    });

    const aiReply = response.choices[0].message.content;

    // ✅ Save AI reply
    conversation.push({
      role: "assistant",
      content: aiReply
    });

    res.json({ reply: aiReply });

  } catch (error) {
    console.error("❌ ERROR:", error.message);

    res.status(500).json({
      reply: "⚠️ Server error ဖြစ်ပါတယ်။ နောက်တစ်ကြိမ် ထပ်ကြိုးစားပါ။"
    });
  }
});

// 🚀 Start server
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
});