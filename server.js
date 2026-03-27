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

// ✅ Rate limit (protect API usage)
app.use(rateLimit({
  windowMs: 60 * 1000,
  max: 20
}));

// ✅ Middleware
app.use(cors());
app.use(express.json());

// ✅ OpenAI setup
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// ✅ Store conversations per user
const userConversations = {};

// ✅ Strong Burmese System Prompt (UPGRADED)
const systemPrompt = {
  role: "system",
  content: `
သင်သည် မြန်မာဘာသာဖြင့် အလွန်ကျွမ်းကျင်သော AI ဆရာတစ်ဦး ဖြစ်သည်။

လိုက်နာရန်စည်းကမ်းများ:
- အမြဲ မြန်မာဘာသာဖြင့်သာ ပြန်ဖြေပါ
- သဘာဝဆန်ပြီး လူတစ်ယောက်လို ပြောပါ
- ရိုးရှင်းစွာရှင်းပြပြီး လိုအပ်ရင် နက်ရှိုင်းစွာဆက်ရှင်းပါ
- ဥပမာများထည့်ပါ
- ဖော်ရွေပြီး သဘောထားကောင်းစွာ ပြန်ဖြေပါ

ရည်ရွယ်ချက်:
- သင်ကြားပေးခြင်း
- နားလည်အောင်ရှင်းပြခြင်း
`
};

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

    // ✅ Identify user
    const userId = req.ip;

    // ✅ Initialize memory
    if (!userConversations[userId]) {
      userConversations[userId] = [systemPrompt];
    }

    const conversation = userConversations[userId];

    // ✅ Add user message
    conversation.push({
      role: "user",
      content: userMessage
    });

    // ✅ Limit memory (safe)
    if (conversation.length > 20) {
      conversation.splice(1, 2);
    }

    // ✅ OpenAI call (STABLE MODEL)
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini", // ✅ stable & cheap & good
      messages: conversation
    });

    const aiReply = response.choices[0].message.content;

    // ✅ Save AI reply
    conversation.push({
      role: "assistant",
      content: aiReply
    });

    // ✅ Send response
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

// 🚀 Start server
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
});