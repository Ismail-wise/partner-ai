import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import dotenv from "dotenv";
dotenv.config();

import express from "express";
import OpenAI from "openai";
import cors from "cors";
import rateLimit from "express-rate-limit";

// Fix __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// ==========================
// ⚙️ MIDDLEWARE
// ==========================
app.use(rateLimit({
  windowMs: 60 * 1000,
  max: 30
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
// 🧠 SIMPLE MEMORY (NO PDF)
// ==========================
const courseMemory = `
Stock investing ဆိုတာသည် ကုမ္ပဏီများ၏ ရှယ်ယာများကို ဝယ်ယူခြင်းဖြစ်သည်။

Risk Management ဆိုတာသည် အရှုံးကိုလျှော့ချရန် နည်းလမ်းများဖြစ်သည်။

Diversification ဆိုတာသည် မတူညီသော assets များတွင် ရင်းနှီးမြှုပ်နှံခြင်းဖြစ်သည်။

Long-term investing သည် အချိန်ကြာရှည်စွာ ရင်းနှီးမြှုပ်နှံခြင်းဖြစ်သည်။
`;

// ==========================
// 🧠 SYSTEM PROMPT
// ==========================
const systemPrompt = {
  role: "system",
  content: `
သင်သည် မြန်မာဘာသာဖြင့် သင်ကြားနိုင်သော AI ဆရာဖြစ်သည်။

စည်းကမ်းများ:
- မြန်မာဘာသာဖြင့်သာ ပြန်ဖြေပါ
- ရိုးရှင်းပြီး နားလည်လွယ်အောင်ရှင်းပြပါ
- ဥပမာများထည့်ပါ
`
};

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
      systemPrompt,
      {
        role: "system",
        content: `
အောက်ပါ content သည် course မှ ဖြစ်သည်။
လိုအပ်ပါက အသုံးပြုပါ:

${courseMemory}
`
      },
      {
        role: "user",
        content: userMessage
      }
    ];

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages,
      temperature: 0.7
    });

    const reply = response.choices[0].message.content;

    res.json({ reply });

  } catch (err) {
    console.error(err.message);
    res.json({ reply: "⚠️ Error ဖြစ်ပါတယ်" });
  }
});

// ==========================
// 🚀 START SERVER
// ==========================
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
});