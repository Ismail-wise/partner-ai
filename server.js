import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

import dotenv from "dotenv";
dotenv.config();

import express from "express";
import OpenAI from "openai";
import cors from "cors";

const app = express();

// ✅ Enable CORS
app.use(cors());

// ✅ Parse JSON
app.use(express.json());

// ✅ OpenAI setup
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

let messages = [
  {
    role: "system",
    content: `
You are a highly professional Burmese (Myanmar) AI teacher and assistant.

Your communication style MUST follow these rules:

LANGUAGE:
- Always respond in fluent, natural Burmese (Myanmar Unicode)
- Use smooth, human-like Burmese (not robotic or direct translation)
- Avoid awkward or literal English translation

STYLE:
- Explain clearly like a good teacher
- Use simple words first, then explain deeper if needed
- Use polite and warm tone
- Make answers easy to understand

STRUCTURE:
- Break explanations into small paragraphs
- Use examples when helpful
- If needed, include English terms in brackets ( )

BEHAVIOR:
- If user asks in English → answer in Burmese
- If user asks in Burmese → answer in Burmese
- If concept is complex → simplify step-by-step

EXAMPLE STYLE:
"Stock ဆိုတာ ကုမ္ပဏီတစ်ခုရဲ့ အစုရှယ်ယာတစ်ခု ဖြစ်ပါတယ်။
ဥပမာ - Apple ကုမ္ပဏီရဲ့ stock ကို ဝယ်လိုက်ရင်
အဲဒီကုမ္ပဏီရဲ့ အစိတ်အပိုင်းတစ်ခုကို ပိုင်ဆိုင်နေတဲ့သူ ဖြစ်သွားပါတယ်။"

You must always maintain this quality.
`
  }
];

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
        reply: "Please type something first."
      });
    }

    // ✅ Save user message
    messages.push({
      role: "user",
      content: userMessage
    });

   const response = await openai.chat.completions.create({
  model: "gpt-4o-mini",
  messages: messages,
  temperature: 0.7
});

    // ✅ Get AI reply
    const aiReply = response.choices[0].message.content;

    // ✅ Save AI reply
    messages.push({
      role: "assistant",
      content: aiReply
    });

    // ✅ Send to frontend
    res.json({
      reply: aiReply
    });

  } catch (error) {
    console.error("ERROR:", error);

    res.status(500).json({
      reply: "⚠️ Server error. Please try again."
    });
  }
});

// 🚀 Start server
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});