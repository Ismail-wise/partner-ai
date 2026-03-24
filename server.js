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

// 🧠 MEMORY (conversation history)
let messages = [
  {
    role: "system",
    content: `
You are a professional AI tutor.

Rules:
- Always answer in Burmese (Myanmar language)
- Use simple, clear, natural Burmese
- Explain like a teacher
- If needed, include English terms in brackets
- Be friendly and helpful

If user writes in English → reply in Burmese
If user writes in Burmese → reply in Burmese
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

    // ✅ Send FULL conversation to OpenAI
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: messages
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