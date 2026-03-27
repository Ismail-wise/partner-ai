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

// ✅ Rate limit (protect API money)
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

// ✅ Your system prompt (FIXED)
const systemPrompt = {
  role: "system",
  content: `
You are a highly professional Burmese (Myanmar) AI teacher and assistant.

LANGUAGE:
- Always respond in fluent Burmese (Myanmar Unicode)
- Natural and human-like

STYLE:
- Clear explanation
- Simple first, deeper if needed
- Warm and polite

STRUCTURE:
- Short paragraphs
- Use examples

BEHAVIOR:
- Always answer in Burmese
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
        reply: "Please type something first."
      });
    }

    if (userMessage.length > 500) {
      return res.status(400).json({
        reply: "Message too long (max 500 characters)"
      });
    }

    // ✅ Identify user (simple)
    const userId = req.ip;

    // ✅ Create conversation if not exists
    if (!userConversations[userId]) {
      userConversations[userId] = [systemPrompt];
    }

    const conversation = userConversations[userId];

    // ✅ Add user message
    conversation.push({
      role: "user",
      content: userMessage
    });

    // ✅ Limit memory (important)
    if (conversation.length > 20) {
      conversation.splice(1, 2);
    }

    // ✅ Call OpenAI
    const response = await openai.chat.completions.create({
      model: "gpt-5-mini", // upgraded model
      messages: conversation,
      temperature: 0.7
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
    console.error("ERROR:", error);

    res.status(500).json({
      reply: "⚠️ Server error. Please try again."
    });
  }
});

// 🚀 Start server
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});