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

// 🧠 MEMORY (IMPORTANT)
let messages = [
  { role: "system", content: "You are a partnership business consultant." }
];

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});
app.post("/chat", async (req, res) => {
  try {
    const userMessage = req.body.message;

    // ✅ Input validation
    if (!userMessage || userMessage.trim() === "") {
      return res.status(400).json({
        reply: "Please type something first."
      });
    }

    // ✅ Save user message to memory
    messages.push({ role: "user", content: userMessage });

    // ✅ Send FULL conversation to AI
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: messages
    });

    // ✅ Get AI reply
    const aiReply = response.choices[0].message.content;

    // ✅ Save AI reply to memory
    messages.push({ role: "assistant", content: aiReply });

    // ✅ Send back to frontend
    res.json({
      reply: aiReply
    });

  } catch (error) {
    console.error(error);

    res.status(500).json({
      reply: "Something went wrong. Check server console."
    });
  }
});

app.get("/", (req, res) => {
  res.send("Server is running ✅");
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});