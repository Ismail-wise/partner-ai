import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function run() {
  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: "You are a smart business consultant." },
      { role: "user", content: "How should I start a partnership business?" }
    ]
  });

  console.log(response.choices[0].message.content);
}

run();