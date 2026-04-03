import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: "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"
});

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