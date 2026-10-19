export const CHAT_INTERFACE_PROTOCOL = 1;
export const REPLY_LANGUAGES = Object.freeze(['auto', 'my', 'en']);

export function replyLanguageInstructions(language = 'auto') {
    if (!REPLY_LANGUAGES.includes(language)) throw new TypeError('Unsupported reply language');
    const choice = {
        auto: 'Match the language of the latest user message: English questions receive English answers, Burmese questions receive Burmese answers. For mixed messages, follow the main language; prefer Burmese only when the main language is unclear. Do not let the language of earlier turns or retrieved sources override the latest question.',
        my: 'The user selected Burmese. Reply in natural, respectful Burmese. Write the explanation in Burmese even when the question, earlier answers or retrieved sources are in English. Familiar English business terms, names and exact source references may remain unchanged.',
        en: 'The user selected English. Reply in clear, natural English. Write the explanation in English even when the question, earlier answers or retrieved sources are in Burmese. Do not add a Burmese greeting, translation or summary. Names, quoted terms and exact source references may remain unchanged.',
    }[language];
    return `${choice}
An explicit request in the latest message to use a different language takes precedence over this preference. Do not translate or repeat every answer in both languages unless asked.
Use concise, conversational language. Start with the answer. Usually a few short paragraphs are enough; provide detail when requested or necessary to answer accurately. Avoid canned greetings, slogans, emoji, decorative headings, repeated disclaimers, summaries of the user's question, and unnecessary follow-up questions. Keep necessary uncertainty and source references.`;
}

// Keep the current turn's preference after potentially long, multilingual RAG
// evidence. Source text and history must not become a reply-language preference.
export function withReplyLanguageInstructions(contextInstructions, language = 'auto') {
    return `${contextInstructions}\n\nREPLY LANGUAGE FOR THIS TURN\n${replyLanguageInstructions(language)}\nThe language of evidence is not a request to use that language. Answer the latest question now, using the reply-language rule above.`;
}
