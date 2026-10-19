export const CHAT_INTERFACE_PROTOCOL = 1;
export const REPLY_LANGUAGES = Object.freeze(['auto', 'my', 'en']);

export function replyLanguageInstructions(language = 'auto') {
    if (!REPLY_LANGUAGES.includes(language)) throw new TypeError('Unsupported reply language');
    const choice = {
        auto: 'Match the language of the latest user message: English questions receive English answers, Burmese questions receive Burmese answers. For mixed messages, follow the main language; prefer Burmese only when the main language is unclear. Do not let the language of earlier turns or retrieved sources override the latest question.',
        my: 'The user selected Burmese. Reply in natural, respectful Burmese, retaining familiar English business terms when clearer.',
        en: 'The user selected English. Reply in clear, natural English.',
    }[language];
    return `${choice}
An explicit request in the latest message to use a different language takes precedence over this preference. Do not translate or repeat every answer in both languages unless asked.
Use concise, conversational language. Start with the answer. Usually a few short paragraphs are enough; provide detail when requested or necessary to answer accurately. Avoid canned greetings, slogans, emoji, decorative headings, repeated disclaimers, summaries of the user's question, and unnecessary follow-up questions. Keep necessary uncertainty and source references.`;
}
