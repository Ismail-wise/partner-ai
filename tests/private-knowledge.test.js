import test from 'node:test';
import assert from 'node:assert/strict';
import { registerPrivateKnowledge, authenticateSecret, validatePrivateFile, privateChatOptions, privateChatRules } from '../private-knowledge.js';

function harness(models) {
    const routes = new Map(), middleware = [];
    const app = {use: (...args) => middleware.push(args), post: (path, handler) => routes.set(path, handler)};
    registerPrivateKnowledge(app, {ai: {models}, secret: 's'.repeat(40), model: 'existing-model', embeddingModel: 'gemini-embedding-2', dimensions: 128, jsonParser: (_req,_res,next) => next()});
    return {routes, middleware};
}
async function call(handler, body) {
    const result = {code: 200};
    const res = {status: code => {result.code=code;return res;}, json: value => {result.body=value;return result;}};
    await handler({body}, res);return result;
}
test('internal authentication denies missing, malformed and mismatched secrets', () => {
    const secret='s'.repeat(40);
    assert.equal(authenticateSecret(undefined, secret),false);
    assert.equal(authenticateSecret(['s'.repeat(40)], secret),false);
    assert.equal(authenticateSecret('t'.repeat(40), secret),false);
    assert.equal(authenticateSecret(secret, secret),true);
    const {middleware}=harness({});let next=false, code;
    middleware[0][1]({headers:{}},{status:n=>({json:()=>{code=n;}})},()=>next=true);
    assert.equal(code,403);assert.equal(next,false);
});
test('private chats omit external search while personal chats retain it', () => {
    assert.deepEqual(privateChatOptions({scope:'owner_private_knowledge_v1'}),{});
    assert.deepEqual(privateChatOptions({scope:'personal_partnership_advice'}),{tools:[{googleSearch:{}}]});
    assert.match(privateChatRules(),/draft as proposed/);assert.match(privateChatRules(),/\[S1\]/);
});
test('private originals reject executable types, MIME mismatch and corrupt base64', () => {
    for(const body of [{mimeType:'image/svg+xml',data:'PHN2Zz4='},{mimeType:'image/png',data:Buffer.from('<script>bad()</script>').toString('base64')},{mimeType:'application/pdf',data:'%bad%'}]) assert.throws(()=>validatePrivateFile(body));
    assert.equal(validatePrivateFile({mimeType:'application/pdf',data:Buffer.from('%PDF-1.7 test').toString('base64')}).mime,'application/pdf');
});
test('transcription uses inline bytes without tools and flags incomplete extraction',async () => {
    let request;
    const {routes}=harness({generateContent:async value=>{request=value;return {text:'မြန်မာ 250,000',candidates:[{finishReason:'MAX_TOKENS'}]};}});
    const result=await call(routes.get('/internal/pbr/private/extract'),{mimeType:'application/pdf',data:Buffer.from('%PDF-1.7 test').toString('base64')});
    assert.equal(result.code,200);assert.equal(result.body.partial,true);assert.equal(result.body.text,'မြန်မာ 250,000');
    assert.equal(request.contents[0].parts[1].inlineData.mimeType,'application/pdf');assert.equal(request.config.tools,undefined);
});
test('embedding preserves one vector per private chunk and never accepts aggregation',async () => {
    let request;
    const {routes}=harness({embedContent:async value=>{request=value;return {embeddings:value.contents.map(()=>({values:Array(128).fill(.25)}))};}});
    const handler=routes.get('/internal/pbr/private/embed');
    const result=await call(handler,{mode:'document',texts:['လုပ်ငန်း A','လုပ်ငန်း B']});
    assert.equal(result.body.vectors.length,2);assert.equal(result.body.embeddingKey,'gemini-embedding-2:128:private-v1');
    assert.equal(request.contents.length,2);assert.match(request.contents[0].parts[0].text,/title:/);
    assert.equal((await call(handler,{mode:'query',texts:['one','two']})).code,422);
    const broken=harness({embedContent:async()=>({embeddings:[{values:Array(128).fill(.1)}]})});
    assert.equal((await call(broken.routes.get('/internal/pbr/private/embed'),{mode:'document',texts:['one','two']})).code,503);
});
test('provider errors return generic failures without reflecting private data',async()=>{
    const {routes}=harness({generateContent:async()=>{throw new Error('PRIVATE PAYROLL 500000 API KEY');},embedContent:async()=>{throw new Error('PRIVATE DATA');}});
    const extract=await call(routes.get('/internal/pbr/private/extract'),{mimeType:'application/pdf',data:Buffer.from('%PDF-1.7 test').toString('base64')});
    const embed=await call(routes.get('/internal/pbr/private/embed'),{mode:'document',texts:['private note']});
    assert.equal(extract.code,503);assert.equal(embed.code,503);assert.doesNotMatch(JSON.stringify([extract,embed]),/PAYROLL|API KEY|PRIVATE DATA/);
});
