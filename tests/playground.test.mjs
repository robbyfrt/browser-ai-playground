import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createServer} from 'node:http';
import {after, before, test} from 'node:test';
import {chromium} from 'playwright';

// All model APIs are mocked. These tests validate the app, not model inference.
const appPath = new URL('../index.html', import.meta.url);
const app = await readFile(appPath, 'utf8');
const prompt = (await readFile(new URL('./fixtures/monthly-savings-v1.txt', import.meta.url), 'utf8')).trimEnd();
let browser, server, url;
before(async () => {
  server = createServer((request, response) => {
    const path = new URL(request.url, 'http://localhost').pathname;
    if (!['/browser-ai-playground/', '/browser-ai-playground/index.html'].includes(path)) {
      response.writeHead(404);
      response.end('Not found');
      return;
    }
    response.writeHead(200, {'Content-Type': 'text/html; charset=utf-8'});
    response.end(app);
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  url = `http://127.0.0.1:${server.address().port}/browser-ai-playground/`;
  browser = await chromium.launch({headless: true});
});
after(async () => {
  await browser?.close();
  if (server?.listening) await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
});

const wasmModule = `
export const env = {backends: {onnx: {wasm: {}}}};
export class InterruptableStoppingCriteria {
  interrupted = false;
  interrupt() { this.interrupted = true; }
}
export class TextStreamer {
  constructor(tokenizer, options) { Object.assign(this, options); }
}
export async function pipeline() {
  if (env.backends.onnx.wasm.numThreads !== 1) throw new Error('WASM must work without isolation headers');
  const pipe = async (messages, options) => {
    const prompt = messages.at(-1).content;
    const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
    if (prompt === 'error-before') throw new Error('mock before output');
    if (prompt === 'hard-stop') await new Promise(() => {});
    await wait(30);
    options.streamer.token_callback_function([1n]);
    options.streamer.callback_function('Partial ');
    if (prompt === 'error-after') throw new Error('mock after output');
    if (prompt === 'stop' || prompt === 'long') {
      while (!options.stopping_criteria[0].interrupted) await wait(10);
      return [{generated_text: [...messages, {role: 'assistant', content: 'Partial '}]}];
    }
    options.streamer.token_callback_function(prompt === 'length'
      ? Array(options.max_new_tokens - 1).fill(4n) : [2n]);
    options.streamer.callback_function('answer');
    return [{generated_text: [...messages, {role: 'assistant', content: 'Partial answer'}]}];
  };
  pipe.tokenizer = {};
  pipe.model = {generation_config: {eos_token_id: 2}};
  return pipe;
}
`;
const webModule = `
export const prebuiltAppConfig = {model_list: [{model_id: 'Qwen3.5-0.8B-q4f16_1-MLC'}]};
export async function deleteModelAllInfoInCache(id) {
  window.mock.deletedIds ??= [];
  window.mock.deletedIds.push(id);
  if (window.mock.deleteError) throw new Error('mock cache removal failure');
  for (const name of ['webllm/model', 'webllm/config', 'webllm/wasm']) {
    const cache = await caches.open(name);
    for (const request of await cache.keys()) {
      if (request.url.includes('/' + id + '/')) await cache.delete(request);
    }
  }
}
export async function CreateMLCEngine(id) {
  window.mock.loadedIds.push(id);
  if (window.mock.loadError) throw new Error('mock load failure');
  let interrupted = false;
  return {
    async unload() { window.mock.unloads++; },
    async resetChat() {
      window.mock.resets++;
      if (window.mock.resetError) throw new Error('mock reset failure');
      await new Promise(resolve => setTimeout(resolve, window.mock.resetDelay));
    },
    async interruptGenerate() { window.mock.interrupts++; interrupted = true; },
    chat: {completions: {create: async function*(request) {
      window.mock.requests.push(request);
      interrupted = false;
      await new Promise(resolve => setTimeout(resolve, 30));
      const text = request.messages.at(-1).content;
      if (text === 'error-before') throw new Error('mock before output');
      if (window.mock.chunks) {
        for (const content of window.mock.chunks) yield {choices: [{delta: {content}}]};
        yield {choices: [{finish_reason: 'stop'}]};
        return;
      }
      yield {choices: [{delta: {content: 'Partial '}}]};
      if (text === 'error-after') throw new Error('mock after output');
      if (text === 'stop' || text === 'long') {
        while (!interrupted) await new Promise(resolve => setTimeout(resolve, 10));
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      if (!interrupted) yield {choices: [{delta: {content: 'answer'}}]};
      yield {choices: [{finish_reason: text === 'unknown' ? null : text === 'length' ? 'length' : 'stop'}]};
      yield {choices: [], usage: {completion_tokens: 7, total_tokens: 24, extra: {decode_tokens_per_s: 12.5}}};
    }}}
  };
}
`;

async function fixture(t, kind, options = {}) {
  const context = await browser.newContext({viewport: {width: 390, height: 844}, acceptDownloads: true});
  t.after(() => context.close());
  await context.route('https://cdn.jsdelivr.net/**', route => route.fulfill({
    contentType: 'text/javascript',
    body: route.request().url().includes('transformers') ? wasmModule : webModule
  }));
  await context.addInitScript(options => {
    window.idle = () => document.querySelector('#send').textContent === 'Send' && !document.querySelector('#send').disabled;
    window.mock = {requests: [], loadedIds: [], resets: 0, creates: [], destroys: 0, interrupts: 0, unloads: 0, resetDelay: 0, ...options};
    Object.defineProperty(navigator, 'gpu', {value: {
      requestAdapter: async () => ({info: {}, features: new Set(['shader-f16']), limits: {maxStorageBufferBindingSize: 1e9}})
    }});
    // Playwright's Chromium exposes a real LanguageModel; remove it to model policy/flag-gated browsers.
    if (window.mock.prompt === 'absent') return void delete self.LanguageModel;
    window.LanguageModel = {
      async availability(options) { window.mock.availabilityOptions = options; return window.mock.prompt || 'available'; },
      async create(options) {
        window.mock.creates.push(options);
        if (window.mock.createError) throw new Error('mock session creation failure');
        return {
          inputUsage: 20, inputQuota: 4096,
          destroy() { window.mock.destroys++; },
          async *promptStreaming(text, {signal}) {
            window.mock.requests.push({text, initialPrompts: options.initialPrompts});
            if (window.mock.reply) { yield window.mock.reply; return; }
            if (text === 'error-before') throw new Error('mock before output');
            await new Promise(resolve => setTimeout(resolve, 30));
            yield 'Partial ';
            if (text === 'error-after') throw new Error('mock after output');
            if (text === 'stop' || text === 'long') {
              await new Promise(resolve => {
                if (signal.aborted) resolve();
                else signal.addEventListener('abort', resolve, {once: true});
              });
              window.mock.interrupts++;
              throw new DOMException('mock aborted', 'AbortError');
            }
            yield 'Partial '; // Repeated chunks must not be mistaken for snapshots.
            yield 'answer';
          }
        };
      }
    };
  }, options);
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  t.after(() => assert.deepEqual(errors, [], 'no unhandled page errors'));
  await page.goto(`${url}?v=${Date.now()}`);
  await page.waitForFunction(() => !document.querySelector('#model').disabled);
  await page.selectOption('#model', kind === 'webllm' ? 'qwen08' : kind === 'prompt' ? '__prompt' : 'wasm');
  await page.waitForFunction(() => !document.querySelector('#load').disabled);
  return page;
}
async function stop(page) {
  await page.waitForFunction(() => document.querySelector('#send').textContent === 'Stop');
  await page.click('#send');
}
async function load(page) {
  await page.click('#load');
  await page.waitForFunction(() => window.idle());
}
async function send(page, text) {
  await page.fill('#input', text);
  await page.click('#send');
  await page.waitForFunction(() => window.idle());
}
async function exported(page) {
  const download = page.waitForEvent('download');
  await page.$eval('#export', button => button.click()); // collapsed in Device details
  const stream = await (await download).createReadStream();
  let json = '';
  for await (const chunk of stream) json += chunk.toString();
  return JSON.parse(json);
}

test('frozen sample, runtime pins and worker remain in one HTML; no automatic deadline', () => {
  assert.equal(app.match(/const SAVINGS_PROMPT=`([\s\S]*?)`;/)[1], prompt);
  assert.doesNotMatch(app, /BUDGET_MS|requestStop\('timeout'\)/);
  assert.match(app, /transformers@3\.8\.1/);
  assert.match(app, /web-llm@0\.2\.87/);
  assert.match(app, /new Blob\(\[`\(\$\{wasmWorker\.toString\(\)\}\)\(\)`\]/);
});

test('static project-path hosting needs no isolation headers or presentation files', async () => {
  const response = await fetch(url);
  assert.equal(response.status, 200);
  assert.match(response.headers.get('content-type'), /text\/html/);
  assert.equal(response.headers.get('cross-origin-opener-policy'), null);
  assert.equal(response.headers.get('cross-origin-embedder-policy'), null);
  assert.equal(await response.text(), app);
  assert.equal((await fetch(new URL('index.html', url))).status, 200);
  assert.equal((await fetch(new URL('missing.html', url))).status, 404);
  assert.doesNotMatch(app, /presentation_blueprint|supporting_files_plans|localhost/);
});

test('startup progressively reveals controls without hiding errors or examples', async t => {
  const page = await fixture(t, 'webllm');
  for (const selector of ['.metrics', '#runStatus', '#clear', '#export', '#preview', '.bar', '#browserNote']) {
    assert.equal(await page.locator(selector).isVisible(), false, `${selector} does not crowd startup`);
  }
  assert.equal(await page.getByRole('button', {name: 'Sort a list', exact: true}).isVisible(), true);
  assert.equal(await page.getByRole('button', {name: 'Monthly savings', exact: true}).isVisible(), false);
  assert.equal(await page.getByRole('button', {name: 'Summarize a notice', exact: true}).isVisible(), false);
  await page.getByText('More examples', {exact: true}).click();
  await page.getByRole('button', {name: 'Summarize a notice', exact: true}).click();
  assert.match(await page.inputValue('#input'), /Summarize this notice/);
  assert.equal(await page.locator('#preview').isVisible(), true);
  await page.fill('#input', '');
  assert.equal(await page.locator('#preview').isVisible(), false);
  await page.setViewportSize({width: 1280, height: 720});
  assert.equal(await page.locator('.cards').evaluate(el => el.open), false);
  await page.getByText('Device details', {exact: true}).click();
  assert.equal(await page.locator('#browserNote').isVisible(), true);
  await page.getByText('Device details', {exact: true}).click();
  await page.setViewportSize({width: 390, height: 844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  const composer = await page.locator('.compose').boundingBox();
  assert.ok(composer.y + composer.height <= 844, 'composer reachable in initial phone viewport');
  await load(page);
  assert.equal(await page.locator('.metrics').isVisible(), false);
  assert.equal(await page.locator('#clear').isVisible(), true);
  await send(page, 'ordinary chat');
  assert.equal(await page.locator('.metrics').isVisible(), true);
  assert.equal(await page.locator('#export').isVisible(), false, 'export stays in collapsed Device details');
  assert.equal(await page.locator('#export').isDisabled(), false);
  assert.equal(await page.textContent('#send'), 'Send');
});

for (const kind of ['wasm', 'webllm', 'prompt']) {
  test(`${kind}: selecting a template starts a new backend conversation`, async t => {
    const page = await fixture(t, kind);
    await load(page);
    await send(page, 'ordinary chat');
    await page.getByRole('button', {name: 'Sort a list', exact: true}).click();
    await page.waitForFunction(() => window.idle());
    assert.match(await page.inputValue('#input'), /Sort these words/);
    assert.equal((await exported(page)).runs.length, 1, 'selection does not send');
    assert.equal(await page.locator('.bot').count(), 0);
    if (kind === 'prompt') {
      assert.equal(await page.evaluate(() => window.mock.destroys), 1);
      assert.equal(await page.evaluate(() => window.mock.creates.length), 2);
    }
    if (kind === 'webllm') assert.equal(await page.evaluate(() => window.mock.resets), 1);
    await page.click('#send');
    await page.waitForFunction(() => window.idle());
    const record = (await exported(page)).runs[1];
    assert.equal(record.messages.length, 2);
    assert.equal(record.freshConversation, true);
    await send(page, 'follow-up');
    assert.equal((await exported(page)).runs[2].messages.length, 4);
  });

  test(`${kind}: fresh workflow, export, honest metrics and ordinary chat`, async t => {
    const page = await fixture(t, kind);
    assert.equal(await page.locator('.cards').evaluate(el => el.open), false);
    await page.getByText('More examples', {exact: true}).click();
    await page.getByRole('button', {name: 'Monthly savings', exact: true}).click();
    assert.equal(await page.inputValue('#input'), prompt);
    assert.equal(await page.textContent('#promptPreview'), prompt);
    await load(page);
    assert.equal(await page.inputValue('#input'), prompt, 'load preserves composer');
    await page.click('#send');
    await page.waitForFunction(() => window.idle());
    let data = await exported(page);
    const first = data.runs[0];
    assert.equal(first.prompt, prompt);
    assert.equal(first.promptVersion, 'monthly-savings-v1');
    assert.equal(first.freshConversation, true);
    assert.equal(first.messages.length, 2);
    assert.equal(first.messages[0].role, 'system');
    assert.equal(first.status, 'completed');
    assert.ok(first.timings.elapsedMs >= 25);
    assert.ok(first.timings.firstTextMs > 0);
    assert.ok(first.startedAt && first.endedAt);
    if (kind === 'wasm') {
      assert.equal(first.settings.max_new_tokens, 2048);
      assert.equal(first.outputTokens.value, 2);
      assert.ok(first.timings.firstTokenMs > 0);
      assert.equal(first.timings.backendDecodeTokensPerSecond, null);
      assert.equal(first.timings.writingSpeedSource, 'measured after first output');
    } else if (kind === 'webllm') {
      assert.equal(first.settings.max_tokens, null);
      assert.equal('max_tokens' in await page.evaluate(() => window.mock.requests[0]), false);
      assert.equal(first.runtimeVersion, '0.2.87');
      assert.equal(first.modelId, 'Qwen3.5-0.8B-q4f16_1-MLC');
      assert.equal(first.outputTokens.value, 7);
      assert.equal(first.timings.backendDecodeTokensPerSecond, 12.5);
      assert.equal(first.timings.writingTokensPerSecond, 12.5);
      assert.match(await page.locator('#mWrite').textContent(), /12\.5 tok\/s/);
      assert.equal(first.timings.firstTokenMs, null);
      assert.equal(first.settings.temperature, null);
      assert.equal(first.settings.enable_thinking, false);
      assert.deepEqual(await page.evaluate(() => window.mock.requests[0].extra_body), {enable_thinking: false});
    } else {
      assert.equal(first.answer, 'Partial Partial answer');
      assert.equal(first.modelId, null);
      assert.equal(first.settings.max_tokens, null);
      assert.equal(first.timings.firstTokenMs, null);
      assert.match(first.outputTokens.source, /estimate/);
      assert.match(first.timings.writingSpeedSource, /^estimate/);
      assert.equal(first.context.limit, 4096);
      assert.deepEqual(await page.evaluate(() => window.mock.availabilityOptions), first.settings.expectedInputs
        ? {expectedInputs: first.settings.expectedInputs, expectedOutputs: first.settings.expectedOutputs} : {});
      assert.equal(await page.evaluate(() => window.mock.destroys), 0, 'first send after load needs no reset');
    }
    await send(page, 'ordinary chat');
    data = await exported(page);
    assert.equal(data.runs[1].messages.length, 4, 'conversation continues by default');
    assert.equal(data.runs[1].freshRequested, false);
    await page.fill('#input', 'keep draft');
    await page.click('#clear');
    await page.waitForFunction(() => window.idle());
    assert.equal(await page.inputValue('#input'), 'keep draft');
    await send(page, 'fresh again');
    data = await exported(page);
    assert.equal(data.runs.length, 3, 'reset retains exports');
    assert.equal(data.runs[2].messages.length, 2);
    if (kind === 'prompt') assert.equal(await page.evaluate(() => window.mock.destroys), 1);
    if (kind === 'webllm') assert.equal(await page.evaluate(() => window.mock.resets), 1);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.selectOption('#model', kind === 'wasm' ? 'qwen08' : 'wasm');
    await page.waitForFunction(() => !document.querySelector('#load').disabled);
    assert.equal(await page.locator('#send').isDisabled(), true);
  });

  test(`${kind}: partial stop, pending export and race guards`, async t => {
    const page = await fixture(t, kind);
    await load(page);
    await page.fill('#input', 'stop');
    await page.click('#send');
    await page.waitForFunction(() => document.querySelector('.bot')?.textContent.includes('Partial'));
    for (const id of ['model', 'load', 'clear', 'input']) assert.equal(await page.locator('#' + id).isDisabled(), true);
    assert.equal(await page.textContent('#send'), 'Stop');
    const pending = (await exported(page)).runs[0];
    assert.equal(pending.status, 'running');
    assert.equal(pending.endedAt, null);
    await stop(page);
    await page.waitForFunction(() => window.idle());
    const record = (await exported(page)).runs[0];
    assert.equal(record.status, 'stopped');
    assert.ok(record.answer.startsWith('Partial'));
    assert.ok(record.cancellation.requestedAt);
    assert.match(record.cancellation.cessation, /settled/);
    assert.match(await page.textContent('#runStatus'), /next send starts a new conversation/);
    await send(page, 'after stop');
    const next = (await exported(page)).runs[1];
    assert.equal(next.freshRequested, true, 'uncertain context is reset automatically');
    assert.equal(next.messages.length, 2);
    assert.equal(next.status, 'completed');
  });

  test(`${kind}: long generation runs past five minutes until Stop`, async t => {
    const page = await fixture(t, kind);
    await load(page);
    await page.clock.install();
    await page.fill('#input', 'long');
    await page.click('#send');
    await page.clock.runFor(100);
    await page.waitForFunction(() => document.querySelector('.bot')?.textContent.includes('Partial'));
    await page.clock.fastForward(600000);
    await page.clock.runFor(500);
    const pending = (await exported(page)).runs[0];
    assert.equal(pending.status, 'running');
    assert.equal(pending.cancellation.requestedAt, null);
    assert.equal(pending.budgetMs, undefined);
    assert.equal(await page.textContent('#send'), 'Stop');
    await page.clock.resume();
    await page.click('#send');
    await page.waitForFunction(() => window.idle());
    const record = (await exported(page)).runs[0];
    assert.equal(record.status, 'stopped');
    assert.ok(record.timings.elapsedMs >= 600000);
    assert.match(record.answer, /Partial/);
  });

  test(`${kind}: error before/after output never fabricates TTFT or loses partial text`, async t => {
    const page = await fixture(t, kind);
    await load(page);
    await send(page, 'error-before');
    await send(page, 'error-after');
    const {runs} = await exported(page);
    assert.equal(runs[0].status, 'error');
    assert.equal(runs[0].timings.firstTokenMs, null);
    assert.equal(runs[0].timings.firstTextMs, null);
    assert.equal(runs[0].answer, '');
    assert.equal(runs[1].status, 'error');
    assert.equal(runs[1].answer, 'Partial ');
    assert.match(runs[1].error.message, /mock after output/);
    assert.match(await page.textContent('.bot'), /Partial/);
  });
}

test('WASM unresponsive generation is terminated, remains exportable and requires reload', async t => {
  const page = await fixture(t, 'wasm');
  await load(page);
  await page.fill('#input', 'hard-stop');
  await page.click('#send');
  await stop(page);
  await page.waitForFunction(() => !document.querySelector('#load').disabled);
  const record = (await exported(page)).runs[0];
  assert.equal(record.status, 'stopped');
  assert.equal(record.timings.firstTokenMs, null);
  assert.match(record.cancellation.cessation, /worker terminated/);
  assert.equal(await page.locator('#send').isDisabled(), true);
  await load(page);
  await send(page, 'reloaded');
});

for (const kind of ['wasm', 'webllm']) {
  test(`${kind}: length status uses backend evidence`, async t => {
    const page = await fixture(t, kind);
    await load(page);
    await send(page, 'length');
    const record = (await exported(page)).runs[0];
    assert.equal(record.status, 'length-limited');
    assert.equal(record.backendFinishReason, 'length');
  });
}

test('WebLLM unknown finish stays unknown; reset/load failures are explicit and exportable', async t => {
  const page = await fixture(t, 'webllm');
  await load(page);
  await send(page, 'unknown');
  assert.equal((await exported(page)).runs[0].status, 'unknown');
  await page.evaluate(() => { window.mock.resetError = true; });
  await send(page, 'reset failure');
  let data = await exported(page);
  assert.equal(data.runs[1].status, 'error');
  assert.equal(data.runs[1].startedAt, null);
  assert.equal(data.runs[1].prompt, 'reset failure');
  assert.equal(await page.inputValue('#input'), 'reset failure');
  await page.selectOption('#model', 'wasm');
  await page.waitForFunction(() => !document.querySelector('#load').disabled);
  await page.selectOption('#model', 'qwen08');
  await page.evaluate(() => { window.mock.loadError = true; });
  await page.click('#load');
  await page.waitForFunction(() => !document.querySelector('#load').disabled);
  data = await exported(page);
  assert.ok(data.events.some(event => event.stage === 'Load failed'));
  assert.equal(await page.locator('#send').isDisabled(), true);
});

for (const [state, label, disabled] of [
  ['absent', 'Built-in browser AI · not exposed', true],
  ['unavailable', 'Chrome built-in Gemini Nano · unavailable', true],
  ['downloadable', 'Chrome built-in Gemini Nano · downloadable', false]
]) {
  test(`Prompt API ${state} is listed with its usable state`, async t => {
    const page = await fixture(t, 'wasm', {prompt: state});
    const option = page.locator('#model option[value="__prompt"]');
    assert.equal(await option.textContent(), label);
    assert.equal(await option.isDisabled(), disabled);
    if (state !== 'absent') assert.deepEqual(await page.evaluate(() => window.mock.availabilityOptions.expectedInputs), [{type: 'text', languages: ['en']}]);
  });
}

for (const [reply, verdict, text] of [
  ['Your longest free block is 15:45–17:00, which is 1 hour 15 minutes.', 'pass', /Rule check: PASS 2\/2/],
  ['Your longest free block is 12:00–13:00, one hour.', 'fail', /Rule check: FAIL 0\/2 · missed: range/]
]) {
  test(`template rule check reports ${verdict}`, async t => {
    const page = await fixture(t, 'prompt', {reply});
    await load(page);
    await page.locator('#examples summary').click();
    await page.getByRole('button', {name: 'Find free time', exact: true}).click();
    await page.click('#send');
    await page.waitForFunction(() => window.idle());
    assert.match(await page.locator('.bot .check').textContent(), text);
    const check = (await exported(page)).runs[0].check;
    assert.equal(check.template, 'Find free time');
    assert.equal(check.verdict, verdict);
  });
}

for (const kind of ['wasm', 'webllm']) {
  test(`${kind}: local dropdown label survives reload; Remove unloads and deletes only that model`, async t => {
    const page = await fixture(t, kind);
    await page.evaluate(async kind => {
      const cache = await caches.open('transformers-cache');
      await cache.put('https://example.com/unrelated', new Response('keep'));
      if (kind === 'wasm') {
        const root = 'https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/';
        for (const file of ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/model_quantized.onnx']) {
          await cache.put(root + file, new Response('{}'));
        }
      } else {
        const root = 'https://huggingface.co/mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC/resolve/main/';
        const weights = await caches.open('webllm/model');
        await weights.put(root + 'tensor-cache.json', new Response(JSON.stringify({records: [{dataPath: 'params_shard_0.bin'}]})));
        await weights.put(root + 'params_shard_0.bin', new Response('weights'));
        await weights.put(root + 'tokenizer.json', new Response('{}'));
        await (await caches.open('webllm/config')).put(root + 'mlc-chat-config.json', new Response('{}'));
      }
    }, kind);
    await page.reload();
    await page.waitForFunction(() => !document.querySelector('#model').disabled);
    const key = kind === 'wasm' ? 'wasm' : 'qwen08';
    await page.selectOption('#model', key);
    await page.waitForFunction(() => !document.querySelector('#load').disabled);
    assert.match(await page.locator(`#model option[value="${key}"]`).textContent(), / · local$/);
    await load(page);
    await page.click('#remove');
    await page.waitForFunction(() => !document.querySelector('#load').disabled);
    assert.doesNotMatch(await page.locator(`#model option[value="${key}"]`).textContent(), / · local$/);
    assert.equal(await page.locator('#remove').isVisible(), false);
    assert.equal(await page.locator('#send').isDisabled(), true);
    assert.equal(await page.evaluate(async () => !!await (await caches.open('transformers-cache')).match('https://example.com/unrelated')), true);
    if (kind === 'webllm') {
      assert.equal(await page.evaluate(() => window.mock.unloads), 1);
      assert.deepEqual(await page.evaluate(() => window.mock.deletedIds), [
        'Qwen3.5-0.8B-q4f16_1-MLC', 'Qwen3.5-0.8B-q4f32_1-MLC'
      ]);
    }
  });
}

test('partial WebLLM download is removable but not labeled local', async t => {
  const page = await fixture(t, 'webllm');
  await page.evaluate(async () => {
    const cache = await caches.open('webllm/model');
    const root = 'https://huggingface.co/mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC/resolve/main/';
    await cache.put(root + 'tensor-cache.json', new Response(JSON.stringify({records: [{dataPath: 'missing.bin'}]})));
  });
  await page.reload();
  await page.waitForFunction(() => !document.querySelector('#model').disabled);
  await page.selectOption('#model', 'qwen08');
  await page.waitForFunction(() => !document.querySelector('#load').disabled);
  assert.doesNotMatch(await page.locator('#model option[value="qwen08"]').textContent(), / · local$/);
  await page.click('#remove');
  await page.waitForFunction(() => !document.querySelector('#load').disabled);
  assert.equal(await page.locator('#remove').isVisible(), false);
});

for (const [range, pass] of [
  ['8:00 AM–1:00 PM', true],
  ['8 a.m.–1 p.m.', true],
  ['08:00–13:00', true],
  ['8 AM to 1 PM', true],
  ['08:00–13:30', false],
  ['8:00 AM–1:30 PM', false],
  ['08:30–13:00', false]
]) {
  test(`notice time rule accepts equivalent times, not different minutes: ${range}`, async t => {
    const reply = `* **What Happens:** Water in Building B is shut off (${range}) for valve replacement.
* **When:** Tuesday, 14 October (last parking time 7:00 PM).
* **Requirements:** Fill containers and leave parking spaces 12–18 free.`;
    const page = await fixture(t, 'prompt', {reply});
    await load(page);
    await page.locator('#examples summary').click();
    await page.getByRole('button', {name: 'Summarize a notice', exact: true}).click();
    await page.waitForFunction(() => window.idle());
    await page.click('#send');
    await page.waitForFunction(() => window.idle());
    const check = (await exported(page)).runs[0].check;
    assert.equal(check.rules.find(item => item.rule === 'time: 08:00 to 13:00').pass, pass);
    assert.equal(check.verdict, pass ? 'pass' : 'fail');
  });
}

for (const [label, chunks, verdict] of [
  ['empty thinking block', ['<thi', 'nk>\n\n</thi', 'nk>\n\napple\nbanana\nfig\nmango\npear'], 'pass'],
  ['thinking with a misleading answer', ['<think>pear, mango, fig, banana, apple</think>', '\napple\nbanana\nfig\nmango\npear'], 'pass'],
  ['wrong final answer', ['<think>apple\nbanana\nfig\nmango\npear</think>', '\npear\nmango'], 'fail'],
  ['unfinished thinking block', ['<think>apple\nbanana\nfig\nmango\npear'], 'fail']
]) {
  test(`WebLLM checks final answer only: ${label}`, async t => {
    const page = await fixture(t, 'webllm', {chunks});
    await load(page);
    await page.getByRole('button', {name: 'Sort a list', exact: true}).click();
    await page.waitForFunction(() => window.idle());
    await page.click('#send');
    await page.waitForFunction(() => window.idle());
    const record = (await exported(page)).runs[0];
    assert.equal(record.rawAnswer, chunks.join(''));
    assert.equal(record.check.verdict, verdict);
    assert.doesNotMatch(record.answer, /think/);
    assert.doesNotMatch(await page.locator('.bot').textContent(), /<think>/);
    if (verdict === 'pass') {
      assert.equal(record.answer, 'apple\nbanana\nfig\nmango\npear');
      await send(page, 'follow-up');
      const next = (await exported(page)).runs[1];
      assert.equal(next.messages[2].content, record.answer);
    }
  });
}

test('every template has rules; free-typed prompts are not checked', async t => {
  const page = await fixture(t, 'prompt');
  assert.deepEqual(await page.locator('#prompts button.chip:not(#clear)').allTextContents(),
    ['Sort a list', 'Summarize a notice', 'Extract a changed meeting', 'Find free time', 'Fix the bug', 'Monthly savings']);
  for (const name of ['Summarize a notice', 'Sort a list', 'Extract a changed meeting', 'Fix the bug', 'Monthly savings']) {
    assert.match(app.match(new RegExp(`name: '${name}'[\\s\\S]*?rules: \\[([\\s\\S]*?)\\]\\}`))[1], /\['/, `${name} has rules`);
  }
  await load(page);
  await send(page, 'hello');
  assert.equal((await exported(page)).runs[0].check, null);
  assert.equal(await page.locator('.bot .check').count(), 0);
});

test('template session failure preserves the draft and leaves no usable stale session', async t => {
  const page = await fixture(t, 'prompt');
  await load(page);
  await page.fill('#input', 'keep my draft');
  await page.evaluate(() => { window.mock.createError = true; });
  await page.getByRole('button', {name: 'Sort a list', exact: true}).click();
  await page.waitForFunction(() => !document.querySelector('#load').disabled);
  assert.equal(await page.locator('#send').isDisabled(), true);
  assert.equal(await page.inputValue('#input'), 'keep my draft');
  assert.match((await exported(page)).events[0].message, /session creation/);
});

test('new conversation blocks model switches and sends until reset settles', async t => {
  const page = await fixture(t, 'webllm', {resetDelay: 400});
  await load(page);
  await page.fill('#input', 'reset race');
  await page.click('#clear');
  for (const id of ['model', 'clear', 'send']) assert.equal(await page.locator('#' + id).isDisabled(), true);
  await page.locator('#send').dispatchEvent('click');
  await page.waitForFunction(() => window.idle());
  assert.equal(await page.evaluate(() => window.mock.requests.length), 0);
  assert.equal(await page.inputValue('#input'), 'reset race');
});
