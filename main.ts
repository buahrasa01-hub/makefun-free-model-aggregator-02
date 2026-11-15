import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { pathToFileURL } from 'node:url';
import { readBrief, makeCatalog, candidates, needFromBody, record, type Obj, type Model, type Need } from './lib/brief.ts';
import { anthropicRequest, anthropicResponse, anthropicStream, anthropicModels, openAIModels, openAIStream } from './lib/translation.ts';

const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const promptPath = process.env.MAKEFUN_PROMPT_FILE ?? '';
const brief = readBrief(promptPath);
const catalog = makeCatalog(brief);
const refusedModels = new Set<string>();
const answered = new Map<string, number>();
let credentialsRejected = false;
let arrival = 0;
const maxRequest = 8 * 1024 * 1024;
const gatewayBase = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const gatewayKey = process.env.OPENAI_API_KEY ?? '';

function sendJson(res: ServerResponse, status: number, value: unknown): void {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(value));
}
function errorBody(wire: 'openai' | 'anthropic', type: string, message: string): Obj {
  return wire === 'anthropic' ? { type: 'error', error: { type, message } } : { error: { message, type, code: null } };
}
async function readBody(req: IncomingMessage): Promise<Obj | null> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const raw of req) {
    const chunk = Buffer.isBuffer(raw) ? raw : Buffer.from(String(raw));
    total += chunk.length;
    if (total > maxRequest) return null;
    chunks.push(chunk);
  }
  try { const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8')); return record(value) ? value : null; }
  catch { return null; }
}
function requestNeed(body: Obj): Need {
  const prompt = brief.prompts[arrival++];
  return prompt?.requires ?? needFromBody(body);
}
function ranked(need: Need): Model[] {
  if (credentialsRejected) return [];
  return candidates(brief.models, need).filter((m) => !refusedModels.has(m.id)).sort((a, b) =>
    b.rating_milli - a.rating_milli || (answered.get(a.id) ?? 0) - (answered.get(b.id) ?? 0) || a.id.localeCompare(b.id));
}
function chatUrl(): string {
  if (!gatewayBase) return '';
  return gatewayBase.endsWith('/v1') ? `${gatewayBase}/chat/completions` : `${gatewayBase}/v1/chat/completions`;
}
function refusalCode(text: string): string {
  try {
    const data: unknown = JSON.parse(text);
    if (record(data)) {
      if (record(data.error)) return String(data.error.code ?? data.error.type ?? 'upstream_refused');
      return String(data.code ?? 'upstream_refused');
    }
  } catch { /* use stable fallback */ }
  return 'upstream_refused';
}
interface Failure { status: number; code: string; model: string | null }
async function dial(model: string, body: Obj): Promise<Response> {
  const headers: Record<string, string> = { 'content-type': 'application/json', authorization: `Bearer ${gatewayKey}` };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}
async function completion(body: Obj, wire: 'openai' | 'anthropic', res: ServerResponse): Promise<void> {
  const need = requestNeed(body);
  const pool = ranked(need);
  if (!pool.length) { sendJson(res, 503, errorBody(wire, 'overloaded_error', 'no eligible free model')); return; }
  const failures: Failure[] = [];
  let success: { model: Model; response: Response } | null = null;
  let request: Obj = wire === 'openai' ? { ...body } : anthropicRequest(body, pool[0].id);
  for (const model of pool) {
    request = wire === 'openai' ? { ...body, model: model.id } : anthropicRequest(body, model.id);
    let upstream: Response;
    try { upstream = await dial(model.id, request); }
    catch {
      failures.push({ status: 599, code: 'gateway_network_error', model: model.id });
      refusedModels.add(model.id);
      continue;
    }
    if (upstream.ok && upstream.body) { success = { model, response: upstream }; break; }
    const text = await upstream.text();
    const code = refusalCode(text);
    failures.push({ status: upstream.status, code, model: model.id });
    if (upstream.status === 401) { credentialsRejected = true; break; }
    const retryable = upstream.status === 429 || upstream.status >= 500 || (upstream.status === 400 && code === 'model_not_allowed');
    if (retryable) refusedModels.add(model.id);
    else break;
  }
  if (!success) {
    if (failures.length) res.setHeader('x-makefun-upstream-failures', JSON.stringify(failures.slice(0, 8)));
    sendJson(res, 502, errorBody(wire, 'upstream_error', 'no eligible model returned a completion'));
    return;
  }
  const { model, response } = success;
  answered.set(model.id, (answered.get(model.id) ?? 0) + 1);
  if (failures.length) res.setHeader('x-makefun-upstream-failures', JSON.stringify(failures.slice(0, 8)));
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    try {
      const stream = response.body!;
      const allowed = new Set(Array.isArray(body.tools) ? body.tools.filter(record).map((t: Obj) => t.name).filter((v: unknown) => typeof v === 'string') : []);
      const events = wire === 'openai' ? openAIStream(stream, model.id, allowed) : anthropicStream(stream, model.id, allowed);
      for await (const event of events) res.write(event);
    } finally { res.end(); }
    return;
  }
  const raw: unknown = await response.json();
  const answer = record(raw) ? raw : {};
  const declared = new Set<string>(Array.isArray(body.tools) ? body.tools.filter(record).map((t: Obj) => t.name).filter((v: unknown): v is string => typeof v === 'string') : []);
  if (wire === 'openai') {
    const choices = Array.isArray(answer.choices) ? answer.choices.map((raw) => {
      if (!record(raw) || !record(raw.message) || !Array.isArray(raw.message.tool_calls)) return raw;
      return { ...raw, message: { ...raw.message, tool_calls: raw.message.tool_calls.filter((call: unknown) => record(call) && record(call.function) && declared.has(call.function.name)) } };
    }) : answer.choices;
    const usage = record(answer.usage) ? answer.usage : {};
    sendJson(res, 200, { ...answer, ...(Array.isArray(answer.choices) ? { choices } : {}), model: model.id, usage: { ...usage, prompt_tokens: Number(usage.prompt_tokens) || 0, completion_tokens: Math.max(1, Number(usage.completion_tokens) || 0) } });
  } else {
    sendJson(res, 200, anthropicResponse(answer, model.id, declared));
  }
}

export async function routeRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const pathname = new URL(req.url ?? '/', 'http://local').pathname;
  if (req.method === 'GET' && pathname === '/v1/models') {
    const models = brief.models.filter((m) => m.free);
    if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModels(models));
    else sendJson(res, 200, openAIModels(models));
    return;
  }
  if (req.method === 'POST' && (pathname === '/v1/chat/completions' || pathname === '/v1/messages')) {
    const wire = pathname.endsWith('/messages') ? 'anthropic' : 'openai';
    const body = await readBody(req);
    if (!body) { sendJson(res, 400, errorBody(wire, 'invalid_request_error', 'body must be a JSON object')); return; }
    await completion(body, wire, res);
    return;
  }
  sendJson(res, 404, errorBody(pathname === '/v1/messages' ? 'anthropic' : 'openai', 'not_found', 'route not found'));
}

async function start(): Promise<void> {
  if (process.argv.includes('--list')) { process.stdout.write(`${JSON.stringify(catalog)}\n`); return; }
  if (!promptPath || !envUrl) { process.stdout.write('{"providers":[]}\n'); return; }
  const server = createServer((req, res) => { void routeRequest(req, res).catch(() => { if (!res.headersSent) sendJson(res, 500, { error: { message: 'internal error' } }); else res.end(); }); });
  await new Promise<void>((resolve, reject) => { server.once('error', reject); server.listen(0, '0.0.0.0', resolve); });
  const address = server.address();
  if (!address || typeof address === 'string' || address.port === 8080) throw new Error('could not acquire a free port');
  const registration = await fetch(`${envUrl}/register`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ protocol_version: 1, port: address.port, catalog }) });
  if (!registration.ok) throw new Error(`register failed: ${registration.status}`);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  const timer = setInterval(() => {
    void fetch(`${envUrl}/work`).then((r) => r.json()).then((work: unknown) => {
      if (record(work) && work.done === true) { clearInterval(timer); server.close(); process.exit(0); }
    }).catch(() => { /* try on the next 100 ms poll */ });
  }, 100);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await start();
