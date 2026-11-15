import { record, type Obj } from './brief.ts';

function textContent(value: unknown): string {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) return '';
  return value.filter((v) => record(v) && v.type === 'text' && typeof v.text === 'string').map((v: Obj) => v.text).join('\n');
}
function jsonObject(value: unknown): Obj {
  if (record(value)) return value;
  if (typeof value === 'string') try { const p: unknown = JSON.parse(value); if (record(p)) return p; } catch { /* invalid args */ }
  return {};
}
function stopToAnthropic(value: unknown): string {
  if (value === 'length') return 'max_tokens';
  if (value === 'tool_calls' || value === 'function_call') return 'tool_use';
  if (value === 'content_filter') return 'refusal';
  return 'end_turn';
}

export function anthropicRequest(body: Obj, model: string): Obj {
  const messages: Obj[] = [];
  const system = textContent(body.system);
  if (system) messages.push({ role: 'system', content: system });
  if (Array.isArray(body.messages)) for (const item of body.messages) {
    if (!record(item)) continue;
    const assistant = item.role === 'assistant';
    if (typeof item.content === 'string') { messages.push({ role: assistant ? 'assistant' : 'user', content: item.content }); continue; }
    if (!Array.isArray(item.content)) continue;
    if (assistant) {
      const calls: Obj[] = [];
      const texts: string[] = [];
      for (const block of item.content) if (record(block)) {
        if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
        if (block.type === 'tool_use') calls.push({ id: typeof block.id === 'string' ? block.id : '', type: 'function', function: { name: block.name, arguments: JSON.stringify(record(block.input) ? block.input : {}) } });
      }
      messages.push({ role: 'assistant', content: texts.join('\n') || null, ...(calls.length ? { tool_calls: calls } : {}) });
    } else {
      const parts: Obj[] = [];
      for (const block of item.content) if (record(block)) {
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
        else if (block.type === 'image' && record(block.source)) {
          const src = block.source;
          if (src.type === 'base64' && typeof src.data === 'string') parts.push({ type: 'image_url', image_url: { url: `data:${typeof src.media_type === 'string' ? src.media_type : 'image/png'};base64,${src.data}` } });
          else if (src.type === 'url' && typeof src.url === 'string') parts.push({ type: 'image_url', image_url: { url: src.url } });
        } else if (block.type === 'tool_result') {
          const content = typeof block.content === 'string' ? block.content : JSON.stringify(block.content ?? '');
          messages.push({ role: 'tool', tool_call_id: block.tool_use_id ?? '', content });
        }
      }
      if (parts.length) messages.push({ role: 'user', content: parts });
    }
  }
  const out: Obj = { model, messages };
  for (const key of ['max_tokens', 'temperature', 'top_p']) if (body[key] !== undefined) out[key] = body[key];
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.filter(record).map((t: Obj) => ({ type: 'function', function: { name: t.name, description: t.description ?? '', parameters: t.input_schema ?? { type: 'object', properties: {} } } }));
    if (tools.length) out.tools = tools;
  }
  if (record(body.tool_choice)) {
    const choice = body.tool_choice;
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) { out.stream = true; out.stream_options = { include_usage: true }; }
  return out;
}

export function anthropicResponse(answer: Obj, model: string, declaredTools: Set<string>): Obj {
  const choice = Array.isArray(answer.choices) && record(answer.choices[0]) ? answer.choices[0] : {};
  const msg = record(choice.message) ? choice.message : {};
  const content: Obj[] = [];
  const text = typeof msg.content === 'string' ? msg.content : textContent(msg.content);
  if (text) content.push({ type: 'text', text });
  if (Array.isArray(msg.tool_calls)) for (const call of msg.tool_calls) if (record(call) && record(call.function) && declaredTools.has(call.function.name)) {
    content.push({ type: 'tool_use', id: call.id ?? 'toolu_0', name: call.function.name, input: jsonObject(call.function.arguments) });
  }
  if (!content.length) content.push({ type: 'text', text: '' });
  const usage = record(answer.usage) ? answer.usage : {};
  return {
    id: typeof answer.id === 'string' ? `msg_${answer.id}` : 'msg_aggregate', type: 'message', role: 'assistant', model, content,
    stop_reason: stopToAnthropic(choice.finish_reason), stop_sequence: null,
    usage: { input_tokens: Number(usage.prompt_tokens) || 0, output_tokens: Number(usage.completion_tokens) || 1 },
  };
}

async function* frames(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let split = buffer.indexOf('\n\n');
      while (split >= 0) { yield buffer.slice(0, split); buffer = buffer.slice(split + 2); split = buffer.indexOf('\n\n'); }
    }
    buffer += decoder.decode();
    if (buffer.trim()) yield buffer;
  } finally { reader.releaseLock(); }
}

export async function* openAIStream(stream: ReadableStream<Uint8Array>, model: string, allowedTools: Set<string>): AsyncGenerator<string> {
  const rejected = new Set<number>();
  for await (const frame of frames(stream)) {
    const lines = frame.split('\n').flatMap((line) => {
      const trimmed = line.trimStart();
      if (!trimmed.startsWith('data:')) return [line];
      const payload = trimmed.slice(5).trim();
      if (!payload || payload === '[DONE]') return [line];
      try {
        const chunk: unknown = JSON.parse(payload);
        if (!record(chunk)) return [line];
        const choices = Array.isArray(chunk.choices) ? chunk.choices.map((raw) => {
          if (!record(raw) || !record(raw.delta) || !Array.isArray(raw.delta.tool_calls)) return raw;
          const delta = { ...raw.delta, tool_calls: raw.delta.tool_calls.flatMap((call: unknown) => {
            if (!record(call)) return [];
            const index = typeof call.index === 'number' ? call.index : 0;
            const fn = record(call.function) ? call.function : {};
            if (typeof fn.name === 'string' && !allowedTools.has(fn.name)) rejected.add(index);
            return rejected.has(index) ? [] : [call];
          }) };
          return { ...raw, delta };
        }) : chunk.choices;
        return [`data: ${JSON.stringify({ ...chunk, ...(Array.isArray(chunk.choices) ? { choices } : {}), model })}`];
      } catch { /* pass through unknown SSE data */ }
      return [line];
    });
    yield `${lines.join('\n')}\n\n`;
  }
}

export async function* anthropicStream(stream: ReadableStream<Uint8Array>, model: string, allowedTools: Set<string>): AsyncGenerator<string> {
  let started = false, textOpen = false, nextBlock = 0, stop = 'end_turn', input = 0, output = 1;
  const tools = new Map<number, { index: number; id: string; name: string; allowed: boolean }>();
  const emit = (event: string, data: Obj): string => `event: ${event}\ndata: ${JSON.stringify({ ...data, model })}\n\n`;
  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [emit('message_start', { type: 'message_start', message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } })];
  };
  for await (const frame of frames(stream)) {
    for (const line of frame.split('\n')) {
      const part = line.trimStart();
      if (!part.startsWith('data:')) continue;
      const raw = part.slice(5).trim();
      if (!raw) continue;
      if (raw === '[DONE]') continue;
      let chunk: unknown;
      try { chunk = JSON.parse(raw); } catch { continue; }
      if (!record(chunk)) continue;
      const out: string[] = [];
      if (record(chunk.usage)) {
        input = Number(chunk.usage.prompt_tokens) || 0;
        output = Number(chunk.usage.completion_tokens) || 1;
      }
      const choice = Array.isArray(chunk.choices) && record(chunk.choices[0]) ? chunk.choices[0] : null;
      if (!choice) continue;
      out.push(...start());
      const delta = record(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string' && delta.content) {
        if (!textOpen) { textOpen = true; const index = nextBlock++; out.push(emit('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } })); }
        out.push(emit('content_block_delta', { type: 'content_block_delta', index: nextBlock - 1, delta: { type: 'text_delta', text: delta.content } }));
      }
      if (Array.isArray(delta.tool_calls)) for (const call of delta.tool_calls) if (record(call)) {
        const upstreamIndex = typeof call.index === 'number' ? call.index : tools.size;
        let t = tools.get(upstreamIndex);
        const fn = record(call.function) ? call.function : {};
        if (!t) {
          const name = typeof fn.name === 'string' ? fn.name : '';
          if (!allowedTools.has(name)) { tools.set(upstreamIndex, { index: -1, id: '', name, allowed: false }); continue; }
          if (textOpen) { out.push(emit('content_block_stop', { type: 'content_block_stop', index: nextBlock - 1 })); textOpen = false; }
          t = { index: nextBlock++, id: typeof call.id === 'string' ? call.id : `toolu_${nextBlock}`, name, allowed: true };
          tools.set(upstreamIndex, t);
          out.push(emit('content_block_start', { type: 'content_block_start', index: t.index, content_block: { type: 'tool_use', id: t.id, name: t.name, input: {} } }));
        }
        if (t.allowed && typeof fn.arguments === 'string' && fn.arguments) out.push(emit('content_block_delta', { type: 'content_block_delta', index: t.index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
      }
      if (choice.finish_reason !== undefined && choice.finish_reason !== null) stop = stopToAnthropic(choice.finish_reason);
      for (const event of out) yield event;
    }
  }
  for (const event of start()) yield event;
  if (textOpen) yield emit('content_block_stop', { type: 'content_block_stop', index: nextBlock - 1 });
  for (const t of tools.values()) if (t.allowed) yield emit('content_block_stop', { type: 'content_block_stop', index: t.index });
  yield emit('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: output } });
  yield emit('message_stop', { type: 'message_stop' });
}

export function anthropicModels(models: Array<{ id: string; display_name: string }>): Obj {
  return { data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: '2026-01-01T00:00:00Z' })), has_more: false };
}
export function openAIModels(models: Array<{ id: string; provider: string }>): Obj {
  return { object: 'list', data: models.map((m) => ({ id: m.id, object: 'model', created: 1767225600, owned_by: m.provider })) };
}
