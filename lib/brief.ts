import { readFileSync } from 'node:fs';

export type Obj = Record<string, any>;
export interface Model {
  id: string;
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}
export interface Need { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number }
export interface Prompt { id: string; api: string; stream: boolean; weight: number; requires: Need }
export interface Brief { models: Model[]; prompts: Prompt[] }
export interface Catalog {
  providers: Array<{
    id: string; base_url: string; key_env: string; wire: 'openai';
    models: Array<{ id: string; free: true; context_length: number | null }>;
  }>;
}

export const record = (v: unknown): v is Obj => !!v && typeof v === 'object' && !Array.isArray(v);

function documents(source: string): Obj[] {
  const found: Obj[] = [];
  for (let from = source.indexOf('{'); from >= 0; from = source.indexOf('{', from + 1)) {
    let depth = 0, quoted = false, escaped = false, end = -1;
    for (let i = from; i < source.length; i++) {
      const ch = source[i];
      if (quoted) {
        if (escaped) escaped = false;
        else if (ch === '\\') escaped = true;
        else if (ch === '"') quoted = false;
      } else if (ch === '"') quoted = true;
      else if (ch === '{') depth++;
      else if (ch === '}' && --depth === 0) { end = i + 1; break; }
    }
    if (end < 0) continue;
    try {
      const value: unknown = JSON.parse(source.slice(from, end));
      if (record(value) && typeof value.schema === 'string' && value.schema.startsWith('free-model-aggregator-02/')) found.push(value);
    } catch { /* A brace in the prose is not a document. */ }
  }
  return found;
}

function num(v: unknown, fallback = 0): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}
function nullableBool(v: unknown): boolean | null { return typeof v === 'boolean' ? v : null; }

export function readBrief(path: string): Brief {
  if (!path) return { models: [], prompts: [] };
  try {
    const docs = documents(readFileSync(path, 'utf8'));
    const modelDoc = docs.find((d) => d.schema.endsWith('/models/v1') || Array.isArray(d.models) || Array.isArray(d.ratings));
    const promptDoc = docs.find((d) => d.schema.endsWith('/prompts/v1') || Array.isArray(d.manifest) || Array.isArray(d.prompts));
    const rawModels = modelDoc && (Array.isArray(modelDoc.models) ? modelDoc.models : modelDoc.ratings);
    const rawPrompts = promptDoc && (Array.isArray(promptDoc.prompts) ? promptDoc.prompts : promptDoc.manifest);
    const models: Model[] = Array.isArray(rawModels) ? rawModels.flatMap((v: unknown) => {
      if (!record(v) || typeof v.id !== 'string') return [];
      return [{
        id: v.id,
        provider: typeof v.provider === 'string' ? v.provider : 'openrouter',
        display_name: typeof v.display_name === 'string' ? v.display_name : v.id,
        rating_milli: num(v.rating_milli),
        context_length: typeof v.context_length === 'number' ? v.context_length : null,
        tools: nullableBool(v.tools), vision: nullableBool(v.vision), json: nullableBool(v.json), free: v.free === true,
      }];
    }) : [];
    const prompts: Prompt[] = Array.isArray(rawPrompts) ? rawPrompts.flatMap((v: unknown) => {
      if (!record(v) || typeof v.id !== 'string') return [];
      const source = record(v.requires) ? v.requires : {};
      const requires: Need = {};
      for (const key of ['tools', 'vision', 'json'] as const) if (source[key] === true) requires[key] = true;
      if (Number.isSafeInteger(source.min_context) && source.min_context > 0) requires.min_context = source.min_context;
      return [{ id: v.id, api: v.api === 'anthropic' ? 'anthropic' : 'openai', stream: v.stream === true, weight: v.weight === 2 ? 2 : 1, requires }];
    }) : [];
    return { models, prompts };
  } catch {
    return { models: [], prompts: [] };
  }
}

export function makeCatalog(brief: Brief): Catalog {
  const models = brief.models.filter((m) => m.free).map((m) => ({ id: m.id, free: true as const, context_length: m.context_length }));
  return models.length ? { providers: [{ id: 'openrouter', base_url: 'https://openrouter.ai/api/v1', key_env: 'OPENROUTER_API_KEY', wire: 'openai', models }] } : { providers: [] };
}

export function candidates(models: Model[], need: Need): Model[] {
  return models.filter((m) => m.free &&
    (need.tools !== true || m.tools === true) &&
    (need.vision !== true || m.vision === true) &&
    (need.json !== true || m.json === true) &&
    (need.min_context === undefined || (m.context_length !== null && m.context_length >= need.min_context)));
}

export function needFromBody(body: Obj): Need {
  let vision = false;
  if (Array.isArray(body.messages)) for (const message of body.messages) {
    if (record(message) && Array.isArray(message.content)) for (const part of message.content) {
      if (record(part) && (part.type === 'image' || part.type === 'image_url')) vision = true;
    }
  }
  const format = record(body.response_format) ? body.response_format.type : undefined;
  return {
    ...(Array.isArray(body.tools) && body.tools.length ? { tools: true } : {}),
    ...(vision ? { vision: true } : {}),
    ...(format === 'json_object' || format === 'json_schema' ? { json: true } : {}),
  };
}
