import type { ModelData, SimResult, FeedbackLoop, AgentTemplate, AgentInstance } from './types';

const BASE = '/api';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!resp.ok) {
    throw new Error(`API ${options?.method || 'GET'} ${url}: ${resp.status}`);
  }
  return resp.json();
}

// ---- Model ----

export function fetchModel(): Promise<ModelData> {
  return request<ModelData>('/model');
}

export function updateModel(model: ModelData): Promise<ModelData> {
  return request<ModelData>('/model', {
    method: 'PUT',
    body: JSON.stringify(model),
  });
}

export function fetchNodes(): Promise<{ nodes: { name: string; kind: string; subsystem: string; operator?: string; inputs?: Record<string, string> }[] }> {
  return request('/model/nodes');
}

export function addStock(name: string, spec: Record<string, unknown>): Promise<void> {
  return request(`/model/stocks?name=${encodeURIComponent(name)}`, {
    method: 'POST',
    body: JSON.stringify(spec),
  });
}

export function deleteStock(name: string): Promise<void> {
  return request(`/model/stocks/${encodeURIComponent(name)}`, { method: 'DELETE' });
}

export function addAuxiliary(name: string, spec: Record<string, unknown>): Promise<void> {
  return request(`/model/auxiliaries?name=${encodeURIComponent(name)}`, {
    method: 'POST',
    body: JSON.stringify(spec),
  });
}

export function deleteAuxiliary(name: string): Promise<void> {
  return request(`/model/auxiliaries/${encodeURIComponent(name)}`, { method: 'DELETE' });
}

export function updateParameter(name: string, value: number): Promise<void> {
  return request(`/model/parameters/${encodeURIComponent(name)}?value=${value}`, { method: 'PUT' });
}

export function reloadModel(): Promise<void> {
  return request('/model/reload', { method: 'POST' });
}

// ---- Simulation ----

export function runSimulation(params: {
  mode: 'deterministic' | 'stochastic';
  n_steps?: number;
  n_ensemble?: number;
  seed?: number;
}): Promise<SimResult> {
  return request<SimResult>('/sim/run', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

// ---- Loops ----

export function fetchLoops(maxLen = 8): Promise<FeedbackLoop[]> {
  return request<{ loops: FeedbackLoop[] }>(`/loops?max_len=${maxLen}`).then(r => r.loops);
}

export function fetchLoopsForVariable(variable: string, maxLen = 8): Promise<FeedbackLoop[]> {
  return request<{ loops: FeedbackLoop[] }>(`/loops/${encodeURIComponent(variable)}?max_len=${maxLen}`).then(r => r.loops);
}

// ---- Agents ----

export function fetchAgentTemplates(): Promise<Record<string, AgentTemplate>> {
  return request<{ templates: Record<string, AgentTemplate> }>('/agents/templates').then(r => r.templates);
}

export function createAgentTemplate(name: string, template: AgentTemplate): Promise<void> {
  return request('/agents/templates', {
    method: 'POST',
    body: JSON.stringify({ name, template }),
  });
}

export function deleteAgentTemplate(name: string): Promise<void> {
  return request(`/agents/templates/${encodeURIComponent(name)}`, { method: 'DELETE' });
}

export function initializeAgents(seed?: number): Promise<{ count: number }> {
  return request('/agents/initialize', {
    method: 'POST',
    body: JSON.stringify({ seed: seed ?? null }),
  });
}

export function fetchAgentPopulation(): Promise<AgentInstance[]> {
  return request<{ agents: AgentInstance[] }>('/agents/population').then(r => r.agents);
}
