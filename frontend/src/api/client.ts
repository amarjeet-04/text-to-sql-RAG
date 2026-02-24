import axios from 'axios';
import type {
  LoginResponse,
  ConnectRequest,
  ConnectResponse,
  QueryResponse,
  NLStatusResponse,
} from '../types';

const api = axios.create({
  baseURL: '/api',
});

// Attach auth token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Redirect to login on 401
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && window.location.pathname !== '/login') {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export async function login(username: string, password: string): Promise<LoginResponse> {
  const { data } = await api.post<LoginResponse>('/auth/login', { username, password });
  return data;
}

export async function connectDb(req: ConnectRequest): Promise<ConnectResponse> {
  const { data } = await api.post<ConnectResponse>('/db/connect', req, {
    // DB handshake + schema checks on remote servers may take longer than typical API calls.
    timeout: 60000,
  });
  return data;
}

export async function getDbStatus(): Promise<{ connected: boolean; tables_count: number; views_count: number }> {
  const { data } = await api.get('/db/status');
  return data;
}

export async function sendQuery(question: string): Promise<QueryResponse> {
  const { data } = await api.post<QueryResponse>('/chat/query', { question });
  return data;
}

export async function fetchNLResponse(
  question: string,
  results: Record<string, unknown>[],
  requestId?: string,
): Promise<{ nl_answer: string }> {
  const { data } = await api.post<{ nl_answer: string }>('/chat/nl-response', {
    question,
    results,
    request_id: requestId,
  });
  return data;
}

export async function pollNLResponse(requestId: string): Promise<NLStatusResponse> {
  const { data } = await api.get<NLStatusResponse>('/chat/nl-response', {
    params: { request_id: requestId },
  });
  return data;
}

interface NLStreamHandlers {
  onToken?: (token: string) => void;
  onDone?: (answer: string) => void;
  onError?: (error: string) => void;
}

export async function streamNLResponse(requestId: string, handlers: NLStreamHandlers): Promise<void> {
  const token = localStorage.getItem('token');
  const response = await fetch(`/api/chat/nl-stream?request_id=${encodeURIComponent(requestId)}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    method: 'GET',
  });
  if (!response.ok || !response.body) {
    throw new Error(`NL stream failed with status ${response.status}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  const handleEvent = (eventBlock: string) => {
    const lines = eventBlock.split('\n');
    let eventName = 'message';
    const dataLines: string[] = [];
    for (const line of lines) {
      if (line.startsWith('event:')) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trim());
      }
    }
    if (dataLines.length === 0) {
      return;
    }
    const rawData = dataLines.join('\n');
    let payload: Record<string, unknown> = {};
    try {
      payload = JSON.parse(rawData) as Record<string, unknown>;
    } catch {
      payload = {};
    }
    if (eventName === 'token') {
      const tokenPiece = String(payload.token ?? '');
      if (tokenPiece) handlers.onToken?.(tokenPiece);
      return;
    }
    if (eventName === 'done') {
      handlers.onDone?.(String(payload.nl_answer ?? ''));
      return;
    }
    if (eventName === 'error') {
      const err = String(payload.error ?? 'nl_stream_error');
      handlers.onError?.(err);
      throw new Error(err);
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let splitIndex = buffer.indexOf('\n\n');
    while (splitIndex >= 0) {
      const eventBlock = buffer.slice(0, splitIndex).trim();
      buffer = buffer.slice(splitIndex + 2);
      if (eventBlock) {
        handleEvent(eventBlock);
      }
      splitIndex = buffer.indexOf('\n\n');
    }
  }
}

export async function clearChat(): Promise<void> {
  await api.post('/chat/clear');
}

export async function clearCache(): Promise<void> {
  await api.post('/chat/clear-cache');
}

// --- Evaluation ---

export interface EvalCase {
  question: string;
  ground_truth_sql: string;
  category: string;
  difficulty: string;
  description: string;
}

export interface EvalResultItem {
  question: string;
  category: string;
  difficulty: string;
  ground_truth_sql: string;
  generated_sql: string;
  execution_success: boolean;
  result_match: boolean;
  column_match: boolean;
  row_count_match: boolean;
  score: number;
  ragas_score: number | null;
  ragas_faithfulness: number | null;
  ragas_answer_correctness: number | null;
  ragas_sql_validity: number | null;
  ragas_context_relevance: number | null;
  ragas_execution_accuracy: number | null;
  ragas_result_similarity: number | null;
  execution_time_ms: number;
  error_message: string;
}

export interface EvalSummary {
  total_cases: number;
  average_score: number;
  execution_success_rate: number;
  result_match_rate: number;
  avg_execution_time_ms: number;
  by_category: Record<string, { count: number; execution_success_rate: number; result_match_rate: number; average_score: number }>;
  by_difficulty: Record<string, { count: number; execution_success_rate: number; result_match_rate: number; average_score: number }>;
  ragas_metrics: Record<string, number> | null;
  failed_cases: Array<{ question: string; generated_sql?: string; error?: string }>;
}

export interface RunEvalResponse {
  results: EvalResultItem[];
  summary: EvalSummary;
}

export async function listEvalFiles(): Promise<{ files: Array<{ name: string; size_kb: number }> }> {
  const { data } = await api.get('/evaluation/files');
  return data;
}

export async function loadEvalFile(filename: string): Promise<{ cases: EvalCase[]; count: number }> {
  const { data } = await api.get(`/evaluation/load/${filename}`);
  return data;
}

export async function getSampleCases(): Promise<{ cases: EvalCase[]; count: number }> {
  const { data } = await api.get('/evaluation/sample-cases');
  return data;
}

export async function uploadEvalFile(file: File): Promise<{ cases: EvalCase[]; count: number }> {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post('/evaluation/upload', formData);
  return data;
}

export async function runEvaluation(cases: EvalCase[]): Promise<RunEvalResponse> {
  const { data } = await api.post<RunEvalResponse>('/evaluation/run', { cases });
  return data;
}

export default api;
