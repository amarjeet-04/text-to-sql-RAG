export interface User {
  username: string;
  role: string;
  name: string;
}

export interface LoginResponse {
  token: string;
  user: User;
}

export interface ConnectRequest {
  host: string;
  port: string;
  username: string;
  password: string;
  database: string;
  llm_provider: string;
  api_key: string;
  model: string;
  temperature: number;
  query_timeout: number;
  view_support: boolean;
}

export interface ConnectResponse {
  success: boolean;
  message: string;
  tables_count: number;
  views_count: number;
}

export interface QueryResponse {
  intent: string | null;
  nl_answer: string | null;
  sql: string | null;
  results: Record<string, unknown>[] | null;
  row_count: number;
  from_cache: boolean;
  error: string | null;
  nl_pending?: boolean;
}

export interface ChatEntry {
  question: string;
  response: QueryResponse;
  timestamp: string;
}
