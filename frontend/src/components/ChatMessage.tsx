import { Typography, Collapse, Alert, Space, Spin, Tooltip } from 'antd';
import {
  UserOutlined,
  RobotOutlined,
  DatabaseOutlined,
  ThunderboltOutlined,
  LoadingOutlined,
} from '@ant-design/icons';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ResultsTable from './ResultsTable';
import type { ChatEntry, QueryComplexity } from '../types';

const { Text, Paragraph } = Typography;

interface Props {
  entry: ChatEntry;
}

// ── complexity badge ─────────────────────────────────────────────────────────
const COMPLEXITY_CONFIG: Record<
  QueryComplexity,
  { color: string; bg: string; border: string; label: string; tooltip: string }
> = {
  deterministic: {
    color:   '#1677ff',
    bg:      '#e6f4ff',
    border:  '#91caff',
    label:   'Instant',
    tooltip: 'deterministic — hard-coded SQL builder, no LLM call needed (~0 ms)',
  },
  simple_llm: {
    color:   '#389e0d',
    bg:      '#f6ffed',
    border:  '#b7eb8f',
    label:   'Fast · gpt-4o-mini',
    tooltip: 'simple_llm — gpt-4o-mini + short direct-answer prompt (~2–4 s)',
  },
  complex_llm: {
    color:   '#d48806',
    bg:      '#fffbe6',
    border:  '#ffe58f',
    label:   'Complex · gpt-4o',
    tooltip: 'complex_llm — gpt-4o + full chain-of-thought prompt (~8–10 s)',
  },
};

function ComplexityBadge({ complexity }: { complexity: QueryComplexity | null | undefined }) {
  if (!complexity || !(complexity in COMPLEXITY_CONFIG)) return null;
  const cfg = COMPLEXITY_CONFIG[complexity];
  return (
    <Tooltip title={cfg.tooltip}>
      <span
        style={{
          display:       'inline-block',
          padding:       '1px 8px',
          borderRadius:  10,
          fontSize:      11,
          fontWeight:    600,
          color:         cfg.color,
          background:    cfg.bg,
          border:        `1px solid ${cfg.border}`,
          cursor:        'default',
          letterSpacing: 0.2,
        }}
      >
        {cfg.label}
      </span>
    </Tooltip>
  );
}

// ── pipeline rows — mirrors the Streamlit timing table exactly ───────────────
const PIPELINE_ROWS: { label: string; key: string }[] = [
  { label: 'setup  (dialect + tables + session + intent)', key: 'intent_detection'         },
  { label: 'schema  get_table_info',                       key: 'schema_loading'            },
  { label: 'stored-proc guidance',                         key: 'stored_procedure_guidance' },
  { label: 'cache lookup',                                 key: 'cache_lookup'              },
  { label: 'RAG  embed + retrieve',                        key: 'rag_retrieval'             },
  { label: 'LLM  prompt + SQL generation',                 key: 'sql_generation'            },
  { label: 'SQL  extract + validate + retry LLM',          key: 'sql_validation'            },
  { label: 'guardrails  regex rewrites',                   key: 'guardrails_applied'        },
  { label: 'dry-run  SET NOEXEC ON',                       key: 'dry_run_validation'        },
  { label: 'DB  execute_query_safe',                       key: 'db_execution'              },
  { label: 'format  _results_to_records',                  key: 'results_formatting'        },
];

function msBar(ms: number): string {
  return '█'.repeat(Math.min(30, Math.floor(ms / 200)));
}

// ── timing table (Streamlit-style) ───────────────────────────────────────────
function TimingTable({
  timing,
  fromCache,
}: {
  timing: Record<string, number>;
  fromCache: boolean;
}) {
  const totalMs  = Number(timing['total'] ?? 0);
  const rows     = PIPELINE_ROWS.map(({ label, key }) => ({
    label,
    key,
    ms: Number(timing[key] ?? 0),
  }));
  const rowsSum      = rows.reduce((s, r) => s + r.ms, 0);
  const unaccounted  = Math.round((totalMs - rowsSum) * 10) / 10;

  const COL_LABEL: React.CSSProperties = {
    flex: 1,
    fontFamily: 'monospace',
    fontSize: 12,
    color: '#555',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  };
  const COL_MS: React.CSSProperties = {
    width: 64,
    textAlign: 'right',
    fontFamily: 'monospace',
    fontSize: 12,
    color: '#222',
  };
  const COL_BAR: React.CSSProperties = {
    width: 160,
    fontFamily: 'monospace',
    fontSize: 10,
    color: '#4F46E5',
    letterSpacing: -1,
    paddingLeft: 8,
  };
  const ROW: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    padding: '2px 0',
    borderBottom: '1px solid #f0f0f0',
  };

  return (
    <div style={{ fontSize: 12 }}>
      {/* cache banner */}
      {fromCache && (
        <div
          style={{
            display:      'flex',
            alignItems:   'center',
            gap:          6,
            padding:      '4px 8px',
            marginBottom: 8,
            borderRadius: 6,
            background:   '#f6ffed',
            border:       '1px solid #b7eb8f',
            color:        '#389e0d',
            fontWeight:   600,
            fontSize:     12,
          }}
        >
          <ThunderboltOutlined />
          Served from cache — LLM + DB skipped
        </div>
      )}

      {/* header */}
      <div style={{ display: 'flex', gap: 4, padding: '2px 0', borderBottom: '2px solid #d9d9d9', fontWeight: 700, fontSize: 11, color: '#888' }}>
        <span style={{ flex: 1 }}>Step</span>
        <span style={{ width: 64, textAlign: 'right' }}>ms</span>
        <span style={{ width: 160, paddingLeft: 8 }}>bar</span>
      </div>

      {/* stage rows */}
      {rows.map(({ label, key, ms }) => {
        const isCacheHit = fromCache && key === 'cache_lookup';
        return (
          <div
            key={key}
            style={{
              ...ROW,
              ...(isCacheHit ? { background: '#f6ffed', borderRadius: 4 } : {}),
            }}
          >
            <span
              style={{
                ...COL_LABEL,
                color:      isCacheHit ? '#389e0d' : ms > 0 ? '#222' : '#bbb',
                fontWeight: isCacheHit ? 700 : 'normal',
              }}
            >
              {label}
              {isCacheHit ? ' ⚡' : ''}
            </span>
            <span
              style={{
                ...COL_MS,
                color:      isCacheHit ? '#389e0d' : ms > 2000 ? '#cf1322' : ms > 500 ? '#d48806' : '#222',
                fontWeight: ms > 2000 ? 700 : 'normal',
              }}
            >
              {ms.toFixed(0)}
            </span>
            <span style={{ ...COL_BAR, color: isCacheHit ? '#389e0d' : '#4F46E5' }}>
              {msBar(ms)}
            </span>
          </div>
        );
      })}

      {/* separator */}
      <div style={{ borderBottom: '2px solid #d9d9d9', margin: '4px 0' }} />

      {/* sum of rows */}
      <div style={{ ...ROW, borderBottom: 'none' }}>
        <span style={{ ...COL_LABEL, color: '#555', fontWeight: 600 }}>sum of rows</span>
        <span style={{ ...COL_MS, fontWeight: 600 }}>{rowsSum.toFixed(0)}</span>
        <span style={COL_BAR} />
      </div>

      {/* ⚠ unaccounted — only shown when gap > 1ms */}
      {Math.abs(unaccounted) > 1 && (
        <div style={{ ...ROW, background: '#fff7e6', borderRadius: 4, borderBottom: 'none' }}>
          <span style={{ ...COL_LABEL, color: '#d48806', fontWeight: 700 }}>⚠ unaccounted</span>
          <span style={{ ...COL_MS, color: '#d48806', fontWeight: 700 }}>{unaccounted.toFixed(0)}</span>
          <span style={{ ...COL_BAR, color: '#d48806', fontSize: 11 }}>← overhead / missing mark</span>
        </div>
      )}

      {/* TOTAL */}
      <div style={{ ...ROW, background: '#f5f5f5', borderRadius: 4, borderBottom: 'none', marginTop: 2 }}>
        <span style={{ ...COL_LABEL, color: '#111', fontWeight: 700 }}>TOTAL  (wall clock)</span>
        <span style={{ ...COL_MS, color: '#111', fontWeight: 700 }}>{totalMs.toFixed(0)}</span>
        <span style={{ ...COL_BAR, color: '#4F46E5', fontWeight: 700 }}>{msBar(totalMs)}</span>
      </div>
    </div>
  );
}

// ── main component ───────────────────────────────────────────────────────────
export default function ChatMessage({ entry }: Props) {
  const { question, response } = entry;
  const hasTiming = response.timing != null && Object.keys(response.timing).length > 0;

  return (
    <div style={{ marginBottom: 24 }}>
      {/* User message */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
        <div
          style={{
            maxWidth:     '70%',
            background:   '#4F46E5',
            color:        'white',
            padding:      '10px 16px',
            borderRadius: '16px 16px 4px 16px',
          }}
        >
          <Space size={8}>
            <UserOutlined />
            <span>{question}</span>
          </Space>
        </div>
      </div>

      {/* Assistant message */}
      <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
        <div
          style={{
            maxWidth:     '85%',
            background:   '#ffffff',
            border:       '1px solid #e8e8e8',
            padding:      '12px 16px',
            borderRadius: '16px 16px 16px 4px',
          }}
        >
          {/* header row: icon + name + badge */}
          <Space size={8} style={{ marginBottom: 8 }}>
            <RobotOutlined style={{ color: '#4F46E5' }} />
            <Text strong style={{ color: '#4F46E5' }}>
              Assistant
            </Text>
            {response.from_cache ? (
              <Tooltip title="Result served instantly from cache — no LLM or DB query needed">
                <span
                  style={{
                    display:       'inline-flex',
                    alignItems:    'center',
                    gap:           4,
                    padding:       '2px 10px',
                    borderRadius:  10,
                    fontSize:      11,
                    fontWeight:    700,
                    color:         '#389e0d',
                    background:    '#f6ffed',
                    border:        '1px solid #b7eb8f',
                    cursor:        'default',
                    letterSpacing: 0.2,
                  }}
                >
                  <ThunderboltOutlined style={{ fontSize: 11 }} />
                  Cached
                </span>
              </Tooltip>
            ) : (
              <ComplexityBadge complexity={response.query_complexity} />
            )}
          </Space>

          {/* NL answer */}
          {response.nl_answer && (
            <Paragraph style={{ marginBottom: 8 }}>{response.nl_answer}</Paragraph>
          )}
          {response.nl_pending && (
            <Paragraph type="secondary" style={{ marginBottom: 8 }}>
              <Spin indicator={<LoadingOutlined style={{ fontSize: 12 }} spin />} size="small" />{' '}
              Generating summary...
            </Paragraph>
          )}

          {/* Error */}
          {response.error && !response.nl_answer && (
            <Alert
              type="error"
              showIcon
              style={{ marginBottom: 8 }}
              description={response.error}
            />
          )}

          {/* SQL */}
          {response.sql && response.intent !== 'CONVERSATION' && (
            <Collapse
              size="small"
              style={{ marginBottom: 8 }}
              items={[
                {
                  key:   'sql',
                  label: (
                    <Space size={4}>
                      <DatabaseOutlined />
                      <span>View SQL</span>
                    </Space>
                  ),
                  children: (
                    <SyntaxHighlighter
                      language="sql"
                      style={oneLight}
                      customStyle={{ margin: 0, borderRadius: 6, fontSize: 13 }}
                    >
                      {response.sql}
                    </SyntaxHighlighter>
                  ),
                },
              ]}
            />
          )}

          {/* Timing — Streamlit-style table */}
          {hasTiming && (
            <Collapse
              size="small"
              style={{ marginBottom: 8 }}
              items={[
                {
                  key:   'timing',
                  label: (
                    <Space size={4}>
                      <ThunderboltOutlined />
                      <span>
                        ⏱ Stage Timing — TOTAL {(response.timing!['total'] ?? 0).toFixed(0)} ms
                      </span>
                    </Space>
                  ),
                  children: (
                    <TimingTable
                      timing={response.timing as Record<string, number>}
                      fromCache={response.from_cache}
                    />
                  ),
                },
              ]}
            />
          )}

          {/* Results table */}
          {response.results && response.results.length > 0 && (
            <>
              <ResultsTable data={response.results} />
              <Text type="secondary" style={{ fontSize: 12 }}>
                {response.row_count} row(s) returned
              </Text>
            </>
          )}

          {/* No results */}
          {response.results &&
            response.results.length === 0 &&
            !response.error &&
            response.intent !== 'CONVERSATION' && (
              <Alert type="info" showIcon description="No results returned" />
            )}
        </div>
      </div>
    </div>
  );
}
