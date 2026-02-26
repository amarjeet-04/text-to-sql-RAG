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

// ── complexity badge ────────────────────────────────────────────────────────
const COMPLEXITY_CONFIG: Record<
  QueryComplexity,
  { color: string; bg: string; border: string; label: string; tooltip: string }
> = {
  deterministic: {
    color:  '#1677ff',
    bg:     '#e6f4ff',
    border: '#91caff',
    label:  'Instant',
    tooltip: 'deterministic — hard-coded SQL builder, no LLM call needed (~0 ms)',
  },
  simple_llm: {
    color:  '#389e0d',
    bg:     '#f6ffed',
    border: '#b7eb8f',
    label:  'Fast · gpt-4o-mini',
    tooltip: 'simple_llm — gpt-4o-mini + short direct-answer prompt (~2–4 s)',
  },
  complex_llm: {
    color:  '#d48806',
    bg:     '#fffbe6',
    border: '#ffe58f',
    label:  'Complex · gpt-4o',
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
          display:      'inline-block',
          padding:      '1px 8px',
          borderRadius: 10,
          fontSize:     11,
          fontWeight:   600,
          color:        cfg.color,
          background:   cfg.bg,
          border:       `1px solid ${cfg.border}`,
          cursor:       'default',
          letterSpacing: 0.2,
        }}
      >
        {cfg.label}
      </span>
    </Tooltip>
  );
}

const STAGE_ORDER = [
  'start',
  'intent_detection',
  'schema_loading',
  'stored_procedure_guidance',
  'cache_lookup',
  'rag_retrieval',
  'sql_generation',
  'sql_validation',
  'guardrails_applied',
  'db_execution',
  'results_formatting',
  'total',
] as const;

function stageLabel(stage: string): string {
  return stage.replace(/_/g, ' ');
}

export default function ChatMessage({ entry }: Props) {
  const { question, response } = entry;
  const stageTiming = response.timing ?? null;
  const knownStageSet = new Set<string>(STAGE_ORDER);
  const timingItems = stageTiming
    ? [
        ...STAGE_ORDER.map((stage) => ({
          stage,
          ms: Number(stageTiming[stage] ?? 0),
        })),
        ...Object.entries(stageTiming)
          .filter(([stage]) => !knownStageSet.has(stage))
          .map(([stage, ms]) => ({
            stage,
            ms: Number(ms ?? 0),
          })),
      ]
    : [];

  return (
    <div style={{ marginBottom: 24 }}>
      {/* User message */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
        <div
          style={{
            maxWidth: '70%',
            background: '#4F46E5',
            color: 'white',
            padding: '10px 16px',
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
            maxWidth: '85%',
            background: '#ffffff',
            border: '1px solid #e8e8e8',
            padding: '12px 16px',
            borderRadius: '16px 16px 16px 4px',
          }}
        >
          <Space size={8} style={{ marginBottom: 8 }}>
            <RobotOutlined style={{ color: '#4F46E5' }} />
            <Text strong style={{ color: '#4F46E5' }}>
              Assistant
            </Text>
            {response.from_cache ? (
              <Tooltip title="Result served instantly from cache — no LLM or DB query needed">
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 4,
                    padding: '2px 10px',
                    borderRadius: 10,
                    fontSize: 11,
                    fontWeight: 700,
                    color: '#389e0d',
                    background: '#f6ffed',
                    border: '1px solid #b7eb8f',
                    cursor: 'default',
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

          {/* Natural language answer */}
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
              message={response.error}
              type="error"
              showIcon
              style={{ marginBottom: 8 }}
            />
          )}

          {/* SQL query */}
          {response.sql && response.intent !== 'CONVERSATION' && (
            <Collapse
              size="small"
              style={{ marginBottom: 8 }}
              items={[
                {
                  key: 'sql',
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
                      customStyle={{
                        margin: 0,
                        borderRadius: 6,
                        fontSize: 13,
                      }}
                    >
                      {response.sql}
                    </SyntaxHighlighter>
                  ),
                },
              ]}
            />
          )}

          {timingItems.length > 0 && (
            <Collapse
              size="small"
              style={{ marginBottom: 8 }}
              items={[
                {
                  key: 'timing',
                  label: (
                    <Space size={4}>
                      <ThunderboltOutlined />
                      <span>Stage Timing (ms)</span>
                    </Space>
                  ),
                  children: (
                    <div style={{ fontSize: 12 }}>
                      {response.from_cache && (
                        <div
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 6,
                            padding: '4px 8px',
                            marginBottom: 8,
                            borderRadius: 6,
                            background: '#f6ffed',
                            border: '1px solid #b7eb8f',
                            color: '#389e0d',
                            fontWeight: 600,
                          }}
                        >
                          <ThunderboltOutlined />
                          Served from cache — LLM + DB skipped
                        </div>
                      )}
                      {timingItems.map((item) => (
                        <div
                          key={item.stage}
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            gap: 12,
                            padding: '2px 0',
                            ...(response.from_cache && item.stage === 'cache_lookup'
                              ? { color: '#389e0d', fontWeight: 600 }
                              : {}),
                          }}
                        >
                          <Text
                            type={response.from_cache && item.stage === 'cache_lookup' ? undefined : 'secondary'}
                            style={response.from_cache && item.stage === 'cache_lookup' ? { color: '#389e0d', fontWeight: 600 } : {}}
                          >
                            {stageLabel(item.stage)}
                            {response.from_cache && item.stage === 'cache_lookup' ? ' ⚡' : ''}
                          </Text>
                          <Text
                            code
                            style={response.from_cache && item.stage === 'cache_lookup' ? { color: '#389e0d' } : {}}
                          >
                            {Number.isFinite(item.ms) ? item.ms.toFixed(2) : '0.00'}
                          </Text>
                        </div>
                      ))}
                    </div>
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
              <Alert message="No results returned" type="info" showIcon />
            )}
        </div>
      </div>
    </div>
  );
}
