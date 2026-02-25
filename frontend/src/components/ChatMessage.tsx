import { Typography, Collapse, Alert, Tag, Space, Spin } from 'antd';
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
import type { ChatEntry } from '../types';

const { Text, Paragraph } = Typography;

interface Props {
  entry: ChatEntry;
}

export default function ChatMessage({ entry }: Props) {
  const { question, response } = entry;
  const stageTiming = response.timing ?? null;
  const timingItems = stageTiming
    ? Object.entries(stageTiming)
        .filter(([stage]) => stage !== 'start')
        .map(([stage, ms]) => ({ stage, ms }))
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
            {response.from_cache && (
              <Tag icon={<ThunderboltOutlined />} color="green">
                Cached
              </Tag>
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
                      {timingItems.map((item) => (
                        <div
                          key={item.stage}
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            gap: 12,
                            padding: '2px 0',
                          }}
                        >
                          <Text type="secondary">{item.stage}</Text>
                          <Text code>{item.ms.toFixed(2)}</Text>
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
