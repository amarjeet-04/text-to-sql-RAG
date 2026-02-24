import { useState, useEffect } from 'react';
import {
  Layout,
  Typography,
  Button,
  Table,
  Card,
  Statistic,
  Row,
  Col,
  Select,
  Space,
  Tag,
  Upload,
  message,
  Spin,
  Collapse,
  Alert,
  Divider,
  Form,
  Input,
} from 'antd';
import {
  PlayCircleOutlined,
  UploadOutlined,
  FileTextOutlined,
  ExperimentOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ArrowLeftOutlined,
  PlusOutlined,
  DeleteOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { User } from '../types';
import {
  listEvalFiles,
  loadEvalFile,
  getSampleCases,
  uploadEvalFile,
  runEvaluation,
} from '../api/client';
import type { EvalCase, EvalResultItem, EvalSummary } from '../api/client';

const { Header, Content } = Layout;
const { Title, Text } = Typography;

export default function EvaluationPage() {
  const navigate = useNavigate();
  const [user] = useState<User>(() => {
    const stored = localStorage.getItem('user');
    return stored ? JSON.parse(stored) : null;
  });

  const [cases, setCases] = useState<EvalCase[]>([]);
  const [availableFiles, setAvailableFiles] = useState<Array<{ name: string; size_kb: number }>>([]);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<EvalResultItem[] | null>(null);
  const [summary, setSummary] = useState<EvalSummary | null>(null);
  const [addForm] = Form.useForm();

  useEffect(() => {
    if (!user || user.role !== 'Admin') {
      message.error('Admin access required');
      navigate('/');
      return;
    }
    // Load available files
    listEvalFiles()
      .then((data) => setAvailableFiles(data.files))
      .catch(() => {});
  }, [user, navigate]);

  const handleLoadFile = async (filename: string) => {
    setLoading(true);
    try {
      const data = await loadEvalFile(filename);
      setCases(data.cases);
      setResults(null);
      setSummary(null);
      message.success(`Loaded ${data.count} cases from ${filename}`);
    } catch {
      message.error('Failed to load file');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async () => {
    setLoading(true);
    try {
      const data = await getSampleCases();
      setCases(data.cases);
      setResults(null);
      setSummary(null);
      message.success(`Loaded ${data.count} sample cases`);
    } catch {
      message.error('Failed to load sample cases');
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (file: File) => {
    try {
      const data = await uploadEvalFile(file);
      setCases(data.cases);
      setResults(null);
      setSummary(null);
      message.success(`Uploaded ${data.count} cases`);
    } catch {
      message.error('Failed to upload file');
    }
    return false; // prevent default upload
  };

  const handleAddCase = () => {
    addForm.validateFields().then((values) => {
      setCases((prev) => [
        ...prev,
        {
          question: values.question,
          ground_truth_sql: values.ground_truth_sql,
          category: values.category || 'general',
          difficulty: values.difficulty || 'medium',
          description: '',
        },
      ]);
      addForm.resetFields();
      message.success('Case added');
    });
  };

  const handleRunEvaluation = async () => {
    if (cases.length === 0) {
      message.warning('No test cases loaded');
      return;
    }
    setRunning(true);
    try {
      const data = await runEvaluation(cases);
      setResults(data.results);
      setSummary(data.summary);
      message.success(`Evaluation complete: ${data.results.length} cases`);
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      message.error(axiosErr.response?.data?.detail || 'Evaluation failed');
    } finally {
      setRunning(false);
    }
  };

  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

  const casesColumns = [
    {
      title: 'Question',
      dataIndex: 'question',
      key: 'question',
      ellipsis: true,
      width: '50%',
    },
    { title: 'Category', dataIndex: 'category', key: 'category', width: 120 },
    {
      title: 'Difficulty',
      dataIndex: 'difficulty',
      key: 'difficulty',
      width: 100,
      render: (d: string) => (
        <Tag color={d === 'easy' ? 'green' : d === 'medium' ? 'orange' : 'red'}>{d}</Tag>
      ),
    },
  ];

  const resultsColumns = [
    {
      title: 'Question',
      dataIndex: 'question',
      key: 'question',
      ellipsis: true,
      width: '35%',
    },
    { title: 'Category', dataIndex: 'category', key: 'category', width: 100 },
    {
      title: 'Exec',
      dataIndex: 'execution_success',
      key: 'exec',
      width: 60,
      render: (v: boolean) =>
        v ? <CheckCircleOutlined style={{ color: '#52c41a' }} /> : <CloseCircleOutlined style={{ color: '#f5222d' }} />,
    },
    {
      title: 'Match',
      dataIndex: 'result_match',
      key: 'match',
      width: 60,
      render: (v: boolean) =>
        v ? <CheckCircleOutlined style={{ color: '#52c41a' }} /> : <CloseCircleOutlined style={{ color: '#f5222d' }} />,
    },
    {
      title: 'Score',
      dataIndex: 'score',
      key: 'score',
      width: 80,
      render: (v: number) => <Tag color={v >= 0.8 ? 'green' : v >= 0.5 ? 'orange' : 'red'}>{pct(v)}</Tag>,
    },
    {
      title: 'RAGAS',
      dataIndex: 'ragas_score',
      key: 'ragas',
      width: 80,
      render: (v: number | null) =>
        v !== null ? <Tag color={v >= 0.8 ? 'green' : v >= 0.5 ? 'orange' : 'red'}>{pct(v)}</Tag> : '-',
    },
    {
      title: 'Time',
      dataIndex: 'execution_time_ms',
      key: 'time',
      width: 80,
      render: (v: number) => `${v.toFixed(0)}ms`,
    },
  ];

  if (!user || user.role !== 'Admin') return null;

  return (
    <Layout style={{ minHeight: '100vh', background: '#f5f5f5' }}>
      <Header
        style={{
          background: '#fff',
          padding: '0 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid #f0f0f0',
          height: 56,
        }}
      >
        <Space>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/')} type="text" />
          <ExperimentOutlined style={{ fontSize: 20, color: '#4F46E5' }} />
          <Title level={4} style={{ margin: 0 }}>
            SQL Generation Evaluation
          </Title>
        </Space>
        <Text type="secondary">Admin Only</Text>
      </Header>

      <Content style={{ padding: 24, maxWidth: 1400, margin: '0 auto', width: '100%' }}>
        {/* Load Cases Section */}
        <Card title="Test Cases" style={{ marginBottom: 24 }}>
          <Space wrap style={{ marginBottom: 16 }}>
            {availableFiles.map((f) => (
              <Button
                key={f.name}
                icon={<FileTextOutlined />}
                onClick={() => handleLoadFile(f.name)}
                loading={loading}
              >
                {f.name} ({f.size_kb}KB)
              </Button>
            ))}
            <Button icon={<ExperimentOutlined />} onClick={handleLoadSample} loading={loading}>
              Sample Cases
            </Button>
            <Upload
              accept=".json"
              showUploadList={false}
              beforeUpload={(file) => handleUpload(file as unknown as File)}
            >
              <Button icon={<UploadOutlined />}>Upload JSON</Button>
            </Upload>
          </Space>

          {/* Add custom case */}
          <Collapse
            size="small"
            style={{ marginBottom: 16 }}
            items={[
              {
                key: 'add',
                label: (
                  <Space>
                    <PlusOutlined />
                    Add Custom Case
                  </Space>
                ),
                children: (
                  <Form form={addForm} layout="vertical" size="small">
                    <Form.Item name="question" label="Question" rules={[{ required: true }]}>
                      <Input placeholder="Natural language question" />
                    </Form.Item>
                    <Form.Item name="ground_truth_sql" label="Ground Truth SQL" rules={[{ required: true }]}>
                      <Input.TextArea rows={3} placeholder="Expected SQL query" />
                    </Form.Item>
                    <Row gutter={16}>
                      <Col span={12}>
                        <Form.Item name="category" label="Category" initialValue="general">
                          <Select
                            options={[
                              { value: 'aggregation', label: 'Aggregation' },
                              { value: 'filter', label: 'Filter' },
                              { value: 'date', label: 'Date' },
                              { value: 'join', label: 'Join' },
                              { value: 'text_search', label: 'Text Search' },
                              { value: 'general', label: 'General' },
                            ]}
                          />
                        </Form.Item>
                      </Col>
                      <Col span={12}>
                        <Form.Item name="difficulty" label="Difficulty" initialValue="medium">
                          <Select
                            options={[
                              { value: 'easy', label: 'Easy' },
                              { value: 'medium', label: 'Medium' },
                              { value: 'hard', label: 'Hard' },
                            ]}
                          />
                        </Form.Item>
                      </Col>
                    </Row>
                    <Button type="primary" icon={<PlusOutlined />} onClick={handleAddCase}>
                      Add Case
                    </Button>
                  </Form>
                ),
              },
            ]}
          />

          {cases.length > 0 && (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <Text strong>{cases.length} test cases loaded</Text>
                <Space>
                  <Button
                    icon={<DeleteOutlined />}
                    size="small"
                    onClick={() => {
                      setCases([]);
                      setResults(null);
                      setSummary(null);
                    }}
                  >
                    Clear
                  </Button>
                  <Button
                    type="primary"
                    icon={<PlayCircleOutlined />}
                    onClick={handleRunEvaluation}
                    loading={running}
                  >
                    Run Evaluation
                  </Button>
                </Space>
              </div>
              <Table
                columns={casesColumns}
                dataSource={cases.map((c, i) => ({ ...c, key: i }))}
                size="small"
                pagination={cases.length > 10 ? { pageSize: 10 } : false}
              />
            </>
          )}
        </Card>

        {/* Running indicator */}
        {running && (
          <Card style={{ marginBottom: 24, textAlign: 'center' }}>
            <Spin size="large" />
            <div style={{ marginTop: 16 }}>
              <Text>Running evaluation on {cases.length} cases...</Text>
            </div>
          </Card>
        )}

        {/* Results Section */}
        {summary && (
          <>
            {/* Key Metrics */}
            <Row gutter={16} style={{ marginBottom: 24 }}>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Overall Score"
                    value={summary.average_score * 100}
                    precision={1}
                    suffix="%"
                    valueStyle={{ color: summary.average_score >= 0.7 ? '#3f8600' : '#cf1322' }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Execution Success"
                    value={summary.execution_success_rate * 100}
                    precision={1}
                    suffix="%"
                    valueStyle={{ color: summary.execution_success_rate >= 0.8 ? '#3f8600' : '#cf1322' }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Result Match"
                    value={summary.result_match_rate * 100}
                    precision={1}
                    suffix="%"
                    valueStyle={{ color: summary.result_match_rate >= 0.7 ? '#3f8600' : '#cf1322' }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Avg Execution Time"
                    value={summary.avg_execution_time_ms}
                    precision={0}
                    suffix="ms"
                  />
                </Card>
              </Col>
            </Row>

            {/* RAGAS Metrics */}
            {summary.ragas_metrics && (
              <Card title="RAGAS Metrics" style={{ marginBottom: 24 }} size="small">
                <Row gutter={16}>
                  <Col span={4}>
                    <Statistic title="RAGAS Score" value={(summary.ragas_metrics.avg_ragas_score ?? 0) * 100} precision={1} suffix="%" />
                  </Col>
                  <Col span={4}>
                    <Statistic title="Faithfulness" value={(summary.ragas_metrics.avg_faithfulness ?? 0) * 100} precision={1} suffix="%" />
                  </Col>
                  <Col span={4}>
                    <Statistic title="Answer Correct" value={(summary.ragas_metrics.avg_answer_correctness ?? 0) * 100} precision={1} suffix="%" />
                  </Col>
                  <Col span={4}>
                    <Statistic title="SQL Validity" value={(summary.ragas_metrics.avg_sql_validity ?? 0) * 100} precision={1} suffix="%" />
                  </Col>
                  <Col span={4}>
                    <Statistic title="Context Relevance" value={(summary.ragas_metrics.avg_context_relevance ?? 0) * 100} precision={1} suffix="%" />
                  </Col>
                  <Col span={4}>
                    <Statistic title="Result Similarity" value={(summary.ragas_metrics.avg_result_similarity ?? 0) * 100} precision={1} suffix="%" />
                  </Col>
                </Row>
              </Card>
            )}

            {/* By Category & Difficulty */}
            <Row gutter={16} style={{ marginBottom: 24 }}>
              <Col span={12}>
                <Card title="By Category" size="small">
                  <Table
                    size="small"
                    pagination={false}
                    dataSource={Object.entries(summary.by_category).map(([cat, data]) => ({
                      key: cat,
                      category: cat,
                      count: data.count,
                      success: pct(data.execution_success_rate),
                      match: pct(data.result_match_rate),
                      score: pct(data.average_score),
                    }))}
                    columns={[
                      { title: 'Category', dataIndex: 'category', key: 'category' },
                      { title: 'Count', dataIndex: 'count', key: 'count' },
                      { title: 'Success', dataIndex: 'success', key: 'success' },
                      { title: 'Match', dataIndex: 'match', key: 'match' },
                      { title: 'Score', dataIndex: 'score', key: 'score' },
                    ]}
                  />
                </Card>
              </Col>
              <Col span={12}>
                <Card title="By Difficulty" size="small">
                  <Table
                    size="small"
                    pagination={false}
                    dataSource={Object.entries(summary.by_difficulty).map(([diff, data]) => ({
                      key: diff,
                      difficulty: diff,
                      count: data.count,
                      success: pct(data.execution_success_rate),
                      match: pct(data.result_match_rate),
                      score: pct(data.average_score),
                    }))}
                    columns={[
                      { title: 'Difficulty', dataIndex: 'difficulty', key: 'difficulty' },
                      { title: 'Count', dataIndex: 'count', key: 'count' },
                      { title: 'Success', dataIndex: 'success', key: 'success' },
                      { title: 'Match', dataIndex: 'match', key: 'match' },
                      { title: 'Score', dataIndex: 'score', key: 'score' },
                    ]}
                  />
                </Card>
              </Col>
            </Row>

            {/* Detailed Results */}
            {results && (
              <Card title="Detailed Results" style={{ marginBottom: 24 }}>
                <Table
                  columns={resultsColumns}
                  dataSource={results.map((r, i) => ({ ...r, key: i }))}
                  size="small"
                  pagination={results.length > 20 ? { pageSize: 20 } : false}
                  expandable={{
                    expandedRowRender: (record: EvalResultItem) => (
                      <div style={{ padding: '8px 0' }}>
                        <Row gutter={16}>
                          <Col span={12}>
                            <Text strong>Ground Truth SQL</Text>
                            <SyntaxHighlighter language="sql" style={oneLight} customStyle={{ fontSize: 12 }}>
                              {record.ground_truth_sql}
                            </SyntaxHighlighter>
                          </Col>
                          <Col span={12}>
                            <Text strong>Generated SQL</Text>
                            <SyntaxHighlighter language="sql" style={oneLight} customStyle={{ fontSize: 12 }}>
                              {record.generated_sql}
                            </SyntaxHighlighter>
                          </Col>
                        </Row>
                        {record.error_message && (
                          <Alert message={record.error_message} type="error" showIcon style={{ marginTop: 8 }} />
                        )}
                        {record.ragas_score !== null && (
                          <>
                            <Divider style={{ margin: '12px 0' }} />
                            <Row gutter={8}>
                              <Col><Tag>RAGAS: {pct(record.ragas_score!)}</Tag></Col>
                              <Col><Tag>Faithfulness: {pct(record.ragas_faithfulness!)}</Tag></Col>
                              <Col><Tag>Answer: {pct(record.ragas_answer_correctness!)}</Tag></Col>
                              <Col><Tag>SQL Valid: {pct(record.ragas_sql_validity!)}</Tag></Col>
                              <Col><Tag>Context: {pct(record.ragas_context_relevance!)}</Tag></Col>
                              <Col><Tag>Similarity: {pct(record.ragas_result_similarity!)}</Tag></Col>
                            </Row>
                          </>
                        )}
                      </div>
                    ),
                  }}
                />
              </Card>
            )}

            {/* Failed Cases */}
            {summary.failed_cases.length > 0 && (
              <Card title={`Failed Cases (${summary.failed_cases.length})`} size="small">
                <Collapse
                  size="small"
                  items={summary.failed_cases.map((fc, i) => ({
                    key: i,
                    label: fc.question,
                    children: (
                      <div>
                        {fc.generated_sql && (
                          <SyntaxHighlighter language="sql" style={oneLight} customStyle={{ fontSize: 12 }}>
                            {fc.generated_sql}
                          </SyntaxHighlighter>
                        )}
                        {fc.error && <Alert message={fc.error} type="error" showIcon />}
                      </div>
                    ),
                  }))}
                />
              </Card>
            )}
          </>
        )}
      </Content>
    </Layout>
  );
}
