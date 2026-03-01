import { useState, useRef, useEffect } from 'react';
import {
  Layout,
  Input,
  Button,
  Typography,
  Tag,
  Space,
  Spin,
  Modal,
  message,
} from 'antd';
import {
  SendOutlined,
  LogoutOutlined,
  UserOutlined,
  DeleteOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  RobotOutlined,
  SettingOutlined,
  TeamOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import ChatMessage from '../components/ChatMessage';
import Sidebar from '../components/Sidebar';
import {
  sendQuery,
  clearChat,
  getDbStatus,
  connectDb,
  fetchNLResponse,
  pollNLResponse,
  streamNLResponse,
} from '../api/client';
import logo from '../assets/globe_logo.png';
import type { User, ChatEntry } from '../types';

const { Header, Content } = Layout;
const { Title, Text, Paragraph } = Typography;

// Default DB + LLM settings used for auto-connect
const DEFAULT_DB_SETTINGS = {
  host: '95.168.168.71',
  port: '1988',
  username: 'withinearth_reader',
  password: 'pass@readerWE#2026',
  database: 'mis_report_data',
  llm_provider: 'openai',
  api_key: '',
  model: 'gpt-4o-mini',
  temperature: 0,
  query_timeout: 60,
  view_support: true,
};

export default function ChatPage() {
  const navigate = useNavigate();
  const [user] = useState<User>(() => {
    const stored = localStorage.getItem('user');
    return stored ? JSON.parse(stored) : null;
  });

  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatEntry[]>([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!user) {
      navigate('/login');
    }
  }, [user, navigate]);

  // Auto-connect: check status first, if not connected then connect automatically
  useEffect(() => {
    if (!user) return;
    let cancelled = false;

    const autoConnect = async () => {
      try {
        const status = await getDbStatus();
        if (status.connected) {
          if (!cancelled) setConnected(true);
          return;
        }
      } catch {
        // Session expired or backend down — try connecting anyway
      }

      // Not connected yet — auto-connect with saved or default settings
      if (cancelled) return;
      setConnecting(true);

      try {
        const saved = localStorage.getItem('dbSettings');
        const settings = saved ? JSON.parse(saved) : DEFAULT_DB_SETTINGS;

        const result = await connectDb({
          host: settings.host,
          port: settings.port,
          username: settings.db_username || settings.username,
          password: settings.db_password || settings.password,
          database: settings.database,
          llm_provider: settings.llm_provider,
          api_key: settings.api_key,
          model: settings.model,
          temperature: settings.temperature,
          query_timeout: settings.query_timeout,
          view_support: settings.view_support,
        });

        if (!cancelled) {
          if (result.success) {
            setConnected(true);
          } else {
            message.error('Auto-connect failed: ' + result.message);
          }
        }
      } catch {
        if (!cancelled) {
          message.error('Failed to connect to database. Open settings to configure.');
        }
      } finally {
        if (!cancelled) setConnecting(false);
      }
    };

    autoConnect();
    return () => { cancelled = true; };
  }, [user]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const handleSend = async () => {
    const q = question.trim();
    if (!q) return;

    if (!connected) {
      message.warning('Database is still connecting. Please wait.');
      return;
    }

    setQuestion('');
    setLoading(true);

    try {
      const response = await sendQuery(q);
      const entry: ChatEntry = {
        question: q,
        response,
        timestamp: new Date().toISOString(),
      };
      setChatHistory((prev) => [...prev, entry]);

      const requestId = response.request_id || '';
      if (response.nl_pending && response.results && response.results.length > 0 && requestId) {
        window.setTimeout(() => {
          pollNLResponse(requestId)
            .then((statusResp) => {
              if (statusResp.status === 'ready' && statusResp.nl_answer) {
                setChatHistory((prev) =>
                  prev.map((e) =>
                    e.response.request_id === requestId
                      ? { ...e, response: { ...e.response, nl_answer: statusResp.nl_answer, nl_pending: false } }
                      : e,
                  ),
                );
                return;
              }
              if (statusResp.status === 'pending') {
                streamNLResponse(requestId, {
                  onToken: (token) => {
                    setChatHistory((prev) =>
                      prev.map((e) =>
                        e.response.request_id === requestId
                          ? {
                              ...e,
                              response: {
                                ...e.response,
                                nl_answer: `${e.response.nl_answer || ''}${token}`,
                                nl_pending: true,
                              },
                            }
                          : e,
                      ),
                    );
                  },
                  onDone: (answer) => {
                    setChatHistory((prev) =>
                      prev.map((e) =>
                        e.response.request_id === requestId
                          ? { ...e, response: { ...e.response, nl_answer: answer, nl_pending: false } }
                          : e,
                      ),
                    );
                  },
                }).catch(() => {
                  fetchNLResponse(q, response.results || [], requestId)
                    .then(({ nl_answer }) => {
                      setChatHistory((prev) =>
                        prev.map((e) =>
                          e.response.request_id === requestId
                            ? { ...e, response: { ...e.response, nl_answer, nl_pending: false } }
                            : e,
                        ),
                      );
                    })
                    .catch(() => {});
                });
              }
            })
            .catch(() => {
              fetchNLResponse(q, response.results || [], requestId)
                .then(({ nl_answer }) => {
                  setChatHistory((prev) =>
                    prev.map((e) =>
                      e.response.request_id === requestId
                        ? { ...e, response: { ...e.response, nl_answer, nl_pending: false } }
                        : e,
                    ),
                  );
                })
                .catch(() => {});
            });
        }, 250);
      }
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      message.error(axiosErr.response?.data?.detail || 'Failed to process query');
    } finally {
      setLoading(false);
    }
  };

  const handleClearChat = async () => {
    try {
      await clearChat();
      setChatHistory([]);
    } catch {
      message.error('Failed to clear chat');
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/login');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (!user) return null;

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Layout>
        {/* Header */}
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
            <img
              src={logo}
              alt="Within Earth"
              style={{
                height: 36,
                width: 36,
                borderRadius: '50%',
                objectFit: 'cover',
                objectPosition: 'center center',
                verticalAlign: 'middle',
                position: 'relative',
                top: 2,
              }}
            />
            <Title level={4} style={{ margin: 0 }}>
              Within Earth Chatbot
            </Title>
          </Space>

          <Space size={16}>
            {user.role === 'Admin' && (
              <Button
                icon={<SettingOutlined />}
                onClick={() => setSettingsOpen(true)}
                size="small"
              >
                Settings
              </Button>
            )}
            {user.role === 'Admin' && (
              <Button
                icon={<TeamOutlined />}
                onClick={() => navigate('/admin')}
                size="small"
              >
                Users
              </Button>
            )}
            {user.role === 'Admin' && (
              <Button
                icon={<ExperimentOutlined />}
                onClick={() => navigate('/evaluation')}
                size="small"
              >
                Evaluation
              </Button>
            )}
            <Space size={4}>
              <UserOutlined />
              <Text strong>{user.name}</Text>
              <Tag color={user.role === 'Admin' ? 'gold' : 'blue'}>{user.role}</Tag>
            </Space>
            <Button icon={<LogoutOutlined />} onClick={handleLogout} size="small">
              Logout
            </Button>
          </Space>
        </Header>

        {/* Content */}
        <Content
          style={{
            display: 'flex',
            flexDirection: 'column',
            height: 'calc(100vh - 56px)',
            background: '#fafafa',
          }}
        >
          {/* Chat header */}
          <div
            style={{
              padding: '12px 24px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              borderBottom: '1px solid #f0f0f0',
              background: '#fff',
            }}
          >
            <Text strong>Conversation</Text>
            {chatHistory.length > 0 && (
              <Button
                icon={<DeleteOutlined />}
                size="small"
                onClick={handleClearChat}
              >
                Clear Chat
              </Button>
            )}
          </div>

          {/* Chat messages */}
          <div
            style={{
              flex: 1,
              overflow: 'auto',
              padding: '24px',
            }}
          >
            {chatHistory.length === 0 && !loading && connecting && (
              <div
                style={{
                  textAlign: 'center',
                  marginTop: '20vh',
                  color: '#aaa',
                }}
              >
                <Spin size="large" />
                <div style={{ marginTop: 16 }}>
                  <Text type="secondary" style={{ fontSize: 16 }}>
                    Connecting to database...
                  </Text>
                </div>
              </div>
            )}

            {chatHistory.length === 0 && !loading && !connecting && !connected && (
              <div
                style={{
                  textAlign: 'center',
                  marginTop: '20vh',
                  color: '#aaa',
                }}
              >
                <DatabaseOutlined style={{ fontSize: 48, marginBottom: 16 }} />
                <div>
                  <Text type="secondary" style={{ fontSize: 16 }}>
                    Could not connect to database
                  </Text>
                </div>
                {user.role === 'Admin' && (
                  <Button
                    type="primary"
                    icon={<SettingOutlined />}
                    onClick={() => setSettingsOpen(true)}
                    style={{ marginTop: 16 }}
                  >
                    Open Settings
                  </Button>
                )}
              </div>
            )}

            {chatHistory.length === 0 && !loading && !connecting && connected && (
              <div style={{ maxWidth: 600, margin: '10vh auto 0' }}>
                <div
                  style={{
                    background: '#fff',
                    border: '1px solid #e8e8e8',
                    borderRadius: 16,
                    padding: '24px 28px',
                    marginBottom: 16,
                  }}
                >
                  <Space size={8} style={{ marginBottom: 12 }}>
                    <RobotOutlined style={{ color: '#4F46E5', fontSize: 20 }} />
                    <Title level={5} style={{ margin: 0, color: '#4F46E5' }}>
                      Welcome{user?.name ? `, ${user.name}` : ''}!
                    </Title>
                  </Space>
                  <Paragraph style={{ marginBottom: 16, color: '#555' }}>
                    I'm the Within Earth Chatbot. I can help you query your data using
                    natural language — no SQL needed. Here are some things you can ask:
                  </Paragraph>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {[
                      'Show top 5 agents by revenue',
                      'Revenue by country this month',
                      'How does my business look?',
                      'Top 10 hotels by bookings',
                    ].map((suggestion) => (
                      <Button
                        key={suggestion}
                        type="default"
                        style={{
                          textAlign: 'left',
                          height: 'auto',
                          padding: '8px 14px',
                          borderRadius: 8,
                          whiteSpace: 'normal',
                        }}
                        onClick={() => {
                          setQuestion(suggestion);
                        }}
                      >
                        {suggestion}
                      </Button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {chatHistory.map((entry, i) => (
              <ChatMessage key={i} entry={entry} />
            ))}

            {loading && (
              <div style={{ display: 'flex', justifyContent: 'center', padding: 24 }}>
                <Spin tip="Processing..." />
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {/* Input area */}
          <div
            style={{
              padding: '16px 24px',
              borderTop: '1px solid #f0f0f0',
              background: '#fff',
            }}
          >
            <Space.Compact style={{ width: '100%' }}>
              <Input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  connected
                    ? 'Ask about your data... (Press Enter to send)'
                    : connecting
                    ? 'Connecting to database...'
                    : 'Database not connected'
                }
                disabled={!connected || loading}
                size="large"
              />
              <Button
                type="primary"
                icon={<SendOutlined />}
                onClick={handleSend}
                disabled={!connected || !question.trim() || loading}
                size="large"
              >
                Send
              </Button>
            </Space.Compact>
            <Text type="secondary" style={{ fontSize: 11, marginTop: 4, display: 'block' }}>
              Press Enter to send. You can ask follow-up questions about previous results.
            </Text>
          </div>
        </Content>
      </Layout>

      {/* Settings Modal (Admin only) */}
      <Modal
        title="Database & LLM Settings"
        open={settingsOpen}
        onCancel={() => setSettingsOpen(false)}
        footer={null}
        width={400}
      >
        <Sidebar
          user={user}
          connected={connected}
          onConnected={() => {
            setConnected(true);
            setSettingsOpen(false);
          }}
        />
      </Modal>
    </Layout>
  );
}
