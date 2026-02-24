import { useState, useEffect } from 'react';
import {
  Form,
  Input,
  Select,
  Slider,
  Switch,
  Button,
  Divider,
  Typography,
  Badge,
  Space,
  message,
} from 'antd';
import {
  ApiOutlined,
  SettingOutlined,
  SafetyOutlined,
  ClearOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
} from '@ant-design/icons';
import { connectDb, clearCache } from '../api/client';
import type { User } from '../types';

const { Title, Text } = Typography;

interface Props {
  user: User;
  connected: boolean;
  onConnected: () => void;
}

export default function Sidebar({ user, connected, onConnected }: Props) {
  const isAdmin = user.role === 'Admin';
  const [loading, setLoading] = useState(false);

  const [form] = Form.useForm();

  // Load saved settings from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('dbSettings');
    if (saved) {
      try {
        form.setFieldsValue(JSON.parse(saved));
      } catch {
        // ignore
      }
    }
  }, [form]);

  const handleConnect = async () => {
    try {
      const values = await form.validateFields();
      setLoading(true);

      // Save settings
      localStorage.setItem('dbSettings', JSON.stringify(values));

      const result = await connectDb({
        host: values.host,
        port: values.port,
        username: values.db_username,
        password: values.db_password,
        database: values.database,
        llm_provider: values.llm_provider,
        api_key: values.api_key,
        model: values.model,
        temperature: values.temperature,
        query_timeout: values.query_timeout,
        view_support: values.view_support,
      });

      if (result.success) {
        message.success(result.message);
        onConnected();
      } else {
        message.error(result.message);
      }
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'response' in err) {
        const axiosErr = err as { response?: { data?: { detail?: string } } };
        message.error(axiosErr.response?.data?.detail || 'Connection failed');
      } else if (err && typeof err === 'object' && 'message' in err) {
        const msg = String((err as { message?: string }).message || 'Connection failed');
        if (msg.toLowerCase().includes('timeout')) {
          message.error('Connection timed out. Please verify backend is running and DB host/port are reachable.');
        } else {
          message.error(msg);
        }
      } else {
        message.error('Connection failed');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleClearCache = async () => {
    try {
      await clearCache();
      message.success('Cache cleared');
    } catch {
      message.error('Failed to clear cache');
    }
  };

  return (
    <div style={{ padding: '16px 12px' }}>
      <Space style={{ marginBottom: 16 }}>
        <SettingOutlined />
        <Title level={5} style={{ margin: 0 }}>
          Configuration
        </Title>
      </Space>

      {!isAdmin && (
        <div
          style={{
            padding: '8px 12px',
            background: '#e6f7ff',
            borderRadius: 6,
            marginBottom: 16,
            fontSize: 12,
          }}
        >
          Read-only mode. Contact admin for settings access.
        </div>
      )}

      <Form
        form={form}
        layout="vertical"
        size="small"
        initialValues={{
          llm_provider: 'DeepSeek',
          model: 'deepseek-chat',
          temperature: 0,
          host: '95.168.168.71',
          port: '1988',
          db_username: 'withinearth_reader',
          db_password: 'pass@readerWE#2026',
          database: 'mis_report_data',
          api_key: 'sk-553b82b11de04693a5a8ad23a1862347',
          query_timeout: 60,
          view_support: true,
        }}
      >
        {/* LLM Settings */}
        <Divider orientationMargin={0} style={{ fontSize: 12 }}>
          LLM Settings
        </Divider>

        <Form.Item label="Provider" name="llm_provider">
          <Select
            disabled={!isAdmin}
            options={[
              { value: 'OpenAI', label: 'OpenAI' },
              { value: 'DeepSeek', label: 'DeepSeek' },
            ]}
          />
        </Form.Item>

        <Form.Item label="API Key" name="api_key" rules={[{ required: true }]}>
          <Input.Password disabled={!isAdmin} placeholder="Enter API key" />
        </Form.Item>

        <Form.Item noStyle shouldUpdate={(prev, curr) => prev.llm_provider !== curr.llm_provider}>
          {({ getFieldValue }) => (
            <Form.Item label="Model" name="model">
              <Select
                disabled={!isAdmin}
                options={
                  getFieldValue('llm_provider') === 'DeepSeek'
                    ? [
                        { value: 'deepseek-chat', label: 'deepseek-chat' },
                        { value: 'deepseek-coder', label: 'deepseek-coder' },
                      ]
                    : [
                        { value: 'gpt-4', label: 'GPT-4' },
                        { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
                        { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' },
                      ]
                }
              />
            </Form.Item>
          )}
        </Form.Item>

        <Form.Item label="Temperature" name="temperature">
          <Slider disabled={!isAdmin} min={0} max={1} step={0.1} />
        </Form.Item>

        {/* Database Settings */}
        <Divider orientationMargin={0} style={{ fontSize: 12 }}>
          Microsoft SQL Server
        </Divider>

        <Form.Item label="Host" name="host" rules={[{ required: true }]}>
          <Input disabled={!isAdmin} />
        </Form.Item>

        <Form.Item label="Port" name="port" rules={[{ required: true }]}>
          <Input disabled={!isAdmin} />
        </Form.Item>

        <Form.Item label="Username" name="db_username" rules={[{ required: true }]}>
          <Input disabled={!isAdmin} />
        </Form.Item>

        <Form.Item label="Password" name="db_password" rules={[{ required: true }]}>
          <Input.Password disabled={!isAdmin} />
        </Form.Item>

        <Form.Item label="Database" name="database" rules={[{ required: true }]}>
          <Input disabled={!isAdmin} />
        </Form.Item>

        {/* Query Protection */}
        <Divider orientationMargin={0} style={{ fontSize: 12 }}>
          <SafetyOutlined /> Query Protection
        </Divider>

        <Form.Item label="Timeout (seconds)" name="query_timeout">
          <Slider disabled={!isAdmin} min={5} max={120} />
        </Form.Item>

        <Form.Item label="View Support" name="view_support" valuePropName="checked">
          <Switch disabled={!isAdmin} />
        </Form.Item>
      </Form>

      <Button
        type="primary"
        icon={<ApiOutlined />}
        onClick={handleConnect}
        loading={loading}
        block
        style={{ marginBottom: 12 }}
      >
        Connect to Database
      </Button>

      {/* Connection Status */}
      <div style={{ textAlign: 'center', marginBottom: 12 }}>
        <Badge
          status={connected ? 'success' : 'warning'}
          text={
            <Text type={connected ? 'success' : 'warning'}>
              {connected ? (
                <>
                  <CheckCircleOutlined /> Connected
                </>
              ) : (
                <>
                  <CloseCircleOutlined /> Not connected
                </>
              )}
            </Text>
          }
        />
      </div>

      {/* Admin actions */}
      {isAdmin && connected && (
        <Button
          icon={<ClearOutlined />}
          onClick={handleClearCache}
          block
          size="small"
          danger
        >
          Clear Query Cache
        </Button>
      )}
    </div>
  );
}
