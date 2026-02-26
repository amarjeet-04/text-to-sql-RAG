import { useEffect, useState } from 'react';
import {
  Table,
  Button,
  Modal,
  Form,
  Input,
  Select,
  Tag,
  Popconfirm,
  Typography,
  Space,
  message,
  Card,
} from 'antd';
import {
  PlusOutlined,
  DeleteOutlined,
  TeamOutlined,
  ArrowLeftOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { listUsers, createUser, deleteUser } from '../api/client';
import type { UserInfo } from '../api/client';

const { Title, Text } = Typography;

export default function AdminPage() {
  const [users, setUsers] = useState<UserInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [form] = Form.useForm();
  const navigate = useNavigate();

  const currentUser: UserInfo | null = (() => {
    try {
      return JSON.parse(localStorage.getItem('user') ?? 'null');
    } catch {
      return null;
    }
  })();

  const fetchUsers = async () => {
    setLoading(true);
    try {
      const data = await listUsers();
      setUsers(data);
    } catch {
      message.error('Failed to load users');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const handleCreate = async (values: {
    name: string;
    username: string;
    password: string;
    role: string;
  }) => {
    setSubmitting(true);
    try {
      await createUser(values.username, values.password, values.role, values.name);
      message.success(`User "${values.username}" created`);
      setModalOpen(false);
      form.resetFields();
      fetchUsers();
    } catch (err: unknown) {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        'Failed to create user';
      message.error(detail);
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (username: string) => {
    try {
      await deleteUser(username);
      message.success(`User "${username}" deleted`);
      fetchUsers();
    } catch (err: unknown) {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        'Failed to delete user';
      message.error(detail);
    }
  };

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (name: string) => <Text strong>{name}</Text>,
    },
    {
      title: 'Username',
      dataIndex: 'username',
      key: 'username',
      render: (username: string) => <Text code>{username}</Text>,
    },
    {
      title: 'Role',
      dataIndex: 'role',
      key: 'role',
      render: (role: string) => (
        <Tag color={role === 'Admin' ? 'gold' : 'blue'}>{role}</Tag>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: unknown, record: UserInfo) => {
        const isSelf = record.username === currentUser?.username;
        return (
          <Popconfirm
            title={`Delete user "${record.username}"?`}
            description="This action cannot be undone."
            onConfirm={() => handleDelete(record.username)}
            okText="Delete"
            okButtonProps={{ danger: true }}
            cancelText="Cancel"
            disabled={isSelf}
          >
            <Button
              danger
              size="small"
              icon={<DeleteOutlined />}
              disabled={isSelf}
              title={isSelf ? 'Cannot delete your own account' : ''}
            >
              Delete
            </Button>
          </Popconfirm>
        );
      },
    },
  ];

  return (
    <div
      style={{
        minHeight: '100vh',
        background: '#f5f5f5',
        padding: '32px 24px',
      }}
    >
      <div style={{ maxWidth: 860, margin: '0 auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 24, gap: 12 }}>
          <Button
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate('/')}
            type="text"
          >
            Back to Chat
          </Button>
        </div>

        <Card bordered={false} style={{ borderRadius: 12, boxShadow: '0 2px 12px rgba(0,0,0,0.06)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
            <Space>
              <TeamOutlined style={{ fontSize: 22, color: '#4F46E5' }} />
              <Title level={4} style={{ margin: 0, color: '#4F46E5' }}>
                User Management
              </Title>
            </Space>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setModalOpen(true)}
            >
              Add User
            </Button>
          </div>

          <Table
            columns={columns}
            dataSource={users}
            rowKey="username"
            loading={loading}
            pagination={false}
            size="middle"
          />
        </Card>
      </div>

      <Modal
        title="Create New User"
        open={modalOpen}
        onCancel={() => {
          setModalOpen(false);
          form.resetFields();
        }}
        footer={null}
        destroyOnClose
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreate}
          initialValues={{ role: 'Analyst' }}
          style={{ marginTop: 16 }}
        >
          <Form.Item
            name="name"
            label="Full Name"
            rules={[{ required: true, message: 'Please enter full name' }]}
          >
            <Input placeholder="e.g. Jane Smith" />
          </Form.Item>

          <Form.Item
            name="username"
            label="Username"
            rules={[
              { required: true, message: 'Please enter a username' },
              { min: 3, message: 'At least 3 characters' },
            ]}
          >
            <Input placeholder="e.g. jsmith" />
          </Form.Item>

          <Form.Item
            name="password"
            label="Password"
            rules={[
              { required: true, message: 'Please enter a password' },
              { min: 6, message: 'At least 6 characters' },
            ]}
          >
            <Input.Password placeholder="Min. 6 characters" />
          </Form.Item>

          <Form.Item name="role" label="Role">
            <Select>
              <Select.Option value="Analyst">Analyst</Select.Option>
              <Select.Option value="Admin">Admin</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item style={{ marginBottom: 0, textAlign: 'right' }}>
            <Space>
              <Button onClick={() => { setModalOpen(false); form.resetFields(); }}>
                Cancel
              </Button>
              <Button type="primary" htmlType="submit" loading={submitting}>
                Create User
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
