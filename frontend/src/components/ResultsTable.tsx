import { Table } from 'antd';
import type { ColumnsType } from 'antd/es/table';

interface Props {
  data: Record<string, unknown>[];
}

export default function ResultsTable({ data }: Props) {
  if (!data || data.length === 0) return null;

  const columns: ColumnsType<Record<string, unknown>> = Object.keys(data[0]).map((key) => ({
    title: key,
    dataIndex: key,
    key,
    ellipsis: true,
    render: (value: unknown) => {
      if (value === null || value === undefined) return '-';
      if (typeof value === 'number') return value.toLocaleString();
      return String(value);
    },
  }));

  return (
    <Table
      columns={columns}
      dataSource={data.map((row, i) => ({ ...row, key: i }))}
      size="small"
      scroll={{ x: 'max-content' }}
      pagination={data.length > 10 ? { pageSize: 10, size: 'small' } : false}
      style={{ marginTop: 8 }}
    />
  );
}
