import React, { useState } from 'react';
import { 
  Upload as AntUpload, 
  Button, 
  Form, 
  Input, 
  Select, 
  Card, 
  Progress, 
  Alert,
  Typography,
  Space,
  Divider
} from 'antd';
import { UploadOutlined, VideoCameraOutlined } from '@ant-design/icons';
import axios from 'axios';

const { Title, Text } = Typography;
const { Option } = Select;

const Upload = () => {
  const [form] = Form.useForm();
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analysisId, setAnalysisId] = useState(null);
  const [error, setError] = useState(null);
  const [fileList, setFileList] = useState([]);

  const speakerRoles = [
    { value: 'first_proposition', label: 'First Proposition' },
    { value: 'first_opposition', label: 'First Opposition' },
    { value: 'second_proposition', label: 'Second Proposition' },
    { value: 'second_opposition', label: 'Second Opposition' },
    { value: 'third_proposition', label: 'Third Proposition' },
    { value: 'third_opposition', label: 'Third Opposition' }
  ];

  const teamSides = [
    { value: 'Proposition', label: 'Proposition' },
    { value: 'Opposition', label: 'Opposition' }
  ];

  const handleUpload = async (values) => {
    if (fileList.length === 0) {
      setError('Please select a video file');
      return;
    }

    setUploading(true);
    setError(null);
    setProgress(0);

    const formData = new FormData();
    formData.append('file', fileList[0]);
    formData.append('speaker_name', values.speaker_name);
    formData.append('speaker_role', values.speaker_role);
    formData.append('debate_topic', values.debate_topic);
    formData.append('team_side', values.team_side);

    try {
      const response = await axios.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const { analysis_id } = response.data;
      setAnalysisId(analysis_id);

      // Poll for progress
      pollProgress(analysis_id);

    } catch (err) {
      setError(err.response?.data?.message || 'Upload failed');
      setUploading(false);
    }
  };

  const pollProgress = async (id) => {
    try {
      const response = await axios.get(`/analysis/${id}/status`);
      const { status, progress: currentProgress } = response.data;

      setProgress(currentProgress * 100);

      if (status === 'completed') {
        setUploading(false);
        // Redirect to results page
        window.location.href = `/analysis/${id}`;
      } else if (status === 'failed') {
        setError('Analysis failed');
        setUploading(false);
      } else {
        // Continue polling
        setTimeout(() => pollProgress(id), 2000);
      }
    } catch (err) {
      setError('Failed to check analysis status');
      setUploading(false);
    }
  };

  const uploadProps = {
    beforeUpload: (file) => {
      const isVideo = file.type.startsWith('video/');
      if (!isVideo) {
        setError('Please upload a video file');
        return false;
      }

      const isLt500M = file.size / 1024 / 1024 < 500;
      if (!isLt500M) {
        setError('Video must be smaller than 500MB');
        return false;
      }

      setFileList([file]);
      setError(null);
      return false; // Prevent automatic upload
    },
    fileList,
    onRemove: () => {
      setFileList([]);
    },
  };

  return (
    <div style={{ maxWidth: 800, margin: '0 auto' }}>
      <Card>
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <div style={{ textAlign: 'center' }}>
            <VideoCameraOutlined style={{ fontSize: 48, color: '#1890ff' }} />
            <Title level={2}>Upload Debate Video</Title>
            <Text type="secondary">
              Upload your debate speech video for AI-powered analysis and feedback
            </Text>
          </div>

          <Divider />

          {error && (
            <Alert
              message="Error"
              description={error}
              type="error"
              showIcon
              closable
              onClose={() => setError(null)}
            />
          )}

          {uploading && (
            <Card>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text strong>Analysis in Progress...</Text>
                <Progress 
                  percent={Math.round(progress)} 
                  status={progress === 100 ? 'success' : 'active'}
                />
                <Text type="secondary">
                  This may take a few minutes depending on video length
                </Text>
              </Space>
            </Card>
          )}

          <Form
            form={form}
            layout="vertical"
            onFinish={handleUpload}
            disabled={uploading}
          >
            <Form.Item
              label="Video File"
              required
            >
              <AntUpload.Dragger {...uploadProps}>
                <p className="ant-upload-drag-icon">
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">
                  Click or drag video file to this area to upload
                </p>
                <p className="ant-upload-hint">
                  Supports MP4, AVI, MOV formats. Maximum file size: 500MB
                </p>
              </AntUpload.Dragger>
            </Form.Item>

            <Form.Item
              name="speaker_name"
              label="Speaker Name"
              rules={[{ required: true, message: 'Please enter speaker name' }]}
            >
              <Input placeholder="Enter the speaker's name" />
            </Form.Item>

            <Form.Item
              name="speaker_role"
              label="Speaker Role"
              rules={[{ required: true, message: 'Please select speaker role' }]}
            >
              <Select placeholder="Select the speaker's role in the debate">
                {speakerRoles.map(role => (
                  <Option key={role.value} value={role.value}>
                    {role.label}
                  </Option>
                ))}
              </Select>
            </Form.Item>

            <Form.Item
              name="team_side"
              label="Team Side"
              rules={[{ required: true, message: 'Please select team side' }]}
            >
              <Select placeholder="Select which side of the debate">
                {teamSides.map(side => (
                  <Option key={side.value} value={side.value}>
                    {side.label}
                  </Option>
                ))}
              </Select>
            </Form.Item>

            <Form.Item
              name="debate_topic"
              label="Debate Topic"
              rules={[{ required: true, message: 'Please enter debate topic' }]}
            >
              <Input.TextArea 
                rows={3}
                placeholder="Enter the debate motion/topic"
              />
            </Form.Item>

            <Form.Item>
              <Button 
                type="primary" 
                htmlType="submit" 
                loading={uploading}
                size="large"
                block
                icon={<UploadOutlined />}
              >
                {uploading ? 'Analyzing...' : 'Upload and Analyze'}
              </Button>
            </Form.Item>
          </Form>

          <div style={{ background: '#f6f6f6', padding: 16, borderRadius: 8 }}>
            <Title level={4}>What happens next?</Title>
            <ul>
              <li>Your video will be processed using advanced AI models</li>
              <li>Speech will be transcribed and analyzed for content quality</li>
              <li>Delivery style and visual presentation will be evaluated</li>
              <li>You'll receive detailed scores and feedback based on WSD rubric</li>
            </ul>
          </div>
        </Space>
      </Card>
    </div>
  );
};

export default Upload;
