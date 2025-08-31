import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout, Menu, Typography } from 'antd';
import { 
  UploadOutlined, 
  BarChartOutlined, 
  HistoryOutlined,
  HomeOutlined 
} from '@ant-design/icons';

import Dashboard from './components/Dashboard';
import Upload from './components/Upload';
import Analytics from './components/Analytics';
import History from './components/History';
import AnalysisResult from './components/AnalysisResult';

import './App.css';

const { Header, Content, Sider } = Layout;
const { Title } = Typography;

function App() {
  const [selectedKey, setSelectedKey] = React.useState('1');

  const menuItems = [
    {
      key: '1',
      icon: <HomeOutlined />,
      label: 'Dashboard',
      path: '/'
    },
    {
      key: '2',
      icon: <UploadOutlined />,
      label: 'Upload Video',
      path: '/upload'
    },
    {
      key: '3',
      icon: <HistoryOutlined />,
      label: 'Analysis History',
      path: '/history'
    },
    {
      key: '4',
      icon: <BarChartOutlined />,
      label: 'Analytics',
      path: '/analytics'
    }
  ];

  return (
    <Router>
      <Layout style={{ minHeight: '100vh' }}>
        <Header style={{ 
          background: '#001529', 
          padding: '0 24px',
          display: 'flex',
          alignItems: 'center'
        }}>
          <Title level={3} style={{ 
            color: 'white', 
            margin: 0,
            fontWeight: 'bold'
          }}>
            🎯 WSD AI Judge
          </Title>
        </Header>
        
        <Layout>
          <Sider width={200} style={{ background: '#fff' }}>
            <Menu
              mode="inline"
              selectedKeys={[selectedKey]}
              style={{ height: '100%', borderRight: 0 }}
              items={menuItems.map(item => ({
                key: item.key,
                icon: item.icon,
                label: (
                  <a 
                    href={item.path}
                    onClick={(e) => {
                      e.preventDefault();
                      setSelectedKey(item.key);
                      window.history.pushState({}, '', item.path);
                    }}
                  >
                    {item.label}
                  </a>
                )
              }))}
            />
          </Sider>
          
          <Layout style={{ padding: '24px' }}>
            <Content style={{
              background: '#fff',
              padding: 24,
              margin: 0,
              minHeight: 280,
              borderRadius: 8
            }}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/upload" element={<Upload />} />
                <Route path="/history" element={<History />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/analysis/:id" element={<AnalysisResult />} />
              </Routes>
            </Content>
          </Layout>
        </Layout>
      </Layout>
    </Router>
  );
}

export default App;
