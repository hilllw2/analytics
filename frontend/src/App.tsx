import { useEffect, useState } from 'react';
import { useStore } from './store/useStore';
import { api } from './services/api';
import { Header } from './components/Header';
import { SheetTabs } from './components/SheetTabs';
import { Sidebar } from './components/Sidebar';
import { ChatPanel } from './components/ChatPanel';
import { DataPanel } from './components/DataPanel';
import { UploadModal } from './components/UploadModal';
import { RightPanel } from './components/RightPanel';
import { SessionBanner } from './components/SessionBanner';

function App() {
  const { 
    sessionId, 
    activeDataset, 
    showDataPanel, 
    showInsightsPanel,
    setSessionId 
  } = useStore();
  
  const [showUpload, setShowUpload] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);

  useEffect(() => {
    // Initialize session
    const initSession = async () => {
      try {
        const session = await api.createSession();
        setSessionId(session.session_id);
      } catch (error) {
        console.error('Failed to create session:', error);
      } finally {
        setIsInitializing(false);
      }
    };

    if (!sessionId) {
      initSession();
    } else {
      setIsInitializing(false);
    }
  }, [sessionId, setSessionId]);

  if (isInitializing) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="w-12 h-12 spinner mx-auto mb-4"></div>
          <p className="text-gray-600">Initializing session...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      <Header onUpload={() => setShowUpload(true)} />
      <SheetTabs />
      <SessionBanner />
      
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <Sidebar onUpload={() => setShowUpload(true)} />
        
        {/* Main Content */}
        <main className="flex-1 flex overflow-hidden">
          {/* Data Panel */}
          {showDataPanel && activeDataset && (
            <div className="w-[45%] border-r overflow-hidden flex flex-col">
              <DataPanel />
            </div>
          )}
          
          {/* Chat Panel */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <ChatPanel />
          </div>
          
          {/* Right Panel (Profile + Charts + Insights) */}
          {showInsightsPanel && activeDataset && (
            <div className="w-[480px] border-l overflow-hidden flex flex-col bg-white">
              <RightPanel />
            </div>
          )}
        </main>
      </div>
      
      {/* Upload Modal */}
      {showUpload && (
        <UploadModal onClose={() => setShowUpload(false)} />
      )}
    </div>
  );
}

export default App;
