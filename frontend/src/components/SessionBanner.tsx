import { useState, useEffect } from 'react';
import { Clock, AlertTriangle, X } from 'lucide-react';
import { useStore } from '../store/useStore';
import { api } from '../services/api';

export function SessionBanner() {
  const { sessionId } = useStore();
  const [expiresIn, setExpiresIn] = useState<number | null>(null);
  const [showWarning, setShowWarning] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  // Check session status periodically
  useEffect(() => {
    const checkSession = async () => {
      if (!sessionId) return;
      
      try {
        const info = await api.getSessionInfo();
        setExpiresIn(info.expires_in_minutes);
        
        // Show warning when less than 10 minutes remain
        if (info.expires_in_minutes <= 10 && !dismissed) {
          setShowWarning(true);
        }
      } catch (error) {
        console.error('Failed to check session:', error);
      }
    };

    checkSession();
    const interval = setInterval(checkSession, 60000); // Check every minute
    
    return () => clearInterval(interval);
  }, [sessionId, dismissed]);

  // Touch session to extend timeout
  const handleExtend = async () => {
    try {
      await api.getSessionInfo(); // This touches the session
      setShowWarning(false);
      setDismissed(false);
    } catch (error) {
      console.error('Failed to extend session:', error);
    }
  };

  if (!showWarning || !expiresIn) return null;

  return (
    <div className="bg-amber-50 border-b border-amber-200 px-4 py-2">
      <div className="flex items-center justify-between max-w-screen-xl mx-auto">
        <div className="flex items-center gap-2 text-amber-700">
          <AlertTriangle className="w-4 h-4" />
          <span className="text-sm">
            <strong>Session expiring soon.</strong>{' '}
            {expiresIn} minute{expiresIn !== 1 ? 's' : ''} remaining.{' '}
            Your data will be deleted when the session ends.
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleExtend}
            className="text-sm font-medium text-amber-700 hover:text-amber-800 underline"
          >
            Keep working
          </button>
          <button
            onClick={() => {
              setShowWarning(false);
              setDismissed(true);
            }}
            className="p-1 hover:bg-amber-100 rounded"
          >
            <X className="w-4 h-4 text-amber-600" />
          </button>
        </div>
      </div>
    </div>
  );
}
