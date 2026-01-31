import { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  RefreshCw, 
  Code, 
  BookOpen,
  Sparkles,
  FileText,
  BarChart,
  AlertCircle
} from 'lucide-react';
import { useStore } from '../store/useStore';
import { api } from '../services/api';
import { ChatMessage } from './ChatMessage';
import { ChartDisplay } from './ChartDisplay';
import { DataTable } from './DataTable';

export function ChatPanel() {
  const { 
    messages, 
    activeDataset, 
    isProcessing, 
    suggestedQuestions,
    addMessage, 
    updateMessage, 
    setIsProcessing,
    setSuggestedQuestions
  } = useStore();
  
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load suggestions when dataset changes
  useEffect(() => {
    const loadSuggestions = async () => {
      if (activeDataset) {
        try {
          const data = await api.getSuggestions();
          setSuggestedQuestions(data.suggestions || []);
        } catch (error) {
          console.error('Failed to load suggestions:', error);
        }
      }
    };
    loadSuggestions();
  }, [activeDataset, setSuggestedQuestions]);

  const handleSend = async () => {
    if (!input.trim() || isProcessing || !activeDataset) return;

    const userMessage = {
      id: `msg_${Date.now()}`,
      role: 'user' as const,
      content: input.trim(),
      timestamp: new Date(),
    };

    const assistantMessageId = `msg_${Date.now() + 1}`;
    const assistantMessage = {
      id: assistantMessageId,
      role: 'assistant' as const,
      content: '',
      timestamp: new Date(),
      isLoading: true,
    };

    addMessage(userMessage);
    addMessage(assistantMessage);
    setInput('');
    setIsProcessing(true);

    try {
      const response = await api.sendMessage(input.trim());
      
      updateMessage(assistantMessageId, {
        content: response.response || 'I processed your request.',
        isLoading: false,
        data: response.data,
        chart: response.chart,
        methodology: response.methodology,
        warnings: response.warnings,
      });
    } catch (error: any) {
      updateMessage(assistantMessageId, {
        content: `Sorry, I encountered an error: ${error.message || 'Unknown error'}`,
        isLoading: false,
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRegenerate = async () => {
    if (isProcessing || messages.length < 2) return;

    setIsProcessing(true);
    
    const newAssistantId = `msg_${Date.now()}`;
    addMessage({
      id: newAssistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isLoading: true,
    });

    try {
      const response = await api.regenerateResponse();
      
      updateMessage(newAssistantId, {
        content: response.response || 'Here is an alternative response.',
        isLoading: false,
        data: response.data,
        chart: response.chart,
        methodology: response.methodology,
        warnings: response.warnings,
      });
    } catch (error: any) {
      updateMessage(newAssistantId, {
        content: `Regeneration failed: ${error.message}`,
        isLoading: false,
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestionClick = (text: string) => {
    setInput(text);
    inputRef.current?.focus();
  };

  // Welcome state
  if (!activeDataset) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center max-w-md">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-100 to-accent-100 flex items-center justify-center mx-auto mb-4">
            <Sparkles className="w-8 h-8 text-primary-600" />
          </div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">
            Welcome to DataChat
          </h2>
          <p className="text-gray-500 mb-6">
            Upload a data file to start exploring your data with natural language queries.
          </p>
          <div className="text-left bg-gray-50 rounded-lg p-4 text-sm text-gray-600">
            <p className="font-medium mb-2">You can:</p>
            <ul className="space-y-1">
              <li>• Ask questions in plain English</li>
              <li>• Get automatic insights and visualizations</li>
              <li>• Explore trends, segments, and anomalies</li>
              <li>• Export charts and reports</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-gray-50">
      {/* Messages - min-h-0 allows flex child to shrink so overflow-y-auto works */}
      <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-8">
            <p className="text-gray-500 mb-4">
              Start by asking a question about your data
            </p>
            
            {/* Suggestions */}
            {suggestedQuestions.length > 0 && (
              <div className="flex flex-wrap justify-center gap-2">
                {suggestedQuestions.slice(0, 6).map((q, i) => (
                  <button
                    key={i}
                    onClick={() => handleSuggestionClick(q.text)}
                    className="chip bg-white border hover:border-primary-300 hover:bg-primary-50 transition-colors"
                  >
                    {q.text}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] ${message.role === 'user' ? '' : 'w-full'}`}>
              <ChatMessage message={message} />
              
              {/* Chart */}
              {message.chart && (
                <div className="mt-3 card p-4">
                  <ChartDisplay chart={message.chart} />
                </div>
              )}
              
              {/* Data Table */}
              {message.data && message.role === 'assistant' && (
                <div className="mt-3 card overflow-hidden">
                  <DataTable data={message.data} />
                </div>
              )}
              
              {/* Methodology */}
              {message.methodology && (
                <div className="mt-2 text-xs text-gray-500 flex items-center gap-1">
                  <BookOpen className="w-3 h-3" />
                  <span>{message.methodology}</span>
                </div>
              )}
              
              {/* Warnings */}
              {message.warnings && message.warnings.length > 0 && (
                <div className="mt-2 p-2 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-700">
                  {message.warnings.map((w, i) => (
                    <div key={i} className="flex items-start gap-1">
                      <AlertCircle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                      <span>{w}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Quick Prompts */}
      {messages.length > 0 && (
        <div className="px-4 pb-2 flex gap-2 overflow-x-auto">
          <button
            onClick={() => handleSuggestionClick('Summarize the dataset')}
            className="chip bg-white border text-xs whitespace-nowrap hover:border-primary-300"
          >
            <FileText className="w-3 h-3 mr-1" />
            Summarize
          </button>
          <button
            onClick={() => handleSuggestionClick('Find anomalies')}
            className="chip bg-white border text-xs whitespace-nowrap hover:border-primary-300"
          >
            <AlertCircle className="w-3 h-3 mr-1" />
            Find anomalies
          </button>
          <button
            onClick={() => handleSuggestionClick('Show top drivers')}
            className="chip bg-white border text-xs whitespace-nowrap hover:border-primary-300"
          >
            <BarChart className="w-3 h-3 mr-1" />
            Top drivers
          </button>
          <button
            onClick={() => handleSuggestionClick('Compare segments')}
            className="chip bg-white border text-xs whitespace-nowrap hover:border-primary-300"
          >
            <BarChart className="w-3 h-3 mr-1" />
            Compare segments
          </button>
        </div>
      )}

      {/* Input */}
      <div className="p-4 bg-white border-t">
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your data..."
              className="input resize-none pr-20"
              rows={2}
              disabled={isProcessing}
            />
            <div className="absolute right-2 bottom-2 flex gap-1">
              {messages.length > 0 && (
                <button
                  onClick={handleRegenerate}
                  disabled={isProcessing}
                  className="p-1.5 text-gray-400 hover:text-gray-600 disabled:opacity-50"
                  title="Regenerate"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>
          <button
            onClick={handleSend}
            disabled={!input.trim() || isProcessing}
            className="btn btn-primary px-6 self-end"
          >
            {isProcessing ? (
              <div className="w-5 h-5 spinner" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
