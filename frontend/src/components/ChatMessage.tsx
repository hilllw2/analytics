import { User, Bot, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Message } from '../store/useStore';

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`chat-message ${message.role}`}>
      <div className="flex items-start gap-3">
        {/* Avatar */}
        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser ? 'bg-primary-500' : 'bg-gradient-to-br from-primary-100 to-accent-100'
        }`}>
          {isUser ? (
            <User className="w-4 h-4 text-white" />
          ) : (
            <Bot className="w-4 h-4 text-primary-600" />
          )}
        </div>
        
        {/* Content */}
        <div className="flex-1 min-w-0">
          {message.isLoading ? (
            <div className="flex items-center gap-2 text-gray-500">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Analyzing...</span>
            </div>
          ) : (
            <div className={`prose prose-sm max-w-none ${isUser ? 'prose-invert' : ''}`}>
              <ReactMarkdown
                components={{
                  // Custom rendering for code blocks
                  code: ({ node, className, children, ...props }) => {
                    const isInline = !className;
                    return isInline ? (
                      <code className="px-1 py-0.5 bg-gray-100 rounded text-sm font-mono" {...props}>
                        {children}
                      </code>
                    ) : (
                      <pre className="bg-gray-900 text-gray-100 p-3 rounded-lg overflow-x-auto">
                        <code className="text-sm font-mono" {...props}>
                          {children}
                        </code>
                      </pre>
                    );
                  },
                  // Custom table styling
                  table: ({ children }) => (
                    <div className="overflow-x-auto my-2">
                      <table className="min-w-full divide-y divide-gray-200 text-sm">
                        {children}
                      </table>
                    </div>
                  ),
                  th: ({ children }) => (
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider bg-gray-50">
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td className="px-3 py-2 whitespace-nowrap">
                      {children}
                    </td>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
