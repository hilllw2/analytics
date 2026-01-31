import { create } from 'zustand';

export interface Dataset {
  name: string;
  filename: string;
  rowCount: number;
  columnCount: number;
  columns: string[];
  inferredTypes: Record<string, string>;
  dateColumns: string[];
  numericColumns: string[];
  categoricalColumns: string[];
  isSampled: boolean;
  fullRowCount?: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  data?: any;
  chart?: any;
  methodology?: string;
  warnings?: string[];
  isLoading?: boolean;
}

export interface ChartInfo {
  id: string;
  type: string;
  title: string;
  plotlyJson: any;
}

export interface Insight {
  id: string;
  category: string;
  title: string;
  description: string;
  importance: string;
  data?: any;
  suggestedAction?: string;
}

interface AppState {
  // Session
  sessionId: string | null;
  sessionExpiry: Date | null;
  
  // Data
  datasets: Dataset[];
  activeDataset: Dataset | null;
  previewData: any[];
  
  // Chat
  messages: Message[];
  isProcessing: boolean;
  
  // Insights & Profile
  insights: Insight[];
  healthReport: any;
  suggestedQuestions: { text: string; category: string }[];
  
  // Charts
  charts: ChartInfo[];
  
  // UI State
  showDataPanel: boolean;
  showInsightsPanel: boolean;
  
  // Actions
  setSessionId: (id: string) => void;
  setDatasets: (datasets: Dataset[]) => void;
  setActiveDataset: (dataset: Dataset | null) => void;
  setPreviewData: (data: any[]) => void;
  addMessage: (message: Message) => void;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  setMessages: (messages: Message[]) => void;
  setIsProcessing: (processing: boolean) => void;
  setInsights: (insights: Insight[]) => void;
  setHealthReport: (report: any) => void;
  setSuggestedQuestions: (questions: { text: string; category: string }[]) => void;
  addChart: (chart: ChartInfo) => void;
  toggleDataPanel: () => void;
  toggleInsightsPanel: () => void;
  clearSession: () => void;
}

export const useStore = create<AppState>((set) => ({
  // Initial state
  sessionId: null,
  sessionExpiry: null,
  datasets: [],
  activeDataset: null,
  previewData: [],
  messages: [],
  isProcessing: false,
  insights: [],
  healthReport: null,
  suggestedQuestions: [],
  charts: [],
  showDataPanel: true,
  showInsightsPanel: true,

  // Actions
  setSessionId: (id) => set({ sessionId: id }),
  
  setDatasets: (datasets) => set({ datasets }),
  
  setActiveDataset: (dataset) => set({ activeDataset: dataset }),
  
  setPreviewData: (data) => set({ previewData: data }),
  
  addMessage: (message) => set((state) => ({
    messages: [...state.messages, message]
  })),
  
  updateMessage: (id, updates) => set((state) => ({
    messages: state.messages.map((msg) =>
      msg.id === id ? { ...msg, ...updates } : msg
    )
  })),
  
  setMessages: (messages) => set({ messages }),
  
  setIsProcessing: (processing) => set({ isProcessing: processing }),
  
  setInsights: (insights) => set({ insights }),
  
  setHealthReport: (report) => set({ healthReport: report }),
  
  setSuggestedQuestions: (questions) => set({ suggestedQuestions: questions }),
  
  addChart: (chart) => set((state) => ({
    charts: [...state.charts, chart]
  })),
  
  toggleDataPanel: () => set((state) => ({ showDataPanel: !state.showDataPanel })),
  
  toggleInsightsPanel: () => set((state) => ({ showInsightsPanel: !state.showInsightsPanel })),
  
  clearSession: () => set({
    sessionId: null,
    sessionExpiry: null,
    datasets: [],
    activeDataset: null,
    previewData: [],
    messages: [],
    insights: [],
    healthReport: null,
    suggestedQuestions: [],
    charts: [],
  }),
}));
