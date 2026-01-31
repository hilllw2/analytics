import axios, { AxiosInstance } from 'axios';

class ApiService {
  private client: AxiosInstance;
  private sessionId: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: '/api',
      timeout: 60000,
    });

    // Add session ID to all requests
    this.client.interceptors.request.use((config) => {
      if (this.sessionId) {
        config.headers['X-Session-ID'] = this.sessionId;
      }
      return config;
    });
  }

  setSessionId(id: string) {
    this.sessionId = id;
  }

  getSessionId() {
    return this.sessionId;
  }

  // Session APIs
  async createSession() {
    const response = await this.client.post('/session/create');
    this.sessionId = response.data.session_id;
    return response.data;
  }

  async getSessionInfo() {
    const response = await this.client.get('/session/info');
    return response.data;
  }

  async endSession() {
    const response = await this.client.delete('/session/end');
    this.sessionId = null;
    return response.data;
  }

  async clearSession() {
    const response = await this.client.delete('/session/clear');
    return response.data;
  }

  // Upload APIs
  async uploadFile(file: File, options?: { sampleMode?: boolean; maxRows?: number; sheetName?: string }) {
    const formData = new FormData();
    formData.append('file', file);

    const params = new URLSearchParams();
    if (options?.sampleMode) params.append('sample_mode', 'true');
    if (options?.maxRows) params.append('max_rows', options.maxRows.toString());
    if (options?.sheetName) params.append('sheet_name', options.sheetName);

    const response = await this.client.post(`/upload/file?${params.toString()}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    // Update session ID if new session was created
    if (response.data.session_id && !this.sessionId) {
      this.sessionId = response.data.session_id;
    }

    return response.data;
  }

  async getExcelSheets(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.get('/upload/excel-sheets', {
      data: formData,
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  async getPreview(options?: {
    offset?: number;
    limit?: number;
    sortColumn?: string;
    sortAscending?: boolean;
    search?: string;
    searchColumn?: string;
  }) {
    const params = new URLSearchParams();
    if (options?.offset) params.append('offset', options.offset.toString());
    if (options?.limit) params.append('limit', options.limit.toString());
    if (options?.sortColumn) params.append('sort_column', options.sortColumn);
    if (options?.sortAscending !== undefined) params.append('sort_ascending', options.sortAscending.toString());
    if (options?.search) params.append('search', options.search);
    if (options?.searchColumn) params.append('search_column', options.searchColumn);

    const response = await this.client.get(`/upload/preview?${params.toString()}`);
    return response.data;
  }

  async getProfile(includeCorrelations = true) {
    const response = await this.client.post(`/upload/profile?include_correlations=${includeCorrelations}`);
    return response.data;
  }

  async getInsights(maxInsights = 15) {
    const response = await this.client.post(`/upload/insights?max_insights=${maxInsights}`);
    return response.data;
  }

  async getYDataProfile(minimal = true) {
    const response = await this.client.post(`/upload/ydata-profile?minimal=${minimal}`);
    return response.data;
  }

  async createQuickChart(params: {
    chartType: string;
    xColumn?: string;
    yColumn?: string;
    colorColumn?: string;
    title?: string;
    aggregation?: string;
  }) {
    const queryParams = new URLSearchParams();
    queryParams.append('chart_type', params.chartType);
    if (params.xColumn) queryParams.append('x_column', params.xColumn);
    if (params.yColumn) queryParams.append('y_column', params.yColumn);
    if (params.colorColumn) queryParams.append('color_column', params.colorColumn);
    if (params.title) queryParams.append('title', params.title);
    if (params.aggregation) queryParams.append('aggregation', params.aggregation);
    
    const response = await this.client.post(`/upload/quick-chart?${queryParams.toString()}`);
    return response.data;
  }

  // Chat APIs
  async sendMessage(message: string, options?: { regenerate?: boolean; style?: string }) {
    const response = await this.client.post('/chat/message', {
      message,
      regenerate: options?.regenerate || false,
      style: options?.style,
    });
    return response.data;
  }

  async regenerateResponse() {
    const response = await this.client.post('/chat/regenerate');
    return response.data;
  }

  async applyStyle(style: 'detailed' | 'simple' | 'technical') {
    const response = await this.client.post(`/chat/style/${style}`);
    return response.data;
  }

  async getSuggestions() {
    const response = await this.client.get('/chat/suggestions');
    return response.data;
  }

  async getChatHistory(limit = 50) {
    const response = await this.client.get(`/chat/history?limit=${limit}`);
    return response.data;
  }

  async pinDefinition(name: string, formula: string, description?: string) {
    const response = await this.client.post('/chat/pin-definition', {
      name,
      formula,
      description,
    });
    return response.data;
  }

  async getDefinitions() {
    const response = await this.client.get('/chat/definitions');
    return response.data;
  }

  // Analytics APIs
  async computeKPIs(metricColumn: string, dateColumn?: string, groupColumn?: string) {
    const response = await this.client.post('/analytics/kpi', {
      metric_column: metricColumn,
      date_column: dateColumn,
      group_column: groupColumn,
    });
    return response.data;
  }

  async analyzeContribution(metricColumn: string, segmentColumn: string, topN = 10) {
    const response = await this.client.post('/analytics/contribution', {
      metric_column: metricColumn,
      segment_column: segmentColumn,
      top_n: topN,
    });
    return response.data;
  }

  async analyzeTimeSeries(metricColumn: string, dateColumn: string, period = 'M') {
    const response = await this.client.post('/analytics/time-series', {
      metric_column: metricColumn,
      date_column: dateColumn,
      period,
    });
    return response.data;
  }

  async analyzeCohorts(userColumn: string, dateColumn: string, metricColumn?: string, period = 'M') {
    const response = await this.client.post('/analytics/cohort', {
      user_column: userColumn,
      date_column: dateColumn,
      metric_column: metricColumn,
      period,
    });
    return response.data;
  }

  async analyzeFunnel(stageColumn: string, stages: string[], countColumn?: string) {
    const response = await this.client.post('/analytics/funnel', {
      stage_column: stageColumn,
      stages,
      count_column: countColumn,
    });
    return response.data;
  }

  async detectAnomalies(metricColumn: string, dateColumn?: string, method = 'iqr', sensitivity = 1.5) {
    const response = await this.client.post('/analytics/anomalies', {
      metric_column: metricColumn,
      date_column: dateColumn,
      method,
      sensitivity,
    });
    return response.data;
  }

  async analyzeDrivers(targetColumn: string, featureColumns: string[], method = 'correlation') {
    const response = await this.client.post('/analytics/drivers', {
      target_column: targetColumn,
      feature_columns: featureColumns,
      method,
    });
    return response.data;
  }

  async getCorrelationMatrix(columns?: string[]) {
    const params = columns ? `?columns=${columns.join(',')}` : '';
    const response = await this.client.get(`/analytics/correlation-matrix${params}`);
    return response.data;
  }

  // Export APIs
  async exportChartPng(chartId: string, width = 1200, height = 800) {
    const response = await this.client.get(`/export/chart/${chartId}/png?width=${width}&height=${height}`, {
      responseType: 'blob',
    });
    return response.data;
  }

  async exportChartSvg(chartId: string) {
    const response = await this.client.get(`/export/chart/${chartId}/svg`, {
      responseType: 'blob',
    });
    return response.data;
  }

  async exportChartData(chartId: string) {
    const response = await this.client.get(`/export/chart/${chartId}/data`, {
      responseType: 'blob',
    });
    return response.data;
  }

  async exportTable(maxRows = 10000) {
    const response = await this.client.get(`/export/table?max_rows=${maxRows}`, {
      responseType: 'blob',
    });
    return response.data;
  }

  async generateReport(options?: { title?: string; format?: 'html' | 'markdown' }) {
    const response = await this.client.post('/export/report', {
      title: options?.title || 'Data Analysis Report',
      format: options?.format || 'html',
      include_insights: true,
      include_charts: true,
      include_tables: true,
    }, {
      responseType: 'blob',
    });
    return response.data;
  }

  async downloadBundle() {
    const response = await this.client.get('/export/bundle', {
      responseType: 'blob',
    });
    return response.data;
  }

  async listCharts() {
    const response = await this.client.get('/export/charts');
    return response.data;
  }
}

export const api = new ApiService();
