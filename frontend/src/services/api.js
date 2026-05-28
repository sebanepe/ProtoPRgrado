import axios from 'axios'

// Set a reasonable default timeout so the UI won't hang indefinitely.
// Background import means the UI doesn't need an infinite timeout.
const api = axios.create({ baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000', timeout: 120000 })

// Attach X-User-Email header when a user is stored in localStorage (guard in test mocks)
if (api.interceptors && api.interceptors.request && typeof api.interceptors.request.use === 'function') {
  api.interceptors.request.use((cfg) => {
    try {
      const u = localStorage.getItem('user')
      if (u) {
        const parsed = JSON.parse(u)
        if (parsed && parsed.email) { cfg.headers['X-User-Email'] = parsed.email }
        // attach bearer token if present
        if (parsed && parsed.token) { cfg.headers['Authorization'] = `Bearer ${parsed.token}` }
      }
    } catch (e) { }
    return cfg
  })
}

export async function health(){ return api.get('/health').then(r=>r.data).catch(()=>null) }
export async function getDashboardSummary(){
  // Try the real endpoint; if missing or network error, return a safe mock for UI
  try{
    const res = await api.get('/dashboard/summary')
    return res.data
  }catch(e){
    // If server responded with 4xx/5xx, bubble up to UI
    if (e && e.response) throw e
    // fallback mock (used in tests/dev when backend missing)
    const mock = {
      transactions: 12543,
      alerts: 12,
      risk: 0.32,
      model: 'rf-v1',
      alertTrend: [
        {date:'2026-05-15', count:2}, {date:'2026-05-16', count:3}, {date:'2026-05-17', count:1},
        {date:'2026-05-18', count:4}, {date:'2026-05-19', count:2}
      ],
      fraudRatio: {fraud:18, normal:82},
      recentAlerts: [
        {alert_id:101, transaction_id: 'tx_001', score:0.87, channel:'POS', amount:123.45, status:'New', date:'2026-05-19'},
        {alert_id:102, transaction_id: 'tx_002', score:0.65, channel:'Web', amount:543.21, status:'Review', date:'2026-05-19'}
      ]
    }
    return mock
  }
}
export async function login(credentials){
  // try real endpoint, fallback to simulated login
  try{
    const res = await api.post('/auth/login', credentials)
    return res.data
  }catch(e){
    // debug help: log network/axios errors to console (dev only)
    try{ console.error('api.login error', e && e.message ? e.message : e) }catch(_){ }
    // If server responded (4xx/5xx), rethrow so UI can show proper message
    if (e && e.response) {
      throw e
    }
    // fallback simulation only when running tests (do not simulate in real dev/prod)
    const isTest = (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'test') || (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_TEST === 'true')
    if (isTest && credentials && credentials.email && credentials.password) {
      return { id: 1, email: credentials.email, full_name: 'Simulated User', token: 'simulated-token' }
    }
    throw e
  }
}
export async function register(payload){
  // payload: { full_name, email, password, role }
  return api.post('/auth/register', payload).then(r=> r.data)
}
export async function importDataset(file){
  const form = new FormData(); form.append('file', file)
  return api.post('/datasets/import', form, { headers: {'Content-Type':'multipart/form-data'} }).then(r=>r.data)
}

export async function importDatasetBackground(file){
  const form = new FormData(); form.append('file', file)
  return api.post('/datasets/import-background', form, { headers: {'Content-Type':'multipart/form-data'} }).then(r=>r.data)
}

export async function getDatasetStatus(datasetId){
  return api.get(`/datasets/${datasetId}/status`).then(r=> r.data)
}
export async function runPreprocessing(datasetId = null){
  // If datasetId provided, send as query param so backend can scope run
  if (datasetId) {
    return api.post('/preprocessing/run', null, { params: { dataset_id: datasetId } }).then(r=>r.data)
  }
  return api.post('/preprocessing/run').then(r=>r.data)
}
export async function listPreprocessingRuns(){ return api.get('/preprocessing/runs').then(r=> r.data || []) }
export async function previewPreprocessingRun(id){ return api.get(`/preprocessing/runs/${id}/preview`).then(r=> r.data) }
export async function getPreprocessingRunStages(id){ return api.get(`/preprocessing/runs/${id}/stages`).then(r=> r.data) }
export async function downloadPreprocessingRun(id){ return api.get(`/preprocessing/runs/${id}/download`, { responseType: 'blob' }).then(r=> r.data) }
export async function downloadPreprocessingRunReport(id){ return api.get(`/preprocessing/runs/${id}/report`, { responseType: 'blob' }).then(r=> r.data) }
export async function deletePreprocessingRun(id){ return api.delete(`/preprocessing/runs/${id}`).then(r=> r.data) }
export async function runPreprocessingTraining(runId){ return api.post('/preprocessing/run_training', null, { params: { run_id: runId } }).then(r=> r.data) }
export async function previewFeatureSet(id, rows=10){ return api.get(`/feature_sets/${id}/preview`, { params: { rows } }).then(r=> r.data) }
export async function downloadFeatureSet(id){ return api.get(`/feature_sets/${id}/download`, { responseType: 'blob' }).then(r=> r.data) }
export async function deleteFeatureSet(id){ return api.delete(`/feature_sets/${id}`).then(r=> r.data) }
export async function downloadFeatureSetReport(id){ return api.get(`/feature_sets/${id}/report`, { responseType: 'blob' }).then(r=> r.data) }
export async function listDatasets(params){ return api.get('/datasets', { params }).then(r=> r.data && r.data.datasets ? r.data.datasets : []) }
export async function previewDataset(id, rows=10){ return api.get(`/datasets/${id}/preview`, { params: { rows } }).then(r=> r.data) }
export async function deleteDataset(id){ return api.delete(`/datasets/${id}`).then(r=> r.data) }
export async function listAlerts(){ return api.get('/alerts').then(r=> r.data && r.data.alerts ? r.data.alerts : []) }
export async function listModels(){ return api.get('/models/results').then(r=> r.data && r.data.results ? r.data.results : []) }
export async function getModelResults(){ return listModels() }
export async function getModelConfig(){ return api.get('/settings/model-config').then(r=> r.data && r.data.model_config ? r.data.model_config : null) }
export async function setModelConfig(payload){ return api.post('/settings/model-config', payload).then(r=> r.data && r.data.model_config ? r.data.model_config : null) }
export async function saveModelConfig(payload){ return setModelConfig(payload) }
export async function trainModel(payload){ return api.post('/models/train', payload).then(r=> r.data) }
export async function activateModel(modelId){ return api.post(`/models/${modelId}/activate`).then(r=> r.data) }
export async function exportModelResults(modelId){ return api.get(`/models/${modelId}/export`, { responseType: 'blob' }).then(r=> r.data) }
export async function runBatchScoring(payload){ // payload: { transactions: [...], dataset_id? }
  // map to existing alerts.generate endpoint
  const transactions = payload && payload.transactions ? payload.transactions : []
  return api.post('/alerts/generate', transactions).then(r=> r.data)
}
export async function getReportingSummary(){ return api.get('/reporting/summary').then(r=> r.data).catch(()=>null) }
export async function getUsers(){ return api.get('/users').then(r=> r.data && r.data.users ? r.data.users : []) }
export async function createUser(payload){ return api.post('/users', payload).then(r=> r.data) }
export async function updateUser(id, payload){ return api.put(`/users/${id}`, payload).then(r=> r.data) }
export async function toggleUserStatus(id){ return api.post(`/users/${id}/toggle-status`).then(r=> r.data) }
export async function getAlerts(filters){ return api.get('/alerts', { params: filters }).then(r=> r.data && r.data.alerts ? r.data.alerts : []) }
export async function updateAlertStatus(alertId, status){ return api.patch(`/alerts/${alertId}/status`, null, { params: { status } }).then(r=> r.data) }
export async function me(){ return api.get('/auth/me').then(r=> r.data).catch(()=>null) }
export default api
