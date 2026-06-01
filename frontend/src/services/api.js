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
  return api.get('/dashboard/summary').then(r=> r.data)
}

export async function getDashboardOverview(params = {}){
  return api.get('/api/dashboard/overview', { params }).then(r=> r.data)
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
export async function runBatchScoring(payload) {
  // payload: { source_run, algorithm, input_dataset_path? }
  return api.post('/api/scoring/batch-run', payload).then(r => r.data)
}
export async function getBatchScoringRuns(params = {}) {
  return api.get('/api/scoring/runs', { params }).then(r => r.data)
}
export async function getBatchScoringRunById(id) {
  return api.get(`/api/scoring/runs/${id}`).then(r => r.data)
}
export async function getBatchScoringResults(params = {}) {
  return api.get('/api/scoring/results', { params }).then(r => r.data)
}
export async function getBatchScoringReport(params = {}) {
  return api.get('/api/scoring/report', { params }).then(r => r.data)
}
export async function getBatchScoringMetadata(params = {}) {
  return api.get('/api/scoring/metadata', { params }).then(r => r.data)
}
export async function getReportingSummary(){ return api.get('/reporting/summary').then(r=> r.data).catch(()=>null) }
export async function getUsers(){ return api.get('/users').then(r=> r.data && r.data.users ? r.data.users : []) }
export async function createUser(payload){ return api.post('/users', payload).then(r=> r.data) }
export async function updateUser(id, payload){ return api.put(`/users/${id}`, payload).then(r=> r.data) }
export async function toggleUserStatus(id){ return api.post(`/users/${id}/toggle-status`).then(r=> r.data) }
export async function getAlerts(filters){ return api.get('/alerts', { params: filters }).then(r=> r.data && r.data.alerts ? r.data.alerts : []) }
export async function updateAlertStatus(alertId, status){ return api.patch(`/alerts/${alertId}/status`, null, { params: { status } }).then(r=> r.data) }
export async function me(){ return api.get('/auth/me').then(r=> r.data).catch(()=>null) }

// ============================================================
// Phase B: Rules & Alerts API Functions
// ============================================================
export async function getPreprocessedRuns(){
  return api.get('/api/rules/preprocessed-runs').then(r=> r.data)
}

export async function analyzeRules(preprocessedRunId, force = false, config = {}){
  return api.post('/api/rules/analyze', {
    preprocessed_run_id: preprocessedRunId,
    force: force,
    config: config
  }).then(r=> r.data)
}

export async function getRulesSummary(runId, params = {}){
  const queryParams = { run_id: runId, ...params }
  return api.get('/api/rules/summary', { params: queryParams }).then(r=> r.data)
}

export async function getSummaryFilterOptions(runId){
  return api.get('/api/rules/summary-filter-options', { params: { run_id: runId } }).then(r=> r.data)
}

export async function getRulesAlerts(runId, params = {}){
  const queryParams = { run_id: runId, ...params }
  return api.get('/api/rules/alerts', { params: queryParams }).then(r=> r.data)
}

export async function getRuleAlertDetail(alertId, runId){
  return api.get(`/api/rules/alerts/${alertId}`, { params: { run_id: runId } }).then(r=> r.data)
}

export async function getCustomerCardLookup(customerHash){
  return api.get('/api/rules/customer-card-lookup', { params: { customer_hash: customerHash } }).then(r=> r.data)
}

export async function getRuleSummaryTransactions(runId, alertId){
  return api.get('/api/rules/summary-transactions', { params: { run_id: runId, alert_id: alertId } }).then(r=>r.data)
}

export async function getRulesReport(runId){
  return api.get('/api/rules/report', { params: { run_id: runId } }).then(r=> r.data)
}

export async function getRulesMetrics(runId){
  return api.get('/api/rules/metrics', { params: { run_id: runId } }).then(r=> r.data)
}

// ============================================================
// Phase C.4: Supervised Models Readiness API Functions
// ============================================================
export async function getHumanLabelSummary(sourceRun = null){
  const params = {}
  if (sourceRun != null && String(sourceRun).trim() !== '') params.source_run = String(sourceRun).trim()
  return api.get('/api/supervised/human-label-summary', { params }).then(r=> r.data)
}

export async function getHumanReadiness(sourceRun = null){
  const params = {}
  if (sourceRun != null && String(sourceRun).trim() !== '') params.source_run = String(sourceRun).trim()
  return api.get('/api/supervised/human-readiness', { params }).then(r=> r.data)
}

export async function getSupervisedTrainingPreflight(sourceRun){
  return api.get('/api/supervised/training-preflight', { params: { source_run: sourceRun } }).then(r=> r.data)
}

export async function buildHumanSupervisedDataset(sourceRun, options = {}){
  return api.post('/api/supervised/build-human-dataset', {
    source_run: sourceRun,
    force: Boolean(options.force)
  }).then(r=> r.data)
}

export async function trainHumanSupervisedModel(payload){
  return api.post('/api/supervised/train-human-model', payload).then(r=> r.data)
}

export async function getSupervisedTrainingRuns(sourceRun){
  return api.get('/api/supervised/training-runs', { params: { source_run: sourceRun } }).then(r=> r.data)
}

export async function getSupervisedModelMetadata(sourceRun, modelType){
  return api.get('/api/supervised/model-metadata', { params: { source_run: sourceRun, model_type: modelType } }).then(r=> r.data)
}

export async function getSupervisedModelReport(sourceRun, modelType){
  return api.get('/api/supervised/model-report', { params: { source_run: sourceRun, model_type: modelType } }).then(r=> r.data)
}

export async function getSupervisedModelPredictions(sourceRun, modelType, params = {}){
  return api.get('/api/supervised/model-predictions', { params: { source_run: sourceRun, model_type: modelType, ...params } }).then(r=> r.data)
}

export async function getHumanDatasetSummary(sourceRun){
  return api.get('/api/supervised/human-dataset-summary', { params: { source_run: sourceRun } }).then(r=> r.data)
}

export async function getHumanDatasetPreview(sourceRun, params = {}){
  return api.get('/api/supervised/human-dataset-preview', { params: { source_run: sourceRun, ...params } }).then(r=> r.data)
}

export async function validateHumanDataset(sourceRun){
  return api.get('/api/supervised/human-dataset-validate', { params: { source_run: sourceRun } }).then(r=> r.data)
}

// ============================================================
// PHASE B.3: Alert Review API Functions
// ============================================================

export async function updateAlertReviewStatus(alertId, runId, newStatus, analystNotes = null){
  return api.patch(`/api/rules/alerts/${alertId}/status`, {
    run_id: runId,
    new_status: newStatus,
    analyst_notes: analystNotes,
    reviewed_by: null
  }).then(r=> r.data)
}

export async function updateSummaryAlertReviewStatus(summaryAlertId, runId, newStatus, analystNotes = null){
  return api.patch(`/api/rules/summary/${summaryAlertId}/status`, {
    run_id: runId,
    new_status: newStatus,
    analyst_notes: analystNotes,
    reviewed_by: null
  }).then(r=> r.data)
}

export async function getAlertReviewHistory(alertId, runId){
  return api.get(`/api/rules/alerts/${alertId}/history`, {
    params: { run_id: runId }
  }).then(r=> r.data)
}

export async function getSummaryAlertReviewHistory(summaryAlertId, runId){
  return api.get(`/api/rules/summary/${summaryAlertId}/history`, {
    params: { run_id: runId }
  }).then(r=> r.data)
}

export async function getAlertReviews(runId, params = {}){
  const queryParams = { run_id: runId, ...params }
  return api.get('/api/rules/reviews', { params: queryParams }).then(r=> r.data)
}

// ============================================================
// Phase C.2/C.3: Unsupervised Anomaly API Functions
// ============================================================

export async function getAnomalyRuns(){
  return api.get('/api/anomaly/runs').then(r => r.data)
}

export async function getAnomalyMetrics(runId){
  return api.get('/api/anomaly/metrics', { params: { run_id: runId } }).then(r => r.data)
}

export async function getAnomalyScores(runId, params = {}){
  return api.get('/api/anomaly/scores', { params: { run_id: runId, ...params } }).then(r => r.data)
}

export async function getTopAnomalies(runId, limit = 20){
  return api.get('/api/anomaly/top', { params: { run_id: runId, limit } }).then(r => r.data)
}

export async function getAnomalyReport(runId){
  return api.get('/api/anomaly/report', { params: { run_id: runId } }).then(r => r.data)
}

export async function getAnomalyModelMetadata(runId){
  return api.get('/api/anomaly/model-metadata', { params: { run_id: runId } }).then(r => r.data)
}

export async function trainAnomalyModel(payload = {}){
  const params = {}
  if (payload.source_run != null && payload.source_run !== '') params.source_run = payload.source_run
  if (payload.model != null && payload.model !== '') params.model = payload.model
  if (payload.contamination != null && payload.contamination !== '') params.contamination = payload.contamination
  if (payload.sample_size != null && payload.sample_size !== '') params.sample_size = payload.sample_size
  if (payload.max_categories != null && payload.max_categories !== '') params.max_categories = payload.max_categories
  if (payload.n_estimators != null && payload.n_estimators !== '') params.n_estimators = payload.n_estimators
  return api.post('/api/anomaly/train', null, { params }).then(r => r.data)
}

export async function trainAutoencoderAnomaly(params = {}){
  return api.post('/api/anomaly/autoencoder/train', params).then(r => r.data)
}

export async function getAutoencoderMetrics(sourceRun){
  return api.get('/api/anomaly/autoencoder/metrics', { params: { source_run: sourceRun } }).then(r => r.data)
}

export async function getAutoencoderScores(sourceRun, params = {}){
  return api.get('/api/anomaly/autoencoder/scores', { params: { source_run: sourceRun, ...params } }).then(r => r.data)
}

export async function getAutoencoderReport(sourceRun){
  return api.get('/api/anomaly/autoencoder/report', { params: { source_run: sourceRun } }).then(r => r.data)
}

export async function getAutoencoderModelMetadata(sourceRun){
  return api.get('/api/anomaly/autoencoder/model-metadata', { params: { source_run: sourceRun } }).then(r => r.data)
}

// ============================================================
// Phase C5.2: Model Evaluation Comparison API Functions
// ============================================================
export async function buildModelEvaluationComparison(payload){
  return api.post('/api/model-evaluation/build-comparison', payload).then(r => r.data)
}

export async function getModelEvaluationSummary(sourceRun){
  return api.get('/api/model-evaluation/summary', { params: { source_run: sourceRun } }).then(r => r.data)
}

export async function getModelEvaluationAlertLevel(sourceRun, params = {}){
  return api.get('/api/model-evaluation/alert-level', { params: { source_run: sourceRun, ...params } }).then(r => r.data)
}

export async function getModelEvaluationTransactionLevel(sourceRun, params = {}){
  return api.get('/api/model-evaluation/transaction-level', { params: { source_run: sourceRun, ...params } }).then(r => r.data)
}

export async function getModelEvaluationReport(sourceRun){
  return api.get('/api/model-evaluation/report', { params: { source_run: sourceRun } }).then(r => r.data)
}

export async function getModelEvaluationMetadata(sourceRun){
  return api.get('/api/model-evaluation/metadata', { params: { source_run: sourceRun } }).then(r => r.data)
}

export async function getModelEvaluationTopCases(sourceRun, limit = 20){
  return api.get('/api/model-evaluation/top-cases', { params: { source_run: sourceRun, limit } }).then(r => r.data)
}

export default api
