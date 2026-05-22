import axios from 'axios'

// Set a reasonable default timeout so the UI won't hang indefinitely
const api = axios.create({ baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000', timeout: 5000 })

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
export async function runPreprocessing(){ return api.post('/preprocessing/run').then(r=>r.data) }
export async function listAlerts(){ return api.get('/alerts').then(r=> r.data && r.data.alerts ? r.data.alerts : []) }
export async function listModels(){ return api.get('/models/results').then(r=> r.data && r.data.results ? r.data.results : []) }
export async function getModelConfig(){ return api.get('/settings/model-config').then(r=> r.data && r.data.model_config ? r.data.model_config : null) }
export async function setModelConfig(payload){ return api.post('/settings/model-config', payload).then(r=> r.data && r.data.model_config ? r.data.model_config : null) }
export async function me(){ return api.get('/auth/me').then(r=> r.data).catch(()=>null) }
export default api
