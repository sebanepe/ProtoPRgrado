import axios from 'axios'

const api = axios.create({ baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000' })

export async function health(){ return api.get('/health').then(r=>r.data).catch(()=>null) }
export async function importDataset(file){
  const form = new FormData(); form.append('file', file)
  return api.post('/datasets/import', form, { headers: {'Content-Type':'multipart/form-data'} }).then(r=>r.data)
}
export async function runPreprocessing(){ return api.post('/preprocessing/run').then(r=>r.data) }
export async function listAlerts(){ return api.get('/alerts').then(r=> r.data && r.data.alerts ? r.data.alerts : []) }
export async function listModels(){ return api.get('/models/results').then(r=> r.data && r.data.results ? r.data.results : []) }
export async function getModelConfig(){ return api.get('/settings/model-config').then(r=> r.data && r.data.model_config ? r.data.model_config : null) }
export async function setModelConfig(payload){ return api.post('/settings/model-config', payload).then(r=> r.data && r.data.model_config ? r.data.model_config : null) }
export default api
