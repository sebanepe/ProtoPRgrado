import React, { useEffect, useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import ImportData from './pages/ImportData'
import Preprocessing from './pages/Preprocessing'
import Models from './pages/Models'
import ModelTraining from './pages/ModelTraining'
import ModelEvaluation from './pages/ModelEvaluation'
import BatchScoring from './pages/BatchScoring'
import Cases from './pages/Cases'
import RulesAlerts from './pages/RulesAlerts'
import Users from './pages/Users'
import Settings from './pages/Settings'

function App() {
  // Start with login screen by default (avoid flashing dashboard).
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem('user') || 'null') } catch { return null }
  })
  const [checking, setChecking] = useState(true)

  useEffect(() => {
    let mounted = true
    async function validate() {
      const token = user && user.token
      if (!token) {
        setChecking(false)
        return
      }
      try {
        // lazy import to avoid circular deps in tests
        const { me } = await import('./services/api')
        const remote = await me()
        if (remote && mounted) {
          // keep token from storage
          const merged = Object.assign({}, remote, { token })
          localStorage.setItem('user', JSON.stringify(merged))
          setUser(merged)
        } else {
          localStorage.removeItem('user')
          setUser(null)
        }
      } catch (e) {
        localStorage.removeItem('user')
        setUser(null)
      } finally {
        if (mounted) setChecking(false)
      }
    }
    validate()
    return () => { mounted = false }
  }, [])

  // Ensure URL is root while unauthenticated or while checking token
  useEffect(() => {
    if (!user || checking) {
      try { window.history.replaceState({}, '', '/') } catch (e) { /* ignore */ }
    }
  }, [user, checking])

  // While validating token, show login to ensure initial screen is login.
  if (!user || checking) {
    return <Login checking={checking} />
  }

  return (
    <div className="app">
      <Sidebar />
      <div className="content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/import" element={<ImportData />} />
          <Route path="/preprocessing" element={<Preprocessing />} />
          <Route path="/models" element={<Models />} />
          <Route path="/models/training" element={<ModelTraining />} />
          <Route path="/models/evaluation" element={<ModelEvaluation />} />
          <Route path="/monitoring/scoring" element={<BatchScoring />} />
          <Route path="/monitoring/cases" element={<Cases />} />
          <Route path="/rules-alerts" element={<RulesAlerts />} />
          <Route path="/users" element={<Users />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </div>
  )
}

export default App
