import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import ImportData from './pages/ImportData'
import Preprocessing from './pages/Preprocessing'
import Models from './pages/Models'
import Alerts from './pages/Alerts'
import Users from './pages/Users'
import Settings from './pages/Settings'

function App() {
  const isLogged = !!localStorage.getItem('user')
  if (!isLogged) {
    return <Login />
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
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/users" element={<Users />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </div>
  )
}

export default App
