import React from 'react'
export default function KPICard({title, value}){
  return (
    <div className="card metric-card">
      <div className="metric-label">{title}</div>
      <div className="metric-value">{value}</div>
    </div>
  )
}
