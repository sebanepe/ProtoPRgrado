import React from 'react'
export default function KPICard({title, value}){
  return (
    <div className="card">
      <div style={{fontSize:12, color:'#6b7280'}}>{title}</div>
      <div style={{fontSize:20, marginTop:6}}>{value}</div>
    </div>
  )
}
