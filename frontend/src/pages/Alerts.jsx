import React, {useEffect, useState} from 'react'
import Table from '../components/Table'
import { listAlerts } from '../services/api'

export default function Alerts(){
  const [alerts, setAlerts] = useState([])
  useEffect(()=>{ listAlerts().then(a=> setAlerts(a||[])).catch(()=>{}) },[])
  const columns = [
    {key:'id', title:'ID'},
    {key:'transaction_id', title:'Transaction'},
    {key:'risk_score', title:'Risk'},
    {key:'risk_level', title:'Level'},
    {key:'status', title:'Status'}
  ]
  return (
    <div>
      <div className="header"><h2>Alerts</h2></div>
      <Table columns={columns} data={alerts} />
    </div>
  )
}
