import React, {useEffect, useState} from 'react'
import Table from '../components/Table'
import { listModels } from '../services/api'

export default function Models(){
  const [models,setModels] = useState([])
  useEffect(()=>{ listModels().then(m=> setModels(m||[])).catch(()=>{}) },[])
  const columns = [
    {key:'model_name', title:'Model'},
    {key:'accuracy', title:'Accuracy'},
    {key:'precision', title:'Precision'},
    {key:'recall', title:'Recall'},
    {key:'f1_score', title:'F1'},
    {key:'roc_auc', title:'ROC AUC'}
  ]
  return (
    <div>
      <div className="header"><h2>Models</h2></div>
      <Table columns={columns} data={models} />
    </div>
  )
}
