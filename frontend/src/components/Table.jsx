import React from 'react'

export default function Table({columns, data=[]}){
  return (
    <div className="card table-card">
      <div className="table-scroll">
        <table className="table">
          <thead>
            <tr>{columns.map(c=> <th key={c.key}>{c.title}</th>)}</tr>
          </thead>
          <tbody>
            {data.map((row, i)=> (
              <tr key={i}>
                {columns.map(c=> <td key={c.key}>{row[c.key]}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
