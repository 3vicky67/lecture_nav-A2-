// Results.tsx (Modified)
"use client"

import type React from "react"
import { useCallback } from "react"

// --- Interface Definitions ---
interface Result {
  video_id?: string
  video_file?: string
  t_start?: number  // Raw seconds
  t_end?: number    // Raw seconds
  start_time: string  // Formatted MM:SS
  end_time: string    // Formatted MM:SS
  title?: string
  snippet?: string
  snippets?: string  // Alternative field name from backend
  score: number
}

interface Props {
  results: Result[]
  aiAnswer?: string | null
  // The signature is now simple: pass the start and end times in seconds
  playSegment: (startSeconds: number, endSeconds: number) => void 
}

// --- Utility Function to Parse Time ---
const parseTimeFormat = (formatted: string): number => {
    const s = formatted.trim()
    // Support HH:MM:SS and MM:SS
    if (s.includes(':')) {
      const parts = s.split(':').map(p => parseInt(p, 10))
      if (parts.some(Number.isNaN)) return 0
      if (parts.length === 3) {
        const [hh, mm, ss] = parts
        return hh * 3600 + mm * 60 + ss
      }
      if (parts.length === 2) {
        const [mm, ss] = parts
        return mm * 60 + ss
      }
    }
    if (s.endsWith('s')) {
      return parseFloat(s.replace('s', '')) || 0
    }
    if (s.endsWith('m')) {
      const [m, sec] = s.replace('m', '').split('.')
      const mins = parseInt(m || '0', 10)
      const secs = parseInt(sec || '0', 10)
      return mins * 60 + secs
    }
    const n = parseFloat(s)
    return Number.isFinite(n) ? n : 0
}


const Results: React.FC<Props> = ({ results, aiAnswer, playSegment }) => {

  // Function to resolve start/end seconds from a result robustly
  const resolveStartEnd = (res: Result): { start: number; end: number } => {
    if (typeof res.t_start === 'number' && typeof res.t_end === 'number') {
      return { start: res.t_start, end: res.t_end }
    }
    const start = parseTimeFormat(res.start_time)
    const end = parseTimeFormat(res.end_time)
    return { start, end }
  }

  const handlePlayResult = useCallback((res: Result) => {
    const { start, end } = resolveStartEnd(res)
    playSegment(start, end)
  }, [playSegment])

  return (
    <div className="results">
      <h2>Search Results</h2>
      
      {/* AI Answer Section */}
      {aiAnswer && (
        <div className="ai-answer-section" style={{
          backgroundColor: '#1a365d',
          border: '1px solid #2d3748',
          borderRadius: '8px',
          padding: '16px',
          marginBottom: '20px',
          borderLeft: '4px solid #4299e1'
        }}>
          <h3 style={{ 
            color: '#63b3ed', 
            margin: '0 0 8px 0', 
            fontSize: '16px',
            fontWeight: '600'
          }}>
            💡 AI Answer:
          </h3>
          <p style={{ 
            color: '#e2e8f0', 
            margin: 0, 
            lineHeight: '1.5',
            fontSize: '14px'
          }}>
            {aiAnswer}
          </p>
        </div>
      )}
      
      {results.length === 0 && <p>No results found</p>}
      {results.map((res, idx) => (
        <div key={idx} className="result-card">
          <div className="result-header">
            <h3>{res.title || `Segment ${idx + 1}`}</h3>
            <span className="relevance-score">{(res.score * 100).toFixed(1)}% match</span>
          </div>

          {/* Metadata: video identifiers */}
          <div className="result-meta" style={{ marginBottom: 8, fontSize: 13, color: '#4a5568' }}>
            <div><strong>Video ID:</strong> {res.video_id ?? '—'}</div>
            <div><strong>Video File:</strong> {res.video_file ?? '—'}</div>
          </div>
          
          <div className="timestamp-info">
            <button
              className="timestamp clickable"
              onClick={() => handlePlayResult(res)}
              title={`Jump to ${res.start_time}`}
            >
              {res.start_time} - {res.end_time}
            </button>
          </div>
          
          <div className="snippet-text">
            <p>{res.snippet || res.snippets || 'No preview available'}</p>
          </div>
          
          <button
            className="play-segment-button"
            onClick={() => handlePlayResult(res)}
          >
            ▶️ Play Segment ({res.start_time} - {res.end_time})
          </button>
        </div>
      ))}
    </div>
  )
}

export default Results