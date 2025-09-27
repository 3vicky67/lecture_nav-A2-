import React, { useState, useEffect } from 'react';


interface EvaluationResult {
  video_id: string;
  query_count: number;
  avg_mrr_at_10: number;
  avg_precision_at_k: number;
  avg_latency: number;
  meets_mrr_target: boolean;
  meets_latency_target: boolean;
  overall_pass: boolean;
  detailed_results: any[];
}

interface AblationResult {
  video_id: string;
  test_queries: string[];
  configurations_tested: number;
  results: Record<string, any>;
}

interface TestSuiteResult {
  video_id: string;
  timestamp: number;
  tests: {
    mrr_evaluation: EvaluationResult | { error: string };
    latency_test: {
      p95_latency: number;
      meets_target: boolean;
      sample_count: number;
    };
    ablation_study: AblationResult;
  };
  overall_assessment: {
    mrr_test_passed: boolean;
    latency_test_passed: boolean;
    all_tests_passed: boolean;
    test_timestamp: number;
  };
}

const EvaluationDashboard: React.FC = () => {
  const [videoId, setVideoId] = useState('');
  const [goldDataset, setGoldDataset] = useState<Record<string, Record<string, number[]>>>({});
  const [evaluationResult, setEvaluationResult] = useState<EvaluationResult | null>(null);
  const [ablationResult, setAblationResult] = useState<AblationResult | null>(null);
  const [testSuiteResult, setTestSuiteResult] = useState<TestSuiteResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [newAnnotation, setNewAnnotation] = useState({ query: '', timestamps: '' });

  // Load gold dataset on component mount
  useEffect(() => {
    loadGoldDataset();
  }, []);

  const loadGoldDataset = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/evaluation/gold_dataset');
      const data = await response.json();
      setGoldDataset(data.dataset || {});
    } catch (error) {
      console.error('Error loading gold dataset:', error);
    }
  };

  const addGoldAnnotation = async () => {
    if (!videoId || !newAnnotation.query || !newAnnotation.timestamps) {
      alert('Please fill in all fields');
      return;
    }

    const timestamps = newAnnotation.timestamps
      .split(',')
      .map(t => parseFloat(t.trim()))
      .filter(t => !isNaN(t));

    if (timestamps.length === 0) {
      alert('Please enter valid timestamps (comma-separated numbers)');
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/api/evaluation/gold_dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: videoId,
          query: newAnnotation.query,
          timestamps: timestamps
        })
      });

      if (response.ok) {
        setNewAnnotation({ query: '', timestamps: '' });
        loadGoldDataset();
        alert('Annotation added successfully!');
      } else {
        const error = await response.json();
        alert(`Error: ${error.error}`);
      }
    } catch (error) {
      console.error('Error adding annotation:', error);
      alert('Error adding annotation');
    }
  };

  const runEvaluation = async () => {
    if (!videoId) {
      alert('Please enter a video ID');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/evaluation/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId })
      });

      const result = await response.json();
      setEvaluationResult(result);
    } catch (error) {
      console.error('Error running evaluation:', error);
      alert('Error running evaluation');
    } finally {
      setLoading(false);
    }
  };

  const runAblationStudy = async () => {
    if (!videoId) {
      alert('Please enter a video ID');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/evaluation/ablation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          video_id: videoId,
          window_sizes: [30, 45, 60],
          overlap_ratios: [0.25, 0.33, 0.5]
        })
      });

      const result = await response.json();
      setAblationResult(result);
    } catch (error) {
      console.error('Error running ablation study:', error);
      alert('Error running ablation study');
    } finally {
      setLoading(false);
    }
  };

  const runFullTestSuite = async () => {
    if (!videoId) {
      alert('Please enter a video ID');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/evaluation/test_suite', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId })
      });

      const result = await response.json();
      setTestSuiteResult(result);
    } catch (error) {
      console.error('Error running test suite:', error);
      alert('Error running test suite');
    } finally {
      setLoading(false);
    }
  };

  const formatTimestamp = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="evaluation-dashboard">
      <h2>🧪 Evaluation Dashboard</h2>
      
      {/* Video ID Input */}
      <div className="input-section">
        <label>
          Video ID:
          <input
            type="text"
            value={videoId}
            onChange={(e) => setVideoId(e.target.value)}
            placeholder="Enter video ID to test"
          />
        </label>
      </div>

      {/* Gold Dataset Management */}
      <div className="section">
        <h3>📊 Gold Dataset Management</h3>
        
        <div className="add-annotation">
          <h4>Add New Annotation</h4>
          <div className="form-group">
            <input
              type="text"
              placeholder="Query (e.g., 'explain machine learning')"
              value={newAnnotation.query}
              onChange={(e) => setNewAnnotation({...newAnnotation, query: e.target.value})}
            />
            <input
              type="text"
              placeholder="Timestamps (e.g., '120,180,240')"
              value={newAnnotation.timestamps}
              onChange={(e) => setNewAnnotation({...newAnnotation, timestamps: e.target.value})}
            />
            <button onClick={addGoldAnnotation}>Add Annotation</button>
          </div>
        </div>

        <div className="current-dataset">
          <h4>Current Gold Dataset</h4>
          {Object.keys(goldDataset).length === 0 ? (
            <p>No annotations yet. Add some to get started!</p>
          ) : (
            <div className="dataset-list">
              {Object.entries(goldDataset).map(([vid, queries]) => (
                <div key={vid} className="video-annotations">
                  <h5>Video: {vid}</h5>
                  {Object.entries(queries).map(([query, timestamps]) => (
                    <div key={query} className="annotation">
                      <strong>"{query}"</strong> → {timestamps.map(t => formatTimestamp(t)).join(', ')}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Test Controls */}
      <div className="section">
        <h3>🚀 Run Tests</h3>
        <div className="test-buttons">
          <button onClick={runEvaluation} disabled={loading}>
            {loading ? 'Running...' : 'Run MRR@10 Evaluation'}
          </button>
          <button onClick={runAblationStudy} disabled={loading}>
            {loading ? 'Running...' : 'Run Ablation Study'}
          </button>
          <button onClick={runFullTestSuite} disabled={loading} className="primary">
            {loading ? 'Running...' : 'Run Full Test Suite'}
          </button>
        </div>
      </div>

      {/* Results Display */}
      {evaluationResult && (
        <div className="section results">
          <h3>📈 MRR@10 Evaluation Results</h3>
          <div className="metrics-grid">
            <div className="metric">
              <span className="label">Average MRR@10:</span>
              <span className={`value ${evaluationResult.meets_mrr_target ? 'pass' : 'fail'}`}>
                {evaluationResult.avg_mrr_at_10.toFixed(3)}
              </span>
            </div>
            <div className="metric">
              <span className="label">Average Precision@K:</span>
              <span className="value">{evaluationResult.avg_precision_at_k.toFixed(3)}</span>
            </div>
            <div className="metric">
              <span className="label">Average Latency:</span>
              <span className={`value ${evaluationResult.meets_latency_target ? 'pass' : 'fail'}`}>
                {evaluationResult.avg_latency.toFixed(3)}s
              </span>
            </div>
            <div className="metric">
              <span className="label">Overall Pass:</span>
              <span className={`value ${evaluationResult.overall_pass ? 'pass' : 'fail'}`}>
                {evaluationResult.overall_pass ? '✅ PASS' : '❌ FAIL'}
              </span>
            </div>
          </div>
        </div>
      )}

      {ablationResult && (
        <div className="section results">
          <h3>🔬 Ablation Study Results</h3>
          <div className="ablation-results">
            <p>Tested {ablationResult.configurations_tested} configurations</p>
            <div className="config-comparison">
              {Object.entries(ablationResult.results).map(([config, data]) => (
                <div key={config} className="config-result">
                  <h4>{config}</h4>
                  {data.error ? (
                    <p className="error">Error: {data.error}</p>
                  ) : (
                    <div className="config-metrics">
                      <p>Window: {data.window_size}s, Overlap: {data.overlap_seconds}s</p>
                      <p>Segments: {data.segments_count}</p>
                      <p>Avg Latency: {data.avg_latency.toFixed(3)}s</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {testSuiteResult && (
        <div className="section results">
          <h3>🎯 Full Test Suite Results</h3>
          <div className="test-summary">
            <div className="overall-status">
              <h4>Overall Assessment</h4>
              <div className={`status ${testSuiteResult.overall_assessment.all_tests_passed ? 'pass' : 'fail'}`}>
                {testSuiteResult.overall_assessment.all_tests_passed ? '✅ ALL TESTS PASSED' : '❌ SOME TESTS FAILED'}
              </div>
            </div>
            
            <div className="test-details">
              <div className="test-item">
                <span className="test-name">MRR@10 Test:</span>
                <span className={`test-status ${testSuiteResult.overall_assessment.mrr_test_passed ? 'pass' : 'fail'}`}>
                  {testSuiteResult.overall_assessment.mrr_test_passed ? 'PASS' : 'FAIL'}
                </span>
              </div>
              <div className="test-item">
                <span className="test-name">P95 Latency Test:</span>
                <span className={`test-status ${testSuiteResult.overall_assessment.latency_test_passed ? 'pass' : 'fail'}`}>
                  {testSuiteResult.overall_assessment.latency_test_passed ? 'PASS' : 'FAIL'}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EvaluationDashboard;
