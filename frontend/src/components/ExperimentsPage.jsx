import React, { useState, useEffect, useMemo } from 'react'
import {
  Activity,
  AlertCircle,
  BarChart2,
  BarChart3,
  Brain,
  CheckCircle2,
  ChevronDown,
  Database,
  Download,
  Filter,
  Layers,
  LineChart,
  Microscope,
  PlayCircle,
  RefreshCw,
  Sliders,
  Sparkles,
  TrendingUp,
  Trophy,
  Zap,
} from 'lucide-react'

const API_BASE = '' // Proxied to backend port 8000

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState([])
  const [summary, setSummary] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [datasetsInfo, setDatasetsInfo] = useState(null)
  const [selectedDatasetTab, setSelectedDatasetTab] = useState('clinical_oral') // 'clinical_oral' | 'ndu_ufes_histo'
  const [selectedExperimentId, setSelectedExperimentId] = useState('')
  const [selectedHistory, setSelectedHistory] = useState(null)
  const [selectedConfusionMatrix, setSelectedConfusionMatrix] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)

  // Fetch all experiment data
  const fetchData = async () => {
    try {
      setErrorMsg(null)
      const [resExps, resSum, resMet, resData] = await Promise.all([
        fetch(`${API_BASE}/api/experiments`).then((r) => r.json()),
        fetch(`${API_BASE}/api/experiments/summary`).then((r) => r.json()),
        fetch(`${API_BASE}/api/experiments/metrics`).then((r) => r.json()),
        fetch(`${API_BASE}/api/experiments/datasets`).then((r) => r.json()),
      ])

      setExperiments(resExps.experiments || [])
      setSummary(resSum || null)
      setMetrics(resMet || null)
      setDatasetsInfo(resData || null)

      if (resExps.experiments && resExps.experiments.length > 0) {
        // Select best experiment by default or the first one
        const bestId = resSum?.best_experiment?.experiment_id || resExps.experiments[0].experiment_id
        setSelectedExperimentId(bestId)
        fetchExperimentDetails(bestId)
      }
    } catch (err) {
      console.error('Error fetching experiments data:', err)
      setErrorMsg('Failed to load experiment data from backend. Please ensure the backend server is running.')
    } finally {
      setIsLoading(false)
      setIsRefreshing(false)
    }
  }

  // Fetch details (history + confusion matrix) for a specific experiment
  const fetchExperimentDetails = async (expId) => {
    if (!expId) return
    try {
      const [resHist, resCm] = await Promise.all([
        fetch(`${API_BASE}/api/experiments/history/${expId}`).then((r) => (r.ok ? r.json() : null)),
        fetch(`${API_BASE}/api/experiments/confusion-matrix/${expId}`).then((r) => (r.ok ? r.json() : null)),
      ])
      setSelectedHistory(resHist)
      setSelectedConfusionMatrix(resCm)
    } catch (err) {
      console.warn(`Error loading details for ${expId}:`, err)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  const handleSelectExperiment = (expId) => {
    setSelectedExperimentId(expId)
    fetchExperimentDetails(expId)
  }

  const currentDataset = datasetsInfo ? datasetsInfo[selectedDatasetTab] : null

  // Format helpers
  const fmtPct = (val) => (val !== undefined && val !== null ? `${(val * 100).toFixed(1)}%` : 'N/A')
  const fmtNum = (val) => (val !== undefined && val !== null ? Number(val).toFixed(4) : 'N/A')

  return (
    <div className="space-y-8 pb-16 animate-fadeIn">
      {/* Page Header */}
      <div className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-bl from-teal-100/40 via-cyan-50/20 to-transparent rounded-full -mr-20 -mt-20 pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-50 border border-teal-200 text-teal-700 text-xs font-semibold mb-3">
              <Sparkles className="w-3.5 h-3.5" />
              Academic Research & Experimentation Suite
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
              Model Experiments & Results
            </h1>
            <p className="text-slate-600 text-sm sm:text-base mt-2 max-w-3xl">
              Comparative analysis of deep convolutional neural network architectures, hyperparameter sweeps (epochs,
              batch size, learning rate), and dataset stratification protocols for early oral cancer detection.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                setIsRefreshing(true)
                fetchData()
              }}
              disabled={isRefreshing}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 font-medium text-sm transition-all shadow-sm active:scale-95"
            >
              <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin text-teal-600' : ''}`} />
              Refresh Results
            </button>
            <a
              href="/results/experiments.csv"
              download="oral_cancer_experiments.csv"
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-teal-600 hover:bg-teal-700 text-white font-medium text-sm transition-all shadow-sm shadow-teal-600/20 active:scale-95"
            >
              <Download className="w-4 h-4" />
              Export CSV
            </a>
          </div>
        </div>

        {errorMsg && (
          <div className="mt-4 p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 text-sm flex items-center gap-3">
            <AlertCircle className="w-5 h-5 flex-shrink-0 text-amber-600" />
            <span>{errorMsg}</span>
          </div>
        )}
      </div>

      {/* SECTION 8: BEST EXPERIMENT SPOTLIGHT (Presented Prominently at Top) */}
      {summary?.best_experiment && (
        <section className="bg-gradient-to-r from-teal-900 via-slate-900 to-indigo-950 text-white rounded-2xl p-6 sm:p-8 shadow-xl relative overflow-hidden border border-teal-500/20">
          <div className="absolute top-0 right-0 w-80 h-80 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />
          <div className="relative z-10">
            <div className="flex items-center justify-between flex-wrap gap-4 border-b border-white/10 pb-4 mb-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-amber-400/20 text-amber-300 flex items-center justify-center border border-amber-400/30">
                  <Trophy className="w-5 h-5" />
                </div>
                <div>
                  <div className="text-xs font-bold uppercase tracking-wider text-teal-400">
                    Faculty Benchmark Spotlight
                  </div>
                  <h2 className="text-xl font-bold text-white">
                    Highest Performing Architecture (Test Set Evaluation)
                  </h2>
                </div>
              </div>
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-white/10 text-white border border-white/20">
                Experiment ID: {summary.best_experiment.experiment_id}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-white/10">
                <div className="text-xs text-slate-400 font-medium">Model Architecture</div>
                <div className="text-lg font-bold text-teal-300 mt-1">{summary.best_experiment.model}</div>
                <div className="text-[11px] text-slate-400 mt-0.5">{summary.best_experiment.trainable_parameters?.toLocaleString()} params</div>
              </div>

              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-white/10">
                <div className="text-xs text-slate-400 font-medium">Test Accuracy</div>
                <div className="text-2xl font-black text-emerald-400 mt-1">
                  {fmtPct(summary.best_experiment.test_accuracy)}
                </div>
                <div className="text-[11px] text-slate-400 mt-0.5">Held-out test set</div>
              </div>

              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-white/10">
                <div className="text-xs text-slate-400 font-medium">Precision (Weighted)</div>
                <div className="text-lg font-bold text-white mt-1">{fmtNum(summary.best_experiment.precision)}</div>
                <div className="text-[11px] text-slate-400 mt-0.5">True positive rate</div>
              </div>

              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-white/10">
                <div className="text-xs text-slate-400 font-medium">Recall / Sensitivity</div>
                <div className="text-lg font-bold text-white mt-1">{fmtNum(summary.best_experiment.recall)}</div>
                <div className="text-[11px] text-slate-400 mt-0.5">Malignancy detection</div>
              </div>

              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-white/10">
                <div className="text-xs text-slate-400 font-medium">F1-Score</div>
                <div className="text-lg font-bold text-cyan-300 mt-1">{fmtNum(summary.best_experiment.f1_score)}</div>
                <div className="text-[11px] text-slate-400 mt-0.5">Harmonic balance</div>
              </div>

              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-white/10">
                <div className="text-xs text-slate-400 font-medium">Optimal Hyperparams</div>
                <div className="text-sm font-semibold text-slate-200 mt-1">
                  {summary.best_experiment.epochs} Ep / BS {summary.best_experiment.batch_size}
                </div>
                <div className="text-[11px] text-teal-400 mt-0.5">LR: {summary.best_experiment.learning_rate}</div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* SECTION 1: DATASET OVERVIEW */}
      <section className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
          <div>
            <div className="flex items-center gap-2 text-teal-700 font-bold text-xs uppercase tracking-wider">
              <Database className="w-4 h-4" /> Section 1
            </div>
            <h2 className="text-xl font-bold text-slate-900 mt-1">Dataset Overview & Stratification</h2>
          </div>

          {/* Dataset Switcher */}
          <div className="flex items-center bg-slate-100 p-1 rounded-xl">
            <button
              onClick={() => setSelectedDatasetTab('clinical_oral')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                selectedDatasetTab === 'clinical_oral'
                  ? 'bg-white text-teal-800 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Clinical Oral Images (1,081)
            </button>
            <button
              onClick={() => setSelectedDatasetTab('ndu_ufes_histo')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                selectedDatasetTab === 'ndu_ufes_histo'
                  ? 'bg-white text-teal-800 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              NDU-UFES Histopathology (237)
            </button>
          </div>
        </div>

        {currentDataset ? (
          <div className="space-y-6">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                <div className="text-xs font-medium text-slate-500">Total Images</div>
                <div className="text-2xl font-bold text-slate-900 mt-1">{currentDataset.total_images}</div>
                <div className="text-[11px] text-slate-400 mt-1">{currentDataset.modality}</div>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                <div className="text-xs font-medium text-slate-500">Number of Classes</div>
                <div className="text-2xl font-bold text-teal-700 mt-1">{currentDataset.num_classes}</div>
                <div className="text-[11px] text-slate-400 mt-1">{currentDataset.classes?.join(' vs ')}</div>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                <div className="text-xs font-medium text-slate-500">Training Samples</div>
                <div className="text-2xl font-bold text-slate-900 mt-1">{currentDataset.training_samples}</div>
                <div className="text-[11px] text-emerald-600 mt-1">
                  {((currentDataset.training_samples / currentDataset.total_images) * 100).toFixed(0)}% partition
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                <div className="text-xs font-medium text-slate-500">Validation Samples</div>
                <div className="text-2xl font-bold text-slate-900 mt-1">{currentDataset.validation_samples}</div>
                <div className="text-[11px] text-slate-400 mt-1">Tuning & Early Stopping</div>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                <div className="text-xs font-medium text-slate-500">Test Samples</div>
                <div className="text-2xl font-bold text-slate-900 mt-1">{currentDataset.test_samples}</div>
                <div className="text-[11px] text-rose-600 mt-1">Zero-leakage held-out</div>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                <div className="text-xs font-medium text-slate-500">Models Evaluated</div>
                <div className="text-2xl font-bold text-indigo-600 mt-1">{summary?.num_models_tested || 5}</div>
                <div className="text-[11px] text-slate-400 mt-1">Deep Learning CNNs</div>
              </div>
            </div>

            {/* Class Distribution Visual Bars */}
            <div className="p-5 rounded-xl bg-slate-50 border border-slate-200 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-slate-800">Class Distribution Breakdown</h3>
                {currentDataset.patient_level_splitting_enforced && (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    Patient-Level Split Enforced (Zero Leakage)
                  </span>
                )}
              </div>

              <div className="space-y-3">
                {Object.entries(currentDataset.class_counts || {}).map(([cname, count]) => {
                  const pct = currentDataset.class_percentages?.[cname] || 0
                  return (
                    <div key={cname} className="space-y-1">
                      <div className="flex items-center justify-between text-xs font-medium">
                        <span className="text-slate-700">{cname}</span>
                        <span className="text-slate-500">
                          {count} images ({pct}%)
                        </span>
                      </div>
                      <div className="w-full h-2.5 rounded-full bg-slate-200 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            cname.toLowerCase().includes('cancer') || cname.toLowerCase().includes('oscc')
                              ? 'bg-rose-500'
                              : 'bg-teal-500'
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-slate-400">Loading dataset statistics...</div>
        )}
      </section>

      {/* SECTION 2: CNN MODEL COMPARISON TABLE */}
      <section className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
          <div>
            <div className="flex items-center gap-2 text-teal-700 font-bold text-xs uppercase tracking-wider">
              <Layers className="w-4 h-4" /> Section 2
            </div>
            <h2 className="text-xl font-bold text-slate-900 mt-1">CNN Model Architecture Comparison</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Empirical metrics obtained from training and held-out test evaluation.
            </p>
          </div>
        </div>

        <div className="overflow-x-auto rounded-xl border border-slate-200">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 text-slate-700 text-xs font-semibold uppercase tracking-wider border-b border-slate-200">
              <tr>
                <th className="py-3.5 px-4">Architecture</th>
                <th className="py-3.5 px-3">Parameters</th>
                <th className="py-3.5 px-3">Epochs</th>
                <th className="py-3.5 px-3">Batch Size</th>
                <th className="py-3.5 px-3">Test Accuracy</th>
                <th className="py-3.5 px-3">Precision</th>
                <th className="py-3.5 px-3">Recall</th>
                <th className="py-3.5 px-3">F1-Score</th>
                <th className="py-3.5 px-3">ROC-AUC</th>
                <th className="py-3.5 px-3 text-right">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-800">
              {experiments.length === 0 ? (
                <tr>
                  <td colSpan={10} className="py-8 text-center text-slate-400">
                    No experiments recorded yet. Run `python -m training.train` to generate benchmark metrics.
                  </td>
                </tr>
              ) : (
                experiments.map((exp) => {
                  const isBest = summary?.best_experiment?.experiment_id === exp.experiment_id
                  return (
                    <tr
                      key={exp.experiment_id}
                      className={`hover:bg-slate-50/80 transition-colors cursor-pointer ${
                        selectedExperimentId === exp.experiment_id ? 'bg-teal-50/50' : ''
                      }`}
                      onClick={() => handleSelectExperiment(exp.experiment_id)}
                    >
                      <td className="py-3.5 px-4 font-semibold text-slate-900 flex items-center gap-2">
                        {isBest && <Trophy className="w-4 h-4 text-amber-500 flex-shrink-0" />}
                        <span>{exp.model}</span>
                        <span className="text-[10px] text-slate-400 font-mono">({exp.experiment_id})</span>
                      </td>
                      <td className="py-3.5 px-3 font-mono text-xs text-slate-600">
                        {exp.trainable_parameters ? `${(exp.trainable_parameters / 1e6).toFixed(2)}M` : 'N/A'}
                      </td>
                      <td className="py-3.5 px-3 text-slate-600">
                        {exp.actual_epochs ? `${exp.actual_epochs} / ${exp.epochs}` : exp.epochs}
                      </td>
                      <td className="py-3.5 px-3 text-slate-600">{exp.batch_size}</td>
                      <td className="py-3.5 px-3 font-bold text-emerald-700">{fmtPct(exp.test_accuracy)}</td>
                      <td className="py-3.5 px-3 text-slate-700 font-mono text-xs">{fmtNum(exp.precision)}</td>
                      <td className="py-3.5 px-3 text-slate-700 font-mono text-xs">{fmtNum(exp.recall)}</td>
                      <td className="py-3.5 px-3 text-slate-700 font-mono text-xs font-semibold">{fmtNum(exp.f1_score)}</td>
                      <td className="py-3.5 px-3 text-slate-700 font-mono text-xs">{fmtNum(exp.auc)}</td>
                      <td className="py-3.5 px-3 text-right font-mono text-xs text-slate-500">
                        {exp.training_time_seconds ? `${exp.training_time_seconds}s` : 'N/A'}
                      </td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* SECTION 3 & 4: CHARTS (ACCURACY & PRECISION/RECALL/F1) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* SECTION 3 — Model vs Test Accuracy Bar Chart */}
        <section className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div>
              <div className="flex items-center gap-2 text-teal-700 font-bold text-xs uppercase tracking-wider">
                <BarChart3 className="w-4 h-4" /> Section 3
              </div>
              <h3 className="text-lg font-bold text-slate-900 mt-0.5">Model vs Test Accuracy</h3>
            </div>
            <span className="text-xs text-slate-400">Held-out test set</span>
          </div>

          {metrics?.model_comparison && metrics.model_comparison.length > 0 ? (
            <div className="space-y-4 pt-2">
              {metrics.model_comparison.map((item) => {
                const acc = item.test_accuracy || 0
                const widthPct = Math.min(100, Math.max(10, acc * 100))
                return (
                  <div key={item.model} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-slate-800">{item.model}</span>
                      <span className="font-bold text-teal-700">{(acc * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full h-7 bg-slate-100 rounded-lg overflow-hidden flex items-center px-2">
                      <div
                        className="h-5 rounded-md bg-gradient-to-r from-teal-500 to-cyan-500 transition-all duration-700 flex items-center justify-end pr-2 text-[10px] text-white font-bold"
                        style={{ width: `${widthPct}%` }}
                      >
                        {acc >= 0.3 && `${(acc * 100).toFixed(1)}%`}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="py-12 text-center text-slate-400 text-sm">No comparison data available yet.</div>
          )}
        </section>

        {/* SECTION 4 — Precision / Recall / F1 Grouped Comparison */}
        <section className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div>
              <div className="flex items-center gap-2 text-teal-700 font-bold text-xs uppercase tracking-wider">
                <BarChart2 className="w-4 h-4" /> Section 4
              </div>
              <h3 className="text-lg font-bold text-slate-900 mt-0.5">Precision, Recall & F1-Score</h3>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <span className="flex items-center gap-1 text-teal-700 font-medium">
                <span className="w-2.5 h-2.5 rounded bg-teal-500 inline-block"></span> Prec
              </span>
              <span className="flex items-center gap-1 text-cyan-700 font-medium">
                <span className="w-2.5 h-2.5 rounded bg-cyan-500 inline-block"></span> Rec
              </span>
              <span className="flex items-center gap-1 text-indigo-700 font-medium">
                <span className="w-2.5 h-2.5 rounded bg-indigo-500 inline-block"></span> F1
              </span>
            </div>
          </div>

          {metrics?.model_comparison && metrics.model_comparison.length > 0 ? (
            <div className="space-y-4 pt-2">
              {metrics.model_comparison.map((item) => (
                <div key={item.model} className="space-y-1">
                  <div className="text-xs font-semibold text-slate-800">{item.model}</div>
                  <div className="grid grid-cols-3 gap-2">
                    <div className="bg-slate-50 p-2 rounded-lg border border-slate-200/60">
                      <div className="text-[10px] text-slate-500">Precision</div>
                      <div className="text-sm font-bold text-teal-700">{fmtNum(item.precision)}</div>
                    </div>
                    <div className="bg-slate-50 p-2 rounded-lg border border-slate-200/60">
                      <div className="text-[10px] text-slate-500">Recall</div>
                      <div className="text-sm font-bold text-cyan-700">{fmtNum(item.recall)}</div>
                    </div>
                    <div className="bg-slate-50 p-2 rounded-lg border border-slate-200/60">
                      <div className="text-[10px] text-slate-500">F1-Score</div>
                      <div className="text-sm font-bold text-indigo-700">{fmtNum(item.f1_score)}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-12 text-center text-slate-400 text-sm">No comparison data available yet.</div>
          )}
        </section>
      </div>

      {/* SECTION 5, 6, 7: HYPERPARAMETER EXPERIMENTS (EPOCHS, BATCH SIZE, LEARNING RATE) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* SECTION 5 — Epoch Experiment */}
        <section className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-4">
          <div className="border-b border-slate-100 pb-3">
            <div className="flex items-center gap-2 text-teal-700 font-bold text-xs uppercase tracking-wider">
              <Sliders className="w-4 h-4" /> Section 5
            </div>
            <h3 className="text-base font-bold text-slate-900 mt-0.5">Epoch Experiment (10, 20, 30, 50)</h3>
            <p className="text-xs text-slate-500 mt-0.5">Impact of training duration on convergence.</p>
          </div>

          {metrics?.epoch_comparison && metrics.epoch_comparison.length > 0 ? (
            <div className="space-y-3">
              {metrics.epoch_comparison.map((ep) => (
                <div key={ep.epochs} className="p-3 rounded-xl bg-slate-50 border border-slate-200/70 space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-slate-900">{ep.epochs} Epochs</span>
                    <span className="font-bold text-emerald-700">{fmtPct(ep.test_accuracy)}</span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-500">
                    <span>Val Loss: {fmtNum(ep.val_loss)}</span>
                    <span>Val Acc: {fmtPct(ep.val_accuracy)}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-8 text-center text-slate-400 text-xs">No epoch sweeps recorded yet.</div>
          )}
        </section>

        {/* SECTION 6 — Batch Size Experiment */}
        <section className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-4">
          <div className="border-b border-slate-100 pb-3">
            <div className="flex items-center gap-2 text-teal-700 font-bold text-xs uppercase tracking-wider">
              <Sliders className="w-4 h-4" /> Section 6
            </div>
            <h3 className="text-base font-bold text-slate-900 mt-0.5">Batch Size Experiment (16, 32, 64)</h3>
            <p className="text-xs text-slate-500 mt-0.5">Gradient stability and generalization trade-off.</p>
          </div>

          {metrics?.batch_comparison && metrics.batch_comparison.length > 0 ? (
            <div className="space-y-3">
              {metrics.batch_comparison.map((bs) => (
                <div key={bs.batch_size} className="p-3 rounded-xl bg-slate-50 border border-slate-200/70 space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-slate-900">Batch Size: {bs.batch_size}</span>
                    <span className="font-bold text-emerald-700">{fmtPct(bs.test_accuracy)}</span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-500">
                    <span>Val Loss: {fmtNum(bs.val_loss)}</span>
                    <span>Time: {bs.training_time ? `${bs.training_time}s` : 'N/A'}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-8 text-center text-slate-400 text-xs">No batch sweeps recorded yet.</div>
          )}
        </section>

        {/* SECTION 7 — Learning Rate Experiment */}
        <section className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-4">
          <div className="border-b border-slate-100 pb-3">
            <div className="flex items-center gap-2 text-teal-700 font-bold text-xs uppercase tracking-wider">
              <Sliders className="w-4 h-4" /> Section 7
            </div>
            <h3 className="text-base font-bold text-slate-900 mt-0.5">Learning Rate (0.0001, 0.001, 0.01)</h3>
            <p className="text-xs text-slate-500 mt-0.5">Optimizer step magnitude and stability.</p>
          </div>

          {metrics?.learning_rate_comparison && metrics.learning_rate_comparison.length > 0 ? (
            <div className="space-y-3">
              {metrics.learning_rate_comparison.map((lr) => (
                <div key={lr.learning_rate} className="p-3 rounded-xl bg-slate-50 border border-slate-200/70 space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-slate-900 font-mono">LR: {lr.learning_rate}</span>
                    <span className="font-bold text-emerald-700">{fmtPct(lr.test_accuracy)}</span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-500">
                    <span>Val Loss: {fmtNum(lr.val_loss)}</span>
                    <span>Val Acc: {fmtPct(lr.val_accuracy)}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-8 text-center text-slate-400 text-xs">No learning rate sweeps recorded yet.</div>
          )}
        </section>
      </div>

      {/* SECTION 9 & 10: TRAINING CURVES & CONFUSION MATRIX */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* SECTION 9 — Interactive Training Curves */}
        <section className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-3">
            <div>
              <div className="flex items-center gap-2 text-teal-700 font-bold text-xs uppercase tracking-wider">
                <LineChart className="w-4 h-4" /> Section 9
              </div>
              <h3 className="text-lg font-bold text-slate-900 mt-0.5">Training & Validation Curves</h3>
            </div>

            {/* Experiment Selector Dropdown */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500">Experiment:</span>
              <select
                value={selectedExperimentId}
                onChange={(e) => handleSelectExperiment(e.target.value)}
                className="px-3 py-1.5 rounded-lg border border-slate-200 bg-slate-50 text-xs font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-teal-500"
              >
                {experiments.map((exp) => (
                  <option key={exp.experiment_id} value={exp.experiment_id}>
                    {exp.experiment_id}: {exp.model} (Acc: {fmtPct(exp.test_accuracy)})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {selectedHistory?.history ? (
            <div className="space-y-6 pt-2">
              {/* Dual Curves: Accuracy & Loss */}
              <div>
                <div className="flex items-center justify-between text-xs font-semibold text-slate-700 mb-2">
                  <span>Accuracy Across Epochs</span>
                  <div className="flex items-center gap-3">
                    <span className="text-teal-600 flex items-center gap-1">
                      <span className="w-2.5 h-0.5 bg-teal-600 inline-block"></span> Train Acc
                    </span>
                    <span className="text-emerald-600 flex items-center gap-1">
                      <span className="w-2.5 h-0.5 bg-emerald-600 inline-block"></span> Val Acc
                    </span>
                  </div>
                </div>

                {/* SVG Line Chart for Accuracy */}
                <div className="h-44 w-full bg-slate-50 rounded-xl p-3 border border-slate-200/60 relative">
                  <svg className="w-full h-full overflow-visible" viewBox="0 0 100 100" preserveAspectRatio="none">
                    {/* Grid lines */}
                    <line x1="0" y1="25" x2="100" y2="25" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="2" />
                    <line x1="0" y1="50" x2="100" y2="50" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="2" />
                    <line x1="0" y1="75" x2="100" y2="75" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="2" />

                    {/* Train Acc Path */}
                    {selectedHistory.history.train_acc && selectedHistory.history.train_acc.length > 1 && (
                      <polyline
                        fill="none"
                        stroke="#0d9488"
                        strokeWidth="2"
                        points={selectedHistory.history.train_acc
                          .map((val, idx) => {
                            const x = (idx / (selectedHistory.history.train_acc.length - 1)) * 100
                            const y = 100 - val * 100
                            return `${x},${y}`
                          })
                          .join(' ')}
                      />
                    )}

                    {/* Val Acc Path */}
                    {selectedHistory.history.val_acc && selectedHistory.history.val_acc.length > 1 && (
                      <polyline
                        fill="none"
                        stroke="#10b981"
                        strokeWidth="2"
                        strokeDasharray="3"
                        points={selectedHistory.history.val_acc
                          .map((val, idx) => {
                            const x = (idx / (selectedHistory.history.val_acc.length - 1)) * 100
                            const y = 100 - val * 100
                            return `${x},${y}`
                          })
                          .join(' ')}
                      />
                    )}
                  </svg>
                </div>
              </div>

              {/* Loss Curve */}
              <div>
                <div className="flex items-center justify-between text-xs font-semibold text-slate-700 mb-2">
                  <span>Cross-Entropy Loss Across Epochs</span>
                  <div className="flex items-center gap-3">
                    <span className="text-rose-600 flex items-center gap-1">
                      <span className="w-2.5 h-0.5 bg-rose-600 inline-block"></span> Train Loss
                    </span>
                    <span className="text-amber-600 flex items-center gap-1">
                      <span className="w-2.5 h-0.5 bg-amber-600 inline-block"></span> Val Loss
                    </span>
                  </div>
                </div>

                <div className="h-44 w-full bg-slate-50 rounded-xl p-3 border border-slate-200/60 relative">
                  <svg className="w-full h-full overflow-visible" viewBox="0 0 100 100" preserveAspectRatio="none">
                    <line x1="0" y1="25" x2="100" y2="25" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="2" />
                    <line x1="0" y1="50" x2="100" y2="50" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="2" />
                    <line x1="0" y1="75" x2="100" y2="75" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="2" />

                    {/* Train Loss Path */}
                    {selectedHistory.history.train_loss && selectedHistory.history.train_loss.length > 1 && (
                      <polyline
                        fill="none"
                        stroke="#e11d48"
                        strokeWidth="2"
                        points={selectedHistory.history.train_loss
                          .map((val, idx) => {
                            const maxL = Math.max(...selectedHistory.history.train_loss, ...selectedHistory.history.val_loss, 1.0)
                            const x = (idx / (selectedHistory.history.train_loss.length - 1)) * 100
                            const y = 100 - Math.min(100, (val / maxL) * 100)
                            return `${x},${y}`
                          })
                          .join(' ')}
                      />
                    )}

                    {/* Val Loss Path */}
                    {selectedHistory.history.val_loss && selectedHistory.history.val_loss.length > 1 && (
                      <polyline
                        fill="none"
                        stroke="#d97706"
                        strokeWidth="2"
                        strokeDasharray="3"
                        points={selectedHistory.history.val_loss
                          .map((val, idx) => {
                            const maxL = Math.max(...selectedHistory.history.train_loss, ...selectedHistory.history.val_loss, 1.0)
                            const x = (idx / (selectedHistory.history.val_loss.length - 1)) * 100
                            const y = 100 - Math.min(100, (val / maxL) * 100)
                            return `${x},${y}`
                          })
                          .join(' ')}
                      />
                    )}
                  </svg>
                </div>
              </div>
            </div>
          ) : (
            <div className="py-16 text-center text-slate-400 text-sm">Select an experiment to view training curves.</div>
          )}
        </section>

        {/* SECTION 10 — Confusion Matrix Heatmap */}
        <section className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div>
              <div className="flex items-center gap-2 text-teal-700 font-bold text-xs uppercase tracking-wider">
                <Brain className="w-4 h-4" /> Section 10
              </div>
              <h3 className="text-lg font-bold text-slate-900 mt-0.5">Confusion Matrix (Held-out Test)</h3>
            </div>
            <span className="text-xs text-slate-500 font-mono">{selectedExperimentId}</span>
          </div>

          {selectedConfusionMatrix?.confusion_matrix ? (
            <div className="space-y-6 pt-2">
              <div className="flex flex-col items-center justify-center p-4 bg-slate-50 rounded-xl border border-slate-200/60">
                <div className="text-xs text-slate-500 mb-3 font-semibold">PREDICTED CLASS</div>
                <div className="flex items-center gap-4">
                  <div className="writing-mode-vertical text-xs text-slate-500 font-semibold rotate-180">
                    ACTUAL CLASS
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-center">
                    {selectedConfusionMatrix.class_names?.map((cname, rowIdx) =>
                      selectedConfusionMatrix.class_names?.map((colName, colIdx) => {
                        const count = selectedConfusionMatrix.confusion_matrix[rowIdx][colIdx]
                        const isDiagonal = rowIdx === colIdx
                        return (
                          <div
                            key={`${rowIdx}-${colIdx}`}
                            className={`p-4 rounded-xl border flex flex-col items-center justify-center min-w-[120px] transition-all ${
                              isDiagonal
                                ? 'bg-teal-50 border-teal-300 text-teal-900 shadow-sm'
                                : 'bg-rose-50/50 border-rose-200 text-rose-900'
                            }`}
                          >
                            <span className="text-2xl font-black">{count}</span>
                            <span className="text-[10px] uppercase font-bold mt-1 text-slate-500">
                              {cname} → {colName}
                            </span>
                          </div>
                        )
                      })
                    )}
                  </div>
                </div>
              </div>

              {/* Class Performance Metrics */}
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs space-y-2">
                <div className="font-bold text-slate-800">Diagnostic Sensitivity & Specificity</div>
                <p className="text-slate-600 leading-relaxed">
                  Diagonal values represent correct classifications (True Positives and True Negatives). Off-diagonal
                  values reflect diagnostic misclassifications on held-out test data without data leakage.
                </p>
              </div>
            </div>
          ) : (
            <div className="py-16 text-center text-slate-400 text-sm">Select an experiment to view its confusion matrix.</div>
          )}
        </section>
      </div>

      {/* SECTION 12: MULTIMODAL COMPARISON ARCHITECTURE CARD */}
      <section className="bg-slate-900 text-white rounded-2xl p-6 sm:p-8 shadow-md border border-slate-800 space-y-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-teal-500/20 text-teal-400 flex items-center justify-center border border-teal-500/30">
            <Zap className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs font-bold uppercase tracking-wider text-teal-400">
              Multimodal Integration
            </div>
            <h2 className="text-xl font-bold text-white">
              Modality Architecture Comparison: Single vs Multimodal Fusion
            </h2>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-5 rounded-xl bg-white/5 border border-white/10 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-teal-400">Branch A</span>
              <span className="text-xs px-2 py-0.5 rounded bg-white/10 text-slate-300">Surface</span>
            </div>
            <h3 className="font-bold text-base text-white">Clinical Oral Photography</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              ResNet-18 / ResNet-50 trained on 1,081 oral photographs for macroscopic mucosal screening with Grad-CAM.
            </p>
            <div className="pt-2 border-t border-white/10 text-xs text-teal-300 font-semibold">
              Weight in Late Fusion: 35%
            </div>
          </div>

          <div className="p-5 rounded-xl bg-white/5 border border-white/10 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-cyan-400">Branch B</span>
              <span className="text-xs px-2 py-0.5 rounded bg-white/10 text-slate-300">Gold Standard</span>
            </div>
            <h3 className="font-bold text-base text-white">Histopathological Biopsy</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Microscopic tissue architecture trained on NDU-UFES cohort for cellular malignancy and dysplasia detection.
            </p>
            <div className="pt-2 border-t border-white/10 text-xs text-cyan-300 font-semibold">
              Weight in Late Fusion: 45% (Precedence)
            </div>
          </div>

          <div className="p-5 rounded-xl bg-white/5 border border-white/10 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-indigo-400">Branch C</span>
              <span className="text-xs px-2 py-0.5 rounded bg-white/10 text-slate-300">Epidemiology</span>
            </div>
            <h3 className="font-bold text-base text-white">Clinical Risk Profiling</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Random Forest Classifier on 23 one-hot patient risk factors (tobacco, alcohol, lesion size, anatomical site).
            </p>
            <div className="pt-2 border-t border-white/10 text-xs text-indigo-300 font-semibold">
              Weight in Late Fusion: 20%
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
