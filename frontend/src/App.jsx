import React, { useState, useEffect, useRef } from 'react'
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  Brain,
  CheckCircle2,
  ChevronRight,
  Database,
  Download,
  Eye,
  FileCheck,
  FileJson,
  FileSpreadsheet,
  FileText,
  Info,
  Layers,
  Microscope,
  Printer,
  RefreshCw,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  Upload,
  User,
  X,
  Zap,
} from 'lucide-react'
import jsPDF from 'jspdf'
import html2canvas from 'html2canvas'
import ExperimentsPage from './components/ExperimentsPage'

const API_BASE = '' // Proxied by Vite to http://127.0.0.1:8000

export default function App() {
  const [currentTab, setCurrentTab] = useState('home') // 'home' | 'detection' | 'experiments' | 'results' | 'report' | 'about'
  const [modelInfo, setModelInfo] = useState(null)
  const [sampleImages, setSampleImages] = useState({ oral: [], histopathology: [] })
  const [apiOnline, setApiOnline] = useState(false)

  // 1. Clinical Oral Image State
  const [oralFile, setOralFile] = useState(null)
  const [oralPreviewUrl, setOralPreviewUrl] = useState(null)
  const [oralSampleName, setOralSampleName] = useState(null)

  // 2. Histopathological Image State
  const [histoFile, setHistoFile] = useState(null)
  const [histoPreviewUrl, setHistoPreviewUrl] = useState(null)
  const [histoSampleName, setHistoSampleName] = useState(null)

  // 3. Clinical Data State (Exact features from ndbufes_TaskII)
  const [includeClinical, setIncludeClinical] = useState(true)
  const [clinicalData, setClinicalData] = useState({
    localization: 'Tongue',
    larger_size: '2.0',
    tobacco_use: 'Yes',
    alcohol_consumption: 'Yes',
    sun_exposure: 'No',
    gender: 'M',
    age_group: '1', // 0: <40, 1: 40-60, 2: >60
  })

  // Inference & Results State
  const [isLoading, setIsLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)
  const [multimodalResult, setMultimodalResult] = useState(null)

  // Explainability Viewers state
  const [oralCamView, setOralCamView] = useState('overlay') // 'overlay' | 'heatmap' | 'original'
  const [histoCamView, setHistoCamView] = useState('overlay')

  const reportRef = useRef(null)

  // Fetch model metadata and sample images on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/model-info`)
      .then((res) => {
        if (!res.ok) throw new Error('API unreachable')
        return res.json()
      })
      .then((data) => {
        setModelInfo(data)
        setApiOnline(true)
      })
      .catch((err) => {
        console.warn('API error:', err)
        setApiOnline(false)
      })

    fetch(`${API_BASE}/api/sample-images`)
      .then((res) => res.json())
      .then((data) => {
        if (data) setSampleImages(data)
      })
      .catch((err) => console.warn('Samples fetch error:', err))
  }, [])

  // File Handlers
  const handleOralFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      if (!file.type.startsWith('image/')) {
        setErrorMsg('Please select a valid image file for the Clinical Oral Image.')
        return
      }
      setOralFile(file)
      setOralSampleName(null)
      setOralPreviewUrl(URL.createObjectURL(file))
      setErrorMsg(null)
    }
  }

  const handleHistoFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      if (!file.type.startsWith('image/')) {
        setErrorMsg('Please select a valid image file for the Histopathological Image.')
        return
      }
      setHistoFile(file)
      setHistoSampleName(null)
      setHistoPreviewUrl(URL.createObjectURL(file))
      setErrorMsg(null)
    }
  }

  // Quick Select Sample Handlers
  const selectOralSample = async (sample) => {
    try {
      setIsLoading(true)
      const res = await fetch(`${API_BASE}${sample.url}`)
      const blob = await res.blob()
      const file = new File([blob], sample.filename, { type: blob.type })
      setOralFile(file)
      setOralSampleName(sample.label)
      setOralPreviewUrl(URL.createObjectURL(blob))
    } catch {
      setErrorMsg('Failed to load sample oral image.')
    } finally {
      setIsLoading(false)
    }
  }

  const selectHistoSample = async (sample) => {
    try {
      setIsLoading(true)
      const res = await fetch(`${API_BASE}${sample.url}`)
      const blob = await res.blob()
      const file = new File([blob], sample.filename, { type: blob.type })
      setHistoFile(file)
      setHistoSampleName(sample.label)
      setHistoPreviewUrl(URL.createObjectURL(blob))
    } catch {
      setErrorMsg('Failed to load sample histopathology image.')
    } finally {
      setIsLoading(false)
    }
  }

  // Reset form
  const handleReset = () => {
    setOralFile(null)
    setOralPreviewUrl(null)
    setOralSampleName(null)
    setHistoFile(null)
    setHistoPreviewUrl(null)
    setHistoSampleName(null)
    setMultimodalResult(null)
    setErrorMsg(null)
  }

  // Submit Multimodal Prediction
  const handleRunMultimodalPrediction = async () => {
    if (!oralFile && !histoFile && !includeClinical) {
      setErrorMsg('Please provide at least one modality (Clinical Oral Image, Histopathological Image, or Clinical Data).')
      return
    }

    setIsLoading(true)
    setErrorMsg(null)

    const formData = new FormData()
    if (oralFile) {
      formData.append('clinical_image', oralFile)
    }
    if (histoFile) {
      formData.append('histopathology_image', histoFile)
    }
    if (includeClinical) {
      formData.append('clinical_data', JSON.stringify(clinicalData))
    }

    try {
      const res = await fetch(`${API_BASE}/api/predict-multimodal`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        throw new Error(errorData.detail || `Server error: ${res.status}`)
      }

      const data = await res.json()
      setMultimodalResult(data)
      setCurrentTab('results')
    } catch (err) {
      setErrorMsg(err.message || 'Multimodal prediction failed.')
    } finally {
      setIsLoading(false)
    }
  }

  // Download JSON report
  const downloadJsonReport = () => {
    if (!multimodalResult) return
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(multimodalResult.structured_report, null, 2))
    const dlAnchor = document.createElement('a')
    dlAnchor.setAttribute('href', dataStr)
    dlAnchor.setAttribute('download', `${multimodalResult.case_id}_Multimodal_Report.json`)
    dlAnchor.click()
  }

  // Download PDF report
  const downloadPdfReport = async () => {
    if (!reportRef.current) return
    try {
      setIsLoading(true)
      const canvas = await html2canvas(reportRef.current, { scale: 2, useCORS: true })
      const imgData = canvas.toDataURL('image/png')
      const pdf = new jsPDF('p', 'mm', 'a4')
      const imgWidth = 210
      const pageHeight = 295
      const imgHeight = (canvas.height * imgWidth) / canvas.width
      let heightLeft = imgHeight
      let position = 0

      pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight)
      heightLeft -= pageHeight

      while (heightLeft >= 0) {
        position = heightLeft - imgHeight
        pdf.addPage()
        pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight)
        heightLeft -= pageHeight
      }

      pdf.save(`${multimodalResult.case_id}_Multimodal_Report.pdf`)
    } catch (err) {
      console.error('PDF export error:', err)
      window.print() // Fallback to browser print dialog
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-800">
      {/* Top Navbar */}
      <header className="sticky top-0 z-40 bg-white/95 backdrop-blur-md border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => setCurrentTab('home')}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-teal-600 to-cyan-500 flex items-center justify-center text-white shadow-md shadow-teal-500/20">
              <Brain className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg text-slate-900 tracking-tight">Oral Cancer AI</span>
                <span className="text-[10px] font-bold tracking-wider uppercase px-2 py-0.5 rounded-full bg-teal-50 text-teal-700 border border-teal-200">
                  Multimodal 3-Tier
                </span>
              </div>
              <p className="text-xs text-slate-500">Multimodal Oral Cancer Detection System</p>
            </div>
          </div>

          <nav className="flex items-center gap-1 sm:gap-2">
            <button
              onClick={() => setCurrentTab('home')}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                currentTab === 'home'
                  ? 'bg-teal-50 text-teal-700 border border-teal-200'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              Dashboard
            </button>
            <button
              onClick={() => setCurrentTab('detection')}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                currentTab === 'detection'
                  ? 'bg-teal-50 text-teal-700 border border-teal-200'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              Multimodal Studio
            </button>
            <button
              onClick={() => setCurrentTab('results')}
              disabled={!multimodalResult}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                currentTab === 'results'
                  ? 'bg-teal-50 text-teal-700 border border-teal-200'
                  : multimodalResult
                  ? 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  : 'text-slate-300 cursor-not-allowed'
              }`}
            >
              Results {multimodalResult && <span className="w-2 h-2 inline-block rounded-full bg-teal-500 ml-1"></span>}
            </button>
            <button
              onClick={() => setCurrentTab('report')}
              disabled={!multimodalResult}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                currentTab === 'report'
                  ? 'bg-teal-50 text-teal-700 border border-teal-200'
                  : multimodalResult
                  ? 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  : 'text-slate-300 cursor-not-allowed'
              }`}
            >
              Report View
            </button>
            <button
              onClick={() => setCurrentTab('about')}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                currentTab === 'about'
                  ? 'bg-teal-50 text-teal-700 border border-teal-200'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              About Models
            </button>
            <button
              onClick={() => setCurrentTab('experiments')}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                currentTab === 'experiments'
                  ? 'bg-teal-50 text-teal-700 border border-teal-200'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              Model Experiments
            </button>

            {/* Model Status Indicator */}
            <div className="hidden md:flex items-center gap-2 pl-3 ml-2 border-l border-slate-200">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  apiOnline ? 'bg-emerald-500 shadow-sm animate-pulse' : 'bg-rose-500'
                }`}
              />
              <span className="text-xs text-slate-500 font-medium">
                {apiOnline ? `3 Models Online (${modelInfo?.system_status?.device || 'Active'})` : 'Connecting...'}
              </span>
            </div>
          </nav>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Alert Banner */}
        {errorMsg && (
          <div className="mb-6 p-4 rounded-xl bg-rose-50 border border-rose-200 flex items-start gap-3 text-rose-800">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5 text-rose-600" />
            <div className="flex-1 text-sm font-medium">{errorMsg}</div>
            <button onClick={() => setErrorMsg(null)} className="text-rose-500 hover:text-rose-700">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* TAB 1: HOME */}
        {currentTab === 'home' && (
          <div className="space-y-10">
            <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-slate-900 via-slate-800 to-teal-950 text-white p-8 sm:p-12 shadow-xl border border-slate-800">
              <div className="relative z-10 max-w-3xl space-y-5">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-500/15 border border-teal-400/30 text-teal-300 text-xs font-semibold">
                  <Sparkles className="w-3.5 h-3.5" />
                  True Multimodal AI Architecture (Late Fusion)
                </div>
                <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight leading-tight">
                  Multimodal Oral Cancer <br />
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-cyan-300">
                    Detection &amp; Biopsy Synthesis
                  </span>
                </h1>
                <p className="text-slate-300 text-base sm:text-lg leading-relaxed">
                  Integrating macroscopic oral lesion photographs, microscopic histopathology biopsy slides, and clinical patient parameters using calibrated Bayesian late fusion and dual Grad-CAM visual interpretability.
                </p>
                <div className="pt-2 flex flex-wrap items-center gap-4">
                  <button
                    onClick={() => setCurrentTab('detection')}
                    className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-semibold shadow-lg shadow-teal-500/25 transition-all"
                  >
                    Open Multimodal Studio
                    <ArrowRight className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setCurrentTab('about')}
                    className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-white/10 hover:bg-white/15 text-white font-medium border border-white/15 transition-colors"
                  >
                    View Model Validation
                  </button>
                </div>
              </div>
            </div>

            {/* 3 Modalities Pillar Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white p-6 rounded-2xl border border-slate-200 card-shadow space-y-3">
                <div className="w-12 h-12 rounded-xl bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-600">
                  <Eye className="w-6 h-6" />
                </div>
                <div className="text-xs font-bold uppercase tracking-wider text-teal-600">Modality 1 (35% Weight)</div>
                <h3 className="font-bold text-slate-900 text-lg">Clinical Oral Photography</h3>
                <p className="text-slate-600 text-sm leading-relaxed">
                  Macroscopic surface image evaluated by ResNet-18 for superficial ulcerations, erythroplakia, and irregular mucosal margins with Grad-CAM.
                </p>
              </div>

              <div className="bg-white p-6 rounded-2xl border border-slate-200 card-shadow space-y-3">
                <div className="w-12 h-12 rounded-xl bg-indigo-50 border border-indigo-200 flex items-center justify-center text-indigo-600">
                  <Microscope className="w-6 h-6" />
                </div>
                <div className="text-xs font-bold uppercase tracking-wider text-indigo-600">Modality 2 (45% Weight)</div>
                <h3 className="font-bold text-slate-900 text-lg">Histopathological Biopsy</h3>
                <p className="text-slate-600 text-sm leading-relaxed">
                  Microscopic biopsy tissue slide evaluated by our NDU-UFES ResNet-18 model (82.1% test accuracy) for epithelial dysplasia and invasive OSCC cells.
                </p>
              </div>

              <div className="bg-white p-6 rounded-2xl border border-slate-200 card-shadow space-y-3">
                <div className="w-12 h-12 rounded-xl bg-cyan-50 border border-cyan-200 flex items-center justify-center text-cyan-600">
                  <Database className="w-6 h-6" />
                </div>
                <div className="text-xs font-bold uppercase tracking-wider text-cyan-600">Modality 3 (20% Weight)</div>
                <h3 className="font-bold text-slate-900 text-lg">Clinical Patient Data</h3>
                <p className="text-slate-600 text-sm leading-relaxed">
                  23 one-hot encoded clinical risk factors (tobacco, alcohol, anatomical site, lesion diameter) processed by Random Forest (98.8% ROC-AUC).
                </p>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: MULTIMODAL DETECTION STUDIO */}
        {currentTab === 'detection' && (
          <div className="space-y-8">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div>
                <h2 className="text-2xl font-bold text-slate-900">Multimodal Input Studio</h2>
                <p className="text-sm text-slate-500 mt-1">
                  Provide one, two, or all three modalities. The validated Late-Fusion engine will synthesize active modalities dynamically.
                </p>
              </div>
              <button
                onClick={handleReset}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg border border-slate-300 text-slate-600 hover:bg-slate-100 text-xs font-medium self-start"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Reset All Modalities
              </button>
            </div>

            {/* Quick Test Samples Gallery */}
            <div className="bg-white p-5 rounded-2xl border border-slate-200 card-shadow space-y-3">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-amber-500" />
                <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                  Quick-Load Verified Test Samples
                </span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
                {/* Oral Samples */}
                <div className="space-y-1.5">
                  <span className="text-slate-500 font-semibold block">Oral Photos:</span>
                  <div className="flex flex-wrap gap-1.5">
                    {sampleImages.oral?.map((s, idx) => (
                      <button
                        key={idx}
                        onClick={() => selectOralSample(s)}
                        className={`px-2.5 py-1 rounded-lg border text-[11px] font-medium transition-all ${
                          oralSampleName === s.label
                            ? 'bg-teal-600 text-white border-teal-600 font-bold'
                            : s.category.includes('cancer')
                            ? 'bg-rose-50 text-rose-700 border-rose-200 hover:bg-rose-100'
                            : 'bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100'
                        }`}
                      >
                        {s.filename} ({s.category.includes('cancer') ? 'Malignant' : 'Benign'})
                      </button>
                    ))}
                  </div>
                </div>

                {/* Histopathology Samples */}
                <div className="space-y-1.5">
                  <span className="text-slate-500 font-semibold block">Biopsy Slides (NDU-UFES):</span>
                  <div className="flex flex-wrap gap-1.5">
                    {sampleImages.histopathology?.map((s, idx) => (
                      <button
                        key={idx}
                        onClick={() => selectHistoSample(s)}
                        className={`px-2.5 py-1 rounded-lg border text-[11px] font-medium transition-all ${
                          histoSampleName === s.label
                            ? 'bg-indigo-600 text-white border-indigo-600 font-bold'
                            : 'bg-indigo-50 text-indigo-700 border-indigo-200 hover:bg-indigo-100'
                        }`}
                      >
                        🔬 {s.filename}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* THREE SEPARATE INPUT SECTIONS */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* SECTION 1: CLINICAL ORAL IMAGE */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 card-shadow flex flex-col space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-lg bg-teal-50 text-teal-700 font-bold text-xs flex items-center justify-center">
                      1
                    </div>
                    <div>
                      <h3 className="font-bold text-slate-900 text-sm">Clinical Oral Image</h3>
                      <p className="text-[11px] text-slate-400">Oral lesion photograph</p>
                    </div>
                  </div>
                  {oralPreviewUrl && (
                    <button
                      onClick={() => {
                        setOralFile(null)
                        setOralPreviewUrl(null)
                        setOralSampleName(null)
                      }}
                      className="text-xs text-rose-500 hover:text-rose-700"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>

                <p className="text-xs text-slate-500">
                  Upload a clinical oral photograph showing the lesion or affected area.
                </p>

                {oralPreviewUrl ? (
                  <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-slate-900 h-56 flex items-center justify-center">
                    <img src={oralPreviewUrl} alt="Oral lesion" className="max-h-full object-contain" />
                    <div className="absolute inset-x-0 bottom-0 bg-black/60 p-2 text-white text-[11px] flex justify-between">
                      <span className="truncate">{oralSampleName || oralFile?.name}</span>
                      <label className="cursor-pointer text-teal-300 hover:underline">
                        Replace
                        <input type="file" accept="image/*" onChange={handleOralFileChange} className="hidden" />
                      </label>
                    </div>
                  </div>
                ) : (
                  <label className="border-2 border-dashed border-slate-300 hover:border-teal-500 rounded-xl h-56 flex flex-col items-center justify-center p-4 cursor-pointer bg-slate-50/50 hover:bg-teal-50/20 transition-all">
                    <Upload className="w-8 h-8 text-teal-600 mb-2" />
                    <span className="text-xs font-semibold text-slate-700">Click to upload photo</span>
                    <span className="text-[10px] text-slate-400 mt-1">PNG, JPG, JPEG, WebP</span>
                    <input type="file" accept="image/*" onChange={handleOralFileChange} className="hidden" />
                  </label>
                )}

                <div className="mt-auto pt-2 text-[11px] text-slate-400 flex items-center gap-1.5">
                  <span className={`w-2 h-2 rounded-full ${oralFile ? 'bg-teal-500' : 'bg-slate-300'}`}></span>
                  Status: {oralFile ? 'Ready for ResNet-18 analysis' : 'Not provided (optional)'}
                </div>
              </div>

              {/* SECTION 2: HISTOPATHOLOGICAL IMAGE */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 card-shadow flex flex-col space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-lg bg-indigo-50 text-indigo-700 font-bold text-xs flex items-center justify-center">
                      2
                    </div>
                    <div>
                      <h3 className="font-bold text-slate-900 text-sm">Histopathological Image</h3>
                      <p className="text-[11px] text-slate-400">Microscopic tissue slide</p>
                    </div>
                  </div>
                  {histoPreviewUrl && (
                    <button
                      onClick={() => {
                        setHistoFile(null)
                        setHistoPreviewUrl(null)
                        setHistoSampleName(null)
                      }}
                      className="text-xs text-rose-500 hover:text-rose-700"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>

                <p className="text-xs text-slate-500">
                  Upload the histopathological image for microscopic tissue analysis.
                </p>

                {histoPreviewUrl ? (
                  <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-slate-900 h-56 flex items-center justify-center">
                    <img src={histoPreviewUrl} alt="Histopathology slide" className="max-h-full object-contain" />
                    <div className="absolute inset-x-0 bottom-0 bg-black/60 p-2 text-white text-[11px] flex justify-between">
                      <span className="truncate">{histoSampleName || histoFile?.name}</span>
                      <label className="cursor-pointer text-indigo-300 hover:underline">
                        Replace
                        <input type="file" accept="image/*" onChange={handleHistoFileChange} className="hidden" />
                      </label>
                    </div>
                  </div>
                ) : (
                  <label className="border-2 border-dashed border-slate-300 hover:border-indigo-500 rounded-xl h-56 flex flex-col items-center justify-center p-4 cursor-pointer bg-slate-50/50 hover:bg-indigo-50/20 transition-all">
                    <Microscope className="w-8 h-8 text-indigo-600 mb-2" />
                    <span className="text-xs font-semibold text-slate-700">Click to upload slide</span>
                    <span className="text-[10px] text-slate-400 mt-1">High-res PNG, JPG</span>
                    <input type="file" accept="image/*" onChange={handleHistoFileChange} className="hidden" />
                  </label>
                )}

                <div className="mt-auto pt-2 text-[11px] text-slate-400 flex items-center gap-1.5">
                  <span className={`w-2 h-2 rounded-full ${histoFile ? 'bg-indigo-500' : 'bg-slate-300'}`}></span>
                  Status: {histoFile ? 'Ready for Histology ResNet-18 analysis' : 'Not provided (optional)'}
                </div>
              </div>

              {/* SECTION 3: CLINICAL PATIENT INFORMATION */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 card-shadow flex flex-col space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-lg bg-cyan-50 text-cyan-700 font-bold text-xs flex items-center justify-center">
                      3
                    </div>
                    <div>
                      <h3 className="font-bold text-slate-900 text-sm">Clinical Patient Information</h3>
                      <p className="text-[11px] text-slate-400">23 NDU-UFES model features</p>
                    </div>
                  </div>
                  <label className="flex items-center gap-1.5 cursor-pointer text-xs font-medium text-slate-600">
                    <input
                      type="checkbox"
                      checked={includeClinical}
                      onChange={(e) => setIncludeClinical(e.target.checked)}
                      className="accent-teal-600 rounded"
                    />
                    Include
                  </label>
                </div>

                <div className={`space-y-2.5 text-xs ${!includeClinical ? 'opacity-40 pointer-events-none' : ''}`}>
                  {/* Anatomical site */}
                  <div>
                    <label className="block font-semibold text-slate-700 mb-0.5">Lesion Localization</label>
                    <select
                      value={clinicalData.localization}
                      onChange={(e) => setClinicalData({ ...clinicalData, localization: e.target.value })}
                      className="w-full p-2 rounded-lg border border-slate-200 bg-slate-50 text-slate-800"
                    >
                      <option value="Tongue">Tongue</option>
                      <option value="Floor of mouth">Floor of mouth</option>
                      <option value="Buccal mucosa">Buccal mucosa</option>
                      <option value="Lip">Lip</option>
                      <option value="Palate">Palate</option>
                      <option value="Gingiva">Gingiva</option>
                    </select>
                  </div>

                  {/* Size */}
                  <div>
                    <div className="flex justify-between font-semibold text-slate-700 mb-0.5">
                      <span>Max Diameter</span>
                      <span className="text-teal-700 font-bold">{clinicalData.larger_size} cm</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="6.0"
                      step="0.1"
                      value={clinicalData.larger_size}
                      onChange={(e) => setClinicalData({ ...clinicalData, larger_size: e.target.value })}
                      className="w-full accent-teal-600"
                    />
                  </div>

                  {/* Tobacco and Alcohol */}
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block font-semibold text-slate-700 mb-0.5">Tobacco</label>
                      <select
                        value={clinicalData.tobacco_use}
                        onChange={(e) => setClinicalData({ ...clinicalData, tobacco_use: e.target.value })}
                        className="w-full p-1.5 rounded-lg border border-slate-200 bg-slate-50"
                      >
                        <option value="Yes">Yes</option>
                        <option value="Former">Former</option>
                        <option value="No">No</option>
                        <option value="Not informed">Not informed</option>
                      </select>
                    </div>
                    <div>
                      <label className="block font-semibold text-slate-700 mb-0.5">Alcohol</label>
                      <select
                        value={clinicalData.alcohol_consumption}
                        onChange={(e) => setClinicalData({ ...clinicalData, alcohol_consumption: e.target.value })}
                        className="w-full p-1.5 rounded-lg border border-slate-200 bg-slate-50"
                      >
                        <option value="Yes">Yes</option>
                        <option value="Former">Former</option>
                        <option value="No">No</option>
                        <option value="Not informed">Not informed</option>
                      </select>
                    </div>
                  </div>

                  {/* Gender and Age Group */}
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block font-semibold text-slate-700 mb-0.5">Gender</label>
                      <select
                        value={clinicalData.gender}
                        onChange={(e) => setClinicalData({ ...clinicalData, gender: e.target.value })}
                        className="w-full p-1.5 rounded-lg border border-slate-200 bg-slate-50"
                      >
                        <option value="M">Male (M)</option>
                        <option value="F">Female (F)</option>
                      </select>
                    </div>
                    <div>
                      <label className="block font-semibold text-slate-700 mb-0.5">Age Group</label>
                      <select
                        value={clinicalData.age_group}
                        onChange={(e) => setClinicalData({ ...clinicalData, age_group: e.target.value })}
                        className="w-full p-1.5 rounded-lg border border-slate-200 bg-slate-50"
                      >
                        <option value="0">&lt; 40 yrs</option>
                        <option value="1">40–60 yrs</option>
                        <option value="2">&gt; 60 yrs</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div className="mt-auto pt-2 text-[11px] text-slate-400 flex items-center gap-1.5">
                  <span className={`w-2 h-2 rounded-full ${includeClinical ? 'bg-cyan-500' : 'bg-slate-300'}`}></span>
                  Status: {includeClinical ? 'Ready for Random Forest model' : 'Omitted by user'}
                </div>
              </div>
            </div>

            {/* Run Multimodal Action Bar */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 card-shadow flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <Layers className="w-5 h-5 text-teal-600" />
                  <span className="font-bold text-slate-900 text-base">Multimodal Fusion Execution</span>
                </div>
                <p className="text-xs text-slate-500 mt-0.5">
                  Active modalities: {[oralFile && 'Oral Photo', histoFile && 'Biopsy Slide', includeClinical && 'Clinical Data'].filter(Boolean).join(' + ') || 'None selected'}
                </p>
              </div>

              <button
                onClick={handleRunMultimodalPrediction}
                disabled={isLoading || (!oralFile && !histoFile && !includeClinical)}
                className={`py-3.5 px-8 rounded-xl font-bold text-sm flex items-center justify-center gap-2 shadow-lg transition-all ${
                  isLoading || (!oralFile && !histoFile && !includeClinical)
                    ? 'bg-slate-200 text-slate-400 cursor-not-allowed shadow-none'
                    : 'bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-500 hover:to-cyan-500 text-white shadow-teal-600/25 active:scale-[0.99]'
                }`}
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Executing Multimodal Inference &amp; Dual Grad-CAM...
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4" />
                    Run Multimodal Detection
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* TAB 3: RESULTS VIEW */}
        {currentTab === 'results' && multimodalResult && (
          <div className="space-y-8">
            {/* Header with Case ID and Actions */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold tracking-wider uppercase text-teal-600">Multimodal Result</span>
                  <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-200 text-slate-700">
                    {multimodalResult.case_id}
                  </span>
                </div>
                <h2 className="text-2xl font-bold text-slate-900">Multimodal Diagnostic Consensus</h2>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setCurrentTab('report')}
                  className="px-4 py-2 rounded-lg bg-teal-600 hover:bg-teal-500 text-white text-xs font-semibold flex items-center gap-1.5 shadow-sm"
                >
                  <FileText className="w-4 h-4" /> View Structured Report
                </button>
                <button
                  onClick={() => setCurrentTab('detection')}
                  className="px-3.5 py-2 rounded-lg bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 text-xs font-medium"
                >
                  <RotateCcw className="w-3.5 h-3.5" /> New Test
                </button>
              </div>
            </div>

            {/* Fused Prediction Banner */}
            <div
              className={`p-6 sm:p-8 rounded-3xl border shadow-lg ${
                multimodalResult.fused_result.is_malignant
                  ? 'bg-gradient-to-br from-rose-50 via-white to-red-50/40 border-rose-200'
                  : 'bg-gradient-to-br from-emerald-50 via-white to-teal-50/40 border-emerald-200'
              }`}
            >
              <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
                <div className="md:col-span-7 space-y-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                        multimodalResult.fused_result.is_malignant
                          ? 'bg-rose-100 text-rose-800 border border-rose-300/60'
                          : 'bg-emerald-100 text-emerald-800 border border-emerald-300/60'
                      }`}
                    >
                      {multimodalResult.fused_result.is_malignant ? (
                        <AlertTriangle className="w-3.5 h-3.5 text-rose-600" />
                      ) : (
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                      )}
                      Final Fused Consensus
                    </span>
                    <span className="text-xs px-2.5 py-1 rounded-full bg-slate-100 border border-slate-200 font-semibold text-slate-700">
                      {multimodalResult.fused_result.active_modalities_count} of 3 Modalities Evaluated
                    </span>
                  </div>

                  <h3 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
                    Predicted Class:{' '}
                    <span
                      className={
                        multimodalResult.fused_result.is_malignant ? 'text-rose-600' : 'text-emerald-600'
                      }
                    >
                      {multimodalResult.fused_result.fused_prediction}
                    </span>
                  </h3>

                  <p className="text-sm text-slate-600">
                    Synthesized using {multimodalResult.fused_result.fusion_method}. Uncertainty level:{' '}
                    <strong>{multimodalResult.fused_result.uncertainty_level}</strong>.
                  </p>

                  {/* Discordance Alert Banner */}
                  {multimodalResult.fused_result.discordance_alert && (
                    <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-300 text-amber-900 text-xs flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                      <p>{multimodalResult.fused_result.discordance_alert}</p>
                    </div>
                  )}
                </div>

                {/* Quantitative Probability Card */}
                <div className="md:col-span-5 bg-white/90 backdrop-blur-sm p-5 rounded-2xl border border-slate-200 space-y-4">
                  <span className="text-xs font-bold text-slate-700 uppercase tracking-wider block">
                    Multimodal Consensus Probability
                  </span>

                  <div className="space-y-1">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="text-rose-700">Malignancy Risk</span>
                      <span className="text-slate-900">
                        {(multimodalResult.fused_result.cancer_probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-rose-500 to-red-600 rounded-full"
                        style={{ width: `${multimodalResult.fused_result.cancer_probability * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  {/* Modality Contribution Chips */}
                  <div className="pt-2 border-t border-slate-100 space-y-1.5">
                    <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider block">
                      Modality Fusion Weights
                    </span>
                    <div className="space-y-1 text-xs">
                      {Object.entries(multimodalResult.fused_result.modality_contributions).map(([mod, details]) => (
                        <div key={mod} className="flex justify-between text-slate-600">
                          <span>{mod}:</span>
                          <span className="font-semibold text-slate-800">
                            {(details.weight * 100).toFixed(0)}% weight &bull; {(details.probability * 100).toFixed(1)}% prob
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* THREE INDIVIDUAL MODALITY CARDS */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Modality 1: Oral Photo Card */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 card-shadow space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                  <div className="flex items-center gap-2">
                    <Eye className="w-4 h-4 text-teal-600" />
                    <h4 className="font-bold text-slate-900 text-sm">Oral Photography</h4>
                  </div>
                  <span className={`text-[11px] font-bold px-2 py-0.5 rounded-full ${
                    multimodalResult.modalities.oral
                      ? multimodalResult.modalities.oral.is_malignant
                        ? 'bg-rose-100 text-rose-800'
                        : 'bg-emerald-100 text-emerald-800'
                      : 'bg-slate-100 text-slate-500'
                  }`}>
                    {multimodalResult.modalities.oral?.predicted_class || 'Not Uploaded'}
                  </span>
                </div>

                {multimodalResult.modalities.oral ? (
                  <div className="space-y-3">
                    <div className="aspect-square rounded-xl overflow-hidden border border-slate-200 bg-slate-900">
                      <img
                        src={
                          oralCamView === 'overlay' && multimodalResult.modalities.oral.gradcam_overlay
                            ? multimodalResult.modalities.oral.gradcam_overlay
                            : oralCamView === 'heatmap' && multimodalResult.modalities.oral.gradcam_heatmap
                            ? multimodalResult.modalities.oral.gradcam_heatmap
                            : oralPreviewUrl
                        }
                        alt="Oral visualization"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    {/* View switcher */}
                    <div className="flex rounded-lg bg-slate-100 p-0.5 text-[10px] font-semibold">
                      {['overlay', 'heatmap', 'original'].map((v) => (
                        <button
                          key={v}
                          onClick={() => setOralCamView(v)}
                          className={`flex-1 py-1 capitalize rounded-md transition-all ${
                            oralCamView === v ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500'
                          }`}
                        >
                          {v}
                        </button>
                      ))}
                    </div>
                    <div className="text-xs space-y-1 text-slate-600">
                      <div className="flex justify-between">
                        <span>Confidence:</span>
                        <span className="font-bold text-slate-800">{multimodalResult.modalities.oral.confidence_percentage}</span>
                      </div>
                      <p className="text-[11px] text-slate-500">{multimodalResult.modalities.oral.findings_summary}</p>
                    </div>
                  </div>
                ) : (
                  <div className="py-12 text-center text-xs text-slate-400">Oral image was omitted in this evaluation.</div>
                )}
              </div>

              {/* Modality 2: Histopathology Card */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 card-shadow space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                  <div className="flex items-center gap-2">
                    <Microscope className="w-4 h-4 text-indigo-600" />
                    <h4 className="font-bold text-slate-900 text-sm">Histopathology Biopsy</h4>
                  </div>
                  <span className={`text-[11px] font-bold px-2 py-0.5 rounded-full ${
                    multimodalResult.modalities.histopathology
                      ? multimodalResult.modalities.histopathology.is_malignant
                        ? 'bg-rose-100 text-rose-800'
                        : 'bg-emerald-100 text-emerald-800'
                      : 'bg-slate-100 text-slate-500'
                  }`}>
                    {multimodalResult.modalities.histopathology?.predicted_class || 'Not Uploaded'}
                  </span>
                </div>

                {multimodalResult.modalities.histopathology ? (
                  <div className="space-y-3">
                    <div className="aspect-square rounded-xl overflow-hidden border border-slate-200 bg-slate-900">
                      <img
                        src={
                          histoCamView === 'overlay' && multimodalResult.modalities.histopathology.gradcam_overlay
                            ? multimodalResult.modalities.histopathology.gradcam_overlay
                            : histoCamView === 'heatmap' && multimodalResult.modalities.histopathology.gradcam_heatmap
                            ? multimodalResult.modalities.histopathology.gradcam_heatmap
                            : histoPreviewUrl
                        }
                        alt="Histopathology visualization"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    {/* View switcher */}
                    <div className="flex rounded-lg bg-slate-100 p-0.5 text-[10px] font-semibold">
                      {['overlay', 'heatmap', 'original'].map((v) => (
                        <button
                          key={v}
                          onClick={() => setHistoCamView(v)}
                          className={`flex-1 py-1 capitalize rounded-md transition-all ${
                            histoCamView === v ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500'
                          }`}
                        >
                          {v}
                        </button>
                      ))}
                    </div>
                    <div className="text-xs space-y-1 text-slate-600">
                      <div className="flex justify-between">
                        <span>Confidence:</span>
                        <span className="font-bold text-slate-800">{multimodalResult.modalities.histopathology.confidence_percentage}</span>
                      </div>
                      <p className="text-[11px] text-slate-500">{multimodalResult.modalities.histopathology.findings_summary}</p>
                    </div>
                  </div>
                ) : (
                  <div className="py-12 text-center text-xs text-slate-400">Histopathology biopsy slide was omitted in this evaluation.</div>
                )}
              </div>

              {/* Modality 3: Clinical Patient Data Card */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 card-shadow space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                  <div className="flex items-center gap-2">
                    <Database className="w-4 h-4 text-cyan-600" />
                    <h4 className="font-bold text-slate-900 text-sm">Clinical Patient Data</h4>
                  </div>
                  <span className={`text-[11px] font-bold px-2 py-0.5 rounded-full ${
                    multimodalResult.modalities.clinical
                      ? multimodalResult.modalities.clinical.is_malignant
                        ? 'bg-rose-100 text-rose-800'
                        : 'bg-emerald-100 text-emerald-800'
                      : 'bg-slate-100 text-slate-500'
                  }`}>
                    {multimodalResult.modalities.clinical?.predicted_class || 'Not Provided'}
                  </span>
                </div>

                {multimodalResult.modalities.clinical ? (
                  <div className="space-y-3 text-xs text-slate-600">
                    <div className="p-3 bg-slate-50 rounded-xl space-y-1">
                      <div className="flex justify-between">
                        <span>Risk Probability:</span>
                        <span className="font-bold text-slate-900">
                          {(multimodalResult.modalities.clinical.cancer_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-500">{multimodalResult.modalities.clinical.findings_summary}</p>
                    </div>

                    <div className="space-y-1.5">
                      <span className="font-bold text-slate-800 text-[11px] uppercase tracking-wider block">
                        Identified Clinical Indicators:
                      </span>
                      {multimodalResult.modalities.clinical.identified_risk_factors.length > 0 ? (
                        <ul className="pl-4 list-disc space-y-1 text-rose-900 text-[11px]">
                          {multimodalResult.modalities.clinical.identified_risk_factors.map((f, i) => (
                            <li key={i}>{f}</li>
                          ))}
                        </ul>
                      ) : (
                        <span className="text-[11px] text-slate-400">No high-risk etiological factors identified.</span>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="py-12 text-center text-xs text-slate-400">Clinical data was omitted in this evaluation.</div>
                )}
              </div>
            </div>

            {/* Disclaimer */}
            <div className="p-4 rounded-2xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-start gap-3">
              <ShieldAlert className="w-5 h-5 text-amber-700 flex-shrink-0 mt-0.5" />
              <div className="space-y-0.5">
                <p className="font-bold">Medical Research Disclaimer</p>
                <p className="text-amber-800">{multimodalResult.structured_report.clinical_recommendation}</p>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: STRUCTURED REPORT VIEW */}
        {currentTab === 'report' && multimodalResult && (
          <div className="space-y-6 max-w-4xl mx-auto">
            {/* Action Bar */}
            <div className="flex items-center justify-between">
              <button
                onClick={() => setCurrentTab('results')}
                className="text-xs font-semibold text-slate-600 hover:text-slate-900 flex items-center gap-1"
              >
                &larr; Back to Results Dashboard
              </button>
              <div className="flex items-center gap-2">
                <button
                  onClick={downloadJsonReport}
                  className="px-3.5 py-2 rounded-lg bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 text-xs font-semibold flex items-center gap-1.5 shadow-sm"
                >
                  <FileJson className="w-4 h-4 text-amber-600" /> Download JSON
                </button>
                <button
                  onClick={downloadPdfReport}
                  disabled={isLoading}
                  className="px-3.5 py-2 rounded-lg bg-teal-600 hover:bg-teal-500 text-white text-xs font-semibold flex items-center gap-1.5 shadow-sm"
                >
                  <Download className="w-4 h-4" /> Download PDF Report
                </button>
                <button
                  onClick={() => window.print()}
                  className="px-3 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold flex items-center gap-1.5"
                >
                  <Printer className="w-4 h-4" /> Print
                </button>
              </div>
            </div>

            {/* Printable Report Document Card */}
            <div
              ref={reportRef}
              className="bg-white p-8 sm:p-12 rounded-3xl border border-slate-300 shadow-xl space-y-8 text-slate-800"
            >
              {/* Header */}
              <div className="flex items-start justify-between border-b-2 border-slate-900 pb-6">
                <div>
                  <div className="flex items-center gap-2 text-teal-700 font-bold text-sm tracking-wider uppercase">
                    <Brain className="w-5 h-5" /> Oral Cancer AI Diagnostic Suite
                  </div>
                  <h1 className="text-2xl sm:text-3xl font-black text-slate-900 mt-1">
                    Multimodal Oral Cancer Detection Report
                  </h1>
                  <p className="text-xs text-slate-500 mt-1">
                    Case ID: <span className="font-mono font-bold text-slate-800">{multimodalResult.case_id}</span> &bull; Generated: {new Date().toUTCString()}
                  </p>
                </div>
                <div className="text-right">
                  <span className="text-xs font-bold px-3 py-1 rounded-full bg-slate-100 border text-slate-800">
                    {multimodalResult.fused_result.active_modalities_count} / 3 Modalities
                  </span>
                </div>
              </div>

              {/* 1. Patient Clinical Information */}
              <div className="space-y-3">
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-900 border-b border-slate-200 pb-1">
                  1. Patient Intake &amp; Clinical Parameters
                </h3>
                {typeof multimodalResult.structured_report.patient_clinical_data === 'object' ? (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                    <div className="p-2.5 bg-slate-50 rounded-lg">
                      <span className="text-slate-400 block">Lesion Site</span>
                      <span className="font-bold text-slate-800">{multimodalResult.structured_report.patient_clinical_data.localization}</span>
                    </div>
                    <div className="p-2.5 bg-slate-50 rounded-lg">
                      <span className="text-slate-400 block">Max Diameter</span>
                      <span className="font-bold text-slate-800">{multimodalResult.structured_report.patient_clinical_data.larger_size} cm</span>
                    </div>
                    <div className="p-2.5 bg-slate-50 rounded-lg">
                      <span className="text-slate-400 block">Tobacco Use</span>
                      <span className="font-bold text-slate-800">{multimodalResult.structured_report.patient_clinical_data.tobacco_use}</span>
                    </div>
                    <div className="p-2.5 bg-slate-50 rounded-lg">
                      <span className="text-slate-400 block">Alcohol Use</span>
                      <span className="font-bold text-slate-800">{multimodalResult.structured_report.patient_clinical_data.alcohol_consumption}</span>
                    </div>
                    <div className="p-2.5 bg-slate-50 rounded-lg">
                      <span className="text-slate-400 block">Gender</span>
                      <span className="font-bold text-slate-800">{multimodalResult.structured_report.patient_clinical_data.gender === 'M' ? 'Male' : 'Female'}</span>
                    </div>
                    <div className="p-2.5 bg-slate-50 rounded-lg">
                      <span className="text-slate-400 block">Age Group</span>
                      <span className="font-bold text-slate-800">
                        {multimodalResult.structured_report.patient_clinical_data.age_group === '2' ? '> 60 years' : multimodalResult.structured_report.patient_clinical_data.age_group === '1' ? '40–60 years' : '< 40 years'}
                      </span>
                    </div>
                    <div className="p-2.5 bg-slate-50 rounded-lg">
                      <span className="text-slate-400 block">Sun Exposure</span>
                      <span className="font-bold text-slate-800">{multimodalResult.structured_report.patient_clinical_data.sun_exposure}</span>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-slate-500 italic">No clinical patient information submitted.</p>
                )}
              </div>

              {/* 2. Modality Summaries Table */}
              <div className="space-y-3">
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-900 border-b border-slate-200 pb-1">
                  2. Individual Modality Evaluations
                </h3>
                <div className="overflow-hidden rounded-xl border border-slate-200 text-xs">
                  <table className="w-full text-left">
                    <thead className="bg-slate-100 text-slate-700 font-bold">
                      <tr>
                        <th className="p-3">Modality Source</th>
                        <th className="p-3">Model Engine</th>
                        <th className="p-3">Predicted Class</th>
                        <th className="p-3">Confidence</th>
                        <th className="p-3">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      <tr>
                        <td className="p-3 font-semibold text-slate-900">Clinical Oral Photo</td>
                        <td className="p-3">ResNet-18</td>
                        <td className="p-3 font-bold">{multimodalResult.structured_report.individual_modality_findings.oral_photography.output}</td>
                        <td className="p-3">{multimodalResult.structured_report.individual_modality_findings.oral_photography.confidence || '—'}</td>
                        <td className="p-3">
                          {multimodalResult.structured_report.individual_modality_findings.oral_photography.evaluated ? '✅ Evaluated' : '❌ Omitted'}
                        </td>
                      </tr>
                      <tr>
                        <td className="p-3 font-semibold text-slate-900">Histopathology Slide</td>
                        <td className="p-3">NDU-UFES ResNet-18</td>
                        <td className="p-3 font-bold">{multimodalResult.structured_report.individual_modality_findings.histopathology_biopsy.output}</td>
                        <td className="p-3">{multimodalResult.structured_report.individual_modality_findings.histopathology_biopsy.confidence || '—'}</td>
                        <td className="p-3">
                          {multimodalResult.structured_report.individual_modality_findings.histopathology_biopsy.evaluated ? '✅ Evaluated' : '❌ Omitted'}
                        </td>
                      </tr>
                      <tr>
                        <td className="p-3 font-semibold text-slate-900">Clinical Data Profile</td>
                        <td className="p-3">Random Forest</td>
                        <td className="p-3 font-bold">{multimodalResult.structured_report.individual_modality_findings.clinical_data.output}</td>
                        <td className="p-3">
                          {multimodalResult.structured_report.individual_modality_findings.clinical_data.probability !== null
                            ? `${(multimodalResult.structured_report.individual_modality_findings.clinical_data.probability * 100).toFixed(1)}%`
                            : '—'}
                        </td>
                        <td className="p-3">
                          {multimodalResult.structured_report.individual_modality_findings.clinical_data.evaluated ? '✅ Evaluated' : '❌ Omitted'}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* 3. Multimodal Synthesis Output */}
              <div className="space-y-3 p-5 rounded-2xl bg-slate-50 border border-slate-200">
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-900 border-b border-slate-200 pb-1">
                  3. Multimodal Late-Fusion Synthesis
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
                  <div>
                    <span className="text-slate-400 block">Final Fused Prediction</span>
                    <span className="text-base font-black text-slate-900 block mt-0.5">
                      {multimodalResult.structured_report.multimodal_synthesis.fused_prediction}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400 block">Consensus Malignancy Prob</span>
                    <span className="text-base font-black text-slate-900 block mt-0.5">
                      {(multimodalResult.structured_report.multimodal_synthesis.cancer_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400 block">Uncertainty State</span>
                    <span className="text-base font-bold text-slate-700 block mt-0.5">
                      {multimodalResult.structured_report.multimodal_synthesis.uncertainty_level}
                    </span>
                  </div>
                </div>

                {multimodalResult.structured_report.multimodal_synthesis.discordance_alert && (
                  <p className="text-xs text-amber-800 font-medium pt-2 border-t border-slate-200">
                    ⚠️ {multimodalResult.structured_report.multimodal_synthesis.discordance_alert}
                  </p>
                )}
              </div>

              {/* 4. Limitations & Safe Recommendations */}
              <div className="space-y-2 text-xs text-slate-600 border-t border-slate-200 pt-4">
                <h4 className="font-bold text-slate-800 uppercase tracking-wider text-[11px]">
                  Scientific Limitations &amp; Model Disclosures
                </h4>
                <ul className="list-disc pl-4 space-y-1 text-[11px]">
                  {multimodalResult.structured_report.scientific_limitations.map((lim, idx) => (
                    <li key={idx}>{lim}</li>
                  ))}
                </ul>

                <div className="mt-4 p-3 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs">
                  <strong>Clinical Advisory:</strong> {multimodalResult.structured_report.clinical_recommendation}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: ABOUT */}
        {currentTab === 'about' && (
          <div className="space-y-8 max-w-4xl mx-auto">
            <div>
              <h2 className="text-2xl font-bold text-slate-900">Multimodal Architecture &amp; Methodology</h2>
              <p className="text-sm text-slate-500 mt-1">
                Technical details on model training, validation splits, and calibrated late fusion
              </p>
            </div>

            <div className="bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 card-shadow space-y-4 text-sm text-slate-600 leading-relaxed">
              <h3 className="font-bold text-slate-900 text-base flex items-center gap-2">
                <Brain className="w-5 h-5 text-teal-600" />
                Decision-Level Bayesian Late Fusion Rationale
              </h3>
              <p>
                In our research repository, <strong>Clinical Oral Photography</strong> (1,081 macroscopic images) and the <strong>NDU-UFES Histopathology Cohort</strong> (237 microscopic biopsy slides + clinical metadata) originate from two separate patient populations. Because no single dataset contains paired oral photos, biopsy slides, and clinical records for the identical individuals, attempting an end-to-end "Early Fusion" (feature concatenation) deep neural network would be scientifically unsound and require synthetic data pairing.
              </p>
              <p>
                Instead, our system implements a mathematically validated <strong>Calibrated Decision-Level Late Fusion Engine</strong>:
              </p>
              <ul className="list-disc pl-5 space-y-1.5 text-xs text-slate-700">
                <li><strong>Biopsy Histopathology ($w = 0.45$):</strong> PyTorch ResNet-18 trained on NDU-UFES TaskII slides (82.1% held-out test accuracy). Reflects tissue cellular architecture.</li>
                <li><strong>Oral Lesion Photography ($w = 0.35$):</strong> PyTorch ResNet-18 trained on oral lesion photographs. Detects superficial erythema, ulceration, and exophytic margins.</li>
                <li><strong>Clinical Demographics &amp; Habits ($w = 0.20$):</strong> Scikit-learn Random Forest trained on 23 one-hot features (87.2% accuracy, 98.8% ROC-AUC). Evaluates systemic oncological risk.</li>
              </ul>
            </div>
          </div>
        )}

        {/* TAB 6: MODEL EXPERIMENTS */}
        {currentTab === 'experiments' && <ExperimentsPage />}
      </main>

      {/* Footer */}
      <footer className="mt-auto border-t border-slate-200 bg-white py-6 text-xs text-slate-500 text-center">
        Oral Cancer AI &bull; True Multimodal Detection System &bull; Academic AI Research Prototype &bull; Not for Clinical Diagnostic Use
      </footer>
    </div>
  )
}
