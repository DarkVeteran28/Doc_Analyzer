import { useRef, useState } from 'react'

function FileUpload({ onAnalyze }) {
  const [selectedFile, setSelectedFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState('')

  const fileInputRef = useRef(null)

  const handleFile = (file) => {
    if (!file) return

    if (file.type !== 'application/pdf') {
      setError('Please select a valid PDF file.')
      setSelectedFile(null)
      return
    }

    setError('')
    setSelectedFile(file)
  }

  const handleFileChange = (event) => {
    const file = event.target.files[0]
    handleFile(file)
  }

  const handleDragOver = (event) => {
    event.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)

    const file = event.dataTransfer.files[0]
    handleFile(file)
  }

  const handleClick = () => {
    fileInputRef.current.click()
  }

  const handleRemoveFile = () => {
    setSelectedFile(null)
    setError('')

    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleAnalyze = () => {
  if (selectedFile) {
    onAnalyze(selectedFile)
  }
}

  return (
    <div className="mt-10 w-full max-w-2xl">
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        onChange={handleFileChange}
        className="hidden"
      />

      {!selectedFile ? (
        <>
          <div
            onClick={handleClick}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition ${
              isDragging
                ? 'border-blue-500 bg-blue-500/10'
                : 'border-slate-700 bg-slate-900/50 hover:border-blue-500/70 hover:bg-slate-900'
            }`}
          >
            <div className="mb-4 text-4xl">📄</div>

            <h3 className="text-lg font-semibold text-white">
              Drop your PDF here
            </h3>

            <p className="mt-2 text-sm text-slate-400">
              Drag and drop your document, or click to browse
            </p>

            <p className="mt-4 text-xs text-slate-500">
              PDF files only
            </p>
          </div>

          {error && (
            <p className="mt-3 text-center text-sm text-red-400">
              {error}
            </p>
          )}
        </>
      ) : (
        <div className="rounded-2xl border border-slate-700 bg-slate-900 px-6 py-6 text-center">
          <div className="mb-3 text-3xl">📄</div>

          <p className="font-medium text-white">
            {selectedFile.name}
          </p>

          <p className="mt-1 text-sm text-slate-400">
            {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
          </p>

          <div className="mt-6 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <button
              onClick={handleAnalyze}
              className="rounded-xl bg-blue-600 px-6 py-3 font-medium text-white transition hover:bg-blue-500"
            >
              Analyze Document
            </button>

            <button
              onClick={handleClick}
              className="rounded-xl border border-slate-700 px-5 py-3 text-sm font-medium text-slate-300 transition hover:border-slate-500 hover:text-white"
            >
              Choose another
            </button>

            <button
              onClick={handleRemoveFile}
              className="px-4 py-3 text-sm font-medium text-red-400 transition hover:text-red-300"
            >
              Remove
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default FileUpload
