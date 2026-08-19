import FileUpload from './components/FileUpload'

function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-white">
      
      {/* Navbar */}
      <header className="border-b border-slate-800">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-600 font-bold">
              D
            </div>

            <h1 className="text-xl font-semibold">
              Document Analyzer
            </h1>
          </div>

          <div className="rounded-full border border-slate-700 px-4 py-2 text-sm text-slate-300">
            AI Powered
          </div>

        </div>
      </header>

      {/* Hero Section */}
      <main className="mx-auto flex min-h-[calc(100vh-81px)] max-w-7xl items-center justify-center px-6">
        
        <section className="max-w-3xl text-center">
          
          <div className="mb-6 inline-flex rounded-full border border-blue-500/30 bg-blue-500/10 px-4 py-2 text-sm text-blue-400">
            AI-powered document analysis
          </div>

          <h2 className="text-5xl font-bold tracking-tight sm:text-6xl">
            Understand your documents
            <span className="block text-blue-500">
              with AI.
            </span>
          </h2>

          <p className="mx-auto mt-6 max-w-2xl text-lg leading-8 text-slate-400">
            Upload your PDF documents and ask intelligent questions.
            Get fast, accurate answers powered by AI.
          </p>

          <div className="translate-x-15">
          <FileUpload />
          </div>

        </section>

      </main>

    </div>
  )
}

export default App

