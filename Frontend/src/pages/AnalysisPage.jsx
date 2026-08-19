import { useState } from 'react'

function AnalysisPage({ file, onBack }) {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])

  const handleSend = () => {
    if (!question.trim()) return

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: question,
    }

    setMessages((prevMessages) => [
      ...prevMessages,
      userMessage,
    ])

    setQuestion('')

    // Temporary AI response
    setTimeout(() => {
      const aiMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content:
          'This is a temporary AI response. Once the backend is connected, I will answer questions using the contents of your uploaded document.',
      }

      setMessages((prevMessages) => [
        ...prevMessages,
        aiMessage,
      ])
    }, 700)
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter') {
      handleSend()
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
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

          <button
            onClick={onBack}
            className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 transition hover:border-slate-500 hover:text-white"
          >
            Upload Another
          </button>
        </div>
      </header>

      {/* Main Layout */}
      <main className="mx-auto grid min-h-[calc(100vh-81px)] max-w-7xl grid-cols-1 md:grid-cols-4">
        
        {/* Sidebar */}
        <aside className="border-b border-slate-800 p-6 md:border-b-0 md:border-r">
          <h2 className="mb-6 text-sm font-semibold uppercase tracking-wider text-slate-400">
            Document
          </h2>

          <div className="rounded-xl border border-slate-700 bg-slate-900 p-5">
            <div className="mb-4 text-3xl">
              📄
            </div>

            <p className="break-words font-medium text-white">
              {file?.name}
            </p>

            <p className="mt-2 text-sm text-slate-400">
              {file
                ? `${(file.size / 1024 / 1024).toFixed(2)} MB`
                : 'No document selected'}
            </p>

            <div className="mt-5 border-t border-slate-700 pt-4">
              <p className="text-xs text-slate-500">
                Status
              </p>

              <p className="mt-1 text-sm font-medium text-green-400">
                Ready for analysis
              </p>
            </div>
          </div>
        </aside>

        {/* Chat Area */}
        <section className="flex min-h-[600px] flex-col md:col-span-3">
          
          {/* Chat Header */}
          <div className="border-b border-slate-800 p-6">
            <h2 className="text-xl font-semibold">
              Ask about your document
            </h2>

            <p className="mt-1 text-sm text-slate-400">
              Ask questions and get answers based on your uploaded document.
            </p>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6">
            {messages.length === 0 ? (
              <div className="flex h-full items-center justify-center">
                <div className="max-w-md text-center">
                  <div className="mb-4 text-5xl">
                    ✨
                  </div>

                  <h3 className="text-lg font-semibold">
                    Your document is ready
                  </h3>

                  <p className="mt-2 text-sm leading-6 text-slate-400">
                    Ask a question to start exploring the contents of your document.
                  </p>
                </div>
              </div>
            ) : (
              <div className="mx-auto flex max-w-3xl flex-col gap-5">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${
                      message.role === 'user'
                        ? 'justify-end'
                        : 'justify-start'
                    }`}
                  >
                    <div
                      className={`max-w-[80%] rounded-2xl px-5 py-3 text-sm leading-6 ${
                        message.role === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'border border-slate-700 bg-slate-900 text-slate-200'
                      }`}
                    >
                      {message.content}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Input */}
          <div className="border-t border-slate-800 p-6">
            <div className="flex gap-3">
              <input
                type="text"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about your document..."
                className="flex-1 rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm text-white outline-none placeholder:text-slate-500 focus:border-blue-500"
              />

              <button
                onClick={handleSend}
                className="rounded-xl bg-blue-600 px-6 py-3 font-medium text-white transition hover:bg-blue-500"
              >
                Send
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

export default AnalysisPage

