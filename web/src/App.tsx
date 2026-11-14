import { useEffect, useState } from 'react'
import Markdown from 'react-markdown'

type Model = 'small' | 'base'

interface Source {
  n: number
  url: string
  title: string
  score: number
  text: string
}

interface Result {
  sources: Source[]
  retrieve_ms: number
  answer?: string
  generate_ms?: number
  input_tokens?: number
  output_tokens?: number
  cost_usd?: number
}

interface Health {
  chunks: Record<Model, number>
  llm: string
}

const EXAMPLES = [
  'What is a Python decorator?',
  'How do I create a thread in Java?',
  'How do I configure CORS in Spring Boot?',
  'How do I deploy a Kubernetes operator written in Rust?',
]

const MODEL_LABEL: Record<Model, string> = {
  small: 'bge-small (384d)',
  base: 'bge-base (768d)',
}

/** Turn [1] citations into links to the source cards, but never inside code, where a[1] is code. */
function linkCitations(md: string): string {
  return md
    .split(/(```[\s\S]*?```)/g)
    .map((part) =>
      part.startsWith('```')
        ? part
        : part
            .split(/(`[^`\n]*`)/g)
            .map((p) => (p.startsWith('`') ? p : p.replace(/\[(\d{1,2})\]/g, '[[$1]](#source-$1)')))
            .join(''),
    )
    .join('')
}

export default function App() {
  const [question, setQuestion] = useState('')
  const [model, setModel] = useState<Model>('small')
  const [topK, setTopK] = useState(5)
  const [result, setResult] = useState<Result | null>(null)
  const [asked, setAsked] = useState('')
  const [loading, setLoading] = useState<'search' | 'ask' | null>(null)
  const [error, setError] = useState('')
  const [health, setHealth] = useState<Health | null>(null)
  const [highlight, setHighlight] = useState<number | null>(null)

  useEffect(() => {
    fetch('/api/health')
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then(setHealth)
      .catch(() => setError('API not reachable. Start it with: uvicorn api.main:app --port 8010'))
  }, [])

  async function run(kind: 'search' | 'ask') {
    const q = question.trim()
    if (loading) return
    if (q.length < 3) {
      setError('Question must be at least 3 characters.')
      return
    }
    setLoading(kind)
    setError('')
    setHighlight(null)
    try {
      const r = await fetch(`/api/${kind}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, model, top_k: topK }),
      })
      const body = await r.json()
      if (!r.ok) {
        const detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
        throw new Error(detail || r.statusText)
      }
      setResult(body)
      setAsked(q)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(null)
    }
  }

  function jumpTo(n: number) {
    setHighlight(n)
    document.getElementById(`source-${n}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  const topScore = result?.sources[0]?.score

  return (
    <main className="page">
      <header className="header">
        <h1>Ask LoveMeSomeCoding</h1>
        <p className="muted">
          RAG test console · {health ? `${health.chunks[model].toLocaleString()} chunks · ${health.llm}` : 'connecting…'}
        </p>
      </header>

      <form
        className="ask"
        onSubmit={(e) => {
          e.preventDefault()
          run('ask')
        }}
      >
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) run('ask')
          }}
          placeholder="Ask a programming question…"
          rows={2}
          aria-label="Question"
        />
        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button type="button" key={ex} className="chip" onClick={() => setQuestion(ex)}>
              {ex}
            </button>
          ))}
        </div>
        <div className="controls">
          <label>
            Embedding
            <select value={model} onChange={(e) => setModel(e.target.value as Model)}>
              {(Object.keys(MODEL_LABEL) as Model[]).map((m) => (
                <option key={m} value={m} disabled={health ? !health.chunks[m] : false}>
                  {MODEL_LABEL[m]}
                  {health && !health.chunks[m] ? ' (not ingested)' : ''}
                </option>
              ))}
            </select>
          </label>
          <label>
            Top k
            <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
              {[3, 5, 8, 10].map((k) => (
                <option key={k}>{k}</option>
              ))}
            </select>
          </label>
          <span className="spacer" />
          <button type="button" className="secondary" disabled={!!loading} onClick={() => run('search')}>
            {loading === 'search' ? 'Searching…' : 'Search only (free)'}
          </button>
          <button type="submit" disabled={!!loading}>
            {loading === 'ask' ? 'Asking Claude…' : 'Ask Claude (~$0.02)'}
          </button>
        </div>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <section className="result">
          <p className="meta">
            “{asked}” · retrieval {Math.round(result.retrieve_ms)} ms
            {result.generate_ms !== undefined && (
              <>
                {' '}· Claude {(result.generate_ms / 1000).toFixed(1)} s · {result.input_tokens} in /{' '}
                {result.output_tokens} out tokens · ${result.cost_usd?.toFixed(4)}
              </>
            )}
          </p>

          {result.answer !== undefined && (
            <article
              className="answer"
              onClick={(e) => {
                const href = (e.target as HTMLElement).closest('a')?.getAttribute('href')
                if (href?.startsWith('#source-')) {
                  e.preventDefault()
                  jumpTo(Number(href.slice(8)))
                }
              }}
            >
              <Markdown
                components={{
                  a: ({ href, children }) =>
                    href?.startsWith('#') ? (
                      <a href={href} className="cite">{children}</a>
                    ) : (
                      <a href={href} target="_blank" rel="noreferrer">{children}</a>
                    ),
                }}
              >
                {linkCitations(result.answer)}
              </Markdown>
            </article>
          )}

          <h2>
            Retrieved sources
            {topScore !== undefined && topScore < 0.75 && (
              <span className="warn"> · top score {topScore.toFixed(3)} is low; the site may not cover this</span>
            )}
          </h2>
          <ol className="sources">
            {result.sources.map((s) => (
              <li key={s.n} id={`source-${s.n}`} className={highlight === s.n ? 'source hl' : 'source'}>
                <div className="source-head">
                  <span className="num">[{s.n}]</span>
                  <span className="score" title="cosine similarity">
                    <span className="bar" style={{ width: `${Math.max(0, (s.score - 0.5) * 200)}%` }} />
                    {s.score.toFixed(3)}
                  </span>
                  <a href={s.url} target="_blank" rel="noreferrer">
                    {s.title}
                  </a>
                </div>
                <div className="url">{s.url.replace('https://lovemesomecoding.com', '')}</div>
                <details open={highlight === s.n}>
                  <summary>Chunk text ({s.text.length.toLocaleString()} chars, exactly what Claude saw)</summary>
                  <pre>{s.text}</pre>
                </details>
              </li>
            ))}
          </ol>
        </section>
      )}
    </main>
  )
}
