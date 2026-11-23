import { useCallback, useEffect, useMemo, useState } from 'react'
import './App.css'
import { fetchMessages, fetchOpportunities, runScan } from './api.ts'
import type { Opportunity } from './api.ts'
import { useLogSocket } from './useLogSocket.ts'
import OpportunitiesTable from './components/OpportunitiesTable.tsx'
import RunScanButton from './components/RunScanButton.tsx'
import ActivityPanel from './components/ActivityPanel.tsx'
import AgentPyramid from './components/AgentPyramid.tsx'
import RunCompleteModal from './components/RunCompleteModal.tsx'
import type { Celebration } from './components/RunCompleteModal.tsx'
import { AGENT_NODE_MAP } from './agentNodes.ts'

const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 0,
})

function Icon({ name }: { name: 'grid' | 'spark' | 'tag' | 'activity' | 'arrow' | 'check' }) {
  const common = {
    width: 18,
    height: 18,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.8,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    'aria-hidden': true as const,
  }

  const paths = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></>,
    spark: <><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" /><path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z" /></>,
    tag: <><path d="M20.6 13.4 13.4 20.6a2 2 0 0 1-2.8 0L3 13V3h10l7.6 7.6a2 2 0 0 1 0 2.8Z" /><circle cx="8" cy="8" r="1.2" /></>,
    activity: <><path d="M3 12h4l3-8 4 16 3-8h4" /></>,
    arrow: <><path d="M7 17 17 7" /><path d="M8 7h9v9" /></>,
    check: <><path d="m5 12 4 4L19 6" /></>,
  }

  return <svg {...common}>{paths[name]}</svg>
}

function App() {
  const [opportunities, setOpportunities] = useState<Opportunity[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)
  const [runNotice, setRunNotice] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [activeSearch, setActiveSearch] = useState('')
  const [loadError, setLoadError] = useState<string | null>(null)
  const [activeAgent, setActiveAgent] = useState<string | null>(null)
  const [visitedAgents, setVisitedAgents] = useState<string[]>([])
  const [hasNewMessage, setHasNewMessage] = useState(false)
  const [messagesVersion, setMessagesVersion] = useState(0)
  const [celebration, setCelebration] = useState<Celebration | null>(null)

  const handleRunComplete = useCallback((opportunity: Opportunity | null) => {
    setIsRunning(false)
    setActiveAgent(null)
    setVisitedAgents([])
    if (opportunity) {
      setOpportunities((prev) => [opportunity, ...prev.filter((item) => item.id !== opportunity.id)])
      setHasNewMessage(true)
      setMessagesVersion((v) => v + 1)
      setRunNotice(
        activeSearch
          ? `Search for “${activeSearch}” complete. A new opportunity was added.`
          : 'Scan complete. A new opportunity was added to your list.',
      )
      fetchMessages()
        .then((messages) => setCelebration({ opportunity, message: messages[0] ?? null }))
        .catch(() => setCelebration({ opportunity, message: null }))
    } else {
      setRunNotice(
        activeSearch
          ? `No opportunity was saved for “${activeSearch}”. Check live activity for feed matches and scan details.`
          : 'This scan finished without saving an opportunity. Check live activity for details.',
      )
    }
  }, [activeSearch])

  const { logs, connected } = useLogSocket({ onRunComplete: handleRunComplete })

  useEffect(() => {
    if (logs.length === 0) return
    const nodeId = AGENT_NODE_MAP[logs[logs.length - 1].agent]
    if (!nodeId) return
    setActiveAgent(nodeId)
    setVisitedAgents((prev) => (prev.includes(nodeId) ? prev : [...prev, nodeId]))
  }, [logs])

  useEffect(() => {
    fetchOpportunities()
      .then(setOpportunities)
      .catch((err) => setLoadError(err instanceof Error ? err.message : 'Failed to load opportunities'))
  }, [])

  const totals = useMemo(() => {
    const savings = opportunities.reduce((sum, item) => sum + item.discount, 0)
    const averagePrice = opportunities.length
      ? opportunities.reduce((sum, item) => sum + item.deal_price, 0) / opportunities.length
      : null
    return { savings, averagePrice }
  }, [opportunities])

  const handleRun = async () => {
    const query = searchQuery.trim()
    setRunError(null)
    setRunNotice(null)
    try {
      await runScan(query)
      setIsRunning(true)
      setActiveSearch(query)
      setActiveAgent(null)
      setVisitedAgents([])
      setCelebration(null)
    } catch (err) {
      setRunError(err instanceof Error ? err.message : 'Could not start the scan. Please try again.')
    }
  }

  const scrollTo = (section: string) => {
    document.getElementById(section)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="Pricer home">
          <span className="brand-mark"><span /></span>
          <span className="brand-name">pricer<span>.</span><small>DEAL INTELLIGENCE</small></span>
        </a>

        <div className="sidebar-label">Workspace</div>
        <nav className="side-nav" aria-label="Main navigation">
          <button className="nav-item nav-item-active" aria-label="Overview" title="Overview" onClick={() => scrollTo('overview')}>
            <Icon name="grid" /><span>Overview</span>
          </button>
          <button className="nav-item" aria-label="Opportunities" title="Opportunities" onClick={() => scrollTo('opportunities')}>
            <Icon name="tag" /><span>Opportunities</span>
            {opportunities.length > 0 && <span className="nav-count">{opportunities.length}</span>}
          </button>
          <button className="nav-item" aria-label="Live activity" title="Live activity" onClick={() => scrollTo('activity')}>
            <Icon name="activity" /><span>Live activity</span>
            <span className={`nav-live ${connected ? 'is-online' : ''}`} />
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-status">
            <span className={`status-led ${connected ? 'is-online' : ''}`} />
            <div><strong>{connected ? 'System connected' : 'Reconnecting'}</strong><small>Local GPU · CUDA</small></div>
          </div>
          <div className="sidebar-user"><span className="user-avatar">P</span><div><strong>Pricer workspace</strong><small>Personal edition</small></div></div>
        </div>
      </aside>

      <main className="main-area" id="overview">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><span className="breadcrumb-sep">/</span><strong>Overview</strong></div>
          <div className="topbar-right">
            <span className={`connection-pill ${connected ? 'connected' : ''}`}>
              <span />{connected ? 'Live connection' : 'Reconnecting'}
            </span>
            <span className="topbar-avatar">P</span>
          </div>
        </header>

        <div className="page-content">
          <section className="page-heading">
            <div>
              <p className="eyebrow">YOUR PERSONAL DEAL DESK</p>
              <h1>Good deals, found.</h1>
              <p className="page-subtitle">Compare live listings against estimated market value, all in one place.</p>
            </div>
            <div className="heading-date"><span className="date-dot" />Local GPU ready</div>
          </section>

          <section className="metric-grid" aria-label="Deal summary">
            <article className="metric-card">
              <div className="metric-top"><span className="metric-icon icon-violet"><Icon name="tag" /></span><span className="metric-caption">ALL TIME</span></div>
              <p className="metric-value">{opportunities.length}</p>
              <p className="metric-label">Opportunities found</p>
            </article>
            <article className="metric-card">
              <div className="metric-top"><span className="metric-icon icon-green"><Icon name="spark" /></span><span className="metric-caption">ESTIMATED</span></div>
              <p className="metric-value">{currencyFormatter.format(totals.savings)}</p>
              <p className="metric-label">Potential savings found</p>
            </article>
            <article className="metric-card">
              <div className="metric-top"><span className="metric-icon icon-blue"><Icon name="activity" /></span><span className="metric-caption">AVERAGE</span></div>
              <p className="metric-value">{totals.averagePrice === null ? '—' : currencyFormatter.format(totals.averagePrice)}</p>
              <p className="metric-label">Deal price</p>
            </article>
            <article className={`metric-card scan-metric ${isRunning ? 'scan-metric-running' : ''}`}>
              <div className="metric-top"><span className="metric-icon icon-amber"><Icon name={isRunning ? 'activity' : 'check'} /></span><span className="metric-caption">SCANNER</span></div>
              <p className="metric-value metric-status">{isRunning ? 'In progress' : 'Ready'}</p>
              <p className="metric-label">7 pricing agents · local inference</p>
            </article>
          </section>

          <section className="scan-hero" aria-labelledby="scan-title">
            <div className="hero-copy">
              <span className="hero-kicker"><Icon name="spark" /> SMARTER SHOPPING STARTS HERE</span>
              <h2 id="scan-title">Find your next<br />great-value purchase.</h2>
                <p>Search current deal listings, then let Pricer's AI ensemble estimate what each item is really worth.</p>
              {(runNotice || runError) && <div className={`run-feedback ${runError ? 'feedback-error' : ''}`} role={runError ? 'alert' : 'status'}>{runError || runNotice}</div>}
            </div>
            <div className="hero-action">
              <label className="search-label" htmlFor="deal-search">What are you looking for?</label>
              <div className="search-input-wrap">
                <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><circle cx="10.8" cy="10.8" r="6.8" /><path d="m16 16 4.5 4.5" /></svg>
                <input
                  id="deal-search"
                  type="search"
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault()
                      void handleRun()
                    }
                  }}
                  maxLength={120}
                  placeholder="e.g. gaming laptop, air fryer"
                  aria-label="Search deal titles and summaries"
                  disabled={isRunning}
                />
              </div>
              <RunScanButton isRunning={isRunning} onRun={handleRun} />
              <span className="hero-footnote"><span className="hero-footnote-dot" />Searches DealNews, Slickdeals, and TechBargains; blank scans broadly.</span>
            </div>
            <div className="hero-decoration hero-decoration-one" />
            <div className="hero-decoration hero-decoration-two" />
          </section>

          <section className="dashboard-grid">
            <div className="dashboard-card agent-card">
              <div className="section-heading">
                <div><p className="eyebrow">VALUATION PIPELINE</p><h2>Agent workspace</h2></div>
                <span className={`small-state ${isRunning ? 'state-working' : ''}`}><span />{isRunning ? 'Working' : 'Standing by'}</span>
              </div>
              <AgentPyramid activeAgent={activeAgent} visitedAgents={visitedAgents} isRunning={isRunning} />
              <div className="agent-card-footer"><span><i className="legend-dot legend-active" />Active agent</span><span><i className="legend-dot legend-done" />Completed</span><span><i className="legend-dot legend-idle" />Waiting</span></div>
            </div>

            <div className="dashboard-card activity-card" id="activity">
              <div className="section-heading activity-heading">
                <div><p className="eyebrow">WHAT'S HAPPENING</p><h2>Live activity</h2></div>
                <span className={`small-state ${connected ? 'state-working' : ''}`}><span />{connected ? 'Live' : 'Offline'}</span>
              </div>
              <ActivityPanel
                logs={logs}
                connected={connected}
                hasNewMessage={hasNewMessage}
                messagesVersion={messagesVersion}
                onMessagesOpened={() => setHasNewMessage(false)}
              />
            </div>
          </section>

          <section className="opportunities-section" id="opportunities">
            <div className="section-heading opportunity-heading">
              <div><p className="eyebrow">YOUR DEAL HISTORY</p><h2>Opportunities</h2></div>
              <span className="opportunity-count">{opportunities.length} {opportunities.length === 1 ? 'opportunity' : 'opportunities'}</span>
            </div>
            {loadError ? <div className="dashboard-card load-error" role="alert">{loadError}</div> : <OpportunitiesTable opportunities={opportunities} />}
          </section>

          <footer className="page-footer"><span>Pricer</span><span>Private, local-first deal intelligence</span><a href="#overview">Back to top <Icon name="arrow" /></a></footer>
        </div>
      </main>

      {celebration && <RunCompleteModal celebration={celebration} onClose={() => setCelebration(null)} />}
    </div>
  )
}

export default App
