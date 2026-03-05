import { useState } from 'react'
import './App.css'
import AnalysisPage from './pages/Analysis/AnalysisPage'
import PortfolioPage from './pages/Portfolio/PortfolioPage'

const VIEWS = {
  DASHBOARD: 'dashboard',
  PORTFOLIO: 'portfolio',
}

function App() {
  const [activeView, setActiveView] = useState(VIEWS.DASHBOARD)

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-title">TradeSense</div>
        <nav className="app-nav">
          <button
            type="button"
            className={`nav-button ${activeView === VIEWS.DASHBOARD ? 'active' : ''}`}
            onClick={() => setActiveView(VIEWS.DASHBOARD)}
          >
            Analysis
          </button>
          <button
            type="button"
            className={`nav-button ${activeView === VIEWS.PORTFOLIO ? 'active' : ''}`}
            onClick={() => setActiveView(VIEWS.PORTFOLIO)}
          >
            Portfolio
          </button>
        </nav>
      </header>
      <main className="app-main">
        {activeView === VIEWS.DASHBOARD ? <AnalysisPage /> : <PortfolioPage />}
      </main>
    </div>
  )
}

export default App
