import { useState } from 'react'
import './App.css'
import DashboardPage from './pages/Dashboard/DashboardPage'

const ROUTE_SECTIONS = {
  '/': 'dashboard',
  '/dashboard': 'dashboard',
  '/portfolio': 'portfolio',
  '/analyze': 'analysis',
  '/analysis': 'analysis',
  '/holdings': 'holdings',
}

function getInitialSection() {
  const pathname =
    typeof window.location.pathname === 'string' && window.location.pathname.trim()
      ? window.location.pathname.trim().toLowerCase()
      : '/'

  return ROUTE_SECTIONS[pathname] || 'dashboard'
}

function App() {
  const [activeSection, setActiveSection] = useState(() => getInitialSection())

  const scrollToSection = (section) => {
    setActiveSection(section)
    const target = document.getElementById(section)
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <div className="app-title">TradeSense</div>
          <p className="app-subtitle">Single intelligent dashboard for portfolio and analysis workflows</p>
        </div>
        <nav className="app-nav">
          <button
            type="button"
            className={`nav-button ${activeSection === 'dashboard' ? 'active' : ''}`}
            onClick={() => scrollToSection('dashboard')}
          >
            Dashboard
          </button>
          <button
            type="button"
            className={`nav-button ${activeSection === 'portfolio' ? 'active' : ''}`}
            onClick={() => scrollToSection('portfolio')}
          >
            Portfolio
          </button>
          <button
            type="button"
            className={`nav-button ${activeSection === 'analysis' ? 'active' : ''}`}
            onClick={() => scrollToSection('analysis')}
          >
            Analyze
          </button>
        </nav>
      </header>
      <main className="app-main">
        <DashboardPage initialSection={activeSection} />
      </main>
    </div>
  )
}

export default App
