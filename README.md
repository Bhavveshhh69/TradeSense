# TradeSense: AI-Powered Portfolio Intelligence

TradeSense is a sophisticated market analysis and portfolio intelligence system designed to provide deep insights into financial markets and personal investment portfolios. It leverages a combination of machine learning, deterministic reasoning, and AI-driven explanations to help users make informed decisions.

## Problem Solved

In today's complex financial markets, investors face a deluge of data and a shortage of clear, actionable insights. TradeSense addresses this by:
- **Automating Analysis**: Replacing manual, time-consuming market research with an automated ML-driven pipeline.
- **Providing Clarity**: Translating complex model outputs into deterministic, easy-to-understand decisions and context.
- **Unifying Portfolio View**: Offering a single, intelligent platform to track, analyze, and get advice on personal stock portfolios, even across different markets like the US and India.
- **Explaining the "Why"**: Going beyond black-box predictions by offering AI-generated explanations for its analysis.

## Key Features

- **AI-Driven Stock Analysis**: Get ML-powered predictions, decisions (BUY/SELL/HOLD), and contextual summaries for any stock.
- **Comprehensive Portfolio Tracking**: Manage your stock holdings, track their performance, and visualize your asset allocation.
- **Advanced Portfolio Intelligence**: Receive insights on portfolio concentration, diversification, volatility, and performance.
- **Actionable Advisor**: Get rule-based recommendations to improve your portfolio's health (e.g., rebalancing concentrated positions).
- **Multi-Currency Support**: Analyze portfolios containing stocks from different markets (e.g., NASDAQ and NSE) with proper currency conversion and analysis.
- **AI Explanation Layer**: Understand the reasoning behind the system's analysis with an optional RAG- and LLM-powered explanation engine.

## System Architecture

TradeSense employs a modern microservice-oriented architecture:

- **Frontend**: A responsive **React** application provides the user interface for portfolio management and stock analysis.
- **Node.js Backend (API Gateway)**: An **Express** server acts as the primary API for the frontend. It handles user requests, manages portfolio data (stored in a JSON file), and communicates with the Python backend. It serves as an orchestration and enrichment layer.
- **Python Backend (ML & Data Service)**: A **FastAPI** server that exposes the core machine learning, data, and intelligence capabilities. This includes the ML prediction pipeline, market data access, and the AI explanation services.

The system is designed with a clear separation of concerns, which allows for scalability and maintainability.

## Technology Stack

- **Frontend**: React, Vite, Axios, Recharts
- **Backend (Node.js)**: Express.js, Axios, Jest
- **Backend (Python)**: FastAPI, Uvicorn, Pandas, NumPy, Scikit-learn, XGBoost, Joblib, Pytest, Transformers, Torch, FAISS-CPU
- **Data Sources**: yfinance (market data), Finnhub (news)

## Data Flow

A typical request, such as analyzing a stock, follows this path:
1. The **React Frontend** sends a request to the **Node.js API Gateway**.
2. The **Node.js Gateway** validates the request, checks its in-memory cache, and then calls the relevant endpoints on the **Python ML Service**. This might involve fetching the latest price and running a prediction.
3. The **Python Service** executes the ML pipeline: it fetches market data from **yfinance**, builds features, and uses the trained XGBoost model to generate a prediction and deterministic insights.
4. The Python service returns the structured data to the Node.js gateway.
5. The **Node.js Gateway** enriches this data (e.g., adding a human-readable recommendation) and sends the final response to the frontend.
6. The **React Frontend** displays the information to the user.

## Machine Learning Pipeline

The core of TradeSense is its ML pipeline for stock prediction:

1.  **Data Ingestion**: Fetches historical market data (OHLCV) using the `yfinance` library.
2.  **Indicator Calculation**: Computes technical indicators like RSI, EMA, and MACD.
3.  **Feature Engineering**: Creates a feature matrix from the raw data and indicators.
4.  **Prediction**: A pre-trained **XGBoost** model (`xgboost.joblib`) predicts the probability of a 5-day price continuation. The model is calibrated to ensure the probabilities are reliable.
5.  **Deterministic Reasoning**: The model's probability output is fed into a rule-based engine that generates a clear `decision` (BUY, SELL, HOLD), `confidence_level`, and contextual summaries about market trends and risks.

## Market Intelligence Layer

This layer is responsible for providing real-time and historical market data. It is primarily implemented in the Python backend and uses the `yfinance` library to fetch data from Yahoo Finance.

## Portfolio Intelligence Engine

Located in the Node.js backend, this engine provides high-level insights into the user's portfolio. It calculates:
- **Total Portfolio Value & P&L**: With correct handling of multi-currency assets.
- **Asset Allocation**: The weight of each holding in the portfolio.
- **Concentration & Diversification**: Metrics to assess portfolio risk.
- **Volatility & Performance**: Analysis of the best and worst-performing assets.

## AI Explanation Layer

For users who want to dig deeper, TradeSense offers an advanced AI explanation layer. When enabled, it uses:
- **RAG (Retrieval-Augmented Generation)**: To retrieve historical context about past predictions for a given stock from a local vector store.
- **LLM Integration (OpenAI/Groq)**: To generate a human-readable explanation of the current analysis, incorporating the latest prediction, deterministic insights, and historical context.

## External APIs Used

- **yfinance**: The primary source for historical and real-time market data.
- **Finnhub**: An optional source for news headlines to power the sentiment analysis feature. Requires a `FINNHUB_API_KEY`.
- **OpenAI/Groq**: Optional for the AI Explanation Layer. Requires an `OPENAI_API_KEY` or `GROQ_API_KEY`.

## How the System Runs (Node + Python interaction)

The two backends work in concert:
- The **Python backend** is the "brain," focusing on complex data processing and machine learning. It runs as a standalone FastAPI server.
- The **Node.js backend** is the "face" for the frontend, providing a stable API, handling user-specific data like portfolios, and orchestrating calls to the Python backend.

This dual-backend architecture allows for using the best tool for the job: Python for data science and Node.js for scalable web services.

## Environment Variables

The system uses a `.env` file in the `backend/node` directory to manage configuration and API keys. Key variables include:

- `PORT`: The port for the Node.js server (e.g., 3000).
- `REASONING_URL`: The URL of the Python FastAPI server (e.g., `http://localhost:8000/predict`).
- `FINNHUB_API_KEY`: For news fetching.
- `GROQ_API_KEY` / `OPENAI_API_KEY`: For the AI explanation layer.

## How to Run the Project

1.  **Start the Python Backend:**
    ```sh
    cd backend/python
    # Install dependencies
    pip install -r requirements.txt
    # Run the server
    uvicorn tradesense.api:app --reload --port 8000
    ```

2.  **Start the Node.js Backend:**
    ```sh
    cd backend/node
    # Install dependencies
    npm install
    # Run the server
    node server/index.js
    ```

3.  **Start the Frontend:**
    ```sh
    cd frontend
    # Install dependencies
    npm install
    # Run the development server
    npm run dev
    ```

You can now access the application at `http://localhost:5173`.

## Testing Overview

The project has a strong emphasis on testing:
- **Python**: Uses `pytest` for unit and integration tests covering data processing, feature engineering, ML modeling, and API endpoints.
- **Node.js**: Uses `Jest` and `Supertest` to test the API endpoints, services, and repository layers.

To run the tests:
- **Python**: `cd backend/python && pytest`
- **Node.js**: `cd backend/node && npm test`

## Limitations

- The portfolio data is stored in a local JSON file, which is not suitable for a multi-user or production environment.
- The system's performance depends on the availability of external APIs like Yahoo Finance.

## Future Improvements

- Migrate portfolio storage to a database for better scalability and multi-user support.
- Expand the range of financial instruments beyond stocks.
- Enhance the Portfolio Advisor with more sophisticated, customizable strategies.
