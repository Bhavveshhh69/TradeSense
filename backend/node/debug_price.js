const axios = require('axios');

(async () => {
  const symbols = ['AAPL', 'NVDA', 'RELIANCE.NS', 'TCS.NS'];
  for (const sym of symbols) {
    try {
      const res = await axios.get(`http://127.0.0.1:8000/market/latest-price/${encodeURIComponent(sym)}`);
      console.log(sym, res.data.price);
    } catch (err) {
      console.error(sym, 'error', err.message);
    }
  }
})();
