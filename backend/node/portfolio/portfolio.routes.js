const express = require('express');

const controller = require('./portfolio.controller');

const router = express.Router();

router.post('/portfolio/add', controller.addHolding);
router.get('/portfolio', controller.getHoldings);
router.post('/portfolio/trades', controller.createTrade);
router.get('/portfolio/transactions', controller.getTransactions);
router.post('/portfolio/positions/:symbol/adjust', controller.adjustPosition);
router.get('/portfolio/history', controller.getHistory);
router.get('/portfolio/insights', controller.getInsights);
router.get('/portfolio/advisor', controller.getAdvisor);
router.delete('/portfolio/:id', controller.deleteHolding);

module.exports = router;
