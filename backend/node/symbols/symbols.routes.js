const express = require('express');

const controller = require('./symbols.controller');

const router = express.Router();

router.get('/symbols/search', controller.searchSymbols);
router.get('/symbols/normalize/:symbol', controller.normalizeSymbol);

module.exports = router;
