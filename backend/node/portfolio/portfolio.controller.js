const portfolioService = require('./portfolio.service');

function handleError(res, error) {
  const status =
    typeof error?.status === 'number' && Number.isInteger(error.status)
      ? error.status
      : 500;

  return res.status(status).json({
    error: error?.message || 'Internal server error',
  });
}

async function addHolding(req, res) {
  try {
    const item = await portfolioService.addHolding(req.body || {});
    return res.status(201).json({ success: true, item });
  } catch (error) {
    return handleError(res, error);
  }
}

async function getHoldings(req, res) {
  try {
    const payload = await portfolioService.getHoldings();
    return res.status(200).json(payload);
  } catch (error) {
    return handleError(res, error);
  }
}

async function getHistory(req, res) {
  try {
    const payload = await portfolioService.getPortfolioHistory(req.query?.days);
    return res.status(200).json(payload);
  } catch (error) {
    return handleError(res, error);
  }
}

async function getInsights(req, res) {
  try {
    const payload = await portfolioService.getPortfolioInsights(req.query?.days);
    return res.status(200).json(payload);
  } catch (error) {
    return handleError(res, error);
  }
}

async function getAdvisor(req, res) {
  try {
    const payload = await portfolioService.getPortfolioAdvisor(req.query?.days);
    return res.status(200).json(payload);
  } catch (error) {
    return handleError(res, error);
  }
}

async function deleteHolding(req, res) {
  try {
    await portfolioService.deleteHolding(req.params.id);
    return res.status(200).json({ success: true });
  } catch (error) {
    return handleError(res, error);
  }
}

module.exports = {
  addHolding,
  getAdvisor,
  deleteHolding,
  getHistory,
  getInsights,
  getHoldings,
};
