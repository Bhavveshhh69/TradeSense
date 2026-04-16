const express = require('express');

const commandCenterService = require('../services/command_center.service');

const router = express.Router();

router.get('/command-center', async (req, res) => {
  try {
    const payload = await commandCenterService.getCommandCenter();
    return res.status(200).json(payload);
  } catch (error) {
    const status =
      typeof error?.status === 'number' && Number.isInteger(error.status) ? error.status : 500;
    return res.status(status).json({
      error: error?.message || 'Unable to load command center',
    });
  }
});

module.exports = router;
