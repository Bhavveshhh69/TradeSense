const request = require('supertest');
const axios = require('axios');

const app = require('../index');
const cache = require('../cache/memoryCache');

jest.mock('axios');

beforeEach(() => {
  cache.clear();
  axios.post.mockReset();
});

test('POST /api/analyze returns 200 for valid input', async () => {
  axios.post.mockResolvedValue({ data: { ok: true } });

  const response = await request(app)
    .post('/api/analyze')
    .send({ symbol: 'AAPL', payload: { foo: 'bar' } });

  expect(response.status).toBe(200);
  expect(response.body).toEqual({ ok: true });
  expect(axios.post).toHaveBeenCalledTimes(1);
  expect(axios.post).toHaveBeenCalledWith(
    'http://localhost:8000/reason',
    { foo: 'bar' },
    expect.objectContaining({ timeout: expect.any(Number) })
  );
});

test('POST /api/analyze reuses cached response', async () => {
  axios.post.mockResolvedValue({ data: { cached: true } });

  const payload = { symbol: 'AAPL', payload: { alpha: 1 } };

  const first = await request(app).post('/api/analyze').send(payload);
  const second = await request(app).post('/api/analyze').send(payload);

  expect(first.status).toBe(200);
  expect(second.status).toBe(200);
  expect(second.body).toEqual({ cached: true });
  expect(axios.post).toHaveBeenCalledTimes(1);
});

test('POST /api/analyze returns 400 for invalid input', async () => {
  const response = await request(app)
    .post('/api/analyze')
    .send({ payload: { foo: 'bar' } });

  expect(response.status).toBe(400);
});
