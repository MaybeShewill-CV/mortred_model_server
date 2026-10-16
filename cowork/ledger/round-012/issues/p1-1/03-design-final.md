# Design P1-1 v1 (user-locked)
## Goal
Idle scrape shows mortred_waiting_jobs=0 and mortred_queue_depth=0; gauges match atomic _m_waiting_jobs.
## Boonds
No packing/Docker. Admission still reads the atomic.
## Root cause
_m_waiting_jobs decremented in SeriesWork callback; set_waiting_jobs/set_queue_depth only on reply path (before decrement) then never again.
## Fix
1. PrometheusMetrics::bind_waiting_depth(const std::atomic<size_t>*) so render() loads live value (derive-at-scrape).
2. BaseAiServerImpl binds &_m_waiting_jobs after init.
3. publish_waiting_gauges() after admit += and series callback -= (and keep reply path).
4. Unit test: bind+idle zero; update e2e comment if needed.
