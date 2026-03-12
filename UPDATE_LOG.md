# Update Log (Post-Hackathon)

## 12-03-2026 : Optimizations applied to `app.py`
To significantly improve the overall latency of the paper synthesis pipeline, the following performance optimizations were made in `app.py` today:

1. **Model Warm-up on Startup:**
   - Added an `@app.on_event("startup")` FastAPI hook.
   - The `SentenceTransformer` representation model is now pre-loaded into memory immediately when the server starts. This eliminates the heavy overhead (~80+ seconds) previously experienced on the very first search query.

2. **Parallel Summarization & Claim Extraction:**
   - Replaced the sequential per-paper summary and claim extraction logic with `concurrent.futures.ThreadPoolExecutor`.
   - Text processing now runs concurrently across background threads, effectively speeding up the enrichment stage of the pipeline.

3. **In-Memory Endpoint Caching:**
   - Introduced a global `SEARCH_CACHE` to store full API responses.
   - Search requests with identical queries (`topic`, `max_results`, `similarity_threshold`) will now return immediately via a direct cache hit, delivering a near-zero latency response without re-triggering network fetches or re-calculating graph elements.

4. **Added Benchmarking Script:**
   - Created `benchmark.py` for granular profiling of the search pipeline.
