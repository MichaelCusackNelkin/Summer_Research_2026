# Security Master Identity Integration Plan

**Status:** Deferred — not implemented. Requires a Databento **Reference Data**
subscription (separate, more expensive product; we do not currently have access).
This document is a ready-to-execute plan for when/if we subscribe.

**Owner context:** feeds the point-in-time (PIT) universe pipeline
(`build_universes.py` + `data_collection/src/data_collection/universe_pipeline.py`).
Downstream use: PIT covariance matrices and mean-return forecasts, rebuilt every
6 months over an 18-month trailing window.

---

## 1. Why we want this

The pipeline currently identifies securities by **`raw_symbol`** (the display
ticker, resolved point-in-time per day by Databento's `map_symbols=True`). That
is correct *per day*, but it has two identity failure modes that a PIT research
pipeline should not silently inherit:

- **Renames** (e.g. `FB` → `META`): one continuous security's history splits
  across two ticker strings. Inside an 18-month training window this truncates
  the name's history (it can fail the 90%-traded filter and drop out, or train a
  covariance/forecast on a partial series).
- **Recycling** (a ticker string reused for a *different* company years later):
  conflates two distinct securities if you ever key across time.

We **cannot** fix this with `instrument_id`: for Nasdaq-style venues
(XNAS.ITCH) the publisher's numeric `instrument_id` is **remapped daily** — the
same integer refers to different stocks on different days. (Confirmed: the
Databento SDK's own `InstrumentMap.resolve(instrument_id, date)` is keyed on the
*(id, date)* pair with date intervals precisely because the integer alone is
ambiguous.) So `instrument_id` is fine as a per-day diagnostic but is **not** a
cross-time identity key.

### Scope honesty
For *window-scoped* covariance/forecast work, only a **rename inside a window**
truly distorts estimation. Recycling across long gaps is naturally harmless
because we never link across the gap (each universe is rebuilt from its own
window). So `raw_symbol` alone is "mostly fine"; the security master is the
**rigorous** version, justified by wanting to fully trust the data and by the
survivorship guarantees below.

---

## 2. The decision

Use the Databento **Security Master** as the identity backbone:

- **Identity key = a permanent identifier**, preferably **`bbg_comp_id`**
  (Bloomberg composite id) or composite **`figi`**. These survive ticker renames
  (FB and META share one) and distinguish recycled tickers.
- **Fetch key stays `raw_symbol`** — batch jobs, `ALL_SYMBOLS`, and the adapter's
  MD5 cache key all speak raw symbols. **No adapter surgery required.**
- A **point-in-time map** bridges them: `(raw_symbol, date) → permanent_id`,
  plus listing/delisting dates.

This **supersedes** the earlier `symbology.resolve` idea — the security master
gives the same identifier mapping *plus* lifecycle dates (survivorship) *plus*
clean effective-dating, in one product.

---

## 3. What the Security Master provides (VERIFIED from the installed SDK)

Accessed via the **Reference client**, not the historical market-data client:

```python
import databento as db
ref = db.Reference(key=os.getenv("DATABENTO_API_KEY"))  # Reference Data product
sm = ref.security_master.get_range(
    start="2018-05-01",
    end=None,                 # None => all data after start
    index="ts_effective",     # or "ts_record"
    symbols="ALL_SYMBOLS",    # or up to 2,000 symbols/request
    stype_in="raw_symbol",    # see identifier options below
    countries="US",           # optional ISO 3166-1 alpha-2 filter
    security_types=None,      # optional SECTYPE filter (e.g. common stock)
)   # -> pandas.DataFrame indexed + sorted by `index`
```

`SecurityMasterHttpAPI.get_range(...)` signature (verified):
`start, end=None, index="ts_effective", symbols=None, stype_in="raw_symbol",
countries=None, security_types=None`. There is also `get_last(...)` for the
latest snapshot.

**Temporal columns (verified, parsed as datetimes):** `ts_record`,
`ts_effective`, `ts_created` — so the schema is point-in-time; query "as-of" by
filtering on `ts_effective`.

**Lifecycle date columns (verified, parsed as dates):** `listing_created_date`,
`listing_date`, `delisting_date`, `shares_outstanding_date` — gives direct,
auditable survivorship (a name's live interval), instead of trusting that
`ALL_SYMBOLS` happened to include delisted names.

**Identifier symbology (verified — valid `stype_in` values):** `raw_symbol`,
`nasdaq_symbol`, `isin`, `us_code` (CUSIP), `bbg_comp_id`, `bbg_comp_ticker`,
`figi`, `figi_ticker`. Per Databento marketing copy, US-equities coverage
includes CUSIP, FIGI, ISIN, CFI, FISN, CIK since 2005.

### TO VERIFY at subscribe time (could not confirm offline)
- **Exact identifier *column names* in the returned DataFrame.** The SDK only
  hard-codes the datetime/date columns to parse; the id columns (likely
  `bbg_comp_id`, `figi`, `cusip`/`us_code`, `isin`, and possibly a single
  `security_id` primary key) must be confirmed by inspecting one real response.
  **First implementation step is to print `sm.columns` and pick the identity
  column.**
- XNAS.ITCH / US-equities coverage completeness and how multi-listing / share
  classes are represented.
- Actual Reference Data **cost** for the ALL_SYMBOLS, full-history pull.

---

## 4. Prerequisites

- Databento **Reference Data** subscription enabled on the API key.
- `DATABENTO_API_KEY` already in `.env` (same key; reuse `_resolve_api_key` in
  `build_universes.py`).
- Confirm `databento.Reference` is importable in `.venv` (it is, in the pinned
  SDK: `databento/reference/api/security.py`).

---

## 5. Integration design (delta on the current pipeline)

Current state to build on: `build_universes.py` + `universe_pipeline.py` key on
`raw_symbol`; edge case K (drop null/non-positive close|volume) is fixed; 62
offline tests pass. Keep all of that; layer identity on top.

### 5.1 New artifact: the PIT identity map
New module, e.g. `data_collection/src/data_collection/security_master.py`:

- `fetch_security_master(start, end, out_path)` — one `get_range` call for
  `ALL_SYMBOLS`, `countries="US"`, `index="ts_effective"`. Persist raw to
  parquet (cache it like the OHLCV pull; it is a paid call — fetch once).
- `build_identity_map(sm_df) -> pl.DataFrame` — normalize into interval rows:
  `[permanent_id, raw_symbol, valid_from, valid_to, listing_date, delisting_date]`
  where `[valid_from, valid_to)` is the half-open span a given `permanent_id`
  traded under a given `raw_symbol` (derived from consecutive `ts_effective`
  records / symbol changes). `permanent_id` = chosen identity column
  (`bbg_comp_id` preferred).
- `as_of(identity_map, date) -> {raw_symbol: permanent_id}` and the inverse
  `{permanent_id: [(raw_symbol, [from,to))]}` helpers.

### 5.2 Selection (`universe_pipeline.select_universe` / `build_pit_universes`)
- Before aggregating, **join** the OHLCV window frame to the identity map
  *as-of each bar's date* (`raw_symbol` + `ts_event.date()` ∈ `[valid_from,
  valid_to)`) to attach `permanent_id`.
- **Group by `permanent_id`** instead of `symbol` for ADV / `n_days` / `last_px`.
  Renames now aggregate into one continuous 18-month history; recycled strings
  split by id. Keep the most-recent `raw_symbol` in the window as a human label.
- Optional survivorship guard: require the security to be live for ≥X% of the
  window using `listing_date`/`delisting_date` (complements the existing
  90%-traded filter; mostly redundant but auditable).
- Output columns gain `permanent_id` (key) alongside `symbol` (label).

### 5.3 Coverage ledger (`universe_pipeline`)
- Key the ledger on **`permanent_id`** rather than the ticker string. All
  existing interval-algebra branches (brand-new / persisting / returning /
  fully-covered) carry over unchanged — only the key type changes.

### 5.4 Download planner
- Required coverage is still `[max(DATA_FLOOR, R-18mo), R+6mo)` per
  `permanent_id`.
- When emitting Databento requests, translate `permanent_id + needed interval`
  → the **`raw_symbol`(s)** that id used during that interval (from the identity
  map). A security renamed mid-interval yields **two** raw_symbol sub-requests
  (e.g. `FB` for `[from, change)`, `META` for `[change, to)`) — assert there is
  **no coverage hole** at the rename boundary.
- Fetch is unchanged: `adapter.get_data(..., materialize_only=True)` keyed on
  raw_symbol; cache keys and the cost estimate path are untouched.

### 5.5 Outputs
- `universes.parquet`: add `permanent_id` column (the key downstream covariance
  code should join on); keep `symbol` as label and `instrument_id` (per-day) as
  diagnostic only.
- Consolidated per-schema parquets already preserve `instrument_id`; add a
  post-step that attaches `permanent_id` via an as-of join on
  `(symbol, ts_event/ts_recv date)` so downstream reads key on the stable id.

---

## 6. Edge cases / gotchas

- **As-of join correctness.** Always join on the half-open effective interval,
  never a plain equality on symbol — that is the whole point.
- **Multiple FIGIs / share classes.** Decide whether to key on composite FIGI
  (share-class-level) vs share-class FIGI; `bbg_comp_id` is the safer
  rename-stable choice. Document the pick.
- **Missing mappings.** Some `raw_symbol`s in OHLCV may not resolve (gaps in the
  master). Fall back to `raw_symbol`-as-identity for those, and **log/count**
  them (no silent drops).
- **Fetch is still by symbol.** Do not attempt `stype_in="figi"` downloads from
  the market-data batch API unless XNAS.ITCH supports it — keep raw_symbol fetch.
- **Cost.** The master is a paid Reference pull; fetch once and cache. The
  market-data cost estimate (Step 2 of `build_universes.py`) is unaffected.
- **Determinism.** Sort identity-map intervals deterministically; keep the
  existing tie-break behavior in selection.

---

## 7. Testing (offline, mirrors `test_universe_pipeline.py` style)

Add synthetic-frame tests (no network):
- `build_identity_map`: consecutive `ts_effective` records with a symbol change
  produce two intervals under one `permanent_id`.
- Rename: one `permanent_id` with `FB`→`META` selected **once** with combined
  18-month ADV; its plan fetches both `FB` and `META` sub-intervals with no hole.
- Recycle: same string, two `permanent_id`s in disjoint date ranges → **two**
  securities.
- As-of join picks the correct `permanent_id` at an interval boundary
  (half-open, no off-by-one).
- Unmapped `raw_symbol` falls back to itself and is counted.
- Ledger dedup branches still hold under the `permanent_id` key.

Keep the existing 62 tests green.

---

## 8. Fallback (current behavior, no Reference Data)

Until subscribed, the pipeline keys on `raw_symbol` (PIT-correct per day) and
relies on window-scoping to neutralize recycling. Residual risk = rare
mid-window renames, which conservatively drop a name rather than introduce
lookahead/survivorship bias. Edge case K (hygiene) is already fixed.

---

## 9. Quick-start checklist (when subscribing)

1. Confirm Reference Data is enabled on the key; `import databento as db;
   db.Reference(...)`.
2. `get_range(..., symbols="ALL_SYMBOLS", index="ts_effective")`; **print
   `.columns`**; pick the identity column (`bbg_comp_id` preferred).
3. Implement `security_master.py` (fetch + cache + identity map + as-of helpers).
4. Re-point `select_universe`/ledger/planner from `symbol` to `permanent_id`
   per §5; add `permanent_id` to outputs.
5. Add the §7 tests; keep all prior tests green; `--help` clean; no network in
   tests.
6. Verify XNAS.ITCH coverage and the rename/recycle behavior on a few known
   cases (FB→META; a known recycled ticker) before trusting universes.

---

## References
- Security master schema — https://databento.com/docs/schemas-and-data-formats/security-master
- Security master product/pricing — https://databento.com/security-master
- Venue/dataset spec — https://databento.com/docs/venues-and-datasets/security-master
- Symbology standards — https://databento.com/docs/standards-and-conventions/symbology
- Installed SDK (authoritative for our version):
  `.venv/.../databento/reference/api/security.py`,
  `.venv/.../databento/common/constants.py` (SECURITY_MASTER_*_COLUMNS).
