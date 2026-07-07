# Reliability Assessment & Path to Profitability

Companion to [`RESULTS.md`](RESULTS.md). That file reports *what* happened in the backtest;
this one answers *how much to trust it*, *what can be inferred*, and *what would have to be
true (and done) for this strategy family to make money — and where*.

Evidence base: the backtest itself, an 8-seed sensitivity study, a code audit for residual
biases, a statistical power analysis, a review of the published Directional-Changes trading
literature, and an adversarial critique of improvement levers.

---

## 1. How reliable are the test results?

**Internal validity: high.** The pipeline acts only at DC confirmation bars, fills at the
next open, fits every statistic on training data only, and touched the test window exactly
once against pre-registered criteria. The test suite proves causality (signals unchanged
when future bars are replaced with noise). An audit found no leakage — and found that every
*unmodeled* friction points the same way: short borrow costs (~5–15% APR on ETH margin),
slippage, and gap-jumps would make reality **worse** than the reported −15.6%. The reported
number is an upper bound.

**Statistical power: low — and this cuts one way.** A 4.2-month test window has a Sharpe
standard error of ~1.7; it can only *demonstrate* an edge if the true Sharpe is ≳ 4.7
(bootstrap CI: [−4.36, +2.34]). So the results cannot prove the strategy is a loser — but
they were never going to prove it a winner either. What they *do* establish:

- **Point estimates are seed lotteries.** Rerunning the identical experiment with 8 RNG
  seeds: mean −13.0%, std 24.1pp, range [−41.5%, +31.1%]. Never quote any single run.
- **The distribution is centered negative**: 2/8 seeds positive long/short, **0/8 positive
  long-only**.
- **Four independent diagnostics agree**: zero-cost return −4.9% (the signal loses before
  any fees); random-timing null p = 0.557 (entries/exits indistinguishable from randomly
  rotating the same action series — which actually lost *less*, median −9.4%); walk-forward
  1/3 folds positive; GA fitness train 1.04 / validation 1.53 / test −1.44 (the signature
  of selecting noise).
- **The asymmetry favors the negative verdict**: GP+GA search pressure inflates results on
  data it can see (visible: val > train) and cannot reach the test window (test collapsed).
  A negative that survives an adversarial search-for-positive is the credible kind. The one
  positive number in RESULTS.md (+5.7% single-threshold 2%) is a post-hoc pick among five
  candidates in a ±30pp-noise regime — the *non*-credible kind.

**Bottom line:** reliable as *"no demonstrated edge, and the fitting machinery demonstrably
selects noise"* — not as *"proven negative edge."* For deployment decisions those imply the
same action: don't.

## 2. How reliable is the system?

As **software**: solid. Deterministic given a seed, 32 tests covering causality and exact
accounting, byte-identical reruns, honest execution model.

As a **trading system**: unreliable in a precisely measurable way — the GP overshoot
predictor beats a constant by only 1–4% MSE in-sample (from a single feature), the
strong/weak signal distinction built on it is noise, and the GA champion depends on the
seed (8 seeds → 8 different champions with a 73pp return spread). The machinery downstream
of the signal is fine; the signal itself carries no measurable timing information at this
frequency.

## 3. What can we infer?

1. **The loss decomposes into two problems, and the smaller one is fatal.** Fee drag is
   ~10.7pp of the −15.6% (60 trades × ~0.2% round trip ≈ 34%/yr hurdle at this turnover).
   But the zero-cost return is −4.9%: eliminate fees entirely and it still loses. Cost
   levers are hard-capped below breakeven.
2. **Returns are regime beta, not alpha.** +17% in the bull fold (B&H +27%), destroyed in
   chop (−36% vs B&H −1%), tracked the market down in the bear fold. Shorts lost money in
   a −37% market.
3. **This replicates the literature rather than contradicting it.** Published DC-trading
   profits exist only on **FX, at tick or 10-minute frequency, at 0–2.5 bps costs** —
   4–8× cheaper than crypto spot — and the honest edges are small: ~0.2–1.2% per
   multi-month test period (Kampouridis/Adegboye line, ESWA 2017/2021, AI Review 2023), or
   ~21% total *unlevered over ~8 years* for the celebrated Alpha Engine (Golub, Glattfelder
   & Olsen 2017; Sharpe ~3 via liquidity-provision economics on ticks, gross of
   commissions). No peer-reviewed paper demonstrates DC profitability on intraday crypto
   bars. Intraday crypto exhibits *reversal*, not momentum (Wen et al. 2022; Zaremba et
   al. 2021), and crypto momentum broadly decayed after mid-2020. Meanwhile Ao & Li
   (Finance Research Letters, 2024) showed DC backtests are systematically inflated by the
   theoretical-vs-achievable confirmation-price gap — on 100 stocks, 100/100 profitable
   with theoretical fills became 61/100 with real ones. Our +1,168%-naive vs −15.6%-honest
   appendix is the same phenomenon, larger.
4. **Bar data structurally handicaps DC.** Close-only detection sees ~40% of the events a
   high/low-aware detector sees at θ=0.5%, and confirmations gap through the threshold. The
   verdict strictly applies to *this bar-close implementation on 15m ETH*; the DC paradigm
   was born on tick streams.

## 4. How do we extract max profitability?

Honestly: **there is no profitability to extract from the current signal** — levers can
shrink the loss, not flip it. The defensible program, in order:

| Step | Lever | Expected effect | Status |
|---|---|---|---|
| 0 | Switch cost model to perp maker/taker (2–5 bps) + drop the 0.5% threshold | +8–10pp, **capped at ≈ −5% (zero-cost gross)** | Mechanical; necessary infrastructure, not profit |
| 1 | **Redesign the prediction target** — classify P(OS *price move* exceeds round-trip cost) with real features (OS/DC scaling ratio, realized vol, multi-threshold event rates, higher-TF regime, time-of-day) instead of GP-regressing OS *duration* on one feature | The only lever that can cross zero | **Kill-gate: out-of-fold AUC ≤ 0.52 on train → abandon** |
| 2 | Higher-timeframe DC direction gate (long only in HTF up-mode) | ±5–10pp, wide error bars | Cheap; honestly tests trend-following, not MTDC |
| 3 | Multi-asset / multi-timeframe evaluation | Finds where (if anywhere) edge exists | Only with multiple-testing discipline: pre-registered grid, per-asset selection on train/val, White's Reality Check / Deflated Sharpe across the family, losers published |
| — | Turnover tweaks, confidence sizing, monthly refits | Shrink loss toward the −5% gross floor | Lipstick on a dead signal until Step 1 produces one |

**Pre-registered abandonment criteria** (from the adversarial review): stop entirely if,
after Steps 0–1, (a) out-of-fold AUC ≤ 0.52 across thresholds, AND (b) the redesigned
signal's random-rotation null p > 0.2 on *validation*. The current evidence already sits
most of the way there.

## 5. On what assets?

- **Best documented habitat for DC proper: FX majors on tick data** at sub-3 bps all-in
  costs, multi-threshold ensembles — expecting *low-single-digit annual unlevered returns*
  whose selling point is drawdown control. That is what the literature actually supports.
- **Best documented adjacent opportunity in crypto: daily-bar, multi-week trend ensembles
  on the 10–20 most liquid coins** with vol targeting (documented net Sharpe ~0.5–1.5,
  regime-dependent, bull-heavy samples). This is trend-following, DC-flavored at best.
- **Worst habitat — the one tested here: intraday bars on a single spot-fee crypto pair,
  post-2020.** Wrong frequency (intraday crypto reverses), wrong cost regime, wrong era.
- If the DC hypothesis is to be tested further in crypto: **1h/4h bars on BTC, ETH, SOL,
  BNB, XRP** (larger per-event moves relative to fees), full history, perp fee model, under
  the Step-3 discipline above. Data path verified: this environment's proxy blocks exchange
  APIs directly, but a GitHub Actions workflow in this repo can fetch Binance's public
  archives (15m+1h, back to each listing date) and commit them to a `data` branch
  (~10 minutes, ~100 MB).

**Expectation setting:** in the best documented cases, this strategy family earns a few
percent per year unlevered. Nothing in the literature or in these results supports treating
it as more than a research vehicle.

## Key references

- Kampouridis & Otero (2017), *Evolving trading strategies using directional changes*, ESWA 73.
- Adegboye & Kampouridis (2021), ESWA 173; Adegboye, Kampouridis & Otero (2023), *Algorithmic trading with directional changes*, AI Review 56 (0.025%/trade costs, FX 10-min).
- Golub, Glattfelder & Olsen (2017), *The Alpha Engine*, SSRN 2951348 (FX ticks; ~21% unlevered total over ~8 yrs, gross; CXO Advisory critique notes crisis-period concentration).
- Glattfelder, Dupuis & Olsen (2011), *12 empirical scaling laws*, Quantitative Finance (arXiv:0809.1040) — the OS ≈ θ overshoot regularity, on FX ticks.
- Ao & Li (2024), Finance Research Letters 60 — DC confirmation-price gap inflates backtests (100/100 → 61/100 profitable).
- Liu & Tsyvinski (2021), RFS; Shen, Urquhart & Wang (2022), Financial Review — crypto momentum lives at 1–4 week horizons; intraday break-even costs 3–10 bps.
- Wen, Bouri, Xu & Zhao (2022), NAJEF — intraday crypto *reversal*.
- Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest Overfitting*, J. Comp. Finance — the discipline required for any Step-3 search.
