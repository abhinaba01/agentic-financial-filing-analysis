# NWSY - 10-K FY2024

> **Research and educational use only. Not investment advice.**

## Assessment

**Favorable** (confidence 0.83, rubric v1.0) - the rubric's signals lean positive.

Aggregate score `+0.621` over 6 factors covering 100% of rubric weight.

| Factor | Score | Rationale |
|---|---:|---|
| profitability | +0.77 | Profitability supports the assessment (Net margin %: strong, Gross margin %: software-like, Operating margin %: healthy). [p.31] [p.31] [p.31] |
| growth | +0.75 | Growth supports the assessment (Revenue yoy %: growing, Net income yoy %: strong growth). [p.31] [p.31] |
| leverage | +0.90 | Leverage supports the assessment (Debt to equity: conservative, Current ratio: comfortable). [p.33] [p.33] |
| cash generation | +0.70 | Cash generation supports the assessment (Free cash flow: cash generative, Ocf to net income: cash-backed earnings). [p.36] [p.36] |
| risk | -0.60 | Risk weighs against the assessment (Risk severity index: elevated risks). |
| tone | +0.60 | Tone supports the assessment (Sentiment score: positive tone). |

## Extracted metrics

| Metric | Value | Method | Confidence | Source |
|---|---:|---|---:|---|
| Capital expenditure | -268.40M | rule_based | 0.70 | p.36 |
| Cost of revenue | 1.65B | rule_based | 0.70 | p.31 |
| Total current assets | 2.91B | rule_based | 0.70 | p.33 |
| Total current liabilities | 1.81B | rule_based | 0.70 | p.33 |
| Depreciation and amortization | 214.70M | rule_based | 0.70 | p.36 |
| EPS (basic) | 3.71 | rule_based | 0.70 | p.31 |
| EPS (diluted) | 3.64 | rule_based | 0.70 | p.31 |
| Gross profit | 3.17B | rule_based | 0.70 | p.31 |
| Income tax expense | 182.50M | rule_based | 0.70 | p.31 |
| Interest expense | -48.60M | rule_based | 0.70 | p.31 |
| Net income | 731.40M | rule_based | 0.70 | p.31 |
| Operating cash flow | 1.10B | rule_based | 0.70 | p.36 |
| Operating expenses | 2.20B | rule_based | 0.70 | p.31 |
| Operating income | 962.50M | rule_based | 0.70 | p.31 |
| Revenue | 4.81B | rule_based | 0.70 | p.31 |
| Shareholders' equity | 2.46B | rule_based | 0.70 | p.33 |
| Total assets | 5.98B | rule_based | 0.70 | p.33 |
| Total debt | 1.18B | rule_based | 0.70 | p.33 |
| Total liabilities | 3.53B | rule_based | 0.70 | p.33 |

## Derived metrics

| Metric | Value | Formula | Operands |
|---|---:|---|---|
| Gross margin % | 65.8 | `gross_profit / revenue * 100` | gross_profit=3.167e+09, revenue=4.813e+09 |
| Operating margin % | 20 | `operating_income / revenue * 100` | operating_income=9.625e+08, revenue=4.813e+09 |
| Net margin % | 15.2 | `net_income / revenue * 100` | net_income=7.314e+08, revenue=4.813e+09 |
| Ebitda | 1.177e+09 | `operating_income + depreciation_amortization` | operating_income=9.625e+08, depreciation_amortization=2.147e+08 |
| Free cash flow | 8.358e+08 | `operating_cash_flow - abs(capital_expenditure)` | operating_cash_flow=1.104e+09, capital_expenditure=-2.684e+08 |
| Debt to equity | 0.4805 | `total_debt / shareholders_equity` | total_debt=1.18e+09, shareholders_equity=2.456e+09 |
| Current ratio | 1.614 | `current_assets / current_liabilities` | current_assets=2.914e+09, current_liabilities=1.806e+09 |
| Roe % | 29.78 | `net_income / shareholders_equity * 100` | net_income=7.314e+08, shareholders_equity=2.456e+09 |
| Roa % | 12.23 | `net_income / total_assets * 100` | net_income=7.314e+08, total_assets=5.982e+09 |
| Ocf to net income | 1.51 | `operating_cash_flow / net_income` | operating_cash_flow=1.104e+09, net_income=7.314e+08 |
| Interest coverage | 19.8 | `operating_income / abs(interest_expense)` | operating_income=9.625e+08, interest_expense=-4.86e+07 |
| Ebitda margin % | 24.46 | `ebitda / revenue * 100` | ebitda=1.177e+09, revenue=4.813e+09 |
| Fcf margin % | 17.37 | `free_cash_flow / revenue * 100` | free_cash_flow=8.358e+08, revenue=4.813e+09 |
| Pe ratio | 25.41 | `market_price_per_share / eps_diluted` | market_price_per_share=92.5, eps_diluted=3.64 |

## Year-over-year change

- Revenue yoy %: **+11.87%**
- Net income yoy %: **+24.35%**
- Gross profit yoy %: **+14.35%**
- Operating income yoy %: **+24.31%**
- Operating cash flow yoy %: **+20.20%**
- Total assets yoy %: **+11.94%**
- Eps diluted yoy %: **+24.23%**

## Sentiment

Overall **positive** (score +0.333), source: `lexicon_fallback` - fallback, not the fine-tuned classifier.

## Risk factors

- **medium** (p.14): Intense competition in the enterprise software market could adversely affect our pricing and our ability to retain customers.
- **high** (p.15): We derive a significant portion of our revenue from a limited number of large customers, and the loss of one or more of them could materially reduce our net sales.
- **high** (p.16): A material breach of our systems could result in loss of customer data, regulatory penalties and reputational harm, and could have a material adverse effect on our business, financial condition and results of operations.

## Findings

- [supported] Capital expenditure is reported as -268,400,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Cost of revenue is reported as 1,645,800,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Total current assets is reported as 2,914,500,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Total current liabilities is reported as 1,806,200,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Depreciation and amortization is reported as 214,700,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] EPS (basic) is reported as 3.71 USD/share. _(chunks: demo_10k-dc34e)_
- [supported] EPS (diluted) is reported as 3.64 USD/share. _(chunks: demo_10k-dc34e)_
- [supported] Gross profit is reported as 3,166,800,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Income tax expense is reported as 182,500,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Interest expense is reported as -48,600,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Net income is reported as 731,400,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Operating cash flow is reported as 1,104,200,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Operating expenses is reported as 2,204,300,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Operating income is reported as 962,500,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Revenue is reported as 4,812,600,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Shareholders' equity is reported as 2,455,700,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Total assets is reported as 5,982,300,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Total debt is reported as 1,180,000,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Total liabilities is reported as 3,526,600,000 USD. _(chunks: demo_10k-dc34e)_
- [supported] Gross margin % computes to 65.80 (gross_profit / revenue * 100). _(chunks: demo_10k-dc34e, demo_10k-dc34e)_
- [supported] Operating margin % computes to 20.00 (operating_income / revenue * 100). _(chunks: demo_10k-dc34e, demo_10k-dc34e)_
- [supported] Net margin % computes to 15.20 (net_income / revenue * 100). _(chunks: demo_10k-dc34e, demo_10k-dc34e)_
- [supported] Ebitda computes to 1,177,200,000.00 (operating_income + depreciation_amortization). _(chunks: demo_10k-dc34e, demo_10k-dc34e)_
- [supported] Free cash flow computes to 835,800,000.00 (operating_cash_flow - abs(capital_expenditure)). _(chunks: demo_10k-dc34e, demo_10k-dc34e)_
- [supported] Debt to equity computes to 0.48 (total_debt / shareholders_equity). _(chunks: demo_10k-dc34e, demo_10k-dc34e)_
- [supported] Current ratio computes to 1.61 (current_assets / current_liabilities). _(chunks: demo_10k-dc34e, demo_10k-dc34e)_
- [supported] Roe % computes to 29.78 (net_income / shareholders_equity * 100). _(chunks: demo_10k-dc34e, demo_10k-dc34e)_
- [supported] Roa % computes to 12.23 (net_income / total_assets * 100). _(chunks: demo_10k-dc34e, demo_10k-dc34e)_
- [supported] Ocf to net income computes to 1.51 (operating_cash_flow / net_income). _(chunks: demo_10k-dc34e, demo_10k-dc34e)_
- [supported] Interest coverage computes to 19.80 (operating_income / abs(interest_expense)). _(chunks: demo_10k-dc34e, demo_10k-dc34e)_
- [supported] Ebitda margin % computes to 24.46 (ebitda / revenue * 100). _(chunks: demo_10k-dc34e)_
- [supported] Fcf margin % computes to 17.37 (free_cash_flow / revenue * 100). _(chunks: demo_10k-dc34e)_
- [supported] Pe ratio computes to 25.41 (market_price_per_share / eps_diluted). _(chunks: demo_10k-dc34e)_

0 claim(s) were dropped for lacking support in their cited passages.

## Retrieval diagnostics

- Chunks retrieved: 8
- Discarded below similarity floor: 0
- Mean similarity: 0.606
- Retries: 0
- Sufficient: True (sufficient evidence)
- Queries tried:
  - `What were the company's revenue, profitability, cash generation, leverage and principal risks for the fiscal year, and how did they change?`

## Warnings

- sentiment from a word lexicon, not the fine-tuned classifier; treat the tone factor as indicative only
- XBRL tagger not loaded

---

Pipeline `0.1.0` - embedder `BAAI/bge-base-en-v1.5`, XBRL tagger `rule_based_only`, sentiment `lexicon_fallback`, reasoner `stub` - generated 2026-09-05T17:21:25.262025+00:00.

Research and educational use only. Not investment advice.