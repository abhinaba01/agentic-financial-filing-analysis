"""Canonical metric names, their filing labels, and their XBRL concepts.

One place where a metric's identity is defined, so the rule-based extractor, the
XBRL tagger and the rubric all mean the same thing by ``revenue``. The XBRL
concept names are US-GAAP taxonomy elements and are the labels the FiNER-139
tagger from section 5.1 predicts, which is what lets the two extractors be
compared value-for-value.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    # Ordered most specific first: "total revenue" must win over "revenue", or a
    # segment line is picked up before the consolidated total.
    patterns: tuple[str, ...]
    xbrl_concepts: tuple[str, ...] = ()
    unit: str = "USD"
    is_per_share: bool = False
    negative_ok: bool = True
    aliases: tuple[str, ...] = field(default=())


EXTRACTED_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="revenue",
        label="Revenue",
        patterns=(
            "total net sales",
            "total revenues",
            "total revenue",
            "net revenues",
            "net revenue",
            "total net revenue",
            "net sales",
            "revenues",
            "revenue",
        ),
        xbrl_concepts=(
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
        ),
    ),
    MetricSpec(
        name="cost_of_revenue",
        label="Cost of revenue",
        patterns=(
            "total cost of sales",
            "total cost of revenue",
            "cost of goods sold",
            "cost of revenues",
            "cost of revenue",
            "cost of sales",
        ),
        xbrl_concepts=("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"),
    ),
    MetricSpec(
        name="gross_profit",
        label="Gross profit",
        patterns=("gross profit", "gross margin"),
        xbrl_concepts=("GrossProfit",),
    ),
    MetricSpec(
        name="operating_income",
        label="Operating income",
        patterns=(
            "operating income (loss)",
            "income from operations",
            "loss from operations",
            "operating income",
            "operating loss",
        ),
        xbrl_concepts=("OperatingIncomeLoss",),
    ),
    MetricSpec(
        name="net_income",
        label="Net income",
        patterns=(
            "net income (loss)",
            "net loss attributable",
            "net income attributable",
            "net earnings",
            "net income",
            "net loss",
        ),
        xbrl_concepts=(
            "NetIncomeLoss",
            "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
        ),
    ),
    MetricSpec(
        name="operating_expenses",
        label="Operating expenses",
        patterns=("total operating expenses", "operating expenses"),
        xbrl_concepts=("OperatingExpenses", "CostsAndExpenses"),
    ),
    MetricSpec(
        name="depreciation_amortization",
        label="Depreciation and amortization",
        patterns=(
            "depreciation and amortization",
            "depreciation, amortization",
            "depreciation & amortization",
        ),
        xbrl_concepts=(
            "DepreciationDepletionAndAmortization",
            "DepreciationAmortizationAndAccretionNet",
        ),
    ),
    MetricSpec(
        name="interest_expense",
        label="Interest expense",
        patterns=("interest expense, net", "interest expense"),
        xbrl_concepts=("InterestExpense", "InterestIncomeExpenseNet"),
    ),
    MetricSpec(
        name="income_tax_expense",
        label="Income tax expense",
        patterns=("provision for income taxes", "income tax expense", "income tax provision"),
        xbrl_concepts=("IncomeTaxExpenseBenefit",),
    ),
    MetricSpec(
        name="eps_basic",
        label="EPS (basic)",
        patterns=(
            "basic earnings per share",
            "earnings per share - basic",
            "basic net income per share",
            "basic",
        ),
        xbrl_concepts=("EarningsPerShareBasic",),
        unit="USD/share",
        is_per_share=True,
    ),
    MetricSpec(
        name="eps_diluted",
        label="EPS (diluted)",
        patterns=(
            "diluted earnings per share",
            "earnings per share - diluted",
            "diluted net income per share",
            "diluted",
        ),
        xbrl_concepts=("EarningsPerShareDiluted",),
        unit="USD/share",
        is_per_share=True,
    ),
    MetricSpec(
        name="total_assets",
        label="Total assets",
        patterns=("total assets",),
        xbrl_concepts=("Assets",),
        negative_ok=False,
    ),
    MetricSpec(
        name="total_liabilities",
        label="Total liabilities",
        patterns=("total liabilities",),
        xbrl_concepts=("Liabilities",),
        negative_ok=False,
    ),
    MetricSpec(
        name="current_assets",
        label="Total current assets",
        patterns=("total current assets",),
        xbrl_concepts=("AssetsCurrent",),
        negative_ok=False,
    ),
    MetricSpec(
        name="current_liabilities",
        label="Total current liabilities",
        patterns=("total current liabilities",),
        xbrl_concepts=("LiabilitiesCurrent",),
        negative_ok=False,
    ),
    MetricSpec(
        name="shareholders_equity",
        label="Shareholders' equity",
        patterns=(
            "total shareholders' equity",
            "total stockholders' equity",
            "total shareholders equity",
            "total stockholders equity",
            "total equity",
        ),
        xbrl_concepts=(
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
    ),
    MetricSpec(
        name="total_debt",
        label="Total debt",
        patterns=("total debt", "total borrowings", "long-term debt", "term debt"),
        xbrl_concepts=(
            "DebtLongtermAndShorttermCombinedAmount",
            "LongTermDebt",
            "LongTermDebtNoncurrent",
        ),
        negative_ok=False,
    ),
    MetricSpec(
        name="operating_cash_flow",
        label="Operating cash flow",
        patterns=(
            "net cash provided by operating activities",
            "net cash generated by operating activities",
            "net cash used in operating activities",
            "cash generated by operating activities",
            "cash flows from operating activities",
        ),
        xbrl_concepts=(
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
    ),
    MetricSpec(
        name="capital_expenditure",
        label="Capital expenditure",
        patterns=(
            "payments for acquisition of property, plant and equipment",
            "purchases of property and equipment",
            "purchases of property, plant and equipment",
            "capital expenditures",
        ),
        xbrl_concepts=(
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
        ),
    ),
    MetricSpec(
        name="shares_outstanding",
        label="Shares outstanding",
        patterns=(
            "weighted-average diluted shares outstanding",
            "weighted average shares outstanding",
            "shares of common stock outstanding",
            "shares outstanding",
        ),
        xbrl_concepts=(
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfSharesOutstandingBasic",
            "CommonStockSharesOutstanding",
        ),
        unit="shares",
        negative_ok=False,
    ),
)

METRICS_BY_NAME: dict[str, MetricSpec] = {m.name: m for m in EXTRACTED_METRICS}

# Reverse index: XBRL concept -> canonical metric name.
CONCEPT_TO_METRIC: dict[str, str] = {
    concept: spec.name for spec in EXTRACTED_METRICS for concept in spec.xbrl_concepts
}

# Metrics whose YoY change the report publishes when both periods are available.
YOY_METRICS: tuple[str, ...] = (
    "revenue",
    "net_income",
    "gross_profit",
    "operating_income",
    "operating_cash_flow",
    "total_assets",
    "eps_diluted",
)

# Metrics expressed in percentage points, so the rubric and the comparison logic
# know which values need convention handling.
PERCENT_METRICS: frozenset[str] = frozenset(
    {
        "gross_margin_pct",
        "operating_margin_pct",
        "net_margin_pct",
        "ebitda_margin_pct",
        "fcf_margin_pct",
        "roe_pct",
        "roa_pct",
        "revenue_yoy_pct",
        "net_income_yoy_pct",
        "gross_profit_yoy_pct",
        "operating_income_yoy_pct",
        "operating_cash_flow_yoy_pct",
        "total_assets_yoy_pct",
        "eps_diluted_yoy_pct",
    }
)


def metric_label(name: str) -> str:
    spec = METRICS_BY_NAME.get(name)
    if spec:
        return spec.label
    return name.replace("_", " ").replace(" pct", " %").strip().capitalize()
