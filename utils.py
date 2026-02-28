"""
utils.py — Core financial calculations for FinStab
"""

import numpy as np


def calc_risk_score(data: np.ndarray, weekly_expense: float) -> tuple[int, str]:
    """
    Composite risk score (0–100).
      - Volatility     : up to 40 pts
      - Deficit freq   : up to 35 pts
      - Downward trend : up to 25 pts
    Returns (score, label) where label ∈ {LOW, MEDIUM, HIGH}
    """
    if len(data) == 0:
        return 0, "LOW"
    m = float(np.mean(data))
    if m == 0:
        return 100, "HIGH"

    s = float(np.std(data))
    volatility   = min(40.0, (s / m) * 80.0)
    deficit_wks  = int(np.sum(data < weekly_expense))
    deficit_freq = (deficit_wks / len(data)) * 35.0

    last4 = data[-4:] if len(data) >= 4 else data
    slope = float(np.polyfit(range(len(last4)), last4, 1)[0]) if len(last4) > 1 else 0.0
    trend_penalty = min(25.0, abs(slope) / m * 250.0) if slope < 0 else 0.0

    score = int(min(100, round(volatility + deficit_freq + trend_penalty)))
    label = "LOW" if score < 35 else ("MEDIUM" if score < 65 else "HIGH")
    return score, label


def calc_emergency_buffer(
    weekly_expense: float,
    avg_weekly_income: float,
    dependents: int,
) -> tuple[float, float, int]:
    """Returns (buffer_amount, monthly_save_goal, buffer_weeks)"""
    buffer_weeks     = min(8, 4 + dependents)
    weekly_shortfall = max(0.0, weekly_expense - avg_weekly_income)
    buffer_amount    = (weekly_expense * buffer_weeks) + (weekly_shortfall * buffer_weeks * 0.5)
    monthly_save     = buffer_amount / 6.0
    return round(buffer_amount, 2), round(monthly_save, 2), buffer_weeks


def moving_average(data: np.ndarray, window: int = 3) -> np.ndarray:
    if len(data) < window:
        return data.copy()
    result = np.convolve(data, np.ones(window) / window, mode="valid")
    pad    = np.full(window - 1, result[0])
    return np.concatenate([pad, result])


def get_forecast(data: np.ndarray) -> list[float]:
    """Weighted MA + linear trend + 10% noise → 3 forecast values."""
    weights  = np.array([0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.18, 0.22])
    weighted = float(np.dot(data, weights))
    last4    = data[-4:] if len(data) >= 4 else data
    slope    = float(np.polyfit(range(len(last4)), last4, 1)[0]) if len(last4) > 1 else 0.0
    forecasts = []
    for i in range(3):
        noise = 1 + (np.random.random() - 0.5) * 0.10
        forecasts.append(max(0.0, round((weighted + slope * 1.5 * (i + 1)) * noise, 2)))
    return forecasts


def build_report_context(
    weekly_income: list,
    worker_type: str,
    city: str,
    dependents: int,
    monthly_exp: float,
    avg_income: float,
    risk_score: int,
    risk_label: str,
    forecast: float,
    buffer_amount: float,
    monthly_save: float,
    buffer_weeks: int,
) -> str:
    """
    Build a structured plain-text report context string to be passed
    to the chatbot as its system knowledge about this specific worker.
    """
    weekly_expense = monthly_exp / 4.33
    deficit_weeks  = [i + 1 for i, v in enumerate(weekly_income) if v < weekly_expense]
    data           = np.array(weekly_income, dtype=float)
    last4          = data[-4:]
    slope          = float(np.polyfit(range(4), last4, 1)[0]) if len(last4) == 4 else 0
    trend          = "upward" if slope > 100 else ("downward" if slope < -100 else "stable")

    weeks_str = "\n".join(f"  Week {i+1}: ₹{v:,.0f}" for i, v in enumerate(weekly_income))

    return f"""=== FinStab — WORKER ANALYSIS REPORT ===

WORKER PROFILE
--------------
Type       : {worker_type}
City       : {city}
Dependents : {dependents}
Monthly Expenses : ₹{monthly_exp:,.0f}
Weekly Expense Target : ₹{weekly_expense:,.0f}

INCOME DATA (Last 8 Weeks)
--------------------------
{weeks_str}

ANALYSIS RESULTS
----------------
Average Weekly Income : ₹{avg_income:,.0f}
Next Week Forecast    : ₹{forecast:,.0f}
Income Trend          : {trend}
Risk Score            : {risk_score}/100 ({risk_label} risk)
Deficit Weeks         : {deficit_weeks if deficit_weeks else "None"} ({len(deficit_weeks)} out of 8)

EMERGENCY BUFFER PLAN
---------------------
Target Emergency Fund : ₹{buffer_amount:,.0f}
Monthly Savings Goal  : ₹{monthly_save:,.0f}/month (6-month plan)
Weeks of Protection   : {buffer_weeks} weeks without income

=== END OF REPORT ===
"""
