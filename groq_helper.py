"""
groq_helper.py — Groq LLaMA3 integration for FinStab
  • analyze_income()  → one-shot AI report insights
  • chat_with_report() → conversational chatbot grounded in report context
"""

import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────
# 🔑 API KEY — paste your free key from console.groq.com
# ─────────────────────────────────────────────────────────
# GROQ_API_KEY = "GROQ_API_KEY"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL        = "llama-3.1-8b-instant"


def _get_client():
    from groq import Groq
    key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("No Groq API key found. Set GROQ_API_KEY in groq_helper.py")
    os.environ["GROQ_API_KEY"] = key
    return Groq()


# ─────────────────────────────────────────────────────────
# ONE-SHOT ANALYSIS
# ─────────────────────────────────────────────────────────
def analyze_income(
    weekly_income, worker_type, city, dependents,
    weekly_expense, avg_income, risk_score, risk_label,
    forecast, lang_instruction="",
) -> str:
    weeks_str     = "\n".join(f"  Week {i+1}: ₹{v:,.0f}" for i, v in enumerate(weekly_income))
    deficit_count = sum(1 for v in weekly_income if v < weekly_expense)

    prompt = f"""You are a compassionate, practical financial advisor helping informal gig workers in India.

Worker Profile:
- Type: {worker_type}
- City: {city}
- Dependents: {dependents}
- Weekly expense target: ₹{weekly_expense:,.0f}

Income Data (last 8 weeks):
{weeks_str}

Statistics:
- Average weekly income: ₹{avg_income:,.0f}
- Next week forecast: ₹{forecast:,.0f}
- Risk Score: {risk_score}/100 ({risk_label} risk)
- Deficit weeks: {deficit_count}/8

{lang_instruction}

Provide a structured, warm analysis using EXACTLY these 4 markdown sections:

#### 📊 Income Pattern
[2-3 sentences about trend, variability, stability]

#### ⚠️ Biggest Risk
[1-2 sentences on the most critical financial risk]

#### 💡 3 Tips to Stabilize
- [Tip 1 — specific and actionable]
- [Tip 2 — specific and actionable]
- [Tip 3 — specific and actionable]

#### 🎯 This Week's Priority
[One clear action for this week]

Keep language simple, warm, and reference their specific numbers. Avoid jargon.
"""

    try:
        client   = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return _fallback_analysis(weekly_income, avg_income, risk_score, risk_label,
                                  forecast, deficit_count, dependents, weekly_expense, str(e))


# ─────────────────────────────────────────────────────────
# CHATBOT — grounded in report context
# ─────────────────────────────────────────────────────────
CHATBOT_SYSTEM = """You are Shieldy, a friendly and knowledgeable financial assistant for FinStab.
You help informal and gig workers in India understand their income, manage financial risk, and plan savings.

You have been given the worker's full income analysis report as context below.
ALWAYS ground your answers in this specific report data — refer to their actual numbers.
Be warm, encouraging, and use simple language. Avoid financial jargon.
Answer in whatever language the user writes in. If they write in Hindi, respond in Hindi. Tamil → Tamil, etc.
Keep responses concise and practical. Use bullet points for lists. Use ₹ for rupee amounts.

If asked about something not covered in the report, use your general financial knowledge while
being clear you're speaking generally, not from their specific data.

WORKER REPORT:
{report_context}
"""

def chat_with_report(
    report_context: str,
    chat_history: list[dict],
    user_message: str,
) -> str:
    """
    Multi-turn chatbot grounded in the analyzed report.

    Args:
        report_context : the full plain-text report string
        chat_history   : list of {"role": "user"|"assistant", "content": "..."} dicts
        user_message   : latest user message

    Returns:
        assistant reply string
    """
    system_prompt = CHATBOT_SYSTEM.format(report_context=report_context)

    messages = [{"role": "system", "content": system_prompt}]
    # Include recent history (last 10 turns to stay within context)
    for msg in chat_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        client   = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=600,
            temperature=0.6,
        )
        return response.choices[0].message.content
    except Exception as e:
        return (
            f"⚠️ I couldn't connect to the AI service right now. Please check your Groq API key.\n\n"
            f"Error: `{str(e)[:120]}`\n\n"
            f"Get a free key at [console.groq.com](https://console.groq.com) and set it in `groq_helper.py`."
        )


# ─────────────────────────────────────────────────────────
# FALLBACK (no API key)
# ─────────────────────────────────────────────────────────
def _fallback_analysis(weekly_income, avg_income, risk_score, risk_label,
                        forecast, deficit_count, dependents, weekly_expense, err="") -> str:
    data  = np.array(weekly_income, dtype=float)
    last4 = data[-4:]
    slope = float(np.polyfit(range(4), last4, 1)[0]) if len(last4) == 4 else 0
    trend = "upward 📈" if slope > 100 else ("downward 📉" if slope < -100 else "stable ➡️")
    save  = max(200, round(avg_income * 0.10))

    note = f"\n\n> ⚙️ *Running in offline mode — set `GROQ_API_KEY` in `groq_helper.py` for full AI insights.*"
    if err:
        note += f" Error: `{err[:80]}`"

    return f"""#### 📊 Income Pattern
Your income trend is **{trend}** over the last 4 weeks. You earned an average of **₹{avg_income:,.0f}/week** with {deficit_count} weeks falling below your expense target. This indicates **{risk_label.lower()} financial risk**.

#### ⚠️ Biggest Risk
{"High income volatility and " + str(deficit_count) + " deficit week(s) put pressure on " + str(dependents) + " dependent(s). Urgent stabilization needed." if risk_label == "HIGH" else str(deficit_count) + " deficit week(s) signal inconsistent cash flow — build your buffer now before a crisis hits." if risk_label == "MEDIUM" else "Good stability overall. Focus on growing your emergency buffer to protect against future dips."}

#### 💡 3 Tips to Stabilize
- **Diversify platforms**: Sign up for 2–3 gig apps to reduce single-source dependency
- **Track daily earnings**: A simple daily log reveals patterns and high-earning days within 2 weeks
- **Auto-save ₹{save:,}/week**: Consistent small savings beat irregular large ones every time

#### 🎯 This Week's Priority
{"Positive momentum — aim for ₹" + f"{forecast:,.0f}" + " this week by targeting high-demand evening and weekend slots." if forecast > avg_income else "Income may dip — cut variable expenses this week and prioritize your highest-paying work orders."}{note}"""
