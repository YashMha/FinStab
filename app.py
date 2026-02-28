"""
FinStab — app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import re
from groq_helper import analyze_income, chat_with_report
from utils import (
    calc_risk_score, calc_emergency_buffer,
    moving_average, get_forecast, build_report_context,
)

LANG = {
    "English": {
        "tagline": "Know your income. Plan your future.",
        "sub": "Track 8 weeks of earnings, see your risk, get an AI financial plan — free.",
        "step1": "Your Details", "step2": "Weekly Income", "step3": "Your Results",
        "name_lbl": "Your Name",
        "worker_lbl": "What kind of work do you do?",
        "city_lbl": "Which city do you work in?",
        "dep_lbl": "How many people depend on you?",
        "exp_lbl": "Your monthly expenses (Rs.)",
        "income_hdr": "How much did you earn each week?",
        "income_sub": "Enter your income for each of the last 8 weeks. Estimates are fine!",
        "analyze_btn": "Get My Financial Report",
        "avg_lbl": "Weekly Average", "forecast_lbl": "Next Week Forecast",
        "risk_lbl": "Risk Level", "buffer_lbl": "Safety Buffer",
        "ai_hdr": "Your Personalized Plan",
        "chat_hdr": "Ask Shieldy — Your AI Financial Assistant",
        "chat_sub": "Shieldy knows your report. Ask anything — in any language!",
        "chat_placeholder": "E.g. How can I save more? What is my risk score?",
        "chat_send": "Send", "chat_thinking": "Shieldy is thinking...",
        "chat_greeting": "Hello! I'm **Shieldy**, your personal finance assistant.\n\nI've read your income report and I'm ready to help. Ask me anything — about your income, savings, risk, or what to do this week!",
        "buffer_hdr": "Your Emergency Savings Plan",
        "fund_lbl": "Target Fund", "save_lbl": "Save Each Month", "protect_lbl": "Weeks Protected",
        "download_lbl": "Download Full Report (PDF)",
        "deficit_warn": "Some weeks you earned less than your expenses.",
        "deficit_detail": "week(s) below your Rs.{exp:.0f}/week expense target.",
        "low_risk": "LOW — You're doing well!", "med_risk": "MEDIUM — Some concern", "high_risk": "HIGH — Take action now",
        "lang_instr": "Please respond entirely in English.",
        "reset_btn": "Start Over",
        "chart_lbl_actual": "Actual Income", "chart_lbl_forecast": "Forecast",
        "chart_lbl_expense": "Expense Line", "chart_title": "Income Overview — 8 Weeks + Forecast",
        "chart_trend": "Trend", "chart_avg": "Your Average",
        "reanalyze_btn": "Re-analyze in", "lang_notice": "Language changed! Re-analyze to get AI insights in the new language.",
    },
    "Hindi": {
        "tagline": "Apni aay jaanen. Apna bhavishy banayen.",
        "sub": "8 haftoon ki kamaai darj karen, jokhim dekhen, AI vittiya yojana paayen — muft.",
        "step1": "Aapki Jaankaari", "step2": "Saaptaahik Aay", "step3": "Aapke Parinaam",
        "name_lbl": "Aapka Naam",
        "worker_lbl": "Aap kis prakar ka kaam karte hain?",
        "city_lbl": "Aap kis shehar mein kaam karte hain?",
        "dep_lbl": "Aap par kitne log nirbhar hain?",
        "exp_lbl": "Aapka maasik kharcha (Rs.)",
        "income_hdr": "Har hafte aapne kitna kamaya?",
        "income_sub": "Pichhle 8 haftoon ki aay darj karen. Anumaan bhi theek hai!",
        "analyze_btn": "Meri Vittiya Report Dekhen",
        "avg_lbl": "Saaptaahik Ausath", "forecast_lbl": "Agle Hafte ka Anumaan",
        "risk_lbl": "Jokhim Sthar", "buffer_lbl": "Suraksha Bachat",
        "ai_hdr": "Aapki Vyaktigat Yojana",
        "chat_hdr": "Shieldy se Puchhen — Aapka AI Vittiya Sahayak",
        "chat_sub": "Shieldy ko aapki report pata hai. Kuchh bhi puchhen!",
        "chat_placeholder": "Udaaharan: Main aur kaise bachat kar sakta hoon?",
        "chat_send": "Bhejen", "chat_thinking": "Shieldy soch raha hai...",
        "chat_greeting": "Namaste! Main **Shieldy** hoon, aapka vittiya sahayak.\n\nMainne aapki aay report padh li hai. Kuchh bhi puchhen!",
        "buffer_hdr": "Aapki Aapatkaleen Bachat Yojana",
        "fund_lbl": "Lakshya Raashi", "save_lbl": "Har Maheene Bachayen", "protect_lbl": "Surakshit Saptaah",
        "download_lbl": "Poori Report Download Karen (PDF)",
        "deficit_warn": "Kuchh hafte aapki kamaai kharche se kam rahi.",
        "deficit_detail": "hafte aapke Rs.{exp:.0f}/saptaah ke lakshya se kam rahe.",
        "low_risk": "Kam — Achha hai!", "med_risk": "Madhyam — Dhyan den", "high_risk": "Uchch — Abhi kadam uthayen",
        "lang_instr": "Please respond entirely in Hindi language. Use simple, warm language suitable for gig workers.",
        "reset_btn": "Phir se Shuroo Karen",
        "chart_lbl_actual": "Vaastvik Aay", "chart_lbl_forecast": "Anumaan",
        "chart_lbl_expense": "Kharcha Rekha", "chart_title": "8 Haftoon ki Aay",
        "chart_trend": "Rukh", "chart_avg": "Aapka Ausath",
        "reanalyze_btn": "Phir se Vishleshan Karen", "lang_notice": "Bhaasha badal gayi! Nayi bhaasha mein insights ke liye Re-analyze karen.",
    },
    "Marathi": {
        "tagline": "Tumchi kamaai jaana. Bhavishy ghadva.",
        "sub": "8 aaThavdyaanchi kamaai naondva, jokheema paha, AI yojana milva — mophata.",
        "step1": "Tumchi Maahiti", "step2": "Saaptaahik Utpanna", "step3": "Tumche Nikaal",
        "name_lbl": "Tumche Naav",
        "worker_lbl": "Tumhi konatyaa prakarache kaam karta?",
        "city_lbl": "Tumhi konatyaa sheharat kaam karta?",
        "dep_lbl": "Tumchyavar kiti jan avalaamboon aahet?",
        "exp_lbl": "Tumcha maasik kharcha (Rs.)",
        "income_hdr": "Dar aaThavdyaat tumhi kiti kamavale?",
        "income_sub": "Maageel 8 aaThavdyaanche utpanna naondva. Andaajaane Theek aahe!",
        "analyze_btn": "Maazha Arthik Ahavaal Pahaa",
        "avg_lbl": "Saaptaahik Saraasar", "forecast_lbl": "Pudheel aaThavdyaachaa Andaaja",
        "risk_lbl": "Jokheema Paatali", "buffer_lbl": "Suraksha Bachat",
        "ai_hdr": "Tumchi Vaiyakteeka Yojana",
        "chat_hdr": "Shieldy la Vichaara — AI Arthik Sahaayyak",
        "chat_sub": "Shieldy la tumcha ahavaal maahit aahe. Kaahee hi vichaara!",
        "chat_placeholder": "Udaa. Mi adhik bachat kashe karu?",
        "chat_send": "Paatthava", "chat_thinking": "Shieldy vichar karat aahe...",
        "chat_greeting": "Namaskar! Mi **Shieldy** aahe, tumcha Arthik sahaayyak.\n\nTumcha ahavaal vaachala aahe. Kaahee hi vichaara!",
        "buffer_hdr": "Tumchi AaNeebaaNee Bachat Yojana",
        "fund_lbl": "Lakshy Nidhi", "save_lbl": "Darmaahaa Bachat", "protect_lbl": "Surakshit aaThavde",
        "download_lbl": "Sampoornn Ahavaal Download Karaa (PDF)",
        "deficit_warn": "Kaahi aaThavde utpanna kharchaapekshaa kami hote.",
        "deficit_detail": "aaThavde Rs.{exp:.0f}/aaThavdaa pekshaa kami.",
        "low_risk": "Kami — Chhaann!", "med_risk": "Madhyam — Lakshy dyaa", "high_risk": "Jaast — Aattaa karaavaai karaa",
        "lang_instr": "Please respond entirely in Marathi language. Use simple, warm Marathi.",
        "reset_btn": "Punhaa Suruu Karaa",
        "chart_lbl_actual": "Vaastaveek Utpanna", "chart_lbl_forecast": "Andaaja",
        "chart_lbl_expense": "Kharcha Reshaa", "chart_title": "8 aaThavdyaanche Utpanna",
        "chart_trend": "Kl", "chart_avg": "Saraasar",
        "reanalyze_btn": "Punhaa Vishleshan Karaa", "lang_notice": "Bhaashaa badali! Navyaa bhaashet insights saaThee Re-analyze karaa.",
    },
    "Tamil": {
        "tagline": "Ungal varumanam ariyungal. Ethirkaalam tittamidungal.",
        "sub": "8 vaarangalin sambaatiyathai padivu seyyungal, AI tittam perungal — ilavacam.",
        "step1": "Ungal Vivaragal", "step2": "Vaaraantira Varumanam", "step3": "Ungal Mudivugal",
        "name_lbl": "Ungal Peyar",
        "worker_lbl": "Neengal enna velai seygireerkal?",
        "city_lbl": "Neengal enda nagarattil panipurikirgal?",
        "dep_lbl": "Ungalai saarntiruppavar ettanai peer?",
        "exp_lbl": "Ungal maadumaantira selavu (Rs.)",
        "income_hdr": "Ovvoru vaaramum evvalavu sambaadittirkal?",
        "income_sub": "Kadanta 8 vaarangalin varumaanattai ulliidungal. Tooraayamana togaiyum sari!",
        "analyze_btn": "En Nidhi Arikkaiyai Paarungal",
        "avg_lbl": "Vaaraantira Saraasar", "forecast_lbl": "Adutta Vaar Mugankanipu",
        "risk_lbl": "Aapattu Nilai", "buffer_lbl": "Paadukaapu Cemippu",
        "ai_hdr": "Ungal Tanippattu Tittam",
        "chat_hdr": "Shieldy-yidam Keelungal — AI Nidhi Utaviyaalar",
        "chat_sub": "Ungal arikkai Shieldy-ku theriyum. Eduvaiyum keelungal!",
        "chat_placeholder": "Utaa. Naan eppadiyum cemippalaam?",
        "chat_send": "Anuppu", "chat_thinking": "Shieldy yosikkiraar...",
        "chat_greeting": "Vanakkam! Naan **Shieldy**, ungal nidhi utaviyaalar.\n\nUngal arikkaiyai padittean. Etuvaiyum keelungal!",
        "buffer_hdr": "Ungal Avasarakaala Cemippu Tittam",
        "fund_lbl": "Ilakku Nidhi", "save_lbl": "Maadumaantira Cemippu", "protect_lbl": "Paadukaapu Vaarangal",
        "download_lbl": "Muzhu Arikkaiyai Padiviraakkavum (PDF)",
        "deficit_warn": "Sila vaarangal varumanam selavai vida kuiraavaaga iruntadu.",
        "deficit_detail": "vaarangal Rs.{exp:.0f}/vaarattirku keele.",
        "low_risk": "Kurai — Nalladu!", "med_risk": "Naduttaram — Kavanang", "high_risk": "Adhikam — Udanadi nadavadikkai",
        "lang_instr": "Please respond entirely in Tamil language. Use simple, warm Tamil language suitable for gig workers.",
        "reset_btn": "Meendum Tudanghu",
        "chart_lbl_actual": "Unmaiyaan Varumanam", "chart_lbl_forecast": "Mugankanipu",
        "chart_lbl_expense": "Selavu Vari", "chart_title": "8 Vaarangalin Varumanam",
        "chart_trend": "Pokkku", "chart_avg": "Saraasar",
        "reanalyze_btn": "Meendum Paguppaayvu", "lang_notice": "Mozhi maari! Pudiya mozhiyil insights perya Re-analyze seyyungal.",
    },
    "Bengali": {
        "tagline": "Aay janun. Bhabishyat garun.",
        "sub": "8 saptaher aay likhun, jhunki dekhun, AI parikolpana pan — binamuulye.",
        "step1": "Aapnar Tottho", "step2": "Saptahik Aay", "step3": "Aapnar Phalaaphal",
        "name_lbl": "Aapnar Naam",
        "worker_lbl": "Aapni ki dharaner kaj koren?",
        "city_lbl": "Aapni kon shahare kaj koren?",
        "dep_lbl": "Aapnar upor kotojoner nirbharshilata ache?",
        "exp_lbl": "Aapnar maasik kharch (Rs.)",
        "income_hdr": "Prati saptahe koto aay korechen?",
        "income_sub": "Gato 8 saptaher aay likhun. Anumani holeo cholbe!",
        "analyze_btn": "Aamar Aarthik Protibedon Dekhun",
        "avg_lbl": "Saptahik Gord", "forecast_lbl": "Porer Saptaher Purbhaas",
        "risk_lbl": "Jhunkir Matra", "buffer_lbl": "Suroksha Sonchoy",
        "ai_hdr": "Aapnar Byaktigat Parikolpana",
        "chat_hdr": "Shieldy-ke Jiggesh Korun — AI Aarthik Sahakari",
        "chat_sub": "Shieldy aapnar protibedon jane. Jekono bhashay jiggesh korun!",
        "chat_placeholder": "Jemon: Aami kibhabe aro sonchoy korbo?",
        "chat_send": "Pathaan", "chat_thinking": "Shieldy bhabchhe...",
        "chat_greeting": "Nomoshkar! Aami **Shieldy**, aapnar aarthik sahakari.\n\nAapnar protibedon porechi. Jekono proshno korun!",
        "buffer_hdr": "Aapnar Joruri Sonchoy Parikolpana",
        "fund_lbl": "Lakkha Tahobil", "save_lbl": "Maasik Sonchoy", "protect_lbl": "Surokkhit Saptah",
        "download_lbl": "Sampurna Protibedon Download Korun (PDF)",
        "deficit_warn": "Kichhu saptah aay khorcheyr cheye kom chilo.",
        "deficit_detail": "saptah Rs.{exp:.0f}/saptaher niche.",
        "low_risk": "Kom — Bhalo!", "med_risk": "Madhyom — Sotokor thakun", "high_risk": "Beshi — Ekhoni podokkhep nin",
        "lang_instr": "Please respond entirely in Bengali language. Use simple, warm Bengali.",
        "reset_btn": "Aabar Shuru Korun",
        "chart_lbl_actual": "Bastob Aay", "chart_lbl_forecast": "Purbhaas",
        "chart_lbl_expense": "Khorcheyr Rekha", "chart_title": "8 Saptaher Aay",
        "chart_trend": "Dharaa", "chart_avg": "Gordo",
        "reanalyze_btn": "Punoray Bishleshhon Korun", "lang_notice": "Bhaasha paltecho! Notun bhaashay insights-er jonno Re-analyze korun.",
    },
}

WORKER_TYPES = {
    "English":  ["Delivery Rider (Food / Parcel)", "Ride-sharing Driver (Ola / Uber)", "Freelancer / Online Work", "Daily Wage / Construction", "Street Vendor / Shop Helper", "Domestic / Household Worker", "Warehouse / Logistics", "Other Gig Work"],
    "Hindi":    ["Delivery Rider", "Ride-sharing Driver", "Freelancer / Online kaam", "Daily Maazdoor / Nirmaann", "Street Vendor", "Gharelu Kaamgaar", "Warehouse / Logistics", "Anya Gig Kaam"],
    "Marathi":  ["Delivery Rider", "Ride-sharing Driver", "Freelancer / Online kaam", "Danik Mazdoor", "Pheriwala", "Gharguti Kaamgaar", "Warehouse / Logistics", "Itar Gig kaam"],
    "Tamil":    ["Delivery Rider", "Ride-sharing Driver", "Freelancer", "Dinak kooli thozhilaalar", "Theru Vitrpanaiyaalar", "Veetu Veelaiyaal", "Warehouse / Logistics", "Mattra Kig Velai"],
    "Bengali":  ["Delivery Rider", "Ride-sharing Driver", "Freelancer / Online kaj", "Dainik Mazdoor", "Pheri Bikreta", "Grihosthali Kormi", "Warehouse / Logistics", "Onyanyo Gig kaj"],
}

WEEK_DEFAULTS = [8200, 9500, 7800, 11200, 6500, 10800, 9100, 12400]
LANG_OPTIONS  = list(LANG.keys())


# ══════════════════════════════════════════════════════════
# IMPROVED CHART BUILDER
# ══════════════════════════════════════════════════════════
def build_income_chart(weekly_income, weekly_expense, avg_income, forecast, L, for_pdf=False):
    data    = np.array(weekly_income, dtype=float)
    ma      = moving_average(data)
    weeks_x = [f"W{i+1}" for i in range(8)]

    C_BLUE      = "#2563EB"
    C_BLUE_SOFT = "#60A5FA"
    C_GREEN     = "#16A34A"
    C_GREEN_LT  = "rgba(22,163,74,0.15)"
    C_RED       = "#DC2626"
    C_RED_LT    = "rgba(220,38,38,0.13)"
    C_AMBER     = "#D97706"
    C_GREY      = "#6B7A9B"
    C_GREY_LT   = "rgba(107,122,155,0.15)"
    C_BG        = "white" if for_pdf else "rgba(0,0,0,0)"
    C_GRID      = "#EEF2F8"

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("", "Week-on-Week Change (Rs.)")
    )

    bar_colors      = [C_GREEN if v >= weekly_expense else C_RED for v in data]
    bar_colors_soft = ["rgba(22,163,74,0.18)" if v >= weekly_expense else "rgba(220,38,38,0.18)" for v in data]

    # Shaded area under trend
    fig.add_trace(go.Scatter(
        x=weeks_x + weeks_x[::-1],
        y=list(ma) + [0] * 8,
        fill="toself",
        fillcolor=C_GREY_LT,
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ), row=1, col=1)

    # Bars
    fig.add_trace(go.Bar(
        x=weeks_x,
        y=data,
        name=L["chart_lbl_actual"],
        marker=dict(
            color=bar_colors_soft,
            line=dict(color=bar_colors, width=2),
        ),
        text=[f"Rs.{v:,.0f}" for v in data],
        textposition="outside",
        textfont=dict(size=10, color=C_GREY, family="Outfit"),
        customdata=[[v, weekly_expense] for v in data],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Income: <b>Rs.%{customdata[0]:,.0f}</b><br>"
            "vs Expense: Rs.%{customdata[1]:,.0f}<br>"
            "<extra></extra>"
        ),
    ), row=1, col=1)

    # Forecast bar
    fig.add_trace(go.Bar(
        x=["W9"],
        y=[forecast],
        name=L["chart_lbl_forecast"],
        marker=dict(
            color="rgba(37,99,235,0.18)",
            line=dict(color=C_BLUE, width=2),
            pattern=dict(shape="/", fgcolor=C_BLUE, size=6),
        ),
        text=[f"Rs.{forecast:,.0f}"],
        textposition="outside",
        textfont=dict(size=10, color=C_BLUE, family="Outfit"),
        hovertemplate="<b>Forecast W9</b><br>Rs.%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    # Trend line
    fig.add_trace(go.Scatter(
        x=weeks_x,
        y=ma,
        mode="lines+markers",
        name=L["chart_trend"],
        line=dict(color=C_GREY, width=2.5, dash="dot"),
        marker=dict(size=5, color=C_GREY, symbol="circle"),
        hovertemplate="Trend: Rs.%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    # Average line
    fig.add_hline(
        y=avg_income, line_color=C_BLUE_SOFT, line_dash="dash", line_width=1.5,
        annotation_text=f"  {L['chart_avg']}: Rs.{avg_income:,.0f}",
        annotation_font=dict(color=C_BLUE_SOFT, size=11, family="Outfit"),
        annotation_position="top left",
        row=1, col=1,
    )

    # Expense line
    fig.add_hline(
        y=weekly_expense, line_color=C_AMBER, line_dash="dash", line_width=1.8,
        annotation_text=f"  {L['chart_lbl_expense']}: Rs.{weekly_expense:,.0f}",
        annotation_font=dict(color=C_AMBER, size=11, family="Outfit"),
        annotation_position="bottom left",
        row=1, col=1,
    )

    y_max = max(max(data), forecast) * 1.22

    # Green zone above expense
    fig.add_hrect(
        y0=weekly_expense, y1=y_max,
        fillcolor=C_GREEN_LT, layer="below", line_width=0,
        row=1, col=1,
    )
    # Red zone below expense
    fig.add_hrect(
        y0=0, y1=weekly_expense,
        fillcolor=C_RED_LT, layer="below", line_width=0,
        row=1, col=1,
    )

    # Week-on-week delta panel
    deltas = [0] + [data[i] - data[i - 1] for i in range(1, 8)]
    delta_colors = [C_GREEN if d >= 0 else C_RED for d in deltas]
    fig.add_trace(go.Bar(
        x=weeks_x,
        y=deltas,
        name="WoW Change",
        marker=dict(
            color=["rgba(22,163,74,0.25)" if d >= 0 else "rgba(220,38,38,0.25)" for d in deltas],
            line=dict(color=delta_colors, width=1.5),
        ),
        text=[f"+{d:,.0f}" if d > 0 else (f"{d:,.0f}" if d < 0 else "–") for d in deltas],
        textposition="outside",
        textfont=dict(size=9, color=delta_colors, family="Outfit"),
        hovertemplate="WoW: Rs.%{y:+,.0f}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    fig.add_hline(y=0, line_color=C_GREY, line_width=1, row=2, col=1)

    fig.update_layout(
        paper_bgcolor=C_BG,
        plot_bgcolor=C_BG,
        font=dict(family="Outfit, sans-serif", color=C_GREY, size=12),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#DDE4EF",
            borderwidth=1,
            orientation="h",
            x=0, y=1.07,
            font=dict(size=12, family="Outfit"),
        ),
        margin=dict(l=8, r=8, t=48, b=8) if not for_pdf else dict(l=40, r=40, t=48, b=40),
        height=400 if not for_pdf else 360,
        barmode="overlay",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#DDE4EF",
            font=dict(family="Outfit", size=12, color="#1A2035"),
        ),
    )

    axis_style = dict(
        gridcolor=C_GRID,
        linecolor="#DDE4EF",
        tickfont=dict(family="Outfit", size=11, color=C_GREY),
        showgrid=True,
        zeroline=False,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_yaxes(tickprefix="Rs.", row=1, col=1, range=[0, y_max])
    fig.update_yaxes(tickprefix="Rs.", row=2, col=1)
    fig.update_xaxes(showticklabels=True, row=2, col=1)
    fig.update_annotations(font=dict(size=11, color=C_GREY, family="Outfit"))

    return fig


# ══════════════════════════════════════════════════════════
# PDF GENERATION
# ══════════════════════════════════════════════════════════
def generate_pdf_report(results, weekly_income, worker_type, city, dependents, monthly_exp, ai_insights, worker_name="", L=None):
    if L is None:
        L = {
            "chart_lbl_actual": "Actual Income",
            "chart_lbl_forecast": "Forecast",
            "chart_lbl_expense": "Expense Line",
            "chart_title": "Income Overview",
            "chart_trend": "Trend",
            "chart_avg": "Your Average",
        }

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.colors import HexColor, white
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        def safe(t):
            return (t or "").encode("ascii", "replace").decode("ascii")

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
            rightMargin=1.8 * cm, leftMargin=1.8 * cm, topMargin=2 * cm, bottomMargin=2 * cm)

        BLUE    = HexColor("#2563EB"); BLUE_LT = HexColor("#EEF3FE")
        GREEN   = HexColor("#16A34A"); AMBER   = HexColor("#D97706")
        RED     = HexColor("#DC2626"); GREY    = HexColor("#6B7A9B")
        LIGHT   = HexColor("#F7F9FC"); BORDER  = HexColor("#DDE4EF")
        DARK    = HexColor("#1A2035")

        risk_color = {"LOW": GREEN, "MEDIUM": AMBER, "HIGH": RED}[results["risk_label"]]
        risk_icon  = {"LOW": "LOW [OK]", "MEDIUM": "MEDIUM [!]", "HIGH": "HIGH [!!!]"}[results["risk_label"]]
        styles = getSampleStyleSheet()
        pw = A4[0] - 3.6 * cm

        def sty(name, **kw):
            return ParagraphStyle(name, parent=styles["Normal"], **kw)

        story = []
        weekly_expense = monthly_exp / 4.33
        avg   = results["avg_income"];  fcast  = results["forecast"]
        rscore = results["risk_score"]; bamt   = results["buffer_amount"]
        msave  = results["monthly_save"]; bwks = results["buffer_weeks"]

        # ── Header ──
        header_content = [
            [Paragraph("FinStab", sty("T", fontSize=20, fontName="Helvetica-Bold", textColor=white, alignment=TA_CENTER))],
            [Paragraph("Your Financial Report", sty("S", fontSize=10, fontName="Helvetica", textColor=white, alignment=TA_CENTER))],
        ]
        if worker_name:
            header_content.append(
                [Paragraph(f"Prepared for: {safe(worker_name)}", sty("N", fontSize=11, fontName="Helvetica", textColor=white, alignment=TA_CENTER))]
            )
        hdr = Table(header_content, colWidths=[pw])
        hdr.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BLUE), ("TOPPADDING", (0, 0), (-1, -1), 16), ("BOTTOMPADDING", (0, 0), (-1, -1), 14)]))
        story += [hdr, Spacer(1, 0.4 * cm)]

        # ── KPIs ──
        kpi = Table([
            [Paragraph(f"Rs.{avg:,.0f}",   sty("KV",  fontSize=17, fontName="Helvetica-Bold", textColor=BLUE,       alignment=TA_CENTER)),
             Paragraph(f"Rs.{fcast:,.0f}", sty("KV2", fontSize=17, fontName="Helvetica-Bold", textColor=BLUE,       alignment=TA_CENTER)),
             Paragraph(f"{rscore}/100",    sty("KV3", fontSize=17, fontName="Helvetica-Bold", textColor=risk_color, alignment=TA_CENTER)),
             Paragraph(f"Rs.{bamt:,.0f}",  sty("KV4", fontSize=17, fontName="Helvetica-Bold", textColor=BLUE,       alignment=TA_CENTER))],
            [Paragraph("Weekly Average",   sty("KL",  fontSize=9,  fontName="Helvetica", textColor=GREY, alignment=TA_CENTER)),
             Paragraph("Forecast",         sty("KL2", fontSize=9,  fontName="Helvetica", textColor=GREY, alignment=TA_CENTER)),
             Paragraph("Risk Score",       sty("KL3", fontSize=9,  fontName="Helvetica", textColor=GREY, alignment=TA_CENTER)),
             Paragraph("Safety Buffer",    sty("KL4", fontSize=9,  fontName="Helvetica", textColor=GREY, alignment=TA_CENTER))],
        ], colWidths=[pw / 4] * 4)
        kpi.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BLUE_LT), ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
            ("LINEBEFORE", (1, 0), (3, -1), 0.5, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story += [kpi, Spacer(1, 0.3 * cm)]

        # ── Risk badge ──
        rb = Table([[Paragraph(f"<b>Risk Level: {risk_icon}  |  Score: {rscore}/100</b>",
            sty("RB", fontSize=11, fontName="Helvetica-Bold", textColor=risk_color, alignment=TA_CENTER))]],
            colWidths=[pw])
        rb.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("BOX", (0, 0), (-1, -1), 1.5, risk_color),
            ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story += [rb, Spacer(1, 0.5 * cm)]

        # ── Profile ──
        story += [
            Paragraph("Worker Profile", sty("SC", fontSize=12, fontName="Helvetica-Bold", textColor=BLUE, spaceBefore=4, spaceAfter=6)),
            HRFlowable(width=pw, thickness=1, color=BORDER), Spacer(1, 0.2 * cm),
        ]
        profile_data = []
        if worker_name:
            profile_data.append(["Name", safe(worker_name)])
        profile_data += [
            ["Worker Type", safe(worker_type)],
            ["City", safe(city)],
            ["Dependents", str(dependents)],
            ["Monthly Expenses", f"Rs. {monthly_exp:,.0f}"],
            ["Weekly Expense Target", f"Rs. {weekly_expense:,.0f}"],
        ]
        prof = Table(profile_data, colWidths=[pw * 0.38, pw * 0.62])
        prof.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (0, -1), GREY), ("TEXTCOLOR", (1, 0), (1, -1), DARK),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT, white]),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER), ("INNERGRID", (0, 0), (-1, -1), 0.25, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story += [prof, Spacer(1, 0.5 * cm)]

        # ── Weekly income table ──
        story += [
            Paragraph("Weekly Income Data", sty("SC2", fontSize=12, fontName="Helvetica-Bold", textColor=BLUE, spaceBefore=4, spaceAfter=6)),
            HRFlowable(width=pw, thickness=1, color=BORDER), Spacer(1, 0.2 * cm),
        ]
        wk_tbl = Table([
            [Paragraph(f"Week {i+1}", sty(f"WH{i}",  fontSize=9,  fontName="Helvetica-Bold", textColor=GREY,  alignment=TA_CENTER)) for i in range(4)],
            [Paragraph(f"Rs. {weekly_income[i]:,.0f}", sty(f"WV{i}", fontSize=11, fontName="Helvetica-Bold",
                textColor=RED if weekly_income[i] < weekly_expense else GREEN, alignment=TA_CENTER)) for i in range(4)],
            [Paragraph(f"Week {i+5}", sty(f"WH2{i}", fontSize=9,  fontName="Helvetica-Bold", textColor=GREY,  alignment=TA_CENTER)) for i in range(4)],
            [Paragraph(f"Rs. {weekly_income[i+4]:,.0f}", sty(f"WV2{i}", fontSize=11, fontName="Helvetica-Bold",
                textColor=RED if weekly_income[i+4] < weekly_expense else GREEN, alignment=TA_CENTER)) for i in range(4)],
        ], colWidths=[pw / 4] * 4)
        wk_tbl.setStyle(TableStyle([
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT, white, LIGHT, white]),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER), ("INNERGRID", (0, 0), (-1, -1), 0.25, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story += [
            wk_tbl, Spacer(1, 0.15 * cm),
            Paragraph(f"Red = below target (Rs. {weekly_expense:,.0f}/wk)  |  Green = on target or above",
                sty("SM", fontSize=8, fontName="Helvetica", textColor=GREY)),
            Spacer(1, 0.5 * cm),
        ]

        # ── Income Chart ──
        try:
            from plotly.io import to_image

            pdf_L = {
                "chart_lbl_actual":   "Actual Income",
                "chart_lbl_forecast": "Forecast",
                "chart_lbl_expense":  "Expense Line",
                "chart_title":        "Income Overview",
                "chart_trend":        "Trend",
                "chart_avg":          "Your Average",
            }
            chart_fig = build_income_chart(
                weekly_income, weekly_expense, avg, fcast, pdf_L, for_pdf=True
            )
            img_bytes = to_image(chart_fig, format="png", width=740, height=370, scale=2)
            img_buf   = io.BytesIO(img_bytes)

            story += [
                Paragraph("Income Chart", sty("SC_ch", fontSize=12, fontName="Helvetica-Bold", textColor=BLUE, spaceBefore=4, spaceAfter=6)),
                HRFlowable(width=pw, thickness=1, color=BORDER),
                Spacer(1, 0.2 * cm),
                RLImage(img_buf, width=pw, height=pw * 0.5),
                Spacer(1, 0.5 * cm),
            ]
        except Exception as chart_err:
            story += [
                Paragraph(f"(Chart unavailable: {chart_err})",
                    sty("CE", fontSize=9, fontName="Helvetica", textColor=GREY)),
                Spacer(1, 0.3 * cm),
            ]

        # ── Buffer ──
        story += [
            Paragraph("Emergency Buffer Plan", sty("SC3", fontSize=12, fontName="Helvetica-Bold", textColor=BLUE, spaceBefore=4, spaceAfter=6)),
            HRFlowable(width=pw, thickness=1, color=BORDER), Spacer(1, 0.2 * cm),
        ]
        buf_tbl = Table([
            [Paragraph("Target Emergency Fund", sty("BL1", fontSize=9, fontName="Helvetica", textColor=GREY, alignment=TA_CENTER)),
             Paragraph("Monthly Savings Goal",  sty("BL2", fontSize=9, fontName="Helvetica", textColor=GREY, alignment=TA_CENTER)),
             Paragraph("Weeks Protected",       sty("BL3", fontSize=9, fontName="Helvetica", textColor=GREY, alignment=TA_CENTER))],
            [Paragraph(f"<b>Rs. {bamt:,.0f}</b>",     sty("BV1", fontSize=14, fontName="Helvetica-Bold", textColor=BLUE, alignment=TA_CENTER)),
             Paragraph(f"<b>Rs. {msave:,.0f}/mo</b>", sty("BV2", fontSize=14, fontName="Helvetica-Bold", textColor=BLUE, alignment=TA_CENTER)),
             Paragraph(f"<b>{bwks} weeks</b>",        sty("BV3", fontSize=14, fontName="Helvetica-Bold", textColor=BLUE, alignment=TA_CENTER))],
        ], colWidths=[pw / 3] * 3)
        buf_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BLUE_LT), ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
            ("LINEBEFORE", (1, 0), (2, -1), 0.5, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ]))
        story += [buf_tbl, Spacer(1, 0.5 * cm)]

        # ── AI Insights ──
        if ai_insights:
            story += [
                Paragraph("AI Personalized Plan", sty("SC4", fontSize=12, fontName="Helvetica-Bold", textColor=BLUE, spaceBefore=4, spaceAfter=6)),
                HRFlowable(width=pw, thickness=1, color=BORDER), Spacer(1, 0.2 * cm),
            ]
            for sec in re.split(r'####\s*', safe(ai_insights)):
                if not sec.strip():
                    continue
                lines = sec.strip().split('\n')
                story.append(Paragraph(f"<b>{safe(lines[0]).strip()}</b>",
                    sty("AIH", fontSize=11, fontName="Helvetica-Bold", textColor=BLUE, spaceBefore=10, spaceAfter=4)))
                for ln in lines[1:]:
                    ln = ln.strip()
                    if not ln:
                        continue
                    ln = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', safe(ln))
                    indent = 10 if (ln.startswith("- ") or ln.startswith("* ")) else 0
                    if indent:
                        ln = "  - " + ln[2:]
                    story.append(Paragraph(ln, sty(f"AIB{hash(ln)&0xFFFF}", fontSize=10, fontName="Helvetica",
                        textColor=DARK, leading=15, spaceAfter=3, leftIndent=indent)))
            story.append(Spacer(1, 0.5 * cm))

        story += [
            HRFlowable(width=pw, thickness=0.5, color=BORDER), Spacer(1, 0.2 * cm),
            Paragraph("Generated by FinStab  |  Free financial planning for gig workers in India",
                sty("FT", fontSize=8, fontName="Helvetica", textColor=GREY, alignment=TA_CENTER)),
        ]
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        lines = ["FinStab Report", "=" * 50, "",
                 f"Name: {worker_name}" if worker_name else "",
                 f"Worker: {worker_type} | City: {city} | Dependents: {dependents}",
                 f"Monthly Expenses: Rs. {monthly_exp:,.0f}", "", "Weekly Income:"]
        for i, v in enumerate(weekly_income):
            lines.append(f"  Week {i+1}: Rs. {v:,.0f}")
        lines += ["", f"Average: Rs. {results['avg_income']:,.0f}",
                  f"Forecast: Rs. {results['forecast']:,.0f}",
                  f"Risk: {results['risk_score']}/100 ({results['risk_label']})",
                  f"Emergency Buffer: Rs. {results['buffer_amount']:,.0f}",
                  "", "AI Insights:", ai_insights or "N/A", f"\n[PDF error: {e}]"]
        return "\n".join(lines).encode("utf-8")


# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(page_title="FinStab", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
  --bg:#F7F9FC; --white:#FFFFFF; --border:#DDE4EF; --text:#1A2035;
  --muted:#6B7A9B; --accent:#2563EB; --accent-lt:#EEF3FE;
  --green:#16A34A; --green-lt:#DCFCE7; --amber:#D97706; --amber-lt:#FEF3C7;
  --red:#DC2626; --red-lt:#FEE2E2; --radius:14px;
  --shadow:0 2px 16px rgba(37,99,235,0.07);
}
html, body, [class*="css"] {
  font-family: 'Outfit', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
.main .block-container { max-width:860px !important; padding:0 1.5rem 4rem !important; }
section[data-testid="stSidebar"] { display:none !important; }
#MainMenu, footer, header { visibility:hidden !important; }
.stDeployButton { display:none !important; }

.topbar { display:flex; align-items:center; justify-content:space-between;
  padding:18px 0 24px; border-bottom:1px solid var(--border); margin-bottom:32px; }
.topbar-brand { display:flex; align-items:center; gap:10px; }
.topbar-logo { width:40px; height:40px; background:var(--accent); border-radius:10px;
  display:flex; align-items:center; justify-content:center; font-size:22px; color:white; }
.topbar-name { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:18px; color:var(--text); }
.topbar-name span { color:var(--accent); }
.topbar-badge { font-size:11px; background:var(--accent-lt); color:var(--accent);
  padding:4px 12px; border-radius:20px; font-weight:600; letter-spacing:0.02em; }

.hero { text-align:center; padding:8px 0 40px; }
.hero-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:clamp(26px,5vw,40px);
  font-weight:800; line-height:1.15; color:var(--text); margin-bottom:12px; }
.hero-title span { color:var(--accent); }
.hero-sub { font-size:16px; color:var(--muted); max-width:480px; margin:0 auto; line-height:1.7; }

.step-pill { display:inline-flex; align-items:center; gap:8px; background:var(--accent-lt);
  color:var(--accent); font-size:13px; font-weight:700; padding:6px 16px;
  border-radius:24px; margin-bottom:16px; }

.card { background:var(--white); border:1px solid var(--border); border-radius:var(--radius);
  padding:28px; margin-bottom:20px; box-shadow:var(--shadow); }
.card-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:17px; font-weight:700;
  color:var(--text); margin-bottom:6px; }
.card-sub { font-size:13px; color:var(--muted); margin-bottom:20px; }

.chart-card { background:var(--white); border:1px solid var(--border); border-radius:var(--radius);
  padding:24px 20px 16px; margin-bottom:20px; box-shadow:var(--shadow); }
.chart-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:16px; font-weight:700;
  color:var(--text); margin-bottom:4px; }
.chart-sub { font-size:12px; color:var(--muted); margin-bottom:16px; }
.chart-legend { display:flex; gap:18px; flex-wrap:wrap; margin-bottom:14px; }
.legend-dot { display:inline-flex; align-items:center; gap:6px; font-size:12px;
  color:var(--muted); font-weight:500; }
.dot { width:10px; height:10px; border-radius:50%; display:inline-block; }

.kpi { background:var(--white); border:1px solid var(--border); border-radius:12px;
  padding:18px 20px; box-shadow:var(--shadow); }
.kpi-label { font-size:12px; font-weight:600; color:var(--muted); text-transform:uppercase;
  letter-spacing:0.06em; margin-bottom:8px; }
.kpi-value { font-family:'Plus Jakarta Sans',sans-serif; font-size:26px; font-weight:800;
  color:var(--text); line-height:1; }
.kpi-delta { font-size:12px; margin-top:5px; font-weight:600; }
.delta-up { color:var(--green); } .delta-down { color:var(--red); } .delta-mid { color:var(--muted); }

.risk-low  { background:var(--green-lt); color:var(--green); }
.risk-med  { background:var(--amber-lt); color:var(--amber); }
.risk-high { background:var(--red-lt);   color:var(--red);   }
.risk-badge { display:inline-flex; align-items:center; gap:6px; padding:5px 14px;
  border-radius:20px; font-size:13px; font-weight:700; margin-top:6px; }

.warn-box { background:var(--amber-lt); border:1px solid #FCD34D;
  border-left:4px solid var(--amber); border-radius:10px; padding:14px 18px;
  font-size:14px; color:#92400E; margin-bottom:20px; line-height:1.6; }

.ai-card { background:var(--accent-lt); border:1px solid #BFDBFE;
  border-left:4px solid var(--accent); border-radius:var(--radius);
  padding:24px 28px; margin-bottom:20px; }
.ai-tag { display:inline-block; font-size:10px; font-weight:700; letter-spacing:0.12em;
  text-transform:uppercase; background:var(--accent); color:white;
  padding:3px 10px; border-radius:6px; margin-bottom:16px; }

.buf-card { background:var(--white); border:1px solid var(--border); border-radius:12px;
  padding:20px 16px; text-align:center; box-shadow:var(--shadow); }
.buf-icon { font-size:28px; margin-bottom:10px; }
.buf-val  { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:20px;
  color:var(--accent); margin-bottom:5px; }
.buf-lbl  { font-size:12px; color:var(--muted); line-height:1.5; }

.lang-notice { background:#FEF3C7; border:1px solid #FCD34D; border-radius:10px;
  padding:10px 16px; font-size:13px; color:#92400E; margin-bottom:16px; }

.greeting-banner { background: linear-gradient(135deg, var(--accent-lt) 0%, #dbeafe 100%);
  border: 1px solid #BFDBFE; border-radius: 12px; padding: 14px 20px;
  font-size: 15px; font-weight: 600; color: var(--accent); margin-bottom: 20px; }

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
  background: #ffffff !important; border: 1.5px solid var(--border) !important;
  border-radius: 10px !important; color: #1A2035 !important;
  font-family: 'Outfit', sans-serif !important; font-size: 15px !important;
}
.stSelectbox > div > div {
  background: #ffffff !important; border: 1.5px solid var(--border) !important;
  border-radius: 10px !important;
}
.stSelectbox * { color: #1A2035 !important; }
[data-baseweb="select"], [data-baseweb="select"] * { color: #1A2035 !important; background-color: transparent !important; }
[data-baseweb="select"] input { color: #1A2035 !important; caret-color: #1A2035 !important; }
.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}
[data-baseweb="popover"], [data-baseweb="popover"] > div,
[data-baseweb="menu"], ul[data-baseweb="menu"] {
  background: #ffffff !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; box-shadow: 0 8px 24px rgba(37,99,235,0.12) !important;
}
[data-baseweb="popover"] *, [data-baseweb="menu"] * { color: #1A2035 !important; background-color: transparent !important; }
[data-baseweb="popover"] li, [data-baseweb="menu"] li, [role="option"] { background: #ffffff !important; }
[role="option"]:hover, [data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] [role="option"]:hover { background: var(--accent-lt) !important; }
[role="option"]:hover *, [data-baseweb="popover"] [role="option"]:hover *,
[data-baseweb="menu"] [role="option"]:hover * { color: var(--accent) !important; }
[role="option"][aria-selected="true"], [data-baseweb="popover"] [aria-selected="true"],
[data-baseweb="menu"] [aria-selected="true"] { background: var(--accent-lt) !important; font-weight: 600 !important; }
[role="option"][aria-selected="true"] *, [data-baseweb="popover"] [aria-selected="true"] *,
[data-baseweb="menu"] [aria-selected="true"] * { color: var(--accent) !important; }
.stSelectbox label, .stTextInput label, .stNumberInput label,
[data-testid="stWidgetLabel"] p { font-size: 14px !important; font-weight: 600 !important; color: #1A2035 !important; }

.stButton > button {
  background: var(--accent) !important; color: white !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 700 !important;
  font-size: 15px !important; border: none !important; border-radius: 10px !important;
  padding: 12px 28px !important; transition: all 0.18s !important;
}
.stButton > button:hover {
  background: #1d4ed8 !important;
  box-shadow: 0 4px 18px rgba(37,99,235,0.28) !important;
  transform: translateY(-1px) !important;
}
.stDownloadButton > button {
  background: var(--accent) !important; color: white !important;
  border: none !important; font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 700 !important; font-size: 15px !important; border-radius: 10px !important;
  padding: 12px 28px !important; transition: all 0.18s !important; width: 100% !important;
}
.stDownloadButton > button:hover {
  background: #1d4ed8 !important;
  box-shadow: 0 4px 18px rgba(37,99,235,0.28) !important;
  transform: translateY(-1px) !important;
}

.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="metric-container"] { display:none; }
hr { border-color: var(--border) !important; margin: 28px 0 !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════
for k, v in [("step", "input"), ("report_context", ""), ("chat_history", []),
              ("analysis_done", False), ("ai_insights", ""), ("results", {}),
              ("analysis_lang", None), ("worker_name", "")]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">
    <div class="topbar-logo">🛡</div>
    <div class="topbar-name">Fin<span>Stab</span></div>
  </div>
  <div class="topbar-badge">AI for Social Good</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# LANGUAGE SELECTOR
# ══════════════════════════════════════════════════════════
lang_col, _ = st.columns([1, 2])
with lang_col:
    lang = st.selectbox("Language", LANG_OPTIONS, key="lang_select", label_visibility="collapsed")
L = LANG[lang]

lang_changed = (
    st.session_state.step == "results"
    and st.session_state.analysis_lang is not None
    and st.session_state.analysis_lang != lang
)


# ══════════════════════════════════════════════════════════
# INPUT SCREEN
# ══════════════════════════════════════════════════════════
if st.session_state.step == "input":

    tagline = L["tagline"]
    for sep in [".", "।"]:
        if sep in tagline:
            idx = tagline.index(sep)
            h1, h2 = tagline[:idx + 1].strip(), tagline[idx + 1:].strip()
            break
    else:
        h1, h2 = tagline, ""

    st.markdown(f"""
    <div class="hero">
      <h1 class="hero-title">{h1}<br><span>{h2}</span></h1>
      <p class="hero-sub">{L["sub"]}</p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="step-pill">① {L["step1"]}</div>', unsafe_allow_html=True)
    with st.container():
        worker_name = st.text_input(L["name_lbl"], value="", placeholder="e.g. Rahul Sharma")
        col_a, col_b = st.columns(2)
        with col_a:
            worker_type = st.selectbox(L["worker_lbl"], WORKER_TYPES.get(lang, WORKER_TYPES["English"]))
            dependents  = st.number_input(L["dep_lbl"], min_value=0, max_value=20, value=2, step=1)
        with col_b:
            city        = st.text_input(L["city_lbl"], value="Mumbai")
            monthly_exp = st.number_input(L["exp_lbl"], min_value=500, value=12000, step=500)

    st.markdown(f'<div class="step-pill">② {L["step2"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="card-title">{L["income_hdr"]}</div>'
                f'<div class="card-sub">{L["income_sub"]}</div>', unsafe_allow_html=True)
    weekly_income = []
    cols4 = st.columns(4)
    for i in range(8):
        with cols4[i % 4]:
            weekly_income.append(st.number_input(f"Week {i+1}", min_value=0,
                value=WEEK_DEFAULTS[i], step=100, key=f"wk_{i}"))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_c, _ = st.columns([1, 2, 1])
    with btn_c:
        go = st.button(L["analyze_btn"], use_container_width=True)

    if go:
        st.session_state.update({
            "worker_name": worker_name, "worker_type": worker_type, "city": city,
            "dependents": dependents, "monthly_exp": monthly_exp, "weekly_income": weekly_income,
        })
        data           = np.array(weekly_income, dtype=float)
        weekly_expense = monthly_exp / 4.33
        avg_income     = float(np.mean(data))
        risk_score, risk_label = calc_risk_score(data, weekly_expense)
        forecast       = get_forecast(data)[0]
        buf_amt, m_save, buf_wks = calc_emergency_buffer(weekly_expense, avg_income, int(dependents))
        st.session_state.results = {
            "avg_income": avg_income, "forecast": forecast,
            "risk_score": risk_score, "risk_label": risk_label,
            "buffer_amount": buf_amt, "monthly_save": m_save,
            "buffer_weeks": buf_wks, "weekly_expense": weekly_expense,
        }
        ctx = build_report_context(weekly_income, worker_type, city, int(dependents),
            monthly_exp, avg_income, risk_score, risk_label, forecast, buf_amt, m_save, buf_wks)
        if worker_name.strip():
            ctx = f"Worker Name: {worker_name}\n" + ctx
        st.session_state.report_context = ctx
        with st.spinner("Generating your personalized plan..."):
            insights = analyze_income(
                weekly_income=weekly_income, worker_type=worker_type, city=city,
                dependents=int(dependents), weekly_expense=weekly_expense,
                avg_income=avg_income, risk_score=risk_score, risk_label=risk_label,
                forecast=forecast, lang_instruction=L["lang_instr"])
        st.session_state.ai_insights  = insights
        st.session_state.analysis_lang = lang
        greeting = L["chat_greeting"]
        if worker_name.strip():
            for old, new in [("Hello!", f"Hello, {worker_name}!"), ("Namaste!", f"Namaste, {worker_name}!"),
                             ("Namaskar!", f"Namaskar, {worker_name}!"), ("Vanakkam!", f"Vanakkam, {worker_name}!"),
                             ("Nomoshkar!", f"Nomoshkar, {worker_name}!")]:
                greeting = greeting.replace(old, new)
        st.session_state.chat_history  = [{"role": "assistant", "content": greeting}]
        st.session_state.analysis_done = True
        st.session_state.step          = "results"
        st.rerun()


# ══════════════════════════════════════════════════════════
# RESULTS SCREEN
# ══════════════════════════════════════════════════════════
else:
    R  = st.session_state.results
    ctx = st.session_state.report_context
    avg_income    = R["avg_income"];    forecast      = R["forecast"]
    risk_score    = R["risk_score"];    risk_label    = R["risk_label"]
    buffer_amount = R["buffer_amount"]; monthly_save  = R["monthly_save"]
    buffer_weeks  = R["buffer_weeks"];  weekly_expense = R["weekly_expense"]
    weekly_income = st.session_state["weekly_income"]
    worker_name   = st.session_state.get("worker_name", "")
    worker_type   = st.session_state["worker_type"]
    city          = st.session_state["city"]
    dependents    = st.session_state["dependents"]
    monthly_exp   = st.session_state["monthly_exp"]

    risk_color = {"LOW": "#16A34A", "MEDIUM": "#D97706", "HIGH": "#DC2626"}[risk_label]
    risk_text  = {"LOW": L["low_risk"], "MEDIUM": L["med_risk"], "HIGH": L["high_risk"]}[risk_label]
    risk_cls   = {"LOW": "risk-low", "MEDIUM": "risk-med", "HIGH": "risk-high"}[risk_label]
    fd  = forecast - avg_income
    fp  = abs(fd / avg_income * 100) if avg_income else 0
    def_wks = [i + 1 for i, v in enumerate(weekly_income) if v < weekly_expense]

    # ── Language changed notice ──
    if lang_changed:
        st.markdown(f'<div class="lang-notice">🌐 {L["lang_notice"]}</div>', unsafe_allow_html=True)
        rc1, rc2 = st.columns(2)
        with rc1:
            if st.button(f"{L['reanalyze_btn']} {lang}", use_container_width=True):
                with st.spinner("Generating insights..."):
                    insights = analyze_income(
                        weekly_income=weekly_income, worker_type=worker_type, city=city,
                        dependents=int(dependents), weekly_expense=weekly_expense,
                        avg_income=avg_income, risk_score=risk_score, risk_label=risk_label,
                        forecast=forecast, lang_instruction=L["lang_instr"])
                st.session_state.ai_insights   = insights
                st.session_state.analysis_lang = lang
                greeting = L["chat_greeting"]
                if worker_name.strip():
                    for old, new in [("Hello!", f"Hello, {worker_name}!"), ("Namaste!", f"Namaste, {worker_name}!"),
                                     ("Namaskar!", f"Namaskar, {worker_name}!"), ("Vanakkam!", f"Vanakkam, {worker_name}!"),
                                     ("Nomoshkar!", f"Nomoshkar, {worker_name}!")]:
                        greeting = greeting.replace(old, new)
                st.session_state.chat_history = [{"role": "assistant", "content": greeting}]
                st.rerun()
        with rc2:
            if st.button(f"← {L['reset_btn']}", use_container_width=True):
                st.session_state.step = "input"; st.session_state.chat_history = []; st.rerun()
    else:
        cb, _ = st.columns([1, 6])
        with cb:
            if st.button(f"← {L['reset_btn']}"):
                st.session_state.step = "input"; st.session_state.chat_history = []; st.rerun()

    st.markdown(f'<div class="step-pill">③ {L["step3"]}</div>', unsafe_allow_html=True)

    if worker_name.strip():
        st.markdown(f'<div class="greeting-banner">👋 Welcome, <strong>{worker_name}</strong>! Here is your personalized financial report.</div>', unsafe_allow_html=True)

    # ── KPI Cards ──
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="kpi">
          <div class="kpi-label">{L["avg_lbl"]}</div>
          <div class="kpi-value">Rs.{avg_income:,.0f}</div>
          <div class="kpi-delta delta-mid">8-week average</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        dc = "delta-up" if fd >= 0 else "delta-down"
        ds = "▲" if fd >= 0 else "▼"
        st.markdown(f"""<div class="kpi">
          <div class="kpi-label">{L["forecast_lbl"]}</div>
          <div class="kpi-value">Rs.{forecast:,.0f}</div>
          <div class="kpi-delta {dc}">{ds} {fp:.1f}% vs avg</div>
        </div>""", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"""<div class="kpi">
          <div class="kpi-label">{L["risk_lbl"]}</div>
          <div class="kpi-value" style="color:{risk_color}">{risk_score}<span style="font-size:14px;color:#6B7A9B">/100</span></div>
          <div class="risk-badge {risk_cls}">{risk_text}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi">
          <div class="kpi-label">{L["buffer_lbl"]}</div>
          <div class="kpi-value" style="color:#16A34A">Rs.{buffer_amount:,.0f}</div>
          <div class="kpi-delta delta-mid">{buffer_weeks}-week target</div>
        </div>""", unsafe_allow_html=True)

    # ── Improved Chart ──
    st.markdown(f"""
    <div class="chart-card">
      <div class="chart-title">📊 {L["chart_title"]}</div>
      <div class="chart-sub">Green bars = above your expense target &nbsp;·&nbsp; Red bars = below target &nbsp;·&nbsp; W9 = AI forecast</div>
      <div class="chart-legend">
        <span class="legend-dot"><span class="dot" style="background:#16A34A"></span> Above Target</span>
        <span class="legend-dot"><span class="dot" style="background:#DC2626"></span> Below Target</span>
        <span class="legend-dot"><span class="dot" style="background:#2563EB;border-radius:2px;width:14px;height:3px"></span> Forecast (W9)</span>
        <span class="legend-dot"><span class="dot" style="background:#6B7A9B"></span> Trend</span>
        <span class="legend-dot"><span class="dot" style="background:#D97706;border-radius:2px;width:14px;height:2px"></span> Expense Line</span>
      </div>
    </div>""", unsafe_allow_html=True)

    fig = build_income_chart(weekly_income, weekly_expense, avg_income, forecast, L)
    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "resetScale2d"],
        "toImageButtonOptions": {"format": "png", "filename": "FinStab_chart", "scale": 2},
    })

    # ── Deficit warning ──
    if def_wks:
        st.markdown(f"""<div class="warn-box">
          <strong>⚠️ {L["deficit_warn"]}</strong><br>
          {len(def_wks)} {L["deficit_detail"].format(exp=weekly_expense)}
          (Weeks: {', '.join(str(w) for w in def_wks)})
        </div>""", unsafe_allow_html=True)

    # ── AI Insights ──
    st.markdown(f'<div class="card-title" style="margin-bottom:12px;">{L["ai_hdr"]}</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="ai-card">
      <div class="ai-tag">✦ AI Powered</div>
      <div style="font-size:14px;color:#1e3a6e;line-height:1.75;">{st.session_state.ai_insights}</div>
    </div>""", unsafe_allow_html=True)

    # ── Buffer Plan ──
    st.markdown(f'<div class="card-title" style="margin-bottom:12px;">{L["buffer_hdr"]}</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    for col, icon, val, lbl in [
        (b1, "🎯", f"Rs.{buffer_amount:,.0f}", L["fund_lbl"]),
        (b2, "📅", f"Rs.{monthly_save:,.0f}",  L["save_lbl"]),
        (b3, "🛡️", f"{buffer_weeks}w",          L["protect_lbl"]),
    ]:
        with col:
            st.markdown(f"""<div class="buf-card">
              <div class="buf-icon">{icon}</div>
              <div class="buf-val">{val}</div>
              <div class="buf-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # ── Download PDF ──
    st.markdown("<br>", unsafe_allow_html=True)
    pdf_bytes = generate_pdf_report(
        R, weekly_income, worker_type, city, dependents, monthly_exp,
        st.session_state.ai_insights, worker_name=worker_name, L=L,
    )
    is_pdf = pdf_bytes[:4] == b'%PDF'
    _, dlc, _ = st.columns([1, 2, 1])
    with dlc:
        st.download_button(
            label=f"📥 {L['download_lbl']}",
            data=pdf_bytes,
            file_name="FinStab_report.pdf" if is_pdf else "FinStab_report.txt",
            mime="application/pdf" if is_pdf else "text/plain",
            use_container_width=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # CHATBOT
    # ══════════════════════════════════════════════════════
    st.markdown(f"""
    <div style="background:#2563EB;color:white;padding:18px 24px;border-radius:14px 14px 0 0;">
      <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:16px;margin-bottom:3px;">
        🤖 {L["chat_hdr"]}
      </div>
      <div style="font-size:13px;opacity:0.85;">{L["chat_sub"]}</div>
    </div>""", unsafe_allow_html=True)

    msgs_html = ""
    for m in st.session_state.chat_history:
        content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', m["content"]).replace("\n", "<br>")
        if m["role"] == "user":
            msgs_html += f"""<div style="display:flex;justify-content:flex-end;margin-bottom:14px;">
              <div style="max-width:78%;padding:12px 16px;border-radius:16px;border-bottom-right-radius:4px;
                          font-size:14px;line-height:1.65;background:#2563EB;color:white;">{content}</div>
            </div>"""
        else:
            msgs_html += f"""<div style="display:flex;justify-content:flex-start;align-items:flex-start;gap:10px;margin-bottom:14px;">
              <div style="width:34px;height:34px;border-radius:50%;background:#2563EB;color:white;
                          display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:16px;">🛡</div>
              <div style="max-width:78%;padding:12px 16px;border-radius:16px;border-bottom-left-radius:4px;
                          font-size:14px;line-height:1.65;background:#F1F5FD;color:#1A2035;">{content}</div>
            </div>"""

    st.markdown(f"""<div style="background:white;border:1px solid #DDE4EF;border-top:none;
      padding:20px 24px;min-height:180px;max-height:380px;overflow-y:auto;">{msgs_html}</div>""",
        unsafe_allow_html=True)
    st.markdown("""<div style="background:#F7F9FC;border:1px solid #DDE4EF;border-top:none;
      border-radius:0 0 14px 14px;padding:10px 20px;"></div>""", unsafe_allow_html=True)

    ic, bc = st.columns([5, 1])
    with ic:
        user_msg = st.text_input("msg", placeholder=L["chat_placeholder"],
            label_visibility="collapsed", key="chat_input_box")
    with bc:
        send = st.button(L["chat_send"], key="chat_send_btn", use_container_width=True)

    if send and user_msg.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_msg.strip()})
        with st.spinner(L["chat_thinking"]):
            reply = chat_with_report(
                report_context=ctx + f"\n\nIMPORTANT: {L['lang_instr']}",
                chat_history=st.session_state.chat_history[:-1],
                user_message=user_msg.strip())
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    # ── Quick prompts ──
    st.markdown("<br>", unsafe_allow_html=True)
    qp_map = {
        "English":  ["What is my biggest financial risk?", "How can I save more?", "Explain my risk score", "What should I do this week?"],
        "Hindi":    ["Mera sabse bada jokhim kya hai?", "Main aur bachat kaise karun?", "Is hafte kya karun?", "Mera jokhim score samjhayen"],
        "Marathi":  ["Maazha sarvat motha dhoka konata?", "Mi adhik bachat kashi karu?", "Ya aaThavdyat kay karu?", "Jokheema score samjava"],
        "Tamil":    ["En miga periya aapatthu enna?", "Naan eppadiyum cemikkalaam?", "Inta vaaram enna seyyalaam?", "Aapatthu mativeen vilakkungal"],
        "Bengali":  ["Aamar shobbcheye boro jhunki ki?", "Aami kibhabe aro sonchoy korbo?", "Ei saptahe ki korbo?", "Jhunki score bujhiye din"],
    }
    prompts = qp_map.get(lang, qp_map["English"])
    qcols   = st.columns(len(prompts))
    for i, (qcol, qp) in enumerate(zip(qcols, prompts)):
        with qcol:
            if st.button(qp, key=f"qp_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": qp})
                with st.spinner(L["chat_thinking"]):
                    reply = chat_with_report(
                        report_context=ctx + f"\n\nIMPORTANT: {L['lang_instr']}",
                        chat_history=st.session_state.chat_history[:-1],
                        user_message=qp)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()