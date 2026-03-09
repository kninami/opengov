import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── 데이터 로드 ────────────────────────────────────────────────────────
total_df = pd.read_csv("total_data.csv", encoding="utf-8-sig")
api_df = pd.read_csv("api_csv.csv", encoding="utf-8-sig")

total_df.columns = [col.strip() for col in total_df.columns]
api_df.columns = [col.strip() for col in api_df.columns]

total_cols = total_df.columns.tolist()
mona_col   = total_cols[2]   # C: monaCode
name_col   = total_cols[6]   # G: 성명
asset_col  = total_cols[13]  # N: 현재가액
change_col = total_cols[15]  # P: 가액변동

api_cols  = api_df.columns.tolist()
code_col  = api_cols[0]   # A: 국회의원코드
party_col = api_cols[8]   # I: 정당명
term_col  = api_cols[13]  # N: 재선구분명

total_sub = total_df[[mona_col, name_col, asset_col, change_col]].copy()
api_sub   = api_df[[code_col, party_col, term_col]].copy()

for col in [asset_col, change_col]:
    total_sub[col] = (
        total_sub[col].astype(str)
        .str.replace(",", "").str.replace('"', "").str.strip()
    )
    total_sub[col] = pd.to_numeric(total_sub[col], errors="coerce")

merged = total_sub.merge(api_sub, left_on=mona_col, right_on=code_col, how="left")
merged[party_col] = merged[party_col].fillna("정보없음")
merged[term_col]  = merged[term_col].fillna("정보없음")
merged[party_col] = merged[party_col].apply(lambda x: x.split("/")[-1].strip())
merged = merged.dropna(subset=[asset_col, change_col])

# ── symlog 변환 ────────────────────────────────────────────────────────
# Plotly는 symlog 미지원 → 수동 변환 후 커스텀 tick으로 원래 값 표시
X_THRESH = 10_000    # 가액변동 선형 구간 임계값
Y_THRESH = 100_000   # 현재가액 선형 구간 임계값

def symlog(arr, linthresh):
    arr = np.asarray(arr, dtype=float)
    safe = np.where(np.abs(arr) > 0, np.abs(arr), 1)
    return np.where(
        np.abs(arr) <= linthresh,
        arr / linthresh,
        np.sign(arr) * (1 + np.log10(safe / linthresh)),
    )

def symlog_ticks(linthresh, data_min, data_max):
    """원래 값 기준 tick 위치·레이블 생성"""
    lin_ticks = [-linthresh, -linthresh // 2, 0, linthresh // 2, linthresh]
    neg_exp = int(np.floor(np.log10(abs(data_min)))) if data_min < -linthresh else None
    pos_exp = int(np.floor(np.log10(abs(data_max)))) if data_max >  linthresh else None

    log_vals = []
    if neg_exp:
        for e in range(int(np.log10(linthresh)) + 1, neg_exp + 1):
            log_vals.append(-10 ** e)
    if pos_exp:
        for e in range(int(np.log10(linthresh)) + 1, pos_exp + 1):
            log_vals.append(10 ** e)

    raw = sorted(set(lin_ticks + log_vals))
    raw = [v for v in raw if data_min * 1.1 <= v <= data_max * 1.1]
    transformed = symlog(raw, linthresh)
    labels = [f"{int(v):,}" for v in raw]
    return transformed.tolist(), labels

merged["x_t"] = symlog(merged[change_col].values, X_THRESH)
merged["y_t"] = symlog(merged[asset_col].values,  Y_THRESH)

x_ticks, x_labels = symlog_ticks(X_THRESH, merged[change_col].min(), merged[change_col].max())
y_ticks, y_labels = symlog_ticks(Y_THRESH, merged[asset_col].min(),  merged[asset_col].max())

# ── 정당 색상 ──────────────────────────────────────────────────────────
party_colors = {
    "더불어민주당": "#0052A5",
    "국민의힘":     "#E61E2B",
    "조국혁신당":   "#00AEEF",
    "개혁신당":     "#FF6600",
    "진보당":       "#AA0000",
    "무소속":       "#888888",
    "사회민주당":   "#F5A623",
}

def get_color(party):
    return party_colors.get(party, "#BBBBBB")

# ── Plotly 차트 ────────────────────────────────────────────────────────
fig = go.Figure()

for party in sorted(merged[party_col].unique()):
    df_p = merged[merged[party_col] == party]
    sign_series = df_p[change_col].apply(lambda v: "+" if v >= 0 else "")
    fig.add_trace(go.Scatter(
        x=df_p["x_t"],
        y=df_p["y_t"],
        mode="markers",
        name=party,
        marker=dict(color=get_color(party), size=8, opacity=0.78,
                    line=dict(width=0.5, color="white")),
        customdata=np.stack([
            df_p[name_col],
            df_p[party_col],
            df_p[term_col],
            df_p[asset_col],
            df_p[change_col],
            sign_series,
        ], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "정당: %{customdata[1]}<br>"
            "선수: %{customdata[2]}<br>"
            "현재가액: %{customdata[3]:,.0f} 천원<br>"
            "가액변동: %{customdata[5]}%{customdata[4]:,.0f} 천원"
            "<extra></extra>"
        ),
    ))

# x=0 기준선 (변환 후 0은 그대로 0)
fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#aaaaaa")

# ── 한국 가구 금융자산 기준선 (단위: 천원) ────────────────────────────
avg_asset     = 136_900    # 평균 금융자산 1억 3,690만 원
top10_asset   = 1_050_000  # 상위 10% 금융자산 10억 5,000만 원

avg_y   = float(symlog(np.array([avg_asset]),   Y_THRESH)[0])
top10_y = float(symlog(np.array([top10_asset]), Y_THRESH)[0])

for y_val, label in [
    (avg_y,   "평균 금융자산"),
    (top10_y, "상위 10% 금융자산"),
]:
    fig.add_hline(
        y=y_val,
        line_width=1.5,
        line_dash="dot",
        line_color="red",
    )
    fig.add_annotation(
        x=1, xref="paper",
        y=y_val, yref="y",
        text=label,
        showarrow=False,
        xanchor="left",
        font=dict(color="red", size=10),
        bgcolor="rgba(255,255,255,0.7)",
    )

fig.update_layout(
    title=dict(text="국회의원 재산 현황 (2025년 3월 기준)", font=dict(size=20)),
    xaxis=dict(
        title="가액변동 (천원)",
        tickvals=x_ticks,
        ticktext=x_labels,
        showgrid=True,
        gridcolor="#e5e5e5",
    ),
    yaxis=dict(
        title="현재가액 (천원)",
        tickvals=y_ticks,
        ticktext=y_labels,
        showgrid=True,
        gridcolor="#e5e5e5",
    ),
    legend=dict(title="정당", x=1.01, y=1),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#ffffff",
    height=700,
    margin=dict(r=160),
)

fig.write_html("asset_chart.html")
print("asset_chart.html 저장 완료")
fig.show()
