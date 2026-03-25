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

# '정보없음'으로 분류된 의원들만 추출
no_info_members = merged[merged[party_col] == "정보없음"]

print(f"\n── 정보없음으로 분류된 의원 명단 (총 {len(no_info_members)}명) ──")
if not no_info_members.empty:
    # 성명, 정당(정보없음으로 나옴), 현재가액 컬럼 출력
    print(no_info_members[[name_col, party_col, asset_col]])
else:
    print("정보없음으로 분류된 의원이 없습니다. 모든 데이터가 잘 매칭되었습니다!")

print(no_info_members.head()) # 전체 컬럼을 다 보여줘서 어떤 값이 비어있는지 확인 가능

y_ticks, y_labels = symlog_ticks(Y_THRESH, merged[asset_col].min(),  merged[asset_col].max())
# ── 3. 스타일 및 색상 설정 ─────────────────────────────────────────────────────
FONT_FAMILY = "Pretendard, 'Noto Sans KR', sans-serif"
party_colors = {
    "더불어민주당": "#0052A5",
    "더불어민주연합": "#0052A5",
    "새로운미래": "#0052A5",
    "국민의힘":     "#E61E2B",
    "국민의미래":     "#E61E2B",
    "조국혁신당":   "#00AEEF",
    "개혁신당":     "#FF6600",
    "진보당":       "#AA0000",
    "무소속":       "#888888",
    "사회민주당":   "#F5A623",
    "정보없음":     "#ADB5BD"
}

# ── 4. Plotly 차트 생성 ──────────────────────────────────────────────────────
fig = go.Figure()

for party in sorted(merged[party_col].unique()):
    df_p = merged[merged[party_col] == party]
    sign_series = df_p[change_col].apply(lambda v: "+" if v >= 0 else "")
    
    fig.add_trace(go.Scatter(
        x=df_p["x_t"],
        y=df_p["y_t"],
        mode="markers",
        name=party,
        marker=dict(
            color=party_colors.get(party, "#ADB5BD"), 
            size=10, 
            opacity=0.75,
            line=dict(width=0.8, color="white")
        ),
        customdata=np.stack([
            df_p[name_col], df_p[party_col], df_p[term_col], 
            df_p[asset_col], df_p[change_col], sign_series
        ], axis=-1),
        hovertemplate=(
            "<span style='font-size:16px;color:black;'><b>%{customdata[0]}</b> (%{customdata[2]})</span><br>"
            "<span style='color:black;'>%{customdata[1]}</span><br>"
            "<span style='color:black;'>현재가액: <b>%{customdata[3]:,.0f}</b> 천원</span><br>"
            "<span style='color:black;'>가액변동: <b>%{customdata[5]}%{customdata[4]:,.0f}</b> 천원</span>"
            "<extra></extra>"
        ),
    ))

# 기준선 (0점)
fig.add_vline(x=0, line_width=1, line_dash="solid", line_color="#dee2e6")

# 일반 가구 금융자산 기준선 (단위: 천원)
avg_asset     = 136_900    # 평균 1억 3,690만 원
top20_asset   = 314_330    # 상위 20% 평균 3억 1,433만 원 (이미지 데이터 기반)

avg_y   = float(symlog(np.array([avg_asset]),   Y_THRESH)[0])
top20_y = float(symlog(np.array([top20_asset]), Y_THRESH)[0])

for y_val, label in [(avg_y, "가구 평균 금융자산"), (top20_y, "가구 상위 20% 평균")]:
    fig.add_hline(y=y_val, line_width=1.5, line_dash="dot", line_color="#dc3545")
    fig.add_annotation(
        x=0, xref="paper", y=y_val, yref="y",
        text=f" {label}", showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(color="#dc3545", size=11, family=FONT_FAMILY),
    )

# ── 5. 레이아웃 (모바일 대응 및 Bootstrap 스타일) ─────────────────────────────
fig.add_annotation(
    text='데이터 출처: <a href="https://cfoi.or.kr/" style="color:#007bff; text-decoration:none;">투명사회를 위한 정보공개센터</a>',
    xref="paper", yref="paper",
    x=1, y=-0.18, 
    showarrow=False,
    xanchor="right",
    font=dict(size=12, color="#6c757d", family=FONT_FAMILY)
)

fig.update_layout(
    title=dict(
        text="<b>국회의원 재산 현황</b><br><span style='font-size:14px; color:gray;'>2026년 3월 정기공개 데이터 기준 (금융자산 중심)</span>",
        font=dict(size=22, family=FONT_FAMILY, color="#212529"),
        x=0.05, y=0.95
    ),
    xaxis=dict(
        title="가액변동 (천원)",
        tickvals=x_ticks, ticktext=x_labels,
        gridcolor="#f1f3f5", zeroline=False,
        title_font=dict(size=13, family=FONT_FAMILY)
    ),
    yaxis=dict(
        title="현재가액 (천원)",
        tickvals=y_ticks, ticktext=y_labels,
        gridcolor="#f1f3f5", zeroline=False,
        title_font=dict(size=13, family=FONT_FAMILY)
    ),
    legend=dict(
        orientation="h",   # 모바일 대응: 하단 가로 배치
        yanchor="bottom",
        y=-0.3,           
        xanchor="center",
        x=0.5,
        font=dict(family=FONT_FAMILY, size=12),
        bgcolor="rgba(255,255,255,0)"
    ),
    margin=dict(l=60, r=40, t=100, b=150),
    plot_bgcolor="white",
    paper_bgcolor="white",
    hoverlabel=dict(
        bgcolor="white",
        font_size=13,
        font_family=FONT_FAMILY,
        bordercolor="#dee2e6"
    ),
    height=850
)

# ── 6. 저장 및 실행 ──────────────────────────────────────────────────────────
fig.write_html("asset_chart_pro.html", include_plotlyjs='cdn')
print("✅ 출처와 모바일 최적화가 적용된 'asset_chart_pro.html' 저장이 완료되었습니다.")
fig.show()