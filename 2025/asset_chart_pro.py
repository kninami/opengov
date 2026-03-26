import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
import os

# ── 1. 데이터 로드 및 전처리 (기존 로직 유지) ──────────────────────────────────
# (CSV 파일이 없는 경우를 대비해 예외 처리를 추가했습니다.)
total_data_path = "2026_total.csv"

if not os.path.exists(total_data_path):
    print(f"❌ 오류: '{total_data_path}' 파일이 필요합니다.")
    print("데이터 파일이 있는 경로에서 코드를 실행해주세요.")
    exit()

try:
    total_df = pd.read_csv(total_data_path, encoding="utf-8-sig")
except Exception as e:
    print(f"❌ 오류: 파일을 읽는 중 문제가 발생했습니다: {e}")
    exit()

# 컬럼명 정리 및 변수 할당
total_df.columns = [col.strip() for col in total_df.columns]

total_cols = total_df.columns.tolist()
name_col   = total_cols[3]   # D: 성명
party_col  = total_cols[4]   # E: 소속 (정당)
term_col   = total_cols[6]   # G: 재선여부
asset_col  = total_cols[10]  # K: 현재가액
change_col = total_cols[12]  # M: 가액변동

total_sub = total_df[[name_col, party_col, term_col, asset_col, change_col]].copy()

# 데이터 정제 (symlog 변환을 위해 필수)
for col in [asset_col, change_col]:
    total_sub[col] = (
        total_sub[col].astype(str)
        .str.replace(",", "").str.replace('"', "").str.strip()
    )
    total_sub[col] = pd.to_numeric(total_sub[col], errors="coerce")

# 결측치 처리 (2026_total.csv에는 정당·재선 정보가 이미 포함됨)
merged = total_sub.copy()
merged[party_col] = merged[party_col].fillna("정보없음")
merged[term_col]  = merged[term_col].fillna("정보없음")
merged[party_col] = merged[party_col].apply(lambda x: x.split("/")[-1].strip())
merged = merged.dropna(subset=[asset_col, change_col])

# ── 2. symlog 변환 (시각화 범위 최적화) ──────────────────────────────────────────
X_THRESH = 10_000    # 가액변동 선형 구간 임계값 (1,000만 원)
Y_THRESH = 100_000   # 현재가액 선형 구간 임계값 (1억 원)

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

# ── 3. 스타일 및 색상 설정 (Bootstrap 기반) ──────────────────────────────────
# Pretendard 폰트가 설치되어 있지 않아도 Noto Sans로 대체되도록 설정
FONT_FAMILY = "'Pretendard Standard', Pretendard, 'Noto Sans KR', sans-serif"
COLOR_DANGER = "#dc3545" # Bootstrap Danger (Red)
GRID_COLOR = "#f1f3f5"

party_colors = {
    "더불어민주당": "#0052A5",
    "국민의힘":     "#E61E2B",
    "조국혁신당":   "#00AEEF",
    "개혁신당":     "#FF6600",
    "진보당":       "#AA0000",
    "기본소득당":   "#00D2C3",
    "무소속":       "#888888",
    "사회민주당":   "#F5A623",
    "정보없음":     "#ADB5BD"
}

# ── 4. Plotly 차트 생성 (증감 색상 및 기호 로직 추가) ──────────────────────────────────
fig = go.Figure()

for party in sorted(merged[party_col].unique()):
    df_p = merged[merged[party_col] == party]
    
    # 증감에 따른 기호 및 색상 리스트 생성
    labels = []
    for v in df_p[change_col]:
        if v > 0:
            # 증가: 빨간색 상승 세모
            labels.append(f"<span style='color:#e52233;'>▲ +{int(v):,}</span>")
        elif v < 0:
            # 감소: 파란색 하락 세모 (절댓값 처리하여 '-' 중복 방지)
            labels.append(f"<span style='color:#0033ff;'>▼ -{int(abs(v)):,}</span>")
        else:
            # 변동 없음
            labels.append("<span style='color:#666;'>- 0</span>")

    fig.add_trace(go.Scatter(
        x=df_p["x_t"],
        y=df_p["y_t"],
        mode="markers",
        name=party,
        marker=dict(
            color=party_colors.get(party, "#ADB5BD"), 
            size=11, 
            opacity=0.75,
            line=dict(width=0.8, color="white")
        ),
        # 툴팁에 넣을 데이터 묶음 (labels 리스트 추가)
        customdata=np.stack([
            df_p[name_col],     # 0
            df_p[party_col],    # 1
            df_p[term_col],     # 2
            df_p[asset_col],    # 3
            labels              # 4 (커스텀 레이블)
        ], axis=-1),
        # 툴팁 템플릿 수정
        hovertemplate=(
            "<b><span style='font-size:16px; color:black;'>%{customdata[0]}</span></b> <span style='color:black;'>(%{customdata[2]})</span><br>"
            "<span style='color:black;'>%{customdata[1]}</span><br>"
            "<span style='color:black;'>현재가액: <b>%{customdata[3]:,.0f}</b> 천원</span><br>"
            "<span style='color:black;'>가액변동: <b>%{customdata[4]}</b> <span style='font-size:11px;'>천원</span></span>"
            "<extra></extra>"
        )
    ))

# 기준선 (0점)
fig.add_vline(x=0, line_width=1, line_dash="solid", line_color="#dee2e6")

# 일반 가구 금융자산 기준선 (단위: 천원)
avg_asset     = 136_900    # 평균 1억 3,690만 원
top20_asset   = 314_330    # 상위 20% 평균 3억 1,433만 원 (이미지 데이터 기반)

avg_y   = float(symlog(np.array([avg_asset]),   Y_THRESH)[0])
top20_y = float(symlog(np.array([top20_asset]), Y_THRESH)[0])

for y_val, label in [(avg_y, "가구 평균 금융자산"), (top20_y, "가구 상위 20% 평균")]:
    fig.add_hline(y=y_val, line_width=1.5, line_dash="dot", line_color=COLOR_DANGER)
    fig.add_annotation(
        x=0, xref="paper", y=y_val, yref="y",
        text=f" {label}", showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(color=COLOR_DANGER, size=11, family=FONT_FAMILY),
    )

# ── 5. 레이아웃 및 툴팁 스타일 ────────────────────────────────────────────────
fig.update_layout(
    # HTML 제목은 파이썬 외부 템플릿에서 처리하므로 여기서는 제거합니다.
    xaxis=dict(
        title="가액변동 (천원)",
        tickvals=x_ticks, ticktext=x_labels,
        gridcolor=GRID_COLOR, zeroline=False,
        title_font=dict(size=13, family=FONT_FAMILY)
    ),
    yaxis=dict(
        title="현재가액 (천원)",
        tickvals=y_ticks, ticktext=y_labels,
        gridcolor=GRID_COLOR, zeroline=False,
        title_font=dict(size=13, family=FONT_FAMILY)
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(family=FONT_FAMILY, size=11),
        bgcolor="rgba(255,255,255,0)",
        itemsizing="constant",
    ),
    margin=dict(l=60, r=40, t=60, b=60),
    plot_bgcolor="white",
    paper_bgcolor="white",
    # 툴팁 모서리 라운딩 및 스타일
    hoverlabel=dict(
        bgcolor="white",
        font_size=13,
        font_family=FONT_FAMILY,
        bordercolor="#dee2e6",
        align="left",
        namelength=0 # 패딩 효과
    ),
    height=750, # 차트 자체 높이
    hovermode="closest" # 중요: 태그 렌더링을 위해 필요
)

# ── 6. HTML 생성 (메타태그 및 Bootstrap 패널 포함) ─────────────────────────────
# 차트 객체를 HTML div 문자열로 변환
chart_div = plot(fig, output_type='div', include_plotlyjs='cdn', config=dict(responsive=True))

# 메타데이터 정의
page_title = "2026 국회의원 금융자산 시각화"
page_description = "2026년 3월 정기공개 데이터를 기반으로 한 대한민국 국회의원의 현재 금융자산 및 가액변동 현황 그래프입니다."
data_source_url = "https://cfoi.or.kr/"

# 완벽한 HTML5 템플릿 정의 (Bootstrap 5 포함)
html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <meta name="description" content="{page_description}">
    <meta property="og:title" content="{page_title}">
    <meta property="og:description" content="{page_description}">
    <meta property="og:type" content="website">
    
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" as="style" crossorigin href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" />
    
    <style>
        body {{
            font-family: {FONT_FAMILY};
            background-color: #f8f9fa; /* 아주 연한 회색 배경 */
            padding-top: 0.1rem;
            padding-bottom: 2rem;
        }}
        .container-xl{{
			padding: 20px;
		}}
        .chart-container {{
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            padding: 2rem;
            margin-bottom: 2rem;
        }}
        .page-header {{
            margin-bottom: 0.75rem;
            padding: 2rem 1rem 1rem;
        }}
        .page-header .eyebrow {{
            display: inline-block;
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #4f8ef7;
            margin-bottom: 0.6rem;
        }}
        .page-header h1 {{
            font-size: 2.4rem;
            font-weight: 800;
            color: #1a1a2e;
            line-height: 1.25;
            margin-bottom: 0.5rem;
        }}
        .page-header .subtitle {{
            font-size: 0.82rem;
            color: #8a94a6;
            margin: 0;
        }}
        .text-muted{{
			color: #6c757d;
		}}
        /* Plotly 툴팁 라운딩 강제 적용 (CSS) */
        .js-plotly-plot .hoverlayer .hovertext path {{
            rx: 6px;
            ry: 6px;
        }}
        h1 small{{
			font-size: 1.2rem;
			font-weight: 400;
		}}
        /* ── 모바일 반응형 ── */
        @media (max-width: 768px) {{
            .container-xl {{
                padding: 16px !important;
            }}
            .chart-container {{
                padding: 0.5rem;
                border-radius: 8px;
            }}
            .page-header h1 {{
                font-size: 1.65rem !important;
            }}
            .page-header .subtitle {{
                font-size: 0.75rem;
            }}
            .alert {{
                padding: 1rem !important;
            }}
            .plotly-graph-div {{
                height: 520px !important;
                min-height: 520px !important;
            }}
        }}
        
    </style>
</head>
<body>

<div class="container-xl">
    <div class="page-header text-center">
        <span class="eyebrow">2026 · 공직자 재산공개</span>
        <h1>{page_title}</h1>
        <p class="subtitle">2026년 3월 정기공개 데이터 기준 &nbsp;·&nbsp; 실물자산 제외, 금융자산 중심</p>
    </div>

    <div class="alert alert-light border shadow-sm rounded-3 mb-4 p-4" role="alert">
        <div class="d-flex align-items-center mb-3">
            <h3 class="alert-heading fw-semibold text-dark m-0">🔍 이런 걸 살펴보면 어때요?</h3>
        </div>
        <ul class="list-unstyled mb-0 text-secondary" style="line-height: 1.8;">
			<li class="text-dark">누가 가장 금융자산이 많을까(차트의 가장 위쪽에 위치한 의원이 누구인지, 어떤 정당인지 찾아보세요.)</li>
            <li class="text-dark">금융자산이 급격하게 늘어난 사람은 왜 누구일까(그리고 왤까🤔)</li>
        </ul>
    </div>

    <div class="chart-container">
        {chart_div}
    </div>

   <footer style="background:#f8f9fa; padding: 2.5rem 1rem 2rem; margin-top: 3rem;">
		<div class="container text-center">
			<div class="d-flex justify-content-center align-items-center gap-4 flex-wrap mb-3">
				<div style="text-align:left;">
					<div style="color:#1a1a2e; font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:2px; opacity:0.5;">Data Source</div>
					<a href="{data_source_url}" target="_blank"
					   style="color:#1a1a2e; font-weight:600; font-size:0.9rem; text-decoration:none; border-bottom:1px solid #1a1a2e; padding-bottom:1px;"
					   onmouseover="this.style.opacity='0.6'" onmouseout="this.style.opacity='1'">
						투명사회를 위한 정보공개센터
					</a>
				</div>
    <br/>
				<div style="text-align:left;">
					<div style="color:#1a1a2e; font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:2px; opacity:0.5;">Made by</div>
					<span style="color:#1a1a2e; font-weight:600; font-size:0.9rem;">갱 <span style="opacity:0.5; font-weight:400;">· 도토리랩스</span></span>
				</div>
			</div>
		</div>
	</footer>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/js/bootstrap.bundle.min.js"></script>
<svg xmlns="http://www.w3.org/2000/svg" style="display: none;">
  <symbol id="info-fill" fill="currentColor" viewBox="0 0 16 16">
    <path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.93-9.412-1 4.705c-.07.34.029.533.304.533.194 0 .487-.07.686-.246l-.088.416c-.287.346-.92.598-1.465.598-.703 0-1.002-.422-.808-1.319l.738-3.468c.064-.293.006-.399-.287-.47l-.451-.081.082-.381 2.29-.287zM8 5.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/>
  </symbol>
</svg>


</body>
</html>
"""

# HTML 파일 저장
output_filename = "asset_chart.html"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"✅ 메타태그와 안내 패널이 포함된 '{output_filename}' 저장이 완료되었습니다.")
# 웹브라우저로 자동 열기 (선택사항)
# import webbrowser
# webbrowser.open('file://' + os.path.realpath(output_filename))