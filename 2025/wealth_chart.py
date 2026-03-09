import pandas as pd
import plotly.graph_objects as go

# ── 데이터 로드 ────────────────────────────────────────────────
try:
    total_df = pd.read_csv('2025/total_data.csv', dtype=str, encoding='utf-8-sig')
except UnicodeDecodeError:
    total_df = pd.read_csv('2025/total_data.csv', dtype=str, encoding='cp949')

try:
    api_df = pd.read_csv('2025/api_csv.csv', dtype=str, encoding='utf-8-sig')
except UnicodeDecodeError:
    api_df = pd.read_csv('2025/api_csv.csv', dtype=str, encoding='cp949')

# ── api_csv에서 필요한 컬럼 추출 (A열, I열, N열) ────────────────
# A=국회의원코드, I=정당명(index 8), N=재선구분명(index 13)
api_cols = api_df.columns.tolist()
key_col   = api_cols[0]   # 국회의원코드 (A열)
party_col = api_cols[8]   # 정당명 (I열)
term_col  = api_cols[13]  # 재선구분명 (N열)

api_subset = api_df[[key_col, party_col, term_col]].copy()
api_subset.columns = ['국회의원코드', '정당명', '재선구분명']
api_subset = api_subset.drop_duplicates(subset='국회의원코드')

# ── total_data와 조인 ────────────────────────────────────────────
merged = total_df.merge(
    api_subset,
    left_on='monaCode',
    right_on='국회의원코드',
    how='left'
)

# ── 현재가액 숫자 변환 (쉼표 제거) ─────────────────────────────
merged['현재가액_num'] = (
    merged['현재가액']
    .str.replace(',', '', regex=False)
    .pipe(pd.to_numeric, errors='coerce')
)
merged = merged.dropna(subset=['현재가액_num'])

# 성명 컬럼 확인 (total_data 헤더: ...성명...)
name_col = '성명'

# ── 재선구분명 정렬 순서 ─────────────────────────────────────────
term_order = ['초선', '재선', '3선', '4선', '5선', '6선', '7선', '8선', '9선', '10선']
existing_terms = [t for t in term_order if t in merged['재선구분명'].dropna().unique()]
# 매칭 안 된 기타값 뒤에 추가
other_terms = [t for t in merged['재선구분명'].dropna().unique() if t not in term_order]
category_array = existing_terms + other_terms

# ── 정당별 색상 맵 ───────────────────────────────────────────────
party_color_map = {
    '더불어민주당': '#0052A5',
    '국민의힘':     '#E61E2B',
    '조국혁신당':   '#00AAFF',
    '개혁신당':     '#FF6B00',
    '진보당':       '#D42B20',
    '기본소득당':   '#7AC943',
    '사회민주당':   '#F6A800',
    '무소속':       '#888888',
}

# ── 차트 생성 ─────────────────────────────────────────────────────
fig = go.Figure()

parties = merged['정당명'].fillna('미확인').unique()

for party in sorted(parties):
    df_party = merged[merged['정당명'].fillna('미확인') == party]
    color = party_color_map.get(party, '#AAAAAA')

    fig.add_trace(go.Scatter(
        x=df_party['재선구분명'].fillna('미확인'),
        y=df_party['현재가액_num'],
        mode='markers',
        name=party,
        marker=dict(
            color=color,
            size=9,
            opacity=0.75,
            line=dict(width=0.5, color='white')
        ),
        customdata=df_party[[name_col, '정당명', '재선구분명', '현재가액_num']].values,
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            '정당: %{customdata[1]}<br>'
            '선수: %{customdata[2]}<br>'
            '현재가액: %{customdata[3]:,.0f} 만원'
            '<extra></extra>'
        )
    ))

fig.update_layout(
    title=dict(
        text='국회의원 재산 현황 (2025)',
        font=dict(size=20)
    ),
    xaxis=dict(
        title='선수 (당선 횟수)',
        categoryorder='array',
        categoryarray=category_array,
        tickfont=dict(size=13),
    ),
    yaxis=dict(
        title='현재가액 (만원)',
        tickformat=',',
        tickfont=dict(size=12),
    ),
    hovermode='closest',
    legend=dict(
        title='정당',
        itemsizing='constant',
        font=dict(size=12),
    ),
    height=720,
    plot_bgcolor='#FAFAFA',
    paper_bgcolor='white',
)

output_path = '2025/wealth_chart.html'
fig.write_html(output_path)
print(f"차트 저장 완료: {output_path}")
fig.show()
