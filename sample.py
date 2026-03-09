import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

# 1. 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 2. 데이터 불러오기
file_path = '2016sample.csv'
df = pd.read_csv(file_path)
def clean_numeric(val):
    if pd.isna(val) or val == '': return 0
    val = str(val).replace(',', '').strip()
    try: return float(val)
    except: return 0

df['현재가액_num'] = df['현재가액'].apply(clean_numeric)
def clean_numeric(val):
    if pd.isna(val) or val == '': return 0
    val = str(val).replace(',', '').strip()
    try: return float(val)
    except: return 0

df['현재가액_num'] = df['현재가액'].apply(clean_numeric)
def clean_numeric(val):
    if pd.isna(val) or val == '': return 0
    val = str(val).replace(',', '').strip()
    try: return float(val)
    except: return 0

df['현재가액_num'] = df['현재가액'].apply(clean_numeric)

# 2. 데이터 필터링 (의원별 예금 합산)
deposit_types = ['예금', '정치자금법에 따른 정치자금의 수입 및 지출을 위한 예금계좌의 예금']
df_filtered = df[(df['소속구분'] == '1.국회의원') & (df['재산구분'].isin(deposit_types))]
member_deposits = df_filtered.groupby('성명')['현재가액_num'].sum().reset_index()

# 3. 정렬 및 순위 부여
member_deposits = member_deposits.sort_values(by='현재가액_num').reset_index(drop=True)
member_deposits['순위'] = member_deposits.index + 1

# 4. 통계 계산 (국민 평균 대비)
national_avg_deposit = 45000 # 국민 평균 예금액 (천원 단위)
above_avg_count = len(member_deposits[member_deposits['현재가액_num'] > national_avg_deposit])
total_count = len(member_deposits)

# 5. Plotly 차트 생성
fig = px.scatter(
    member_deposits, 
    x='순위', 
    y='현재가액_num',
    hover_name='성명',
    hover_data={'순위': False, '현재가액_num': ':,.0f'},
    labels={'현재가액_num': '예금액(천원)', '순위': '의원 순번'},
    title='2016 국회의원 예금 분포 및 국민 평균 비교',
    template='plotly_white'
)

# 6. 국민 평균선 추가 (빨간 점선)
fig.add_hline(
    y=national_avg_deposit, 
    line_dash="dash", 
    line_color="red", 
    annotation_text=f"국민 평균 예금: {national_avg_deposit:,.0f}천원",
    annotation_position="top left"
)

# 7. ★ 요청하신 통계 요약 텍스트 추가 (Annotation) ★
fig.add_annotation(
    x=total_count * 0.05,             # X축 위치 (전체의 5% 지점)
    y=member_deposits['현재가액_num'].max() * 0.1,  # Y축 위치 (최대값의 10% 지점)
    text=f"국민 평균보다 예금이 많은 의원: <b>{above_avg_count}명</b> / 총 {total_count}명",
    showarrow=False,
    font=dict(size=14, color="black"),
    bgcolor="white",                  # 배경색
    bordercolor="gray",               # 테두리색
    borderwidth=1,
    borderpad=10,
    opacity=0.9,
    align="left"
)

# 8. 스타일 및 로그 스케일 적용
fig.update_yaxes(type="log") # 자산 격차를 표현하기 위해 로그 스케일 사용
fig.update_traces(marker=dict(size=12, opacity=0.6, color='royalblue', line=dict(width=1, color='DarkSlateGrey')))

# 9. HTML 저장 및 출력
fig.write_html('congress_deposit_interactive_final.html')
fig.show()