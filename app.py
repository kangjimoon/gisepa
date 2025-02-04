import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import pycountry

# GitHub에서 CSV 데이터 로드
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/kangjimoon/gisepa/main/esports_raw.csv"
    return pd.read_csv(url)

df = load_data()

# 국가별 플랫폼 비율 계산
platform_ratios = df.groupby('CountryCode').agg({'USDPrizeR': 'sum'}).reset_index()

for platform in ['PC', 'Mobile', 'Console']:
    platform_prize = df[df['Platform'] == platform].groupby('CountryCode')['USDPrizeR'].sum().reset_index()
    platform_ratios = platform_ratios.merge(platform_prize, on='CountryCode', how='left', suffixes=('', f'_{platform}'))
    platform_ratios[f'{platform}_ratio'] = platform_ratios[f'USDPrizeR_{platform}'] / platform_ratios['USDPrizeR'] * 100
    platform_ratios[f'{platform}_ratio'] = platform_ratios[f'{platform}_ratio'].fillna(0)

# 클러스터링 수행
X = platform_ratios[['PC_ratio', 'Mobile_ratio', 'Console_ratio']].values
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
platform_ratios['Cluster'] = kmeans.fit_predict(X)

# 국가명 변환
def get_country_name(code):
    try:
        return pycountry.countries.get(alpha_2=code.upper()).name
    except AttributeError:
        return code

platform_ratios['CountryName'] = platform_ratios['CountryCode'].apply(get_country_name)

# Plotly 3D 시각화
fig = go.Figure()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for cluster in range(4):
    cluster_data = platform_ratios[platform_ratios['Cluster'] == cluster]
    fig.add_trace(go.Scatter3d(
        x=cluster_data['PC_ratio'],
        y=cluster_data['Mobile_ratio'],
        z=cluster_data['Console_ratio'],
        mode='markers+text',
        marker=dict(size=8, color=colors[cluster], opacity=0.8),
        text=cluster_data['CountryName'],
        textposition="top center",
        name=f'Cluster {cluster+1}'
    ))

fig.update_layout(
    title='국가별 플랫폼 비율 클러스터 분석',
    scene=dict(xaxis_title='PC (%)', yaxis_title='Mobile (%)', zaxis_title='Console (%)')
)

# Streamlit에 표시
st.title('Esports 플랫폼 비율 클러스터링')
st.plotly_chart(fig)
