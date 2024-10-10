import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

# 어투 변환을 위한 그룹별 데이터와 매칭 딕셔너리
tone_dict = {
    '스마트투자마스터': '인플루언서',
    '위험감수챌린저': '선비',
    '트렌드속안정주자': '어린이',
    '침착한모험가': '할아버지',
    '핫트렌드선구자': '아저씨'
}

group_metrics = pd.read_csv('그룹별지표.csv', encoding='cp949')

# 고객의 지표를 기반으로 그룹을 분류하고 어투를 반환하는 함수
def classify_customer_tone(customer_metrics_df, group_metrics_df=group_metrics, tone_dict=tone_dict):
    metrics_columns = ['베타계수', '수익률표준편차', '트렌드지수', '투자심리지수']

    # 고객 지표 표준화
    scaler_customer = StandardScaler()
    scaled_customer_metrics = scaler_customer.fit_transform(customer_metrics_df[metrics_columns])

    # 그룹 지표 표준화
    scaler_group = StandardScaler()
    scaled_group_metrics = scaler_group.fit_transform(group_metrics_df[metrics_columns])

    # 코사인 유사도 계산
    tone_result = {}
    for idx, customer_vector in enumerate(scaled_customer_metrics):
        customer_vector = customer_vector.reshape(1, -1)
        similarities = cosine_similarity(customer_vector, scaled_group_metrics)
        most_similar_group_idx = np.argmax(similarities)
        most_similar_group = group_metrics_df.iloc[most_similar_group_idx]['그룹명']
        
        # 고객 ID와 매칭된 어투 저장
        customer_id = customer_metrics_df.iloc[idx]['고객ID']
        matched_tone = tone_dict[most_similar_group]
        tone_result[customer_id] = matched_tone
    st.write(customer_metrics_df)
    
    
    return tone_result, most_similar_group
