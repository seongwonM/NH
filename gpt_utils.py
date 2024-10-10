from openai import OpenAI
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st
from yake import KeywordExtractor


# SentenceTransformer 모델 로드
@st.cache_resource
def load_model(model_path):
    return SentenceTransformer(model_path)

model = load_model('jhgan/ko-sroberta-multitask')

import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 불러오기
load_dotenv()

# 환경 변수 사용
API_KEY = os.getenv('API_KEY')

client = OpenAI(api_key=API_KEY)

# GPT 응답 호출 함수
def get_gpt_response(prompt_type, user_input):
    if prompt_type == "포트폴리오 입력":
        prompt = f"""
        사용자가 다음과 같은 포트폴리오를 입력했습니다: '{user_input}'.
        종목명을 티커종목코드로 변환해주고, 티커종목코드와 종목수량을 반환해주세요.
        반환 예시: AAPL 10, TSLA 5, NVDA 3    
        """
    elif prompt_type == "포트폴리오 업데이트":
        prompt = f"""
        사용자가 다음과 같은 포트폴리오를 입력했습니다: '{user_input}'.
        종목명을 티커종목코드로 변환해주고, 매수/매도를 판단하여 티커종목코드와 변화량을 +, -로 반환해주세요.
        반환 예시: AAPL 10, TSLA -5, NVDA 3
        """
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful financial assistant."},
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo"
    )
    return response.choices[0].message.content.strip()

# GPT 응답을 파싱하는 함수
def parse_portfolio_response(response):
    stock_counts = {}
    matches = re.findall(r'([A-Z]+)\s([+-]?[\d.]+)', response)
    for match in matches:
        ticker = match[0]
        change = float(match[1])
        stock_counts[ticker] = change
    return stock_counts

# 기본 GPT 응답 생성 함수
def gpt_response_basic(prompt):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo"
    )
    return response.choices[0].message.content.strip()

# 20. 고객 분류 및 어투 변환 함수
# 그룹 지표와 어투 매칭 딕셔너리 생성 (그룹명과 어투의 매칭)
tone_dict = {
    '스마트투자마스터': '인플루언서',
    '위험감수챌린저': '선비',
    '트렌드속안정주자': '어린이',
    '침착한모험가': '할아버지',
    '핫트렌드선구자': '아저씨'
}

group_metrics = pd.read_csv('그룹별지표.csv', encoding='cp949')

def classify_customer_tone(customer_metrics_df, group_metrics_df=group_metrics, tone_dict=tone_dict):
    """
    고객의 지표를 기반으로 그룹과 유사도를 비교하여 고객을 분류하고, 그에 맞는 어투를 반환하는 함수.
    
    Args:
        customer_metrics_df (DataFrame): 고객의 지표 데이터프레임
        group_metrics_df (DataFrame): 그룹 지표 데이터프레임
        tone_dict (dict): 그룹별로 어투를 정의한 딕셔너리
    
    Returns:
        dict: 고객 ID와 해당 고객의 어투 매칭 결과
    """
    metrics_columns = ['베타계수', '수익률표준편차', '트렌드지수', '투자심리지수']

    # 고객 지표 표준화
    scaler_customer = StandardScaler()
    scaled_customer_metrics = scaler_customer.fit_transform(customer_metrics_df[metrics_columns])

    # 그룹 지표 표준화
    scaler_group = StandardScaler()
    scaled_group_metrics = scaler_group.fit_transform(group_metrics_df[metrics_columns])

    # 코사인 유사도를 계산하고 각 고객에 대해 가장 유사한 그룹 찾기
    tone_result = {}
    for idx, customer_vector in enumerate(scaled_customer_metrics):
        customer_vector = customer_vector.reshape(1, -1)  # 고객 벡터를 2차원으로 변환
        similarities = cosine_similarity(customer_vector, scaled_group_metrics)  # 유사도 계산
        most_similar_group_idx = np.argmax(similarities)  # 가장 유사한 그룹의 인덱스 찾기
        most_similar_group = group_metrics_df.iloc[most_similar_group_idx]['그룹명']
        
        # 고객ID와 매칭된 어투를 저장
        customer_id = customer_metrics_df.iloc[idx]['고객ID']
        matched_tone = tone_dict[most_similar_group]
        tone_result[customer_id] = matched_tone
    
    return tone_result, most_similar_group

# 21. 답변 양식 일치 평가 함수
metrics_list = [
    '트렌드 지수',
    '베타 계수',
    '수익률 표준 편차',
    '투자 심리 지수',
    '섹터',
    '고객님의 트렌드 지수',
    '고객님의 베타 계수',
    '고객님의 수익률 표준 편차',
    '고객님의 투자 심리 지수',
    '가장 적은 차이를 보이는 지표'
]

def evaluate_clarity(response, metrics=metrics_list):
    # 설명 명확성 평가: 응답이 얼마나 많은 지표 정보를 포함하고 있는지
    metric_count = sum(1 for metric in metrics if metric in response)  # 포함된 지표 수

    return round(metric_count/len(metrics), 3) * 100

def keyphrase_similarity(query, long_document, model=model):
    # 긴 문장에서 핵심어 추출
    kw_extractor = KeywordExtractor(lan="ko", n=1, top=5)  # 영어에 최적화되어 있으며, n=1은 단일 단어 핵심어 추출
    keywords = kw_extractor.extract_keywords(long_document)
    keyphrases = [kw[0] for kw in keywords]
    
    # 핵심어와 짧은 문장 사이의 유사도 계산
    similarities = []
    for keyphrase in keyphrases:
        query_embedding = model.encode([query])
        keyphrase_embedding = model.encode([keyphrase])
        cosine_sim = cosine_similarity(query_embedding, keyphrase_embedding)
        similarities.append(cosine_sim[0][0])
    return max(similarities)

# 문장 2개의 코사인 유사도 계산 함수
def calculate_cosine_similarity(sentence1, sentence2, model=model):
    # 입력 문장 임베딩
    sentence1_embedding = model.encode([sentence1])
    sentence2_embedding = model.encode([sentence2])

    # 벡터 정규화
    sentence1_embedding = sentence1_embedding / np.linalg.norm(sentence1_embedding, axis=1, keepdims=True)
    sentence2_embedding = sentence2_embedding / np.linalg.norm(sentence2_embedding, axis=1, keepdims=True)

    # 코사인 유사도 계산
    cosine_similarity = np.dot(sentence1_embedding, sentence2_embedding.T)[0][0]
    
    return cosine_similarity

# 19. 유사 문장 검색 함수
def find_similar_document(query, index, chunks, model, top_k=1):
    # 질문을 임베딩
    query_embedding = model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    # 유사한 문장 검색 (top_k개의 유사 문장)
    distances, indices = index.search(np.array(query_embedding), top_k)
    
    # 가장 유사한 문장 반환
    similar_documents = [chunks[idx] for idx in indices[0]]
    
    return similar_documents, keyphrase_similarity(query, similar_documents[0])

# ETF 추천 설명용 GPT 프롬프트
def get_etf_recommendation_with_json(etfs, user_metrics, index, documents, model):
    explanations = []
    db_scores = []
    answer_scores = []
    answer_db_scores = []
    for _, etf_info in etfs.iterrows():
        etf_name = etf_info['티커종목코드']
        trend = etf_info['트렌드지수']
        beta = etf_info['베타계수']
        volatility = etf_info['수익률표준편차']
        invest = etf_info['투자심리지수']
        sector = etf_info['섹터분류명']

        # 사용자 지표와 비교
        user_trend = user_metrics['트렌드지수'].values[0]
        user_beta = user_metrics['베타계수'].values[0]
        user_volatility = user_metrics['수익률표준편차'].values[0]
        user_invest = user_metrics['투자심리지수'].values[0]

        description, desc_dist = find_similar_document(f"{etf_name} 설명", index, documents, model, top_k=3)

        metrics = {
            '트렌드지수': abs(trend-user_trend),
            '베타계수': abs(beta-user_beta),
            '수익률표준편차': abs(volatility-user_volatility),
            '투자심리지수': abs(invest-user_invest)
        }

        min_metric = min(metrics, key=metrics.get)

        metric_description, metric_dist = find_similar_document(f"{min_metric} 설명", index, documents, model, top_k=1)

        # GPT에게 전달할 프롬프트
        prompt = f"""
        ETF {etf_name}는 다음과 같은 특징을 가지고 있습니다:
        - 트렌드 지수: {trend}
        - 베타 계수: {beta}
        - 수익률 표준 편차: {volatility}
        - 투자 심리 지수: {invest}
        - 섹터: {sector}
        설명: {description}

        고객님의 포트폴리오와 비교:
        - 고객님의 트렌드 지수: {user_trend}
        - 고객님의 베타 계수: {user_beta}
        - 고객님의 수익률 표준 편차: {user_volatility}
        - 고객님의 투자 심리 지수: {user_invest}

        가장 적은 차이를 보이는 지표:
        - 지표명: {min_metric}
        설명: {metric_description}

        주의 사항:
        - 각 지표는 표준화된 지표로 양수이면 전체 평균보다 높다는 것이고 음수이면 전체 평균보다 낮다는 것입니다.

        메인 요청 사항:
        - {etf_name}에 대한 전체적인 분석
        - 트렌드지수, 베타계수, 수익률표준편차, 투자심리지수에 대한 설명과 차이 설명
        - {min_metric}에 대해 자세한 설명

        찾아야 하는 정보를 찾지 못했을 경우, '-'로 표시해주세요.
        '설명: ' 외에 다른 지표나 지수 설명에는 문장이 아닌 단어 또는 숫자 형식으로 넣어주세요.
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-3.5-turbo"
        )

        explanation = response.choices[0].message.content
        explanations.append(f"{etf_name} 추천 이유:\n{explanation}\n\n")
        db_scores.append(round((desc_dist.mean()+metric_dist.mean())/2, 3)*100)
        answer_scores.append(evaluate_clarity(explanation))
        answer_db_scores.append(round(calculate_cosine_similarity(explanation, prompt), 3)*100)

    return explanations, db_scores, answer_scores, answer_db_scores
