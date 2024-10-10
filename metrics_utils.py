import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta

# 1. 종가 불러오기 함수
def get_stock_prices(tickers):
    stock_prices = {}
    now = datetime.now()
    one_week_ago = now - timedelta(days=7)  # 7일 전부터 현재까지의 데이터
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        history = stock.history(start=one_week_ago.strftime('%Y-%m-%d'), end=now.strftime('%Y-%m-%d'))
        stock_prices[ticker] = history['Close'].iloc[-1] # 가장 최근 종가
    return stock_prices

# 포트폴리오 비중 계산 함수
def calculate_portfolio(portfolio):
    stock_prices = get_stock_prices(portfolio.keys())
    portfolio_data = []
    total_value = 0

    for ticker, count in portfolio.items():
        price = stock_prices[ticker]
        value = price * count
        portfolio_data.append({'티커종목코드': ticker, '보유수량': count, '종가': price, '총가치': value})
        total_value += value

    for data in portfolio_data:
        data['비중'] = data['총가치'] / total_value

    return pd.DataFrame(portfolio_data)


# 고객 지표 계산 함수
def calculate_customer_metrics(customer_id, portfolio_df, stock_info):
    merged_df = pd.merge(stock_info, portfolio_df, on='티커종목코드')
    group_result = {'고객ID': customer_id}
    weights = merged_df['비중']

    weighted_beta = (merged_df['베타계수'] * weights).sum()
    weighted_std = np.sqrt((merged_df['수익률표준편차']**2 * weights).sum())
    weighted_invest = (merged_df['투자심리지수'] * weights).sum()
    weighted_trend = (merged_df['트렌드지수'] * weights).sum()
    weighted_sector = merged_df.groupby('섹터분류명')['비중'].sum().sort_values(ascending=False).index[0]

    group_result['베타계수'] = weighted_beta
    group_result['수익률표준편차'] = weighted_std
    group_result['트렌드지수'] = weighted_trend
    group_result['투자심리지수'] = weighted_invest
    group_result['최대보유섹터'] = weighted_sector

    return group_result

주식별지표=pd.read_csv('주식별지표.csv', encoding='cp949')

# 포트폴리오 및 고객 지표 처리 함수
def process_portfolio_and_metrics(customer_id, portfolio_df, stock_info=주식별지표):
    customer_metrics = calculate_customer_metrics(customer_id, portfolio_df, stock_info)
    
    return pd.DataFrame([customer_metrics])

큐레이션지표=pd.read_csv('큐레이션지표.csv', encoding='cp949')

# ETF 추천 함수 (코사인 유사도 기반)
def recommend_etfs(customer_metrics, curation_metrics=큐레이션지표, top_n=5):
    metrics_columns = ['트렌드지수', '베타계수', '수익률표준편차', '투자심리지수']
    customer_vector = customer_metrics[metrics_columns].values
    etf_vectors = curation_metrics[metrics_columns].values
    similarities = cosine_similarity(customer_vector, etf_vectors)
    top_n_indices = similarities[0].argsort()[::-1][:top_n]
    recommended_etfs = curation_metrics.iloc[top_n_indices]
    return recommended_etfs

