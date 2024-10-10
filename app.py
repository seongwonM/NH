import streamlit as st
import pandas as pd
import json
import faiss
from sentence_transformers import SentenceTransformer
from gpt_utils import get_gpt_response, parse_portfolio_response, gpt_response_basic, get_etf_recommendation_with_json, find_similar_document, keyphrase_similarity
from db_utils import check_customer_id_and_password, save_portfolio_to_db, delete_existing_portfolio, update_existing_portfolio, load_portfolio_from_db, register_new_customer, is_customer_id_exist, fetch_latest_customer_metrics, save_customer_metrics_to_db
from metrics_utils import calculate_portfolio, process_portfolio_and_metrics, recommend_etfs
from tone_utils import classify_customer_tone

# JSON 파일에서 데이터 불러오기
with open('documents.json', 'r') as f:
    documents = json.load(f)

# FAISS 인덱스 로드
@st.cache_resource
def load_faiss_index(index_path):
    return faiss.read_index(index_path)

# SentenceTransformer 모델 로드
@st.cache_resource
def load_model(model_path):
    return SentenceTransformer(model_path)

# Streamlit에서 저장된 모델과 인덱스 불러오기
index = load_faiss_index('document_index.index')
model = load_model('jhgan/ko-sroberta-multitask')
ment = pd.read_csv('어투변환_데이터.csv', encoding='utf-8')

# 세션 상태 초기화
if 'step' not in st.session_state:
    st.session_state.step = 0  # 각 단계를 관리할 변수
if 'customer_id' not in st.session_state:
    st.session_state.customer_id = ''
if 'password' not in st.session_state:
    st.session_state.password = ''

# 메인 함수
def portfolio_input_loop(index=index, documents=documents, model=model, ment=ment):
    st.title("포트폴리오 관리 시스템")

    customer_type = st.radio("기존 고객이신가요? 아니면 신규 고객이신가요?", ('기존', '신규'), index=1)
    if st.checkbox('확인', key='check_button'):
        if customer_type == "기존":
            handle_existing_customer()
        elif customer_type == "신규":
            handle_new_customer()

# 기존 고객 로그인 처리
def handle_existing_customer():
    # 고객 ID와 비밀번호 입력을 session_state로 관리
    st.session_state.customer_id = st.text_input("고객 ID를 입력하세요", value=st.session_state.customer_id)
    st.session_state.password = st.text_input("비밀번호를 입력하세요", type="password", value=st.session_state.password)

    # 로그인 버튼 클릭 시 처리
    if st.checkbox("로그인", key='login'):
        if check_customer_id_and_password(st.session_state.customer_id, st.session_state.password):
            st.success(f"{st.session_state.customer_id}님, 반갑습니다!")
            handle_portfolio_management()
        else:
            st.error("아이디 또는 비밀번호가 잘못되었습니다.")

# 신규 고객 등록 처리
def handle_new_customer():
    st.subheader("신규 고객 등록")
    st.session_state.customer_id = st.text_input("고객 ID를 입력하세요", value=st.session_state.customer_id)
    
    if st.session_state.customer_id:
        if is_customer_id_exist(st.session_state.customer_id):
            st.error("해당 고객 ID는 이미 존재합니다. 다른 ID를 입력하세요.")
        else:
            st.success(f"사용 가능한 고객 ID입니다: {st.session_state.customer_id}")

    st.session_state.password = st.text_input("비밀번호를 입력하세요", type="password", value=st.session_state.password)
    customer_name = st.text_input("성명을 입력하세요")
    customer_age = st.number_input("나이를 입력하세요", min_value=1)
    customer_balance = st.number_input("보유자산을 입력하세요 (단위: 만원)", min_value=1)
    customer_skill = st.selectbox("투자 실력을 선택하세요", ["고수", "일반"])

    if st.checkbox("신규 등록", key='new_register'):
        try:
            if not is_customer_id_exist(st.session_state.customer_id):
                register_new_customer(st.session_state.customer_id, customer_name, customer_age, customer_balance, customer_skill, st.session_state.password)
                st.success("신규 고객이 성공적으로 등록되었습니다.")
                portfolio_input_loop()
        except Exception as e:
            st.error(f"고객 등록 중 오류 발생: {str(e)}")

# 포트폴리오 관리 처리
def handle_portfolio_management():
    st.subheader(f"{st.session_state.customer_id}님의 포트폴리오 관리")
    portfolio_df = load_portfolio_from_db(st.session_state.customer_id)
    if portfolio_df.empty:
        st.warning("현재 등록된 포트폴리오가 없습니다.")
        portfolio_choice = st.radio("포트폴리오를 등록하시겠습니까?", ('네', '아니오'), index=0)

        if st.checkbox('확인', key='second_button') and portfolio_choice=='네':
            handle_portfolio_input(st.session_state.customer_id)
    else:
        st.dataframe(portfolio_df)
        portfolio_choice = st.radio("포트폴리오를 수정하시겠습니까?", ('네', '아니오'), index=0)

        if st.checkbox('확인', key='third_button') and portfolio_choice=='네':
            handle_portfolio_modification(st.session_state.customer_id, portfolio_choice)
            st.session_state.portfolio_df = portfolio_df
    if st.checkbox('추가 기능 실행', key='run_additional_features'):
        st.session_state.portfolio_df = portfolio_df
        handle_additional_features()

# 추가 기능 처리: ETF 추천, GPT 포트폴리오 업데이트 등
def handle_additional_features():
    st.write("추가 기능 실행 중...")

    if st.checkbox("고객 분류 결과 확인", value=False):
        handle_customer_classification(st.session_state.customer_id)

    # 필요에 따라 함수 호출
    if st.checkbox("ETF 추천을 받으시겠습니까?", value=False):
        handle_etf_recommendation(st.session_state.customer_id, st.session_state.portfolio_df)


def handle_portfolio_modification(customer_id, portfolio_choice):
    if portfolio_choice == "네":
        choice = st.selectbox("수정 방식을 선택하세요:", ["새로 입력", "포트폴리오 변경", "GPT로 포트폴리오 변경"], index=1)
        if choice == "새로 입력":
            handle_new_portfolio(customer_id)
        elif choice == "포트폴리오 변경":
            handle_portfolio_update(customer_id)
        elif choice == "GPT로 포트폴리오 변경":
            handle_gpt_portfolio_update(customer_id)

def handle_portfolio_input(customer_id):
    st.subheader("포트폴리오 입력")
    stock_names = st.text_input("보유 주식명을 쉼표로 구분하여 입력하세요 (예: AAPL, TSLA)")
    stock_counts = st.text_input("보유 주식 개수를 쉼표로 구분하여 입력하세요 (예: 10, 5)")

    if st.checkbox("포트폴리오 저장", key='new_portfolio_save'):
        portfolio = dict(zip([stock_name.strip() for stock_name in stock_names.split(',')], map(int, [stock_count.strip() for stock_count in stock_counts.split(',')])))
        st.session_state.portfolio_df = calculate_portfolio(portfolio)
        save_portfolio_to_db(customer_id, st.session_state.portfolio_df)

def handle_new_portfolio(customer_id):
    delete_existing_portfolio(customer_id)
    st.success("기존 포트폴리오가 삭제되었습니다.")
    handle_portfolio_input(customer_id)

def handle_portfolio_update(customer_id):
    ticker = st.text_input("변경할 주식의 티커종목코드를 입력하세요 (예: AAPL)")
    change_type = st.selectbox("매수/매도 여부", ["매수", "매도"], index=0)
    change_amount = st.number_input("매수/매도 수량을 입력하세요", min_value=1)

    if st.checkbox("포트폴리오 업데이트", key='portfolio_update'):
        try:
            update_existing_portfolio(customer_id, {ticker: change_amount * (1 if change_type == "매수" else -1)})
            st.success(f"{ticker} 주식의 포트폴리오가 성공적으로 업데이트되었습니다.")
        except Exception as e:
            st.error(f"포트폴리오 업데이트 중 오류 발생: {str(e)}")

def handle_gpt_portfolio_update(customer_id):
    user_input = st.text_area("챗봇에게 보낼 포트폴리오 입력 (예: 애플 10주 추가 매수, 테슬라 5주 매도)")
    if st.checkbox("GPT 포트폴리오 업데이트", key='portfolio_update_gpt'):
        try:
            gpt_response = get_gpt_response("포트폴리오 업데이트", user_input)
            parsed_portfolio = parse_portfolio_response(gpt_response)
            update_existing_portfolio(customer_id, parsed_portfolio)
            st.success("포트폴리오가 GPT의 도움으로 업데이트되었습니다.")
            st.write("업데이트된 포트폴리오:")
            st.dataframe(load_portfolio_from_db(customer_id))
        except Exception as e:
            st.error(f"GPT 포트폴리오 업데이트 중 오류 발생: {str(e)}")

def handle_etf_recommendation(customer_id, portfolio_df):
    st.write("ETF 추천을 위한 로직 실행 중...")
    # try:
    if portfolio_df is not None and not portfolio_df.empty:
        customer_metrics = process_portfolio_and_metrics(customer_id, portfolio_df)
        recommended_etfs = recommend_etfs(customer_metrics)
        st.write("추천된 ETF 목록:")
        st.dataframe(recommended_etfs)

        etf_explanations, db_scores, answer_scores, answer_db_scores = get_etf_recommendation_with_json(
            recommended_etfs, customer_metrics, index, documents, model)

        for i, (explanation, db_score, answer_score, answer_db_score) in enumerate(zip(etf_explanations, db_scores, answer_scores, answer_db_scores)):
            st.write(f"{i+1}번째 ETF")
            if (db_score + answer_score + answer_db_score) / 3 >= 50:
                st.write(explanation)
            else:
                st.write('생성된 문장이 완벽하지 않아서 출력하지 않습니다.')
            st.write(f"데이터베이스에서 추출된 자료의 신뢰도: {db_score}%")
            st.write(f"문장 완성도: {answer_score}%")
            st.write(f"문장과 데이터베이스의 유사도: {answer_db_score}%")

        # 고객ID와 티커종목코드가 동일한 열처럼 취급될 수 있도록 이름을 통일
        customer = customer_metrics.rename(columns={'고객ID': 'ID'})
        etf = recommended_etfs.rename(columns={'티커종목코드': 'ID'})

        # 두 데이터프레임을 수직 결합
        merged_df = pd.concat([customer, etf], ignore_index=True)

        # ID를 인덱스로 설정
        merged_df.set_index('ID', inplace=True)

        # 필요한 열만 선택하여 시각화
        st.bar_chart(merged_df[['베타계수', '수익률표준편차', '트렌드지수', '투자심리지수']])

    else:
        st.error("포트폴리오가 비어 있어 ETF 추천을 진행할 수 없습니다.")
    # except Exception as e:
    #     st.error(f"ETF 추천 처리 중 오류 발생: {e}")

def handle_customer_classification(customer_id):
    # try:
    customer_metrics = process_portfolio_and_metrics(customer_id, st.session_state.portfolio_df)
    if st.checkbox("지표를 DB에 저장하시겠습니까?", value=False):
        save_customer_metrics_to_db(customer_id, customer_metrics)
    customer_metrics_df = fetch_latest_customer_metrics()
    customer_metrics_df=pd.concat([customer_metrics_df, customer_metrics])
    tone_result, group_result = classify_customer_tone(customer_metrics_df)
    st.write(f"고객님은 '{tone_result[customer_id]}' 말투로 분류되었습니다.")
    similar_documents, similar_score = find_similar_document(f"{group_result} 그룹에 대한 설명", index, documents, model, top_k=3)

    prompt = f'''
        **말투 데이터의 내용은 절대 사용하지 말고, 말투만 참고하세요.**
        **출력할 내용은 아래의 그룹 설명에서 가져와야 하며, 절대 변경하지 마세요.**

        ### 1. 말투 데이터 (말투 학습용 데이터로, 내용은 무시하세요):
        - 사용할 말투: {ment[tone_result[customer_id]].dropna()[:5]}
        (이 말투는 학습용이며, 내용은 무시하세요.)

        ### 2. 출력할 실제 내용 (이 내용만 사용하세요):
        - 고객이 속한 그룹: {group_result}
        - 그룹 설명: {similar_documents[0]}

        **중요 지시사항**:
        1. "말투 데이터"는 **문체만 학습**하고, **내용은 절대 사용하지 마세요**.
        2. "출력할 내용"은 "그룹 설명"에서만 가져오세요.
        3. "그룹 설명"의 **내용은 변경하지 말고**, 오직 말투만 변형하세요.
        4. 고객은 위 그룹에 속한 고객입니다. 고객이 충분히 이해하도록 "고객이 {group_result} 그룹에 속했다는 것"을 언급하고 "위에서 학습한 말투"로 인사를 하고 "학습한 말투"로 "고객의 그룹"을 설명하세요.

        답변 구성: 학습한 말투로 인사, 학습한 말투로 고객이 어떤 그룹에 속해있는지, 학습한 말투로 그룹 설명
    '''
    gpt_reply = gpt_response_basic(prompt)
    st.write(gpt_reply)
    # except Exception as e:
    #     st.error(f"고객분류는 회원가입을 하셔야 가능합니다.")

def chatbot_interaction(customer_id, ment=ment):
    st.subheader("챗봇과 대화해보세요")
    customer_response = st.text_input("자유롭게 질문하세요. 로그인을 하시면 고객 분류에 따라 말투가 변화합니다.")
    if customer_response:
        if customer_response.lower() == '종료':
            st.write("대화를 종료합니다.")
        else:
            try:
                customer_metrics_df = fetch_latest_customer_metrics()
                tone_result, _ = classify_customer_tone(customer_metrics_df)
                similar_documents, similar_score = find_similar_document(customer_response, index, documents, model, top_k=3)
                if keyphrase_similarity(customer_response, similar_documents[0]) > 0.5:
                    st.write("이 문장은 데이터베이스에서 유사한 정보를 가져온 후 작성된 문장입니다.")
                else:
                    st.write("이 문장은 데이터베이스에서 유사한 정보를 찾지 못하고 작성된 문장입니다.")
                if customer_id!='':
                    gpt_prompt = f'''
                    **말투 데이터의 내용은 절대 사용하지 말고, 말투만 참고하세요.**
                    **출력할 내용은 아래의 출력할 실제 내용 설명에서 가져와야 하며, 절대 변경하지 마세요.**

                    ### 1. 말투 데이터 (말투 학습용 데이터로, 내용은 무시하세요):
                    - 사용할 말투: {ment[tone_result[customer_id]].dropna()[:5]}
                    (이 말투는 학습용이며, 내용은 무시하세요.)

                    ### 2. 출력할 실제 내용 (이 내용만 사용하세요):
                    사용자 질문을 바탕으로 유사한 문장을 참고해서 답변을 하세요.
                    - 사용자의 질문: {customer_response}
                    - 유사한 문장: {similar_documents[0]}

                    **중요 지시사항**:
                    1. "말투 데이터"는 **문체만 학습**하고, **내용은 절대 사용하지 마세요**.
                    2. "출력할 내용"은 "2. 출력할 실제 내용"에서만 가져오세요.

                    답변 구성: 유사한 문장을 바탕으로 한 사용자 질문에 대한 응답
                    '''
                else:
                    gpt_prompt = f'''
                    ### 출력할 내용:
                    사용자 질문을 바탕으로 유사한 문장을 참고해서 답변을 하세요.
                    - 사용자의 질문: {customer_response}
                    - 유사한 문장: {similar_documents[0]}

                    답변 구성: 유사한 문장을 바탕으로 한 사용자 질문에 대한 응답
                    '''
                gpt_reply = gpt_response_basic(gpt_prompt)
                st.write(f"사용자: {customer_response}")
                st.write(f"챗봇 답변: {gpt_reply}")
                st.write(f"유사 문장 신뢰도: {round(similar_score, 3) * 100}%")
            except Exception as e:
                st.error(f"챗봇 처리 중 오류가 발생했습니다: {str(e)}")


if __name__ == '__main__':
    portfolio_input_loop()
    chatbot_interaction(st.session_state.customer_id)
