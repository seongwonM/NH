import sqlite3
import pandas as pd
# DB 연결 함수
def connect_to_db():
    conn = sqlite3.connect('customer_portfolio.db')
    return conn

# 1. 고객 ID와 비밀번호 확인 함수
def check_customer_id_and_password(customer_id, password):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT 비밀번호 FROM 고객DB WHERE 고객ID = ?", (customer_id,))
    result = cursor.fetchone()
    conn.close()  # DB 연결 닫기
    if result and result[0] == password:
        return True
    return False

# 2. 포트폴리오 저장 함수 (DB에 저장)
def save_portfolio_to_db(customer_id, portfolio_df):
    conn = connect_to_db()
    cursor = conn.cursor()
    for _, row in portfolio_df.iterrows():
        cursor.execute('''
        INSERT INTO 포트폴리오DB (고객ID, 티커종목코드, 보유수량, 비중)
        VALUES (?, ?, ?, ?)
        ''', (customer_id, row['티커종목코드'], row['보유수량'], row['비중']))
    conn.commit()
    conn.close()  # DB 연결 닫기

# 3. 포트폴리오 삭제 후 새로 입력 (DB에서 기존 포트폴리오 삭제)
def delete_existing_portfolio(customer_id):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM 포트폴리오DB WHERE 고객ID = ?", (customer_id,))
    conn.commit()
    conn.close()  # DB 연결 닫기

# 4. 기존 포트폴리오에서 종목 업데이트 함수
def update_existing_portfolio(customer_id, parsed_changes):
    conn = connect_to_db()
    cursor = conn.cursor()

    # 기존 포트폴리오 조회
    existing_portfolio_query = "SELECT 티커종목코드, 보유수량 FROM 포트폴리오DB WHERE 고객ID = ?"
    existing_portfolio_df = pd.read_sql(existing_portfolio_query, conn, params=(customer_id,))

    # 데이터프레임을 딕셔너리로 변환 (주식명 -> 보유수량)
    existing_portfolio = dict(zip(existing_portfolio_df['티커종목코드'], existing_portfolio_df['보유수량']))

    # 종목 추가/삭제 업데이트
    for ticker, change in parsed_changes.items():
        if ticker in existing_portfolio:
            # 기존 종목은 수량을 업데이트 (매수는 더하고, 매도는 뺌)
            existing_portfolio[ticker] += change
            # 매도로 인해 보유수량이 0 이하가 되면 포트폴리오에서 제거
            if existing_portfolio[ticker] <= 0:
                cursor.execute("DELETE FROM 포트폴리오DB WHERE 고객ID = ? AND 티커종목코드 = ?", (customer_id, ticker))
        else:
            # 새로운 종목은 추가
            existing_portfolio[ticker] = change

    # 업데이트된 포트폴리오를 DB에 저장 (0 이하인 종목은 제거)
    for ticker, count in existing_portfolio.items():
        if count > 0:
            cursor.execute('''
            UPDATE 포트폴리오DB
            SET 보유수량 = ?
            WHERE 고객ID = ? AND 티커종목코드 = ?
            ''', (count, customer_id, ticker))

    conn.commit()
    conn.close()  # DB 연결 닫기

# 5. 신규 고객 등록 함수
def register_new_customer(customer_id, name, age, balance, skill_level, password):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO 고객DB (고객ID, 이름, 나이, 보유자금, 투자실력, 비밀번호)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (customer_id, name, age, balance, skill_level, password))
    conn.commit()
    conn.close()  # DB 연결 닫기

# 6. 고객 포트폴리오 불러오기 함수
def load_portfolio_from_db(customer_id):
    conn = connect_to_db()
    query = '''
    SELECT 티커종목코드, 보유수량, 비중
    FROM 포트폴리오DB
    WHERE 고객ID = ?
    '''
    portfolio_df = pd.read_sql_query(query, conn, params=(customer_id,))
    conn.close()  # DB 연결 닫기
    return portfolio_df

# 7. 고객 ID 중복 여부 확인 함수
def is_customer_id_exist(customer_id):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM 고객DB WHERE 고객ID = ?", (customer_id,))
    exists = cursor.fetchone() is not None
    conn.close()  # DB 연결 닫기
    return exists

# 8. 고객 지표 저장 함수
def save_customer_metrics_to_db(customer_id, metrics):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO 고객지표DB (고객ID, 베타계수, 수익률표준편차, 트렌드지수, 투자심리지수, 최대보유섹터)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (customer_id, metrics['베타계수'].iloc[0], metrics['수익률표준편차'].iloc[0], metrics['트렌드지수'].iloc[0], metrics['투자심리지수'].iloc[0], metrics['최대보유섹터'].iloc[0]))
    conn.commit()
    conn.close()  # DB 연결 닫기

# 9. 모든 고객의 최신 지표 불러오기 함수
def fetch_latest_customer_metrics():
    conn = connect_to_db()
    query = '''
    SELECT 고객ID, 베타계수, 수익률표준편차, 트렌드지수, 투자심리지수
    FROM 고객지표DB
    WHERE 지표ID IN (
        SELECT MAX(지표ID) 
        FROM 고객지표DB
        GROUP BY 고객ID
    )
    '''
    customer_metrics_df = pd.read_sql_query(query, conn)
    conn.close()  # DB 연결 닫기
    return customer_metrics_df
