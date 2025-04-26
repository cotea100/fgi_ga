

# 파트 1: 기본 설정 및 백테스팅 함수
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib
import os
import random
import time
from datetime import datetime
from fpdf import FPDF

# Configure matplotlib
matplotlib.use("Agg")
st.set_page_config(layout="wide", page_title="FGI Strategy Optimization")

# Streamlit 앱 시작되는지 확인용 로그
st.write("App is loading...")

# 파일 업로드 함수
def upload_file():
    """Streamlit 파일 업로더를 사용하여 파일 업로드"""
    st.title("FGI Strategy Optimization with Genetic Algorithm")
    st.write("This app helps you find optimal Fear & Greed Index (FGI) trading strategies using genetic algorithms.")
    uploaded_file = st.file_uploader("Upload Excel or CSV file containing date, price, and FGI data", 
                                   type=["csv", "xlsx", "xls"])
    
    if uploaded_file is None:
        st.info("Please upload a file to continue")
        st.stop()
    
    st.success(f"Uploaded: {uploaded_file.name}")
    return uploaded_file.name, {"file": uploaded_file}

# 업로드된 파일 읽기 함수
def read_uploaded_file(uploaded):
    """업로드된 파일을 pandas DataFrame으로 읽기"""
    file = uploaded["file"]
    
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xls") or file.name.endswith(".xlsx"):
        return pd.read_excel(file, engine="openpyxl")
    else:
        raise ValueError("Unsupported file format. Only CSV, XLS, XLSX files are supported.")

# FGI 전략 백테스팅 함수
def backtest_fgi_strategy(df, buy_threshold, sell_threshold, initial_capital=10_000_000, commission_rate=0.0025):
    """
    FGI 전략 백테스팅 실행
    
    매수 조건: FGI가 buy_threshold 아래로 떨어졌다가 다시 위로 올라갈 때
    매도 조건: FGI가 sell_threshold 위로 올라갔다가 다시 아래로 내려올 때
    """
    cash = initial_capital
    units = 0
    current_state = 'CASH'
    portfolio = []
    dates = []
    trade_count = 0
    trade_history = []  # 거래 시각화용
    
    # 초기 플래그
    fgi_low_seen_since_sell = False  # 매도 이후 FGI < buy_threshold를 본 적 있는지
    fgi_high_seen_since_buy = False  # 매수 이후 FGI > sell_threshold를 본 적 있는지
    
    # 시뮬레이션을 위해 최소 2일의 데이터 필요
    if len(df) < 2:
        st.warning("Data is too short. At least 2 days of data are required.")
        return {
            "Strategy": f"Buy<{buy_threshold}Then>{buy_threshold}_Sell>{sell_threshold}Then<{sell_threshold}",
            "ROI": 0,
            "CAGR": 0,
            "MDD": 0,
            "Calmar": 0,
            "Trades": 0,
            "Final Value": initial_capital,
            "portfolio_values": [initial_capital],
            "trade_history": []
        }
    
    # 첫날 강제 매수 (0일차 결정, 1일차 실행)
    pending_action = ('Buy', 0)
    
    # 메인 시뮬레이션 루프
    for i in range(len(df)):
        date_i = df['date'].iloc[i]
        fgi_i = df['fgi'].iloc[i]
        price_i = df['price'].iloc[i]
        
        # 다음날 존재 확인
        has_next_day = (i < len(df) - 1)
        
        # 전날(i-1) 결정 오늘(i) 실행
        if pending_action is not None and pending_action[1] == i - 1:
            action_type = pending_action[0]
            
            if action_type == 'Buy':
                if not pd.isna(price_i) and price_i > 0:
                    # 수수료 고려하여 매수금액 계산
                    buy_amount = cash / (1 + commission_rate)
                    units = buy_amount / price_i
                    cash = 0
                    current_state = 'STOCK'
                    trade_count += 1
                    
                    # 시각화를 위한 거래 기록
                    trade_history.append({
                        'date': date_i,
                        'action': 'Buy',
                        'price': price_i,
                        'fgi': fgi_i
                    })
                    
                    # 매수 후 FGI 플래그 초기화
                    fgi_low_seen_since_sell = False
                    fgi_high_seen_since_buy = False
            
            elif action_type == 'Sell':
                if not pd.isna(price_i) and price_i > 0:
                    # 수수료 고려하여 매도금액 계산
                    sell_amount = units * price_i
                    cash = sell_amount * (1 - commission_rate)
                    units = 0
                    current_state = 'CASH'
                    trade_count += 1
                    
                    # 시각화를 위한 거래 기록
                    trade_history.append({
                        'date': date_i,
                        'action': 'Sell',
                        'price': price_i,
                        'fgi': fgi_i
                    })
                    
                    # 매도 후 FGI 플래그 초기화
                    fgi_low_seen_since_sell = False
                    fgi_high_seen_since_buy = False
            
            # 실행 후 pending_action 초기화
            pending_action = None
        
        # 오늘(i) FGI 기반 내일(i+1) 결정
        if has_next_day and pending_action is None and not pd.isna(fgi_i):
            if current_state == 'CASH':
                # 매수 전략 로직
                if fgi_i < buy_threshold:
                    fgi_low_seen_since_sell = True
                
                if fgi_low_seen_since_sell and fgi_i > buy_threshold:
                    pending_action = ('Buy', i)
            
            elif current_state == 'STOCK':
                # 매도 전략 로직
                if fgi_i > sell_threshold:
                    fgi_high_seen_since_buy = True
                
                if fgi_high_seen_since_buy and fgi_i < sell_threshold:
                    pending_action = ('Sell', i)
        
        # 포트폴리오 가치 계산 및 기록
        if current_state == 'CASH':
            portfolio_value = cash
        else:  # 'STOCK'
            if pd.isna(price_i) or price_i <= 0:
                portfolio_value = np.nan
            else:
                portfolio_value = units * price_i
        
        portfolio.append(portfolio_value)
        dates.append(date_i)
    
    # 결과 계산
    series = pd.Series(portfolio, index=dates)
    valid_series = series.dropna()
    
    if len(valid_series) <= 1:
        return {
            "Strategy": f"Buy<{buy_threshold}Then>{buy_threshold}_Sell>{sell_threshold}Then<{sell_threshold}",
            "ROI": 0,
            "CAGR": 0,
            "MDD": 0,
            "Calmar": 0,
            "Trades": trade_count,
            "Final Value": initial_capital,
            "portfolio_values": portfolio,
            "dates": dates,
            "trade_history": trade_history
        }
    
    # 최종 포트폴리오 가치
    final_portfolio_value = valid_series.iloc[-1]
    
    # ROI 계산
    roi = (final_portfolio_value / initial_capital) - 1
    
    # 기간(년) 계산
    start_date = valid_series.index[0]
    end_date = valid_series.index[-1]
    n_years = (end_date - start_date).days / 365.25
    if n_years <= 0:
        n_years = 1/365  # 매우 짧은 기간 예외 처리
    
    # CAGR 계산
    cagr = (final_portfolio_value / initial_capital) ** (1 / n_years) - 1
    
    # MDD 계산
    mdd = ((valid_series - valid_series.cummax()) / valid_series.cummax()).min()
    
    # Calmar 비율 계산
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    
    return {
        "Strategy": f"Buy<{buy_threshold}Then>{buy_threshold}_Sell>{sell_threshold}Then<{sell_threshold}",
        "ROI": roi,
        "CAGR": cagr,
        "MDD": mdd,
        "Calmar": calmar,
        "Trades": trade_count,
        "Final Value": final_portfolio_value,
        "portfolio_values": portfolio,
        "dates": dates,
        "trade_history": trade_history
    }
```

# 파트 2: 유전 알고리즘 관련 함수

```python
# 다중 목표 적합도 계산 함수
def calculate_multi_fitness(individual, df, initial_capital, commission_rate, weights=None):
    """
    개체의 다중 목표 적합도 계산
    
    Parameters:
    -----------
    individual : tuple
        (buy_threshold, sell_threshold) 튜플
    df : pandas DataFrame
        날짜, 가격, FGI 열이 있는 DataFrame
    initial_capital : float
        초기 자본금
    commission_rate : float
        거래 수수료율 (예: 0.25%는 0.0025)
    weights : dict, optional
        다양한 지표의 가중치 딕셔너리. 기본 가중치:
        {
            'roi': 0.5,       # 투자수익률
            'mdd': 0.2,       # 최대 낙폭 (부정적)
            'calmar': 0.15,   # 칼마 비율
            'trades': 0.05,   # 거래 빈도 최적화
            'volatility': 0.1 # 수익률 변동성
        }
    """
    buy_threshold, sell_threshold = individual
    
    # 가중치가 제공되지 않은 경우 기본값 사용
    if weights is None:
        weights = {
            'roi': 0.5,       # 투자수익률 (높을수록 좋음)
            'mdd': 0.2,       # 최대 낙폭 (낮을수록 좋음)
            'calmar': 0.15,   # 칼마 비율 (높을수록 좋음)
            'trades': 0.05,   # 거래 빈도 최적화 (최적 범위가 가장 좋음)
            'volatility': 0.1 # 수익률 변동성 (낮을수록 좋음)
        }
    
    # 가중치 합이 1.0이 되도록 보장
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:  # 작은 오차 허용
        # 가중치 정규화
        for key in weights:
            weights[key] = weights[key] / weight_sum
    
    # 백테스팅 실행
    result = backtest_fgi_strategy(df, buy_threshold, sell_threshold, initial_capital, commission_rate)
    
    # 백테스팅 결과에서 기본 지표 추출
    roi = result['ROI']
    mdd = result['MDD']
    calmar = result['Calmar']
    trades = result['Trades']
    
    # 원래 백테스팅 결과에 없을 수 있는 추가 지표 계산
    
    # 1. 수익률 변동성 계산
    if 'portfolio_values' in result:
        portfolio_values = result['portfolio_values']
        # 수익률로 변환
        returns = pd.Series(portfolio_values).pct_change().dropna()
        volatility = returns.std() if len(returns) > 1 else 0.0
    else:
        # 포트폴리오 값을 사용할 수 없는 경우 MDD에서 변동성 추정
        volatility = abs(mdd) / 2.0  # 대략적인 추정
    
    # 2. 거래 지표 계산 - 너무 적거나 너무 많은 거래에 페널티 부여
    if trades < 3:  # 거래가 너무 적음
        trades_score = 0.3
    elif trades < 8:  # 최적보다 적음
        trades_score = 0.7
    elif trades <= 20:  # 최적 범위
        trades_score = 1.0
    elif trades <= 50:  # 최적보다 많지만 여전히 합리적
        trades_score = 0.8
    else:  # 거래가 너무 많음
        trades_score = 0.5
    
    # 개별 적합도 구성 요소 계산 (필요한 경우 정규화)
    # 각 구성 요소는 값이 클수록 좋게 설계
    roi_fitness = roi
    mdd_fitness = -abs(mdd)  # 부정적 값 (낙폭을 최소화하려고 함)
    calmar_fitness = min(calmar, 10.0) / 10.0  # 정규화 및 극도로 높은 값 제한
    trades_fitness = trades_score
    volatility_fitness = -min(volatility, 0.5) / 0.5  # 부정적 & 정규화 (높을수록 좋음)
    
    # 모든 개별 적합도 지표 저장
    fitness_metrics = {
        'roi': roi_fitness,
        'mdd': mdd_fitness,
        'calmar': calmar_fitness,
        'trades': trades_fitness,
        'volatility': volatility_fitness,
    }
    
    # 가중치를 사용하여 복합 적합도 계산
    composite_fitness = 0.0
    for metric, value in fitness_metrics.items():
        if metric in weights:
            composite_fitness += weights[metric] * value
    
    # 종합적인 적합도 결과 생성 및 반환
    fitness_result = {
        'individual': individual,
        'metrics': {
            'roi': roi,
            'mdd': mdd,
            'calmar': calmar,
            'trades': trades,
            'volatility': volatility if 'portfolio_values' in result else 'estimated',
        },
        'fitness_components': fitness_metrics,
        'weights': weights,
        'composite': composite_fitness
    }
    
    return fitness_result

# 다양한 지표에 기반한 엘리트 개체 선택 함수
def select_diverse_elite(population, fitness_results, num_elite=5):
    """다양한 지표에서 뛰어난 엘리트 개체 선택"""
    if not fitness_results or len(fitness_results) < num_elite:
        return population[:min(num_elite, len(population))]
    
    # 개체를 적합도 결과에 매핑하는 딕셔너리 생성
    fitness_dict = {tuple(result['individual']): result for result in fitness_results}
    
    elite = []
    
    # 복합 점수 기준 최고 개체 선택
    best_composite = max(fitness_results, key=lambda x: x['composite'])
    elite.append(tuple(best_composite['individual']))
    
    # 더 많은 엘리트 개체가 필요한 경우, 다양한 지표별 최고 개체 선택
    if num_elite > 1:
        metrics = ['roi', 'mdd', 'calmar', 'trades', 'volatility']
        
        for metric in metrics:
            if len(elite) >= num_elite:
                break
                
            # MDD와 변동성은 최소값을 원함 (그러나 적합도는 음수)
            if metric in ['mdd', 'volatility']:
                best = max(fitness_results, key=lambda x: x['fitness_components'][metric])
            else:
                best = max(fitness_results, key=lambda x: x['fitness_components'][metric])
                
            best_individual = tuple(best['individual'])
            if best_individual not in elite:
                elite.append(best_individual)
                
    # 여전히 더 필요한 경우, 무작위 다양한 개체 선택
    remaining_population = [ind for ind in population if tuple(ind) not in elite]
    while len(elite) < num_elite and remaining_population:
        # 무작위 개체 선택
        random_idx = random.randint(0, len(remaining_population) - 1)
        elite.append(tuple(remaining_population.pop(random_idx)))
    
    return elite

# 다중 목표 최적화를 위한 선택 함수
def selection_multi_objective(population, fitness_results, num_parents):
    """다중 목표 최적화를 위한 방법 조합으로 부모 선택"""
    if not fitness_results:
        return random.sample(population, min(num_parents, len(population)))
    
    # 개체를 적합도 결과에 매핑
    fitness_dict = {tuple(result['individual']): result for result in fitness_results}
    
    # 1. 먼저 엘리트 개체 일부 선택 (부모의 20%)
    num_elite = max(1, int(num_parents * 0.2))
    elite = select_diverse_elite(population, fitness_results, num_elite)
    parents = list(elite)
    
    # 2. 나머지는 토너먼트 선택으로 다중 목표 사용
    remaining_slots = num_parents - len(parents)
    if remaining_slots > 0:
        for _ in range(remaining_slots):
            # 토너먼트 크기 (인구의 약 15%)
            tournament_size = max(3, int(len(population) * 0.15))
            
            # 무작위 토너먼트 선택
            tournament = random.sample(population, tournament_size)
            
            # 토너먼트 점수 계산
            tournament_scores = []
            for individual in tournament:
                if tuple(individual) in fitness_dict:
                    # 토너먼트에 복합 적합도 점수 사용
                    score = fitness_dict[tuple(individual)]['composite']
                    tournament_scores.append((individual, score))
                else:
                    # 적합도를 찾을 수 없는 경우 낮은 점수 할당
                    tournament_scores.append((individual, -1.0))
            
            # 승자 선택 (최고 점수)
            winner, _ = max(tournament_scores, key=lambda x: x[1])
            parents.append(tuple(winner))
    
    return parents

# 초기 모집단 생성 함수
def create_initial_population(population_size, buy_threshold_range, sell_threshold_range):
    """전략의 초기 모집단 생성"""
    population = []
    
    for _ in range(population_size):
        buy_threshold = random.randint(buy_threshold_range[0], buy_threshold_range[1])
        sell_threshold = random.randint(sell_threshold_range[0], sell_threshold_range[1])
        population.append((buy_threshold, sell_threshold))
    
    return population

# 교차 함수
def crossover(parents, offspring_size):
    """균등 교차를 사용하여 부모로부터 자손 생성"""
    offspring = []
    
    while len(offspring) < offspring_size:
        # 무작위로 두 부모 선택
        parent1, parent2 = random.sample(parents, 2)
        
        # 파라미터별 교차
        buy_threshold = parent1[0] if random.random() < 0.5 else parent2[0]
        sell_threshold = parent1[1] if random.random() < 0.5 else parent2[1]
        
        offspring.append((buy_threshold, sell_threshold))
    
    return offspring

# 변이 함수
def mutation(offspring, mutation_rate, buy_threshold_range, sell_threshold_range):
    """자손에 변이 적용"""
    mutated_offspring = []
    
    for individual in offspring:
        buy_threshold, sell_threshold = individual
        
        # 매수 임계값 변이
        if random.random() < mutation_rate:
            # 현재 값에서 -3에서 +3 사이 변화
            change = random.randint(-3, 3)
            buy_threshold = max(buy_threshold_range[0], min(buy_threshold + change, buy_threshold_range[1]))
        
        # 매도 임계값 변이
        if random.random() < mutation_rate:
            change = random.randint(-3, 3)
            sell_threshold = max(sell_threshold_range[0], min(sell_threshold + change, sell_threshold_range[1]))
        
        mutated_offspring.append((buy_threshold, sell_threshold))
    
    return mutated_offspring

# 최고 전략 분석 함수
def analyze_best_strategies(fitness_results, n=5):
    """다양한 지표 기준 상위 N개 전략 분석"""
    if not fitness_results or len(fitness_results) == 0:
        return {}
    
    # 원본 데이터 수정 방지를 위한 복사본 생성
    results_copy = fitness_results.copy()
    
    # 다양한 지표별 최고 전략 찾기
    best_by = {
        'composite': sorted(results_copy, key=lambda x: x['composite'], reverse=True)[:n],
        'roi': sorted(results_copy, key=lambda x: x['metrics']['roi'], reverse=True)[:n],
        'mdd': sorted(results_copy, key=lambda x: x['metrics']['mdd'])[:n],  # MDD는 낮을수록 좋음
        'calmar': sorted(results_copy, key=lambda x: x['metrics']['calmar'], reverse=True)[:n],
        'trades': sorted(results_copy, key=lambda x: 
                         abs(x['metrics']['trades'] - 15))[:n]  # 최적 거래 횟수에 가까울수록 좋음
    }
    
    return best_by
```

# 파트 3: 시각화, 유전 알고리즘 실행 및 메인 함수

```python
# Streamlit을 사용한 진행 상황 시각화
def plot_ga_progress(best_fitness_history, avg_fitness_history, buy_thresh_history, sell_thresh_history, generation_count):
    """Streamlit을 사용하여 유전 알고리즘 진행 상황 시각화"""
    st.subheader("Genetic Algorithm Progress")
    
    col1, col2 = st.columns(2)
    
    # 1. 최고 및 평균 적합도 차트
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(range(1, generation_count + 1), best_fitness_history, 'b-', linewidth=2, label='Best Fitness')
    ax1.plot(range(1, generation_count + 1), avg_fitness_history, 'r-', linewidth=2, label='Average Fitness')
    ax1.set_title('Fitness Trend')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.grid(True)
    ax1.legend()
    col1.pyplot(fig1)
    
    # 2. 임계값 추세
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(range(1, generation_count + 1), buy_thresh_history, 'g-', linewidth=2, label='Buy Threshold')
    ax2.plot(range(1, generation_count + 1), sell_thresh_history, 'm-', linewidth=2, label='Sell Threshold')
    ax2.set_title('Threshold Trends')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Threshold Values')
    ax2.grid(True)
    ax2.legend()
    col2.pyplot(fig2)

# Streamlit을 사용한 전략 결과 시각화
def plot_strategy_results(df, buy_threshold, sell_threshold, initial_capital, result):
    """Streamlit을 사용하여 전략 백테스팅 결과 시각화"""
    st.subheader(f"Strategy Results: Buy<{buy_threshold}Then>{buy_threshold}_Sell>{sell_threshold}Then<{sell_threshold}")
    
    # 차트용 두 열 생성
    col1, col2 = st.columns(2)
    
    # 1. 거래 표시가 있는 FGI 및 가격 차트
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    # 왼쪽 y축에 가격
    ax1.plot(df['date'], df['price'], 'b-', linewidth=1.5, label='Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 오른쪽 y축에 FGI
    ax2 = ax1.twinx()
    ax2.plot(df['date'], df['fgi'], 'r-', linewidth=1, label='FGI', alpha=0.7)
    ax2.axhline(y=buy_threshold, color='g', linestyle='--', label=f'Buy Threshold ({buy_threshold})')
    ax2.axhline(y=sell_threshold, color='m', linestyle='--', label=f'Sell Threshold ({sell_threshold})')
    ax2.set_ylabel('FGI', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 차트에 거래 표시
    trade_history = result.get('trade_history', [])
    buy_dates = [trade['date'] for trade in trade_history if trade['action'] == 'Buy']
    buy_prices = [trade['price'] for trade in trade_history if trade['action'] == 'Buy']
    
    sell_dates = [trade['date'] for trade in trade_history if trade['action'] == 'Sell']
    sell_prices = [trade['price'] for trade in trade_history if trade['action'] == 'Sell']
    
    ax1.scatter(buy_dates, buy_prices, color='g', marker='^', s=100, label='Buy', zorder=5)
    ax1.scatter(sell_dates, sell_prices, color='r', marker='v', s=100, label='Sell', zorder=5)
    
    # 범례 통합
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'Price, FGI, and Trades')
    plt.grid(True, alpha=0.3)
    col1.pyplot(fig1)
    
    # 2. 시간 경과에 따른 포트폴리오 가치
    if 'portfolio_values' in result and 'dates' in result:
        fig2, ax3 = plt.subplots(figsize=(12, 7))
        portfolio_values = result['portfolio_values']
        dates = result['dates']
        
        # 비교를 위한 매수 후 보유 계산
        buy_and_hold_values = (df['price'] / df['price'].iloc[0]) * initial_capital
        
        # 포트폴리오 가치 플롯
        ax3.plot(dates, portfolio_values, 'b-', linewidth=2, label='Strategy Portfolio')
        ax3.plot(df['date'], buy_and_hold_values, 'g--', linewidth=1.5, label='Buy & Hold')
        
        # 포트폴리오 차트에도 거래 표시
        for trade in trade_history:
            if trade['action'] == 'Buy':
                ax3.axvline(x=trade['date'], color='g', linestyle=':', alpha=0.5)
            else:  # 'Sell'
                ax3.axvline(x=trade['date'], color='r', linestyle=':', alpha=0.5)
        
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Portfolio Value')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Portfolio Value')
        ax3.set_title('Portfolio Value Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        col2.pyplot(fig2)
    
    # 성과 지표
    metrics_cols = st.columns(5)
    
    metrics_cols[0].metric("ROI", f"{result['ROI']:.2%}")
    metrics_cols[1].metric("CAGR", f"{result['CAGR']:.2%}")
    metrics_cols[2].metric("MDD", f"{result['MDD']:.2%}")
    metrics_cols[3].metric("Calmar Ratio", f"{result['Calmar']:.2f}")
    metrics_cols[4].metric("Trades", f"{result['Trades']}")
    
    # 매수 후 보유 비교
    bh_roi = (buy_and_hold_values.iloc[-1] / initial_capital) - 1 if len(buy_and_hold_values) > 0 else 0
    st.metric("Strategy vs Buy & Hold", f"{result['ROI'] - bh_roi:.2%}", 
             f"{result['ROI']:.2%} vs {bh_roi:.2%}")

@st.cache_data
def run_genetic_algorithm(df, params):
    """성능을 위한 캐싱을 사용하여 유전 알고리즘 실행"""
    # 파라미터 추출
    initial_capital = params['initial_capital']
    commission_rate = params['commission_rate']
    population_size = params['population_size']
    num_generations = params['num_generations']
    mutation_rate = params['mutation_rate']
    num_parents = params['num_parents']
    buy_threshold_range = params['buy_threshold_range']
    sell_threshold_range = params['sell_threshold_range']
    fitness_weights = params['fitness_weights']
    
    # 추적을 위한 설정
    best_fitness_history = []
    avg_fitness_history = []
    best_individual_history = []
    buy_thresh_history = []
    sell_thresh_history = []
    all_results = []
    all_fitness_results = []
    
    # 상태 표시 플레이스홀더 생성
    status = st.empty()
    progress_bar = st.progress(0)
    
    # 초기 모집단 생성
    population = create_initial_population(population_size, buy_threshold_range, sell_threshold_range)
    
    # 유전 알고리즘 세대 루프
    start_time = time.time()
    for generation in range(1, num_generations + 1):
        generation_start = time.time()
        
        # 상태 업데이트
        status.text(f"Running generation {generation}/{num_generations}...")
        progress_bar.progress(generation / num_generations)
        
        # 모집단의 적합도 계산
        fitness_results = [
            calculate_multi_fitness(individual, df, initial_capital, commission_rate, fitness_weights)
            for individual in population
        ]
        
        # 적합도 값 추출
        fitnesses = [result['composite'] for result in fitness_results]
        avg_fitness = np.mean(fitnesses)
        
        # 최고 개체 찾기
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_individual = population[best_idx]
        best_fitness_result = fitness_results[best_idx]
        
        # 추적을 위한 저장
        all_fitness_results.append(best_fitness_result)
        
        # 이 세대의 최고 개체 백테스팅
        buy_threshold, sell_threshold = best_individual
        best_result = backtest_fgi_strategy(df, buy_threshold, sell_threshold, initial_capital, commission_rate)
        all_results.append(best_result)
        
        # 진행 상황 기록
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        best_individual_history.append(best_individual)
        buy_thresh_history.append(buy_threshold)
        sell_thresh_history.append(sell_threshold)
        
        # 선택
        parents = selection_multi_objective(population, fitness_results, num_parents)
        
        # 교차
        offspring_size = population_size - len(parents)
        offspring = crossover(parents, offspring_size)
        
        # 변이
        offspring = mutation(offspring, mutation_rate, buy_threshold_range, sell_threshold_range)
        
        # 다음 세대
        population = parents + offspring
    
    # 최고 결과 찾기
    best_generation = np.argmax(best_fitness_history)
    ultimate_best_individual = best_individual_history[best_generation]
    ultimate_best_fitness = best_fitness_history[best_generation]
    
    # 최종 결과
    total_time = time.time() - start_time
    
    # 결과 반환
    return {
        'best_fitness_history': best_fitness_history,
        'avg_fitness_history': avg_fitness_history,
        'buy_thresh_history': buy_thresh_history,
        'sell_thresh_history': sell_thresh_history,
        'all_results': all_results,
        'all_fitness_results': all_fitness_results,
        'best_individual': ultimate_best_individual,
        'best_fitness': ultimate_best_fitness,
        'best_generation': best_generation,
        'total_time': total_time,
        'num_generations': num_generations
    }

# Streamlit 앱을 위한 메인 함수
def main():
    st.set_page_config(layout="wide", page_title="FGI Strategy Optimization")
    
    st.title("FGI Strategy Optimization with Genetic Algorithm")
    st.write("""
    This app helps you find optimal Fear & Greed Index (FGI) trading strategies using genetic algorithms.
    Upload your data file containing date, price, and FGI values to get started.
    """)
    
    # 파일 업로드 및 설정을 위한 사이드바
    with st.sidebar:
        st.header("Data Input")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            # 파일 읽기
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # 샘플 데이터 표시
            st.write("Data Preview:")
            st.dataframe(df.head(3))
            
            # 열 선택
            st.header("Column Selection")
            date_col = st.selectbox("Date Column", options=df.columns)
            price_col = st.selectbox("Price Column", options=df.columns)
            fgi_col = st.selectbox("FGI Column", options=df.columns)
            
            # 설정
            st.header("Configuration")
            initial_capital = st.number_input("Initial Capital", value=10000000, step=1000000)
            commission_rate = st.number_input("Commission Rate", value=0.0025, format="%.4f", step=0.0001)
            
            # 유전 알고리즘 파라미터
            st.header("Genetic Algorithm Parameters")
            population_size = st.slider("Population Size", min_value=10, max_value=100, value=50)
            num_generations = st.slider("Number of Generations", min_value=10, max_value=200, value=80)
            mutation_rate = st.slider("Mutation Rate", min_value=0.01, max_value=0.5, value=0.2, step=0.01)
            num_parents = st.slider("Number of Parents", min_value=5, max_value=int(population_size/2), value=int(population_size*0.25))
            
            buy_threshold_min = st.slider("Min Buy Threshold", min_value=0, max_value=50, value=5)
            buy_threshold_max = st.slider("Max Buy Threshold", min_value=buy_threshold_min, max_value=100, value=25)
            sell_threshold_min = st.slider("Min Sell Threshold", min_value=0, max_value=50, value=5)
            sell_threshold_max = st.slider("Max Sell Threshold", min_value=sell_threshold_min, max_value=100, value=25)
            
            # 적합도 가중치
            st.header("Fitness Weights")
            st.write("Set the importance of different metrics (total should be close to 1.0)")
            col1, col2 = st.columns(2)
            with col1:
                roi_weight = st.slider("ROI Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
                mdd_weight = st.slider("MDD Weight", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
                calmar_weight = st.slider("Calmar Weight", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
            with col2:
                trades_weight = st.slider("Trades Weight", min_value=0.0, max_value=1.0, value=0.05, step=0.05)
                volatility_weight = st.slider("Volatility Weight", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
            
            # 가중치 정규화
            total_weight = roi_weight + mdd_weight + calmar_weight + trades_weight + volatility_weight
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"Total weight is {total_weight:.2f}, normalizing to 1.0")
            
            # 실행 버튼
            run_button = st.button("Run Optimization")
        else:
            st.info("Please upload a file to continue")
            run_button = False
    
    # 메인 컨텐츠 영역
    if 'uploaded_file' in locals() and uploaded_file is not None:
        # 데이터 준비
        try:
            # 데이터 처리
            processed_df = df.copy()
            processed_df = processed_df[[date_col, price_col, fgi_col]].copy()
            processed_df.columns = ['date', 'price', 'fgi']
            processed_df['date'] = pd.to_datetime(processed_df['date'])
            processed_df = processed_df.sort_values('date').reset_index(drop=True)
            processed_df[['price', 'fgi']] = processed_df[['price', 'fgi']].interpolate()
            
            # 데이터 요약 표시
            st.header("Data Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Period", f"{processed_df['date'].min().strftime('%Y-%m-%d')} to {processed_df['date'].max().strftime('%Y-%m-%d')}")
            col2.metric("Records", f"{len(processed_df)}")
            col3.metric("Price Range", f"{processed_df['price'].min():,.2f} to {processed_df['price'].max():,.2f}")
            col4.metric("FGI Range", f"{processed_df['fgi'].min():.2f} to {processed_df['fgi'].max():.2f}")
            
            # 버튼 클릭 시 최적화 실행
            if run_button:
                st.header("Running Optimization...")
                fitness_weights = {
                    'roi': roi_weight / total_weight,
                    'mdd': mdd_weight / total_weight,
                    'calmar': calmar_weight / total_weight,
                    'trades': trades_weight / total_weight,
                    'volatility': volatility_weight / total_weight
                }
                
                params = {
                    'initial_capital': initial_capital,
                    'commission_rate': commission_rate,
                    'population_size': population_size,
                    'num_generations': num_generations,
                    'mutation_rate': mutation_rate,
                    'num_parents': num_parents,
                    'buy_threshold_range': (buy_threshold_min, buy_threshold_max),
                    'sell_threshold_range': (sell_threshold_min, sell_threshold_max),
                    'fitness_weights': fitness_weights
                }
                
                # 유전 알고리즘 실행
                with st.spinner('Running genetic algorithm optimization...'):
                    results = run_genetic_algorithm(processed_df, params)
                
                # 결과 표시
                st.header("Optimization Results")
                
                # 최고 개체 및 결과 가져오기
                best_individual = results['best_individual']
                best_generation = results['best_generation']
                best_fitness = results['best_fitness']
                
                # 최고 개체에 대한 최종 백테스팅 실행
                best_buy_threshold, best_sell_threshold = best_individual
                best_result = backtest_fgi_strategy(processed_df, best_buy_threshold, best_sell_threshold, initial_capital, commission_rate)
                
                # 지표 및 시각화 표시
                st.subheader(f"Best Strategy Found (Generation {best_generation+1}/{num_generations})")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Buy Threshold", f"{best_buy_threshold}")
                col2.metric("Sell Threshold", f"{best_sell_threshold}")
                col3.metric("Fitness Score", f"{best_fitness:.4f}")
                col4.metric("Total Time", f"{results['total_time']:.1f} seconds")
                
                # 진행 상황 그리기
                plot_ga_progress(
                    results['best_fitness_history'],
                    results['avg_fitness_history'],
                    results['buy_thresh_history'],
                    results['sell_thresh_history'],
                    num_generations
                )
                
                # 전략 결과 그리기
                plot_strategy_results(processed_df, best_buy_threshold, best_sell_threshold, initial_capital, best_result)
                
                # 최고 전략 분석 표시
                best_strategies = analyze_best_strategies(results['all_fitness_results'], 5)
                
                # 다양한 결과 보기를 위한 탭
                tab1, tab2, tab3 = st.tabs(["Best Strategies", "Performance Comparison", "Trade Analysis"])
                
                with tab1:
                    st.subheader("Top 3 Strategies by Different Metrics")
                    
                    # 복합 점수 기준 최고
                    st.write("**Best by Composite Score:**")
                    for i, result in enumerate(best_strategies.get('composite', [])[:3]):
                        individual = result['individual']
                        metrics = result['metrics']
                        st.write(f"{i+1}. Buy: {individual[0]}, Sell: {individual[1]} - " +
                               f"ROI: {metrics['roi']:.2%}, MDD: {metrics['mdd']:.2%}, Trades: {metrics['trades']}")
                    
                    # ROI 기준 최고
                    st.write("**Best by ROI:**")
                    for i, result in enumerate(best_strategies.get('roi', [])[:3]):
                        individual = result['individual']
                        metrics = result['metrics']
                        st.write(f"{i+1}. Buy: {individual[0]}, Sell: {individual[1]} - " +
                               f"ROI: {metrics['roi']:.2%}, MDD: {metrics['mdd']:.2%}, Trades: {metrics['trades']}")
                    
                    # MDD 기준 최고
                    st.write("**Best by MDD (lowest drawdown):**")
                    for i, result in enumerate(best_strategies.get('mdd', [])[:3]):
                        individual = result['individual']
                        metrics = result['metrics']
                        st.write(f"{i+1}. Buy: {individual[0]}, Sell: {individual[1]} - " +
                               f"ROI: {metrics['roi']:.2%}, MDD: {metrics['mdd']:.2%}, Trades: {metrics['trades']}")
                
                with tab2:
                    st.subheader("Performance Comparison")
                    # 매수 후 보유 비교 생성
                    buy_hold_series = (processed_df['price'] / processed_df['price'].iloc[0]) * initial_capital
                    buy_hold_roi = (buy_hold_series.iloc[-1] / initial_capital) - 1
                    
                    # 성과 지표 계산
                    metrics_df = pd.DataFrame({
                        'Metric': ['ROI', 'CAGR', 'MDD', 'Calmar', 'Trades'],
                        'Optimal Strategy': [
                            f"{best_result['ROI']:.2%}", 
                            f"{best_result['CAGR']:.2%}", 
                            f"{best_result['MDD']:.2%}", 
                            f"{best_result['Calmar']:.2f}", 
                            f"{best_result['Trades']}"
                        ],
                        'Buy & Hold': [
                            f"{buy_hold_roi:.2%}", 
                            f"N/A", 
                            f"N/A", 
                            f"N/A", 
                            f"1"
                        ]
                    })
                    
                    st.table(metrics_df)
                
                with tab3:
                    st.subheader("Trade Analysis")
                    if 'trade_history' in best_result and best_result['trade_history']:
                        trades_df = pd.DataFrame(best_result['trade_history'])
                        st.dataframe(trades_df)
                    else:
                        st.write("No trade data available")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()
#
#
#이제 세 부분으로 나뉜 전체 코드가 모두 제공되었습니다. 이 코드를 다음과 같이 사용하세요:
#
#1. `app.py` 파일을 만들고 위 세 부분의 코드를 순서대로 모두 붙여넣습니다.
#2. GitHub 저장소에 이 파일을 업로드하고, 이전에 제공해 드린 `requirements.txt`와 `README.md` 파일도 함께 업로드합니다.
#3. Streamlit Cloud에서 이 저장소를 배포합니다.
#
#이 코드는 Streamlit의 파일 업로더, 사이드바 설정, 메인 컨텐츠 영역, 데이터 처리, 유전 알고리즘 실행 및 결과 시각화의 전체 흐름을 완전히 구현합니다. 사용자는 FGI 데이터를 업로드하고 파라미터를 조정한 다음 최적의 전략을 찾을 수 있습니다.