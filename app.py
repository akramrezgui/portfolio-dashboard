import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.title("Portfolio Settings")
assets = st.sidebar.text_input("Asset tickers (comma-separated)", "AAPL,MSFT,TSLA,NVDA").split(",")
benchmark = st.sidebar.text_input("Benchmark ticker", "SPY")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
rf = st.sidebar.number_input("Risk-free rate", value=0.02)

# -------------------------------
# Data Download
# -------------------------------
data = yf.download(assets + [benchmark], start=start_date)

if "Adj Close" in data.columns:
    prices = data["Adj Close"][assets]
    benchmark_prices = data["Adj Close"][benchmark]
else:
    prices = data["Close"][assets]
    benchmark_prices = data["Close"][benchmark]

returns = prices.pct_change().dropna()
benchmark_returns = benchmark_prices.pct_change().dropna()

# -------------------------------
# Step 3: Monte Carlo Efficient Frontier
# -------------------------------
mean_returns = returns.mean().to_numpy() * 252
cov_matrix = returns.cov().to_numpy() * 252

num_portfolios = 5000
results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(len(assets)))
    port_return = np.dot(weights, mean_returns)
    port_std = np.sqrt(weights @ cov_matrix @ weights)
    sharpe = (port_return - rf) / port_std
    results[0,i] = port_return
    results[1,i] = port_std
    results[2,i] = sharpe
    weights_record.append(weights)

max_sharpe_idx = results[2].argmax()
best_return, best_std, best_sharpe = results[:, max_sharpe_idx]
best_weights = weights_record[max_sharpe_idx]

# Efficient Frontier Plot
fig_frontier = px.scatter(
    x=results[1,:], y=results[0,:], color=results[2,:],
    labels={'x':'Risk (Std Dev)','y':'Return','color':'Sharpe Ratio'},
    title="Efficient Frontier (Monte Carlo)"
)
fig_frontier.add_scatter(x=[best_std], y=[best_return], mode='markers',
    marker=dict(color='red', size=12, symbol='star'), name='Max Sharpe')
fig_frontier.add_annotation(x=best_std, y=best_return, text="Max Sharpe",
    showarrow=True, arrowhead=2, xshift=-80, yshift=40, bgcolor="white")

st.plotly_chart(fig_frontier)

# -------------------------------
# Step 4: Capital Allocation Line
# -------------------------------
cal_x = np.linspace(0, max(results[1,:])*1.2, 100)
cal_y = rf + (best_return - rf)/best_std * cal_x
cal_slope = (best_return - rf) / best_std

fig_cal = go.Figure()
fig_cal.add_trace(go.Scatter(x=cal_x, y=cal_y, mode='lines',
    name='Capital Allocation Line', line=dict(color='green')))
fig_cal.add_trace(go.Scatter(x=[0], y=[rf], mode='markers',
    marker=dict(color='black', size=10), name='Risk-Free'))
fig_cal.add_trace(go.Scatter(x=[best_std], y=[best_return], mode='markers',
    marker=dict(color='red', size=12, symbol='star'), name='Tangency Portfolio'))
fig_cal.add_annotation(x=best_std, y=best_return, text="Tangency Portfolio",
    showarrow=True, arrowhead=2, xshift=-60, yshift=40, bgcolor="white")
fig_cal.add_annotation(x=cal_x[-1]*0.7, y=cal_y[-1]*0.7,
    text=f"Slope = Sharpe {cal_slope:.2f}", showarrow=False, bgcolor="lightyellow")

st.plotly_chart(fig_cal)

# -------------------------------
# Step 5: Portfolio vs Benchmark
# -------------------------------
portfolio_returns = (returns @ best_weights)
portfolio_cum = (1 + portfolio_returns).cumprod()
benchmark_cum = (1 + benchmark_returns).cumprod()
drawdown = portfolio_cum / portfolio_cum.cummax() - 1

fig_benchmark = go.Figure()
fig_benchmark.add_trace(go.Scatter(y=portfolio_cum, x=portfolio_cum.index,
    mode='lines', name='Optimized Portfolio', line=dict(color='red')))
fig_benchmark.add_trace(go.Scatter(y=benchmark_cum, x=benchmark_cum.index,
    mode='lines', name=benchmark, line=dict(color='blue')))
fig_benchmark.update_layout(title="Portfolio vs Benchmark (Cumulative Returns)")
st.plotly_chart(fig_benchmark)

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(y=drawdown, x=drawdown.index, mode='lines',
    name='Drawdown', line=dict(color='purple')))
fig_dd.update_layout(title="Portfolio Drawdown")
st.plotly_chart(fig_dd)

# -------------------------------
# Step 6: Rolling Metrics
# -------------------------------
window = 90
rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
rolling_ret = portfolio_returns.rolling(window).mean() * 252
rolling_sharpe = (rolling_ret - rf) / rolling_vol

fig_roll = go.Figure()
fig_roll.add_trace(go.Scatter(y=rolling_vol, x=rolling_vol.index, mode='lines',
    name='Rolling Volatility', line=dict(color='orange')))
fig_roll.add_trace(go.Scatter(y=rolling_ret, x=rolling_ret.index, mode='lines',
    name='Rolling Return', line=dict(color='green')))
fig_roll.add_trace(go.Scatter(y=rolling_sharpe, x=rolling_sharpe.index, mode='lines',
    name='Rolling Sharpe', line=dict(color='blue')))
fig_roll.update_layout(title="Rolling Metrics (90-Day Window)")
st.plotly_chart(fig_roll)

# -------------------------------
# Step 7: Risk Metrics Comparison
# -------------------------------
port_mean = portfolio_returns.mean() * 252
port_vol = portfolio_returns.std() * np.sqrt(252)
port_sharpe = (port_mean - rf) / port_vol
port_max_dd = drawdown.min()

bench_mean = benchmark_returns.mean() * 252
bench_vol = benchmark_returns.std() * np.sqrt(252)
bench_sharpe = (bench_mean - rf) / bench_vol
bench_max_dd = (benchmark_cum / benchmark_cum.cummax() - 1).min()

metrics_df = pd.DataFrame({
    "Metric": ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Max Drawdown"],
    "Optimized Portfolio": [f"{port_mean:.2%}", f"{port_vol:.2%}", f"{port_sharpe:.2f}", f"{port_max_dd:.2%}"],
    "Benchmark (SPY)": [f"{bench_mean:.2%}", f"{bench_vol:.2%}", f"{bench_sharpe:.2f}", f"{bench_max_dd:.2%}"]
})
st.subheader("📊 Risk Metrics Comparison")
st.dataframe(metrics_df)

# -------------------------------
# Portfolio Weights
# -------------------------------
weights_df = pd.DataFrame({'Asset': assets, 'Weight': best_weights})
weights_df['Weight'] = (weights_df['Weight'] * 100).round(2).astype(str) + '%'
st.subheader("🔑 Max Sharpe Portfolio Weights")
st.dataframe(weights_df)

fig_pie = px.pie(values=best_weights, names=assets, title="Portfolio Allocation (Max Sharpe)")
st.plotly_chart(fig_pie)
