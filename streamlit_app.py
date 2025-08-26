"""
Streamlit Dashboard for Grid Trading Strategy Analysis

Interactive web app to configure and test grid trading strategies on Ethereum data.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, date
import polars as pl
from dataclasses import dataclass
from typing import List, Optional
import os

# Page configuration
st.set_page_config(
    page_title="Grid Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive { border-left-color: #2ca02c; }
    .negative { border-left-color: #d62728; }
    .neutral { border-left-color: #ff7f0e; }
</style>
""", unsafe_allow_html=True)

# Configuration classes
@dataclass
class GridTradingConfig:
    """Configuration for grid trading strategy"""
    initial_capital: float = 5000.0
    initial_eth_holdings: float = 0.0
    price_range_low: float = 3000.0
    price_range_high: float = 10000.0
    num_grids: int = 20
    volatility_threshold: float = 0.05
    max_position_size: float = 0.10
    min_position_size: float = 0.01
    target_allocation: float = 0.5
    rebalance_threshold: float = 0.1
    transaction_fee: float = 0.001

@dataclass
class TradingState:
    """Current state of the trading portfolio"""
    cash: float
    eth_holdings: float
    total_value: float
    last_price: float
    trades_executed: int = 0
    total_fees_paid: float = 0.0

class GridTradingStrategy:
    """Grid trading strategy implementation"""
    
    def __init__(self, config: GridTradingConfig, starting_price: float = None):
        self.config = config
        
        # Calculate initial portfolio value including ETH holdings
        initial_eth_value = 0.0
        if config.initial_eth_holdings > 0 and starting_price:
            initial_eth_value = config.initial_eth_holdings * starting_price
        
        total_initial_value = config.initial_capital + initial_eth_value
        
        self.state = TradingState(
            cash=config.initial_capital,
            eth_holdings=config.initial_eth_holdings,
            total_value=total_initial_value,
            last_price=starting_price or 0.0
        )
        self.trade_history: List[dict] = []
        self.portfolio_history: List[dict] = []
    
    def calculate_position_size(self, price: float, action: str) -> float:
        """Calculate position size based on price position in range"""
        price_range = self.config.price_range_high - self.config.price_range_low
        
        if action == "buy":
            distance_from_low = (price - self.config.price_range_low) / price_range
            intensity = 1.0 - distance_from_low
            position_size = self.config.min_position_size + (
                intensity * (self.config.max_position_size - self.config.min_position_size)
            )
        else:  # sell
            distance_from_low = (price - self.config.price_range_low) / price_range
            intensity = distance_from_low
            position_size = self.config.min_position_size + (
                intensity * (self.config.max_position_size - self.config.min_position_size)
            )
        
        return min(position_size, self.config.max_position_size)
    
    def should_rebalance(self, current_price: float) -> Optional[str]:
        """Check if portfolio needs rebalancing based on target allocation"""
        if self.state.total_value <= 0:
            return None
            
        eth_value = self.state.eth_holdings * current_price
        current_eth_allocation = eth_value / self.state.total_value
        
        deviation = abs(current_eth_allocation - self.config.target_allocation)
        
        if deviation > self.config.rebalance_threshold:
            if current_eth_allocation > self.config.target_allocation:
                return "sell"
            else:
                return "buy"
        
        return None
    
    def execute_trade(self, price: float, action: str, amount: float, reason: str) -> bool:
        """Execute a trade and update state"""
        if action == "buy":
            cost = amount * price
            fee = cost * self.config.transaction_fee
            total_cost = cost + fee
            
            if total_cost <= self.state.cash:
                self.state.cash -= total_cost
                self.state.eth_holdings += amount
                self.state.trades_executed += 1
                self.state.total_fees_paid += fee
                
                self.trade_history.append({
                    "action": "buy",
                    "price": price,
                    "amount": amount,
                    "cost": cost,
                    "fee": fee,
                    "reason": reason,
                    "cash_after": self.state.cash,
                    "eth_after": self.state.eth_holdings
                })
                return True
        
        elif action == "sell":
            if amount <= self.state.eth_holdings:
                proceeds = amount * price
                fee = proceeds * self.config.transaction_fee
                net_proceeds = proceeds - fee
                
                self.state.cash += net_proceeds
                self.state.eth_holdings -= amount
                self.state.trades_executed += 1
                self.state.total_fees_paid += fee
                
                self.trade_history.append({
                    "action": "sell",
                    "price": price,
                    "amount": amount,
                    "proceeds": proceeds,
                    "fee": fee,
                    "reason": reason,
                    "cash_after": self.state.cash,
                    "eth_after": self.state.eth_holdings
                })
                return True
        
        return False
    
    def process_price_update(self, date: str, price: float) -> None:
        """Process a new price update and make trading decisions"""
        if price <= 0:
            return
            
        # Update total portfolio value
        eth_value = self.state.eth_holdings * price
        self.state.total_value = self.state.cash + eth_value
        
        # Check for volatility-based trading
        if self.state.last_price > 0:
            price_change = (price - self.state.last_price) / self.state.last_price
            
            if abs(price_change) >= self.config.volatility_threshold:
                if price_change < 0:  # Price dropped
                    position_size = self.calculate_position_size(price, "buy")
                    amount_to_buy = (self.state.cash * position_size) / price
                    if amount_to_buy > 0:
                        self.execute_trade(
                            price, "buy", amount_to_buy, 
                            f"Volatility buy: {price_change:.2%} drop"
                        )
                
                elif price_change > 0:  # Price increased
                    position_size = self.calculate_position_size(price, "sell")
                    amount_to_sell = self.state.eth_holdings * position_size
                    if amount_to_sell > 0:
                        self.execute_trade(
                            price, "sell", amount_to_sell,
                            f"Volatility sell: {price_change:.2%} gain"
                        )
        
        # Check for rebalancing
        rebalance_action = self.should_rebalance(price)
        if rebalance_action:
            eth_value = self.state.eth_holdings * price
            target_eth_value = self.state.total_value * self.config.target_allocation
            
            if rebalance_action == "buy":
                amount_to_buy = (target_eth_value - eth_value) / price
                if amount_to_buy > 0 and amount_to_buy * price <= self.state.cash:
                    self.execute_trade(
                        price, "buy", amount_to_buy, "Rebalancing: buy ETH"
                    )
            
            elif rebalance_action == "sell":
                amount_to_sell = (eth_value - target_eth_value) / price
                if amount_to_sell > 0 and amount_to_sell <= self.state.eth_holdings:
                    self.execute_trade(
                        price, "sell", amount_to_sell, "Rebalancing: sell ETH"
                    )
        
        # Record portfolio state
        self.portfolio_history.append({
            "date": date,
            "price": price,
            "cash": self.state.cash,
            "eth_holdings": self.state.eth_holdings,
            "eth_value": eth_value,
            "total_value": self.state.total_value,
            "eth_allocation": eth_value / self.state.total_value if self.state.total_value > 0 else 0
        })
        
        self.state.last_price = price

# Load ETH data
@st.cache_data
def load_eth_data():
    """Load and cache ETH price data"""
    # Try different possible paths
    possible_paths = [
        "data/eth-prices.csv",
        "./data/eth-prices.csv", 
        "/mount/src/crypto-trading/data/eth-prices.csv",
        "crypto-trading/data/eth-prices.csv"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        st.error("Could not find ETH price data file. Please ensure data/eth-prices.csv exists.")
        st.stop()
    
    try:
        df = pl.read_csv(
            data_path,
            separator="\t",
            has_header=True,
            schema_overrides={"Value": pl.Float64}
        )
        
        # Clean and process data
        df = df.filter(pl.col("Value") > 0)  # Remove zero prices
        df = df.with_columns([
            pl.col("Date(UTC)").str.strptime(pl.Date, "%m/%d/%y").alias("date"),
            pl.col("Value").alias("price")
        ])
        
        return df.select(["date", "price"]).sort("date")
        
    except Exception as e:
        st.error(f"Error loading data file: {str(e)}")
        st.stop()

# Run backtest
@st.cache_data
def run_backtest(
    initial_capital, initial_eth_holdings, price_range_low, price_range_high,
    num_grids, volatility_threshold, max_position_size, min_position_size,
    target_allocation, rebalance_threshold, transaction_fee,
    start_date, end_date
):
    """Run backtest with caching for better performance"""
    config = GridTradingConfig(
        initial_capital=initial_capital,
        initial_eth_holdings=initial_eth_holdings,
        price_range_low=price_range_low,
        price_range_high=price_range_high,
        num_grids=num_grids,
        volatility_threshold=volatility_threshold,
        max_position_size=max_position_size,
        min_position_size=min_position_size,
        target_allocation=target_allocation,
        rebalance_threshold=rebalance_threshold,
        transaction_fee=transaction_fee
    )
    
    # Load and filter data
    df = load_eth_data()
    
    if start_date:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        df = df.filter(pl.col("date") >= start_date_obj)
    if end_date:
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        df = df.filter(pl.col("date") <= end_date_obj)
    
    if len(df) == 0:
        raise ValueError("No data available for the specified date range")
    
    # Initialize strategy
    starting_price = df["price"][0]
    strategy = GridTradingStrategy(config, starting_price=starting_price)
    
    # Run simulation
    for row in df.iter_rows(named=True):
        strategy.process_price_update(str(row["date"]), row["price"])
    
    # Calculate metrics
    portfolio_df = pl.DataFrame(strategy.portfolio_history)
    initial_value = portfolio_df["total_value"][0]
    final_value = strategy.state.total_value
    
    total_return = (final_value - initial_value) / initial_value
    
    # Buy & hold comparison
    first_price = portfolio_df["price"][0]
    last_price = portfolio_df["price"][-1]
    initial_cash = strategy.config.initial_capital
    initial_eth = strategy.config.initial_eth_holdings
    total_eth_if_bought_all = initial_eth + (initial_cash / first_price)
    buy_hold_final_value = total_eth_if_bought_all * last_price
    buy_hold_return = (buy_hold_final_value - initial_value) / initial_value
    
    # Sharpe ratio calculation
    portfolio_df = portfolio_df.with_columns([
        pl.col("total_value").pct_change().alias("daily_return")
    ])
    daily_returns = portfolio_df["daily_return"].drop_nulls()
    
    if len(daily_returns) > 1:
        avg_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(365) if std_daily_return > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Max drawdown
    portfolio_df = portfolio_df.with_columns([
        pl.col("total_value").cum_max().alias("peak_value")
    ])
    portfolio_df = portfolio_df.with_columns([
        ((pl.col("total_value") - pl.col("peak_value")) / pl.col("peak_value")).alias("drawdown")
    ])
    max_drawdown = portfolio_df["drawdown"].min()
    
    metrics = {
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "excess_return": total_return - buy_hold_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "total_trades": strategy.state.trades_executed,
        "total_fees": strategy.state.total_fees_paid,
        "final_cash": strategy.state.cash,
        "final_eth": strategy.state.eth_holdings,
        "final_value": final_value
    }
    
    return strategy, metrics, config

def create_performance_chart(strategy):
    """Create interactive performance visualization"""
    if not strategy.portfolio_history:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(strategy.portfolio_history)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Portfolio Value vs Buy & Hold', 'ETH Price', 
                       'Portfolio Allocation', 'ETH Allocation %'],
        vertical_spacing=0.12
    )
    
    # Portfolio Value vs Buy & Hold
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['total_value'], name='Strategy', 
                  line=dict(color='green', width=2)),
        row=1, col=1
    )
    
    # Buy & hold calculation
    first_price = df['price'].iloc[0]
    initial_eth = strategy.config.initial_eth_holdings
    initial_cash = strategy.config.initial_capital
    total_eth = initial_eth + (initial_cash / first_price)
    buy_hold_values = total_eth * df['price']
    
    fig.add_trace(
        go.Scatter(x=df['date'], y=buy_hold_values, name='Buy & Hold',
                  line=dict(color='red', width=2, dash='dash')),
        row=1, col=1
    )
    
    # ETH Price
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['price'], name='ETH Price',
                  line=dict(color='blue', width=1)),
        row=1, col=2
    )
    
    # Portfolio Allocation
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['cash'], name='Cash',
                  line=dict(color='lightblue'), fill='tozeroy'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['eth_value'], name='ETH Value',
                  line=dict(color='orange'), fill='tozeroy'),
        row=2, col=1
    )
    
    # ETH Allocation %
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['eth_allocation'] * 100, name='ETH %',
                  line=dict(color='purple', width=2)),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, 
                     title_text="Grid Trading Strategy Performance")
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=2)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="Allocation (%)", row=2, col=2)
    
    return fig

def main():
    st.title("📈 Grid Trading Strategy Dashboard")
    st.markdown("Interactive analysis tool for cryptocurrency grid trading strategies")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Strategy Configuration")
    
    # Portfolio Settings
    st.sidebar.subheader("💰 Portfolio Settings")
    initial_capital = st.sidebar.number_input(
        "Initial Cash ($)", min_value=1000, max_value=100000, value=5000, step=1000
    )
    initial_eth_holdings = st.sidebar.number_input(
        "Initial ETH Holdings", min_value=0.0, max_value=50.0, value=2.0, step=0.5
    )
    
    # Price Range Settings  
    st.sidebar.subheader("📊 Price Range Settings")
    price_range_low = st.sidebar.number_input(
        "Price Range Low ($)", min_value=500, max_value=5000, value=3000, step=100
    )
    price_range_high = st.sidebar.number_input(
        "Price Range High ($)", min_value=5000, max_value=20000, value=10000, step=500
    )
    
    # Trading Settings
    st.sidebar.subheader("⚡ Trading Settings")
    volatility_threshold = st.sidebar.slider(
        "Volatility Threshold (%)", min_value=1, max_value=15, value=5, step=1
    ) / 100.0
    
    max_position_size = st.sidebar.slider(
        "Max Position Size (%)", min_value=1, max_value=25, value=10, step=1
    ) / 100.0
    
    min_position_size = st.sidebar.slider(
        "Min Position Size (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1
    ) / 100.0
    
    # Allocation Settings
    st.sidebar.subheader("⚖️ Allocation Settings")
    target_allocation = st.sidebar.slider(
        "Target ETH Allocation (%)", min_value=20, max_value=80, value=50, step=5
    ) / 100.0
    
    rebalance_threshold = st.sidebar.slider(
        "Rebalance Threshold (%)", min_value=5, max_value=25, value=10, step=1
    ) / 100.0
    
    transaction_fee = st.sidebar.slider(
        "Transaction Fee (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05
    ) / 100.0
    
    num_grids = st.sidebar.slider(
        "Number of Grid Levels", min_value=10, max_value=50, value=20, step=5
    )
    
    # Date Range Settings
    st.sidebar.subheader("📅 Date Range")
    
    preset_ranges = {
        "Crypto Winter (2018-2020)": ("2018-01-01", "2020-12-31"),
        "COVID Bull Run (2020-2022)": ("2020-03-01", "2022-01-01"),
        "Recent Volatility (2022-2024)": ("2022-01-01", "2024-12-31"),
        "2022 Bear Market": ("2021-11-01", "2022-12-31"),
        "Custom": None
    }
    
    selected_preset = st.sidebar.selectbox(
        "Date Range Preset", options=list(preset_ranges.keys()), index=2
    )
    
    if preset_ranges[selected_preset]:
        start_date, end_date = preset_ranges[selected_preset]
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date(2022, 1, 1)).strftime("%Y-%m-%d")
        with col2:
            end_date = st.date_input("End Date", value=date(2024, 12, 31)).strftime("%Y-%m-%d")
    
    # Run Strategy Button
    if st.sidebar.button("🚀 Run Strategy", type="primary", use_container_width=True):
        with st.spinner("Running backtest..."):
            try:
                strategy, metrics, config = run_backtest(
                    initial_capital, initial_eth_holdings, price_range_low, price_range_high,
                    num_grids, volatility_threshold, max_position_size, min_position_size,
                    target_allocation, rebalance_threshold, transaction_fee,
                    start_date, end_date
                )
                
                st.session_state.strategy = strategy
                st.session_state.metrics = metrics
                st.session_state.config = config
                st.session_state.has_results = True
                
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")
                return
    
    # Display Results
    if hasattr(st.session_state, 'has_results') and st.session_state.has_results:
        strategy = st.session_state.strategy
        metrics = st.session_state.metrics
        config = st.session_state.config
        
        # Performance Metrics
        st.header("📊 Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{metrics['total_return']:.1%}")
        with col2:
            excess_return = metrics['excess_return']
            st.metric("Excess Return vs B&H", f"{excess_return:+.1%}", 
                     delta=f"{excess_return:.1%}")
        with col3:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        with col4:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
        
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Final Portfolio", f"${metrics['final_value']:,.0f}")
        with col6:
            st.metric("Total Trades", f"{metrics['total_trades']:,}")
        with col7:
            st.metric("Final ETH Holdings", f"{metrics['final_eth']:.3f} ETH")
        with col8:
            st.metric("Fees Paid", f"${metrics['total_fees']:,.0f}")
        
        # Performance Chart
        st.header("📈 Performance Visualization")
        fig = create_performance_chart(strategy)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent Trades
        st.header("📋 Recent Trades")
        if strategy.trade_history:
            trades_df = pd.DataFrame(strategy.trade_history[-20:])  # Last 20 trades
            trades_df['Action'] = trades_df['action'].str.upper()
            trades_df['Amount'] = trades_df['amount'].apply(lambda x: f"{x:.4f} ETH")
            trades_df['Price'] = trades_df['price'].apply(lambda x: f"${x:,.2f}")
            
            display_df = trades_df[['Action', 'Amount', 'Price', 'reason']].copy()
            display_df.columns = ['Action', 'Amount', 'Price', 'Reason']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades executed during this period.")
    
    else:
        # Initial state
        st.header("👋 Welcome to the Grid Trading Dashboard!")
        st.markdown("""
        ### 🎯 How to Use:
        1. **Configure your strategy** using the sidebar controls
        2. **Select a time period** to test
        3. **Click "Run Strategy"** to see the results
        
        ### 💡 Tips:
        - **Crypto Winter (2018-2020)** shows grid trading at its best
        - **Starting with some ETH** gives more realistic results
        - **Lower volatility thresholds** = more trades
        """)

if __name__ == "__main__":
    main()
