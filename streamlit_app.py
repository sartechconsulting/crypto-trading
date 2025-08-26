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
    asset: str = "ETH"
    initial_capital: float = 5000.0
    initial_asset_holdings: float = 0.0
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
    asset_holdings: float
    total_value: float
    last_price: float
    trades_executed: int = 0
    total_fees_paid: float = 0.0

class GridTradingStrategy:
    """Grid trading strategy implementation"""
    
    def __init__(self, config: GridTradingConfig, starting_price: float = None):
        self.config = config
        
        # Calculate initial asset holdings based on target allocation
        if starting_price and starting_price > 0:
            # Calculate how much of the initial capital should be in the asset
            target_asset_value = config.initial_capital * config.target_allocation
            initial_asset_holdings = target_asset_value / starting_price
            initial_cash = config.initial_capital - target_asset_value
        else:
            # Fallback if no starting price
            initial_asset_holdings = 0.0
            initial_cash = config.initial_capital
        
        # Store calculated initial holdings in config for reference
        self.config.initial_asset_holdings = initial_asset_holdings
        
        self.state = TradingState(
            cash=initial_cash,
            asset_holdings=initial_asset_holdings,
            total_value=config.initial_capital,
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
            
        asset_value = self.state.asset_holdings * current_price
        current_eth_allocation = asset_value / self.state.total_value
        
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
                self.state.asset_holdings += amount
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
                    "eth_after": self.state.asset_holdings
                })
                return True
        
        elif action == "sell":
            if amount <= self.state.asset_holdings:
                proceeds = amount * price
                fee = proceeds * self.config.transaction_fee
                net_proceeds = proceeds - fee
                
                self.state.cash += net_proceeds
                self.state.asset_holdings -= amount
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
                    "eth_after": self.state.asset_holdings
                })
                return True
        
        return False
    
    def process_price_update(self, date: str, price: float) -> None:
        """Process a new price update and make trading decisions"""
        if price <= 0:
            return
            
        # Update total portfolio value
        asset_value = self.state.asset_holdings * price
        self.state.total_value = self.state.cash + asset_value
        
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
                    amount_to_sell = self.state.asset_holdings * position_size
                    if amount_to_sell > 0:
                        self.execute_trade(
                            price, "sell", amount_to_sell,
                            f"Volatility sell: {price_change:.2%} gain"
                        )
        
        # Check for rebalancing
        rebalance_action = self.should_rebalance(price)
        if rebalance_action:
            asset_value = self.state.asset_holdings * price
            target_asset_value = self.state.total_value * self.config.target_allocation
            
            if rebalance_action == "buy":
                amount_to_buy = (target_asset_value - asset_value) / price
                if amount_to_buy > 0 and amount_to_buy * price <= self.state.cash:
                    self.execute_trade(
                        price, "buy", amount_to_buy, "Rebalancing: buy ETH"
                    )
            
            elif rebalance_action == "sell":
                amount_to_sell = (asset_value - target_asset_value) / price
                if amount_to_sell > 0 and amount_to_sell <= self.state.asset_holdings:
                    self.execute_trade(
                        price, "sell", amount_to_sell, "Rebalancing: sell ETH"
                    )
        
        # Record portfolio state
        self.portfolio_history.append({
            "date": date,
            "price": price,
            "cash": self.state.cash,
            "asset_holdings": self.state.asset_holdings,
            "asset_value": asset_value,
            "total_value": self.state.total_value,
            "eth_allocation": asset_value / self.state.total_value if self.state.total_value > 0 else 0
        })
        
        self.state.last_price = price

# Grid Trading Configuration
@dataclass
class PureGridConfig:
    """Configuration for pure grid trading strategy"""
    asset: str = "ETH"
    initial_capital: float = 5000.0
    grid_spacing: float = 0.05  # 5% spacing between grid levels
    buy_amount: float = 0.10  # 10% of available cash per buy
    sell_amount: float = 0.10  # 10% of holdings per sell
    num_buy_levels: int = 5  # How many buy levels below current price
    num_sell_levels: int = 5  # How many sell levels above current price
    reset_frequency: str = "never"  # never, daily, weekly
    transaction_fee: float = 0.001  # 0.1% fee per trade

class PureGridStrategy:
    """Pure grid trading strategy implementation"""
    
    def __init__(self, config: PureGridConfig, starting_price: float = None):
        self.config = config
        
        # Start with all capital in cash - no initial allocation
        self.state = TradingState(
            cash=config.initial_capital,
            asset_holdings=0.0,
            total_value=config.initial_capital,
            last_price=starting_price or 0.0
        )
        
        self.trade_history: List[dict] = []
        self.portfolio_history: List[dict] = []
        self.last_trade_price = starting_price or 0.0
        self.active_buy_levels: List[float] = []
        self.active_sell_levels: List[float] = []
        
        # Initialize grid levels
        if starting_price:
            self._update_grid_levels(starting_price)
    
    def _update_grid_levels(self, current_price: float):
        """Update active buy and sell grid levels"""
        self.active_buy_levels = []
        self.active_sell_levels = []
        
        # Create buy levels below current price
        for i in range(1, self.config.num_buy_levels + 1):
            buy_level = current_price * (1 - i * self.config.grid_spacing)
            self.active_buy_levels.append(buy_level)
        
        # Create sell levels above current price
        for i in range(1, self.config.num_sell_levels + 1):
            sell_level = current_price * (1 + i * self.config.grid_spacing)
            self.active_sell_levels.append(sell_level)
    
    def should_execute_buy(self, current_price: float) -> bool:
        """Check if current price hits any buy level"""
        for level in self.active_buy_levels:
            if current_price <= level:
                return True
        return False
    
    def should_execute_sell(self, current_price: float) -> bool:
        """Check if current price hits any sell level"""
        for level in self.active_sell_levels:
            if current_price >= level:
                return True
        return False
    
    def execute_grid_trade(self, price: float, action: str) -> bool:
        """Execute a grid trade (buy or sell)"""
        if action == "buy" and self.state.cash > 0:
            # Buy with percentage of available cash
            amount_to_spend = self.state.cash * self.config.buy_amount
            fee = amount_to_spend * self.config.transaction_fee
            net_amount = amount_to_spend - fee
            asset_amount = net_amount / price
            
            if asset_amount > 0:
                self.state.cash -= amount_to_spend
                self.state.asset_holdings += asset_amount
                self.state.total_fees_paid += fee
                self.state.trades_executed += 1
                
                self.trade_history.append({
                    "date": "",  # Will be filled by process_price_update
                    "action": "BUY",
                    "amount": asset_amount,
                    "price": price,
                    "value": net_amount,
                    "fee": fee,
                    "reason": f"Grid buy at ${price:.2f}",
                    "cash_before": self.state.cash + amount_to_spend,
                    "asset_before": self.state.asset_holdings - asset_amount,
                    "cash_after": self.state.cash,
                    "asset_after": self.state.asset_holdings
                })
                return True
                
        elif action == "sell" and self.state.asset_holdings > 0:
            # Sell percentage of current holdings
            asset_amount = self.state.asset_holdings * self.config.sell_amount
            gross_amount = asset_amount * price
            fee = gross_amount * self.config.transaction_fee
            net_amount = gross_amount - fee
            
            if asset_amount > 0:
                self.state.asset_holdings -= asset_amount
                self.state.cash += net_amount
                self.state.total_fees_paid += fee
                self.state.trades_executed += 1
                
                self.trade_history.append({
                    "date": "",  # Will be filled by process_price_update
                    "action": "SELL",
                    "amount": asset_amount,
                    "price": price,
                    "value": net_amount,
                    "fee": fee,
                    "reason": f"Grid sell at ${price:.2f}",
                    "cash_before": self.state.cash - net_amount,
                    "asset_before": self.state.asset_holdings + asset_amount,
                    "cash_after": self.state.cash,
                    "asset_after": self.state.asset_holdings
                })
                return True
        
        return False
    
    def process_price_update(self, date: str, price: float) -> None:
        """Process daily price update and execute grid trades"""
        # Check for grid trades
        executed_trade = False
        
        if self.should_execute_buy(price):
            if self.execute_grid_trade(price, "buy"):
                executed_trade = True
                # Update grid levels after buy
                self._update_grid_levels(price)
        
        elif self.should_execute_sell(price):
            if self.execute_grid_trade(price, "sell"):
                executed_trade = True
                # Update grid levels after sell
                self._update_grid_levels(price)
        
        # Update portfolio state
        asset_value = self.state.asset_holdings * price
        self.state.total_value = self.state.cash + asset_value
        
        # Add date to any trades executed today
        for trade in self.trade_history:
            if trade["date"] == "":
                trade["date"] = date
        
        # Record portfolio state
        self.portfolio_history.append({
            "date": date,
            "price": price,
            "cash": self.state.cash,
            "asset_holdings": self.state.asset_holdings,
            "asset_value": asset_value,
            "total_value": self.state.total_value
        })
        
        self.state.last_price = price

# Load ETH data
@st.cache_data
def load_crypto_data(asset: str = "ETH"):
    """Load and cache cryptocurrency price data"""
    if asset == "ETH":
        filename = "eth-prices.csv"
    elif asset == "BTC":
        filename = "btc-prices.csv"
    else:
        st.error(f"Unsupported asset: {asset}")
        st.stop()
    
    # Try different possible paths
    possible_paths = [
        f"data/{filename}",
        f"./data/{filename}", 
        f"/mount/src/crypto-trading/data/{filename}",
        f"crypto-trading/data/{filename}"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        st.error(f"Could not find {asset} price data file. Please ensure data/{filename} exists.")
        st.stop()
    
    try:
        if asset == "ETH":
            # ETH format: Date(UTC), UnixTimeStamp, Value (tab-separated)
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
            
        elif asset == "BTC":
            # BTC format: Date, Close/Last, Volume, Open, High, Low (comma-separated)
            df = pl.read_csv(data_path, has_header=True)
            # Clean and process data
            df = df.with_columns([
                pl.col("Date").str.strptime(pl.Date, "%m/%d/%Y").alias("date"),
                pl.col("Close/Last").cast(pl.Float64).alias("price")
            ])
            df = df.filter(pl.col("price") > 0)  # Remove zero prices
        
        return df.select(["date", "price"]).sort("date")
        
    except Exception as e:
        st.error(f"Error loading {asset} data file: {str(e)}")
        st.stop()

# Run backtest - handles both strategy types
@st.cache_data
def run_backtest(
    strategy_type, asset, initial_capital, 
    # Rebalancing strategy params
    price_range_low, price_range_high, num_grids, volatility_threshold, 
    max_position_size, min_position_size, target_allocation, rebalance_threshold,
    # Grid trading params  
    grid_spacing, buy_amount, sell_amount, num_buy_levels, num_sell_levels,
    # Common params
    transaction_fee, start_date, end_date
):
    """Run backtest with caching for better performance"""
    
    if strategy_type == "Rebalancing Strategy":
        config = GridTradingConfig(
            asset=asset,
            initial_capital=initial_capital,
            initial_asset_holdings=0.0,  # Will be calculated in strategy init
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
    else:  # Grid Trading Strategy
        config = PureGridConfig(
            asset=asset,
            initial_capital=initial_capital,
            grid_spacing=grid_spacing,
            buy_amount=buy_amount,
            sell_amount=sell_amount,
            num_buy_levels=int(num_buy_levels),
            num_sell_levels=int(num_sell_levels),
            transaction_fee=transaction_fee
        )
    
    # Load and filter data
    df = load_crypto_data(asset)
    
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
    if strategy_type == "Rebalancing Strategy":
        strategy = GridTradingStrategy(config, starting_price=starting_price)
    else:  # Grid Trading Strategy
        strategy = PureGridStrategy(config, starting_price=starting_price)
    
    # Run simulation
    for row in df.iter_rows(named=True):
        strategy.process_price_update(str(row["date"]), row["price"])
    
    # Calculate metrics
    portfolio_df = pl.DataFrame(strategy.portfolio_history)
    initial_value = portfolio_df["total_value"][0]
    final_value = strategy.state.total_value
    
    total_return = (final_value - initial_value) / initial_value
    
    # Buy & hold comparison - convert all initial capital to asset at first price
    first_price = portfolio_df["price"][0]
    last_price = portfolio_df["price"][-1]
    initial_total_capital = strategy.config.initial_capital
    # Buy & hold: all capital invested in asset at starting price
    buy_hold_asset_amount = initial_total_capital / first_price
    buy_hold_final_value = buy_hold_asset_amount * last_price
    buy_hold_return = (buy_hold_final_value - initial_total_capital) / initial_total_capital
    
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
        "final_asset": strategy.state.asset_holdings,
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
        subplot_titles=['Portfolio Value vs Buy & Hold', f'{strategy.config.asset} Price', 
                       'Portfolio Allocation', f'{strategy.config.asset} Allocation %'],
        vertical_spacing=0.12
    )
    
    # Portfolio Value vs Buy & Hold
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['total_value'], name='Strategy', 
                  line=dict(color='green', width=2)),
        row=1, col=1
    )
    
    # Buy & hold calculation - should start with same total value as strategy
    # Buy & hold assumes taking the initial total capital and buying asset at first price
    first_price = df['price'].iloc[0]
    initial_total_capital = strategy.config.initial_capital
    # Buy & hold: convert all initial capital to asset at starting price
    buy_hold_asset_amount = initial_total_capital / first_price
    buy_hold_values = buy_hold_asset_amount * df['price']
    
    fig.add_trace(
        go.Scatter(x=df['date'], y=buy_hold_values, name='Buy & Hold',
                  line=dict(color='red', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Asset Price
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['price'], name=f'{strategy.config.asset} Price',
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
        go.Scatter(x=df['date'], y=df['asset_value'], name='ETH Value',
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
    
    # Strategy Selection
    st.sidebar.subheader("🎯 Strategy Type")
    strategy_type = st.sidebar.selectbox(
        "Choose Trading Strategy",
        ["Rebalancing Strategy", "Grid Trading Strategy"],
        index=0,
        help="Select the type of trading strategy to analyze"
    )
    
    # Asset Selection
    st.sidebar.subheader("🪙 Asset Selection")
    selected_asset = st.sidebar.selectbox(
        "Choose Cryptocurrency",
        ["ETH", "BTC"],
        index=0,
        help="Select the cryptocurrency for backtesting"
    )
    
    # Strategy-specific controls
    if strategy_type == "Rebalancing Strategy":
        # Target Allocation (moved to top)
        st.sidebar.subheader("⚖️ Target Allocation")
        target_allocation = st.sidebar.slider(
            f"Target {selected_asset} Allocation (%)", 
            min_value=20, max_value=80, value=50, step=5,
            help=f"Percentage of portfolio value to hold in {selected_asset}"
        ) / 100.0
    else:  # Grid Trading Strategy
        st.sidebar.subheader("📊 Grid Trading Settings")
        target_allocation = 0.0  # Not used in grid trading
        
        grid_spacing = st.sidebar.slider(
            "Grid Spacing (%)", 
            min_value=1, max_value=20, value=5, step=1,
            help="Percentage spacing between grid levels"
        ) / 100.0
        
        buy_amount = st.sidebar.slider(
            "Buy Amount (% of cash)", 
            min_value=5, max_value=50, value=10, step=5,
            help="Percentage of available cash to spend on each buy trigger"
        ) / 100.0
        
        sell_amount = st.sidebar.slider(
            "Sell Amount (% of holdings)", 
            min_value=5, max_value=50, value=10, step=5,
            help="Percentage of current holdings to sell on each sell trigger"
        ) / 100.0
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            num_buy_levels = st.sidebar.number_input(
                "Buy Levels", min_value=1, max_value=10, value=5,
                help="Number of buy levels below current price"
            )
        with col2:
            num_sell_levels = st.sidebar.number_input(
                "Sell Levels", min_value=1, max_value=10, value=5,
                help="Number of sell levels above current price"
            )
    
    # Dynamic defaults based on asset
    if selected_asset == "BTC":
        default_low, default_high = 30000, 100000
        asset_name = "Bitcoin"
    else:
        default_low, default_high = 3000, 10000
        asset_name = "Ethereum"
    
    # Portfolio Settings
    st.sidebar.subheader("💰 Portfolio Settings")
    initial_capital = st.sidebar.number_input(
        "Initial Cash ($)", min_value=1000, max_value=100000, value=5000, step=1000,
        help="Starting cash amount for the strategy"
    )
    
    # Show strategy-specific info
    if strategy_type == "Rebalancing Strategy":
        st.sidebar.info(
            f"💡 **Allocation Strategy**\n\n"
            f"• {target_allocation:.0%} will be allocated to {selected_asset}\n"
            f"• {(1-target_allocation):.0%} will remain as cash\n"
            f"• Initial {selected_asset} amount will be calculated from starting price"
        )
    else:  # Grid Trading Strategy
        st.sidebar.info(
            f"📊 **Grid Trading Strategy**\n\n"
            f"• Start with 100% cash\n"
            f"• Buy {buy_amount:.0%} of cash when price drops {grid_spacing:.0%}\n"
            f"• Sell {sell_amount:.0%} of holdings when price rises {grid_spacing:.0%}\n"
            f"• {num_buy_levels} buy levels, {num_sell_levels} sell levels"
        )
    
    # Common settings for both strategies
    transaction_fee = st.sidebar.slider(
        "Transaction Fee (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05
    ) / 100.0
    
    # Strategy-specific settings
    if strategy_type == "Rebalancing Strategy":
        # Price Range Settings  
        st.sidebar.subheader("📊 Price Range Settings")
        price_range_low = st.sidebar.number_input(
            "Price Range Low ($)", min_value=500, max_value=50000, value=default_low, step=100
        )
        price_range_high = st.sidebar.number_input(
            "Price Range High ($)", min_value=5000, max_value=200000, value=default_high, step=500
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
        
        # Rebalancing Settings
        st.sidebar.subheader("🔄 Rebalancing Settings")
        rebalance_threshold = st.sidebar.slider(
            "Rebalance Threshold (%)", min_value=5, max_value=25, value=10, step=1
        ) / 100.0
        
        num_grids = st.sidebar.slider(
            "Number of Grid Levels", min_value=10, max_value=50, value=20, step=5
        )
    else:
        # Grid Trading doesn't need these settings
        price_range_low = 0
        price_range_high = 0  
        volatility_threshold = 0
        max_position_size = 0
        min_position_size = 0
        rebalance_threshold = 0
        num_grids = 0
    
    # Date Range Settings
    st.sidebar.subheader("📅 Date Range")
    
    # Calculate relative dates
    from datetime import datetime, timedelta
    today = datetime.now().date()
    
    preset_ranges = {
        "Last 6 Months": (today - timedelta(days=180), today),
        "Last Year": (today - timedelta(days=365), today),
        "Last 2 Years": (today - timedelta(days=730), today),
        "Last 3 Years": (today - timedelta(days=1095), today),
        "COVID Bull Run (2020-2022)": (date(2020, 3, 1), date(2022, 1, 1)),
        "2022 Bear Market": (date(2021, 11, 1), date(2022, 12, 31)),
        "Recent Volatility (2022-2024)": (date(2022, 1, 1), date(2024, 12, 31)),
        "All Available Data": (None, None),
        "Custom": "custom"
    }
    
    selected_preset = st.sidebar.selectbox(
        "Date Range Preset", 
        options=list(preset_ranges.keys()), 
        index=1,  # Default to "Last Year"
        help="Choose a preset time period or select Custom for manual dates"
    )
    
    if preset_ranges[selected_preset] == "custom":
        # Custom date selection
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date_input = st.date_input("Start Date", value=date(2023, 1, 1))
        with col2:
            end_date_input = st.date_input("End Date", value=today)
        start_date = start_date_input.strftime("%Y-%m-%d")
        end_date = end_date_input.strftime("%Y-%m-%d")
    elif preset_ranges[selected_preset] == (None, None):
        # All available data
        start_date = None
        end_date = None
    elif preset_ranges[selected_preset]:
        # Use preset dates
        start_date_obj, end_date_obj = preset_ranges[selected_preset]
        start_date = start_date_obj.strftime("%Y-%m-%d")
        end_date = end_date_obj.strftime("%Y-%m-%d")
    
    # Show selected date range
    if start_date and end_date:
        st.sidebar.info(f"📅 **Selected Period:**\n{start_date} to {end_date}")
    elif not start_date and not end_date:
        st.sidebar.info("📅 **Selected Period:**\nAll available data")
    
    # Run Strategy Button
    if st.sidebar.button("🚀 Run Strategy", type="primary", use_container_width=True):
        with st.spinner("Running backtest..."):
            try:
                # Set grid trading parameters with defaults if not defined
                if strategy_type == "Grid Trading Strategy":
                    grid_params = (grid_spacing, buy_amount, sell_amount, num_buy_levels, num_sell_levels)
                else:
                    grid_params = (0.05, 0.1, 0.1, 5, 5)  # Default values for rebalancing strategy
                
                strategy, metrics, config = run_backtest(
                    strategy_type, selected_asset, initial_capital, 
                    # Rebalancing strategy params
                    price_range_low, price_range_high, num_grids, volatility_threshold,
                    max_position_size, min_position_size, target_allocation, rebalance_threshold,
                    # Grid trading params
                    *grid_params,
                    # Common params
                    transaction_fee, start_date, end_date
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
        
        # Show calculated starting allocation
        st.header("🎯 Starting Allocation")
        
        starting_price = strategy.portfolio_history[0]['price'] if strategy.portfolio_history else 0
        initial_asset_value = config.initial_asset_holdings * starting_price
        initial_cash_used = config.initial_capital - initial_asset_value
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                f"Starting {config.asset}", 
                f"{config.initial_asset_holdings:.4f} {config.asset}",
                help=f"Calculated from {config.target_allocation:.0%} allocation"
            )
        with col2:
            st.metric(
                f"Starting {config.asset} Value", 
                f"${initial_asset_value:,.2f}",
                help=f"At starting price of ${starting_price:,.2f}"
            )
        with col3:
            st.metric(
                "Starting Cash", 
                f"${initial_cash_used:,.2f}",
                help=f"Remaining cash after {config.asset} purchase"
            )
        
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
            st.metric(f"Final {config.asset} Holdings", f"{metrics['final_asset']:.3f} {config.asset}")
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
