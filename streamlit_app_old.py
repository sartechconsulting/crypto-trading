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
from datetime import datetime, date
import sys
import os

# Import our trading strategy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load the method-analysis module directly
exec(open('method-analysis.py').read())

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

# Cache the ETH data loading
@st.cache_data
def get_eth_data():
    """Load and cache ETH price data"""
    # Use relative path that works in both local and cloud environments
    data_path = "data/eth-prices.csv"
    if not os.path.exists(data_path):
        # Fallback paths for different deployment scenarios
        possible_paths = [
            "data/eth-prices.csv",
            "./data/eth-prices.csv", 
            "/mount/src/crypto-trading/data/eth-prices.csv",
            "crypto-trading/data/eth-prices.csv"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        else:
            st.error("Could not find ETH price data file. Please ensure data/eth-prices.csv exists.")
            st.stop()
    
    return load_eth_data_streamlit(data_path)

# Create a modified version of load_eth_data for Streamlit
def load_eth_data_streamlit(file_path: str):
    """Load and preprocess ETH price data for Streamlit"""
    import polars as pl
    
    df = pl.read_csv(
        file_path,
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

# Cache the backtest computation
@st.cache_data
def run_cached_backtest(
    initial_capital, initial_eth_holdings, price_range_low, price_range_high,
    num_grids, volatility_threshold, max_position_size, min_position_size,
    target_allocation, rebalance_threshold, transaction_fee,
    start_date, end_date
):
    """Run backtest with caching for better performance"""
    from datetime import datetime
    
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
    
    # Load data with flexible path
    data_path = "data/eth-prices.csv"
    if not os.path.exists(data_path):
        possible_paths = [
            "data/eth-prices.csv",
            "./data/eth-prices.csv", 
            "/mount/src/crypto-trading/data/eth-prices.csv",
            "crypto-trading/data/eth-prices.csv"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
    
    df = load_eth_data_streamlit(data_path)
    
    # Filter date range if specified
    if start_date:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        df = df.filter(pl.col("date") >= start_date_obj)
    if end_date:
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        df = df.filter(pl.col("date") <= end_date_obj)
    
    if len(df) == 0:
        raise ValueError("No data available for the specified date range")
    
    # Get starting price for initial portfolio calculation
    starting_price = df["price"][0]
    
    # Initialize strategy with starting price
    strategy = GridTradingStrategy(config, starting_price=starting_price)
    
    # Run simulation
    for row in df.iter_rows(named=True):
        strategy.process_price_update(
            str(row["date"]), 
            row["price"]
        )
    
    metrics = calculate_performance_metrics(strategy)
    
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
        rows=3, cols=2,
        subplot_titles=[
            'Portfolio Value vs Buy & Hold', 'ETH Price',
            'Portfolio Allocation (Cash vs ETH)', 'ETH Allocation %',
            'Daily Returns', 'Cumulative Trades'
        ],
        specs=[[{"colspan": 1}, {"colspan": 1}],
               [{"colspan": 1}, {"colspan": 1}],
               [{"colspan": 1}, {"colspan": 1}]],
        vertical_spacing=0.08
    )
    
    # 1. Portfolio Value vs Buy & Hold
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['total_value'], name='Strategy Portfolio', 
                  line=dict(color='green', width=2)),
        row=1, col=1
    )
    
    # Calculate buy & hold for comparison
    first_price = df['price'].iloc[0]
    initial_eth = strategy.config.initial_eth_holdings
    initial_cash = strategy.config.initial_capital
    total_eth_if_bought_all = initial_eth + (initial_cash / first_price)
    buy_hold_values = total_eth_if_bought_all * df['price']
    
    fig.add_trace(
        go.Scatter(x=df['date'], y=buy_hold_values, name='Buy & Hold',
                  line=dict(color='red', width=2, dash='dash')),
        row=1, col=1
    )
    
    # 2. ETH Price
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['price'], name='ETH Price',
                  line=dict(color='blue', width=1)),
        row=1, col=2
    )
    
    # Add price range lines
    fig.add_hline(y=strategy.config.price_range_low, line_dash="dot", 
                  line_color="gray", opacity=0.5, row=1, col=2)
    fig.add_hline(y=strategy.config.price_range_high, line_dash="dot", 
                  line_color="gray", opacity=0.5, row=1, col=2)
    
    # 3. Portfolio Allocation
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['cash'], name='Cash',
                  line=dict(color='lightblue'), fill='tonexty'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['eth_value'], name='ETH Value',
                  line=dict(color='orange'), fill='tozeroy'),
        row=2, col=1
    )
    
    # 4. ETH Allocation %
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['eth_allocation'] * 100, name='ETH %',
                  line=dict(color='purple', width=2)),
        row=2, col=2
    )
    fig.add_hline(y=strategy.config.target_allocation * 100, line_dash="dash",
                  line_color="gray", opacity=0.7, row=2, col=2)
    
    # 5. Daily Returns
    df['daily_return'] = df['total_value'].pct_change() * 100
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['daily_return'], name='Daily Return %',
                  mode='lines', line=dict(color='darkgreen', width=1)),
        row=3, col=1
    )
    fig.add_hline(y=0, line_color="black", opacity=0.3, row=3, col=1)
    
    # 6. Cumulative Trades
    trades_df = pd.DataFrame(strategy.trade_history) if strategy.trade_history else pd.DataFrame()
    if not trades_df.empty:
        # Count cumulative trades by date (this is simplified)
        cumulative_trades = list(range(1, len(trades_df) + 1))
        # For simplicity, spread trades across time period
        trade_dates = pd.date_range(df['date'].min(), df['date'].max(), periods=len(trades_df))
        
        fig.add_trace(
            go.Scatter(x=trade_dates, y=cumulative_trades, name='Cumulative Trades',
                      line=dict(color='brown', width=2)),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Grid Trading Strategy Performance Dashboard",
        title_x=0.5
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=2)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="ETH Price ($)", row=1, col=2)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="ETH Allocation (%)", row=2, col=2)
    fig.update_yaxes(title_text="Daily Return (%)", row=3, col=1)
    fig.update_yaxes(title_text="Cumulative Trades", row=3, col=2)
    
    return fig

def create_trades_table(strategy):
    """Create a table of recent trades"""
    if not strategy.trade_history:
        return pd.DataFrame()
    
    trades_df = pd.DataFrame(strategy.trade_history)
    
    # Format the trades for display
    trades_df['formatted_action'] = trades_df['action'].str.upper()
    trades_df['formatted_amount'] = trades_df['amount'].apply(lambda x: f"{x:.4f} ETH")
    trades_df['formatted_price'] = trades_df['price'].apply(lambda x: f"${x:,.2f}")
    
    if 'cost' in trades_df.columns:
        trades_df['formatted_value'] = trades_df.apply(
            lambda row: f"${row.get('cost', row.get('proceeds', 0)):,.2f}", axis=1
        )
    else:
        trades_df['formatted_value'] = trades_df['proceeds'].apply(lambda x: f"${x:,.2f}")
    
    # Select and rename columns for display
    display_df = trades_df[['formatted_action', 'formatted_amount', 'formatted_price', 
                           'formatted_value', 'reason']].copy()
    display_df.columns = ['Action', 'Amount', 'Price', 'Value', 'Reason']
    
    return display_df.tail(20)  # Show last 20 trades

def main():
    st.title("📈 Grid Trading Strategy Dashboard")
    st.markdown("Interactive analysis tool for cryptocurrency grid trading strategies")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Strategy Configuration")
    
    # Portfolio Settings
    st.sidebar.subheader("💰 Portfolio Settings")
    initial_capital = st.sidebar.number_input(
        "Initial Cash ($)", 
        min_value=1000, max_value=100000, value=5000, step=1000,
        help="Starting cash amount"
    )
    initial_eth_holdings = st.sidebar.number_input(
        "Initial ETH Holdings", 
        min_value=0.0, max_value=50.0, value=2.0, step=0.5,
        help="ETH you already own at strategy start"
    )
    
    # Price Range Settings  
    st.sidebar.subheader("📊 Price Range Settings")
    price_range_low = st.sidebar.number_input(
        "Price Range Low ($)", 
        min_value=500, max_value=5000, value=3000, step=100,
        help="Lower bound for grid trading range"
    )
    price_range_high = st.sidebar.number_input(
        "Price Range High ($)", 
        min_value=5000, max_value=20000, value=10000, step=500,
        help="Upper bound for grid trading range"
    )
    
    # Trading Settings
    st.sidebar.subheader("⚡ Trading Settings")
    volatility_threshold = st.sidebar.slider(
        "Volatility Threshold (%)", 
        min_value=1, max_value=15, value=5, step=1,
        help="Price change % that triggers trading action"
    ) / 100.0
    
    max_position_size = st.sidebar.slider(
        "Max Position Size (%)", 
        min_value=1, max_value=25, value=10, step=1,
        help="Maximum % of cash to use per trade"
    ) / 100.0
    
    min_position_size = st.sidebar.slider(
        "Min Position Size (%)", 
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="Minimum % of cash to use per trade"
    ) / 100.0
    
    # Allocation Settings
    st.sidebar.subheader("⚖️ Allocation Settings")
    target_allocation = st.sidebar.slider(
        "Target ETH Allocation (%)", 
        min_value=20, max_value=80, value=50, step=5,
        help="Target % of portfolio in ETH"
    ) / 100.0
    
    rebalance_threshold = st.sidebar.slider(
        "Rebalance Threshold (%)", 
        min_value=5, max_value=25, value=10, step=1,
        help="Allocation deviation % that triggers rebalancing"
    ) / 100.0
    
    transaction_fee = st.sidebar.slider(
        "Transaction Fee (%)", 
        min_value=0.0, max_value=1.0, value=0.1, step=0.05,
        help="Fee charged per trade"
    ) / 100.0
    
    num_grids = st.sidebar.slider(
        "Number of Grid Levels", 
        min_value=10, max_value=50, value=20, step=5,
        help="Number of price levels in the grid"
    )
    
    # Date Range Settings
    st.sidebar.subheader("📅 Date Range")
    
    # Get available date range
    df = get_eth_data()
    min_date = pd.to_datetime(df['date'].min()).date()
    max_date = pd.to_datetime(df['date'].max()).date()
    
    # Preset date ranges
    preset_ranges = {
        "Crypto Winter (2018-2020)": (date(2018, 1, 1), date(2020, 12, 31)),
        "COVID Bull Run (2020-2022)": (date(2020, 3, 1), date(2022, 1, 1)),
        "Recent Volatility (2022-2024)": (date(2022, 1, 1), date(2024, 12, 31)),
        "2022 Bear Market": (date(2021, 11, 1), date(2022, 12, 31)),
        "Full History": (min_date, max_date),
        "Custom": None
    }
    
    selected_preset = st.sidebar.selectbox(
        "Date Range Preset",
        options=list(preset_ranges.keys()),
        index=2  # Default to "Recent Volatility"
    )
    
    if preset_ranges[selected_preset]:
        start_date, end_date = preset_ranges[selected_preset]
    else:
        # Custom date selection
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date", 
                value=date(2022, 1, 1),
                min_value=min_date,
                max_value=max_date
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=date(2024, 12, 31),
                min_value=min_date,
                max_value=max_date
            )
    
    # Run Strategy Button
    if st.sidebar.button("🚀 Run Strategy", type="primary", use_container_width=True):
        # Convert dates to strings
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Run the backtest
        with st.spinner("Running backtest..."):
            try:
                strategy, metrics, config = run_cached_backtest(
                    initial_capital, initial_eth_holdings, price_range_low, price_range_high,
                    num_grids, volatility_threshold, max_position_size, min_position_size,
                    target_allocation, rebalance_threshold, transaction_fee,
                    start_str, end_str
                )
                
                # Store results in session state
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
            st.metric(
                "Total Return",
                f"{metrics['total_return']:.1%}",
                help="Overall strategy return"
            )
        
        with col2:
            excess_return = metrics['excess_return']
            st.metric(
                "Excess Return vs B&H",
                f"{excess_return:+.1%}",
                delta=f"{excess_return:.1%}",
                delta_color="normal" if excess_return > 0 else "inverse",
                help="Return vs buy and hold strategy"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.2f}",
                help="Risk-adjusted return measure"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{metrics['max_drawdown']:.1%}",
                help="Largest peak-to-trough decline"
            )
        
        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric(
                "Final Portfolio Value",
                f"${metrics['final_value']:,.0f}",
                help="Total portfolio value at end"
            )
        
        with col6:
            st.metric(
                "Total Trades",
                f"{metrics['total_trades']:,}",
                help="Number of trades executed"
            )
        
        with col7:
            st.metric(
                "Final ETH Holdings",
                f"{metrics['final_eth']:.3f} ETH",
                help="ETH amount at strategy end"
            )
        
        with col8:
            st.metric(
                "Fees Paid",
                f"${metrics['total_fees']:,.0f}",
                help="Total transaction fees"
            )
        
        # Performance Chart
        st.header("📈 Performance Visualization")
        fig = create_performance_chart(strategy)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent Trades Table
        st.header("📋 Recent Trades")
        trades_df = create_trades_table(strategy)
        if not trades_df.empty:
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades executed during this period.")
        
        # Configuration Summary
        with st.expander("⚙️ Configuration Used"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Portfolio Settings:**")
                st.write(f"- Initial Cash: ${config.initial_capital:,.0f}")
                st.write(f"- Initial ETH: {config.initial_eth_holdings:.2f} ETH")
                st.write(f"- Price Range: ${config.price_range_low:,.0f} - ${config.price_range_high:,.0f}")
                
                st.write("**Trading Settings:**")
                st.write(f"- Volatility Threshold: {config.volatility_threshold:.1%}")
                st.write(f"- Position Size: {config.min_position_size:.1%} - {config.max_position_size:.1%}")
            
            with col2:
                st.write("**Allocation Settings:**")
                st.write(f"- Target ETH Allocation: {config.target_allocation:.1%}")
                st.write(f"- Rebalance Threshold: {config.rebalance_threshold:.1%}")
                st.write(f"- Transaction Fee: {config.transaction_fee:.2%}")
                st.write(f"- Grid Levels: {config.num_grids}")
    
    else:
        # Initial state - show instructions
        st.header("👋 Welcome to the Grid Trading Dashboard!")
        
        st.markdown("""
        ### 🎯 How to Use:
        1. **Configure your strategy** using the sidebar controls
        2. **Select a time period** to test (presets available for key market periods)
        3. **Click "Run Strategy"** to see the results
        
        ### 📊 What You'll See:
        - **Performance metrics** vs buy & hold
        - **Interactive charts** showing portfolio performance
        - **Trade history** with reasons for each trade
        - **Risk-adjusted returns** (Sharpe ratio, max drawdown)
        
        ### 💡 Tips:
        - **Crypto Winter (2018-2020)** and **Recent Volatility (2022-2024)** periods show grid trading at its best
        - **Starting with some ETH** gives more realistic results for existing holders
        - **Lower volatility thresholds** = more trades but potentially better performance
        - **Higher target ETH allocation** = more aggressive strategy
        """)
        
        # Show sample configuration
        st.subheader("📋 Sample Configuration")
        st.code("""
        # Conservative Strategy
        Initial Cash: $5,000
        Initial ETH: 2.0 ETH
        Volatility Threshold: 3%
        Target ETH Allocation: 40%
        
        # Aggressive Strategy  
        Initial Cash: $5,000
        Initial ETH: 2.0 ETH
        Volatility Threshold: 7%
        Target ETH Allocation: 70%
        """)

if __name__ == "__main__":
    main()
