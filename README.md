# Crypto Grid Trading Strategy

A comprehensive backtesting and analysis tool for grid trading strategies on cryptocurrency assets (specifically Ethereum). This project implements various volatility harvesting and mean-reversion algorithms to automatically buy low and sell high in volatile markets.

## Overview

This project implements a configurable grid trading strategy that:

- **Buys more when prices drop** and **sells when prices rise**
- Uses **percentage-based position sizing** based on price distance from range boundaries
- Implements **automatic rebalancing** to maintain target asset allocation
- Includes **transaction costs** and **realistic trading constraints**
- Provides **comprehensive performance analysis** and **visualization tools**

## Strategy Types Implemented

1. **Grid Trading**: Places buy/sell orders at preset intervals above and below current price
2. **Volatility-Based Trading**: Triggers trades when price moves beyond volatility threshold
3. **Constant-Mix Rebalancing**: Maintains target allocation between ETH and cash
4. **Martingale-Style Averaging**: Scales into positions more aggressively as price drops

## Key Features

- ✅ **Configurable Parameters**: Easy to adjust all trading parameters
- ✅ **Historical Backtesting**: Test strategies on 10+ years of ETH data (2015-2025)
- ✅ **Performance Metrics**: Sharpe ratio, max drawdown, excess returns vs buy & hold
- ✅ **Visualization**: Comprehensive charts showing price action, trades, and portfolio performance
- ✅ **Parameter Sensitivity Analysis**: Test how different settings affect performance
- ✅ **Multiple Strategy Presets**: Conservative, aggressive, high-frequency, and more
- ✅ **Export Capabilities**: Save results to CSV for further analysis

## Quick Start

### 🌟 **NEW: Interactive Web Dashboard**
```bash
uv run streamlit run streamlit_app.py
# OR
python run_dashboard.py
```

**Features:**
- ⚡ **Real-time parameter adjustment** with sliders and inputs
- 📊 **Interactive charts** with Plotly visualizations  
- 📅 **Date range presets** for key market periods
- 💰 **Starting ETH configuration** for realistic scenarios
- 📋 **Live trade history** and performance metrics
- 🎯 **Instant results** as you change parameters

### 1. Run Basic Analysis (Command Line)
```bash
uv run method-analysis.py
```

This will:
- Run a backtest with default parameters (now includes starting ETH)
- Display performance report with key metrics
- Generate visualization charts (`strategy_performance.png`)
- Export detailed results to CSV files
- Run parameter sensitivity analysis

### 2. Compare Different Strategies
```bash
uv run config_examples.py
```

This compares 5 different strategy configurations:
- **Conservative**: Lower risk, smaller position sizes
- **Aggressive**: Higher risk, larger position sizes  
- **High Frequency**: Very sensitive to small price movements
- **Large Range**: Captures major price swings
- **Crypto Winter**: Optimized for bear markets

### 3. Test with Starting ETH Holdings
```bash
uv run test_with_eth.py
```

Compare performance starting with different amounts of ETH (0, 1, 2, 5 ETH)

## Configuration Parameters

### Core Trading Parameters

```python
config = GridTradingConfig(
    initial_capital=5000.0,        # Starting cash amount
    initial_eth_holdings=2.0,      # Starting ETH holdings (NEW!)
    price_range_low=3000.0,        # Lower bound of trading range
    price_range_high=10000.0,      # Upper bound of trading range
    num_grids=20,                  # Number of grid levels
    
    # Volatility Trading
    volatility_threshold=0.05,      # 5% price change triggers trade
    max_position_size=0.10,         # Max 10% of cash per trade
    min_position_size=0.01,         # Min 1% of cash per trade
    
    # Rebalancing
    target_allocation=0.5,          # Target 50% ETH, 50% cash
    rebalance_threshold=0.1,        # Rebalance when 10% off target
    
    # Costs
    transaction_fee=0.001           # 0.1% fee per trade
)
```

### Position Sizing Logic

The strategy uses **intelligent position sizing** based on price location within the range:

- **Buying**: More aggressive as price approaches lower bound
- **Selling**: More aggressive as price approaches upper bound
- **Range**: Position size scales from `min_position_size` to `max_position_size`

## Performance Results (2020-2025)

### Default Configuration Results:
- **Total Return**: 951.42%
- **Annualized Return**: 168.33%
- **Sharpe Ratio**: 1.15
- **Max Drawdown**: -54.32%
- **Total Trades**: 327
- **Buy & Hold Return**: 3,557.24%
- **Excess Return**: -2,605.82%

### Key Insights:
1. **Grid trading underperforms buy & hold** in strong bull markets like 2020-2025
2. **Higher target ETH allocation** (70%) performed better than conservative (30%)
3. **10% volatility threshold** provided best risk-adjusted returns
4. **Strategy excels at risk management** with positive Sharpe ratios across all configurations

## Files Generated

- `strategy_performance.png` - Comprehensive performance visualization
- `strategy_results_portfolio.csv` - Daily portfolio values and allocation
- `strategy_results_trades.csv` - Complete trade history with reasons

## Customization

### Create Your Own Configuration

```python
my_config = GridTradingConfig(
    initial_capital=10000.0,
    price_range_low=2000.0,
    price_range_high=15000.0,
    volatility_threshold=0.08,     # 8% threshold
    target_allocation=0.6,         # 60% ETH target
    max_position_size=0.15,        # 15% max per trade
)

strategy = run_backtest(my_config, start_date="2018-01-01", end_date="2023-12-31")
metrics = calculate_performance_metrics(strategy)
print_performance_report(metrics, my_config)
```

### Test Different Time Periods

```python
# Test bear market performance
strategy = run_backtest(config, start_date="2021-11-01", end_date="2022-12-31")

# Test recent bull run
strategy = run_backtest(config, start_date="2023-01-01", end_date="2024-12-31")
```

## Technical Details

### Data Source
- **Historical ETH prices**: Daily data from 2015-2025
- **Format**: Date, Unix timestamp, USD price
- **Quality**: Excludes zero/invalid prices

### Strategy Logic
1. **Volatility Detection**: Compare current price to previous day
2. **Position Sizing**: Calculate based on price position in range
3. **Trade Execution**: Account for transaction fees
4. **Rebalancing**: Maintain target allocation when deviation exceeds threshold
5. **Portfolio Tracking**: Record all state changes for analysis

### Performance Metrics
- **Total Return**: (Final Value - Initial Capital) / Initial Capital
- **Sharpe Ratio**: Annualized return / standard deviation of daily returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Excess Return**: Strategy return minus buy & hold return

## Future Enhancements

Potential improvements you could implement:

1. **Dynamic Range Adjustment**: Automatically adjust price ranges based on market conditions
2. **Multiple Assets**: Extend to trade multiple cryptocurrencies
3. **Options Integration**: Add options strategies for enhanced returns
4. **Machine Learning**: Use ML to optimize parameters based on market regime
5. **Real-Time Trading**: Connect to exchanges for live trading
6. **Risk Management**: Add stop-losses and position limits

## Disclaimer

This is for educational and research purposes only. Past performance does not guarantee future results. Cryptocurrency trading involves substantial risk and may not be suitable for all investors.

## 🌟 Interactive Dashboard Features

### Real-Time Configuration
- **Portfolio Settings**: Adjust starting cash and ETH holdings
- **Price Range**: Set upper and lower trading bounds
- **Trading Parameters**: Fine-tune volatility thresholds and position sizes
- **Allocation Strategy**: Configure target ETH allocation and rebalancing

### Advanced Visualizations
- **Performance Charts**: Interactive Plotly charts with zoom/pan
- **Portfolio Allocation**: Real-time cash vs ETH breakdown
- **Trade Analysis**: Visual trade markers on price charts
- **Risk Metrics**: Drawdown analysis and Sharpe ratio tracking

### Date Range Presets
- **Crypto Winter (2018-2020)**: Bear market where grid trading excels
- **COVID Bull Run (2020-2022)**: Strong bull market favoring buy & hold
- **Recent Volatility (2022-2024)**: Recent period with grid trading advantage
- **2022 Bear Market**: Isolated bear market analysis
- **Custom Range**: Select any date range from 2015-2025

### Performance Comparison
- **Strategy vs Buy & Hold**: Side-by-side performance comparison
- **Excess Returns**: Clear indication of outperformance
- **Risk-Adjusted Metrics**: Sharpe ratio and maximum drawdown
- **Trade History**: Detailed log of all trading decisions

## Dependencies

- **polars**: Fast DataFrame library for data processing
- **numpy**: Numerical computations
- **matplotlib**: Static visualization and charting
- **streamlit**: Interactive web dashboard framework
- **plotly**: Interactive charting library

Install with: `uv sync`

## 🌍 Hosting Your Dashboard

### 🚀 **Easy Deployment (Recommended)**

**Streamlit Community Cloud - FREE:**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file: `streamlit_app.py`
5. Deploy!

Your dashboard will be live at: `https://yourname-crypto-trading-streamlit-app-xyz.streamlit.app`

### 📋 **Deployment Preparation**

Check if you're ready to deploy:
```bash
python3 prepare_deploy.py
```

This will verify all files are present and provide deployment guidance.

### 🐳 **Other Hosting Options**

- **Railway**: $5/month, more resources and custom domains
- **Heroku**: Free tier available, easy scaling
- **Docker + VPS**: Full control, custom server specs
- **GitHub Codespaces**: Quick testing and sharing

See `deploy_guide.md` for detailed instructions for each option.
