import polars as pl
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec


@dataclass
class GridTradingConfig:
    """Configuration for grid trading strategy"""
    initial_capital: float = 5000.0
    initial_eth_holdings: float = 0.0  # Starting ETH amount
    price_range_low: float = 3000.0
    price_range_high: float = 10000.0
    num_grids: int = 20  # Number of grid levels
    
    # Volatility-based trading parameters
    volatility_threshold: float = 0.05  # 5% price change to trigger action
    max_position_size: float = 0.10  # Max 10% of cash per trade
    min_position_size: float = 0.01  # Min 1% of cash per trade
    
    # Rebalancing parameters
    target_allocation: float = 0.5  # Target 50% ETH, 50% cash
    rebalance_threshold: float = 0.1  # Rebalance when allocation deviates by 10%
    
    # Trading costs
    transaction_fee: float = 0.001  # 0.1% fee per trade


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
        
    def calculate_grid_levels(self) -> List[float]:
        """Calculate grid levels between price range"""
        return np.linspace(
            self.config.price_range_low,
            self.config.price_range_high,
            self.config.num_grids
        ).tolist()
    
    def calculate_position_size(self, price: float, action: str) -> float:
        """Calculate position size based on price position in range"""
        price_range = self.config.price_range_high - self.config.price_range_low
        
        if action == "buy":
            # Buy more aggressively as price approaches lower bound
            distance_from_low = (price - self.config.price_range_low) / price_range
            # Invert so we buy more when price is lower
            intensity = 1.0 - distance_from_low
            position_size = self.config.min_position_size + (
                intensity * (self.config.max_position_size - self.config.min_position_size)
            )
        else:  # sell
            # Sell more aggressively as price approaches upper bound
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
                return "sell"  # Too much ETH, sell some
            else:
                return "buy"   # Too little ETH, buy some
        
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
                    # Buy opportunity
                    position_size = self.calculate_position_size(price, "buy")
                    amount_to_buy = (self.state.cash * position_size) / price
                    if amount_to_buy > 0:
                        self.execute_trade(
                            price, "buy", amount_to_buy, 
                            f"Volatility buy: {price_change:.2%} drop"
                        )
                
                elif price_change > 0:  # Price increased
                    # Sell opportunity
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


def load_eth_data(file_path: str) -> pl.DataFrame:
    """Load and preprocess ETH price data"""
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


def run_backtest(config: GridTradingConfig, start_date: str = None, end_date: str = None) -> GridTradingStrategy:
    """Run backtest on historical data"""
    # Load data
    df = load_eth_data("/Users/jesse/repos/crypto-trading/data/eth-prices.csv")
    
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
    
    return strategy


def calculate_performance_metrics(strategy: GridTradingStrategy) -> dict:
    """Calculate performance metrics for the strategy"""
    if not strategy.portfolio_history:
        return {}
    
    portfolio_df = pl.DataFrame(strategy.portfolio_history)
    
    # Use the actual initial portfolio value (cash + initial ETH value)
    initial_value = portfolio_df["total_value"][0]
    final_value = strategy.state.total_value
    
    # Calculate returns
    total_return = (final_value - initial_value) / initial_value
    
    # Calculate daily returns for Sharpe ratio
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
    
    # Calculate max drawdown
    portfolio_df = portfolio_df.with_columns([
        pl.col("total_value").cum_max().alias("peak_value")
    ])
    portfolio_df = portfolio_df.with_columns([
        ((pl.col("total_value") - pl.col("peak_value")) / pl.col("peak_value")).alias("drawdown")
    ])
    
    max_drawdown = portfolio_df["drawdown"].min()
    
    # Buy and hold comparison - calculate what would happen if we just held initial portfolio
    first_price = portfolio_df["price"][0]
    last_price = portfolio_df["price"][-1]
    
    # Calculate buy & hold with initial ETH + converting all cash to ETH at start
    initial_cash = strategy.config.initial_capital
    initial_eth = strategy.config.initial_eth_holdings
    total_eth_if_bought_all = initial_eth + (initial_cash / first_price)
    
    buy_hold_final_value = total_eth_if_bought_all * last_price
    buy_hold_return = (buy_hold_final_value - initial_value) / initial_value
    
    return {
        "total_return": total_return,
        "annualized_return": total_return / (len(portfolio_df) / 365),
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "buy_hold_return": buy_hold_return,
        "excess_return": total_return - buy_hold_return,
        "total_trades": strategy.state.trades_executed,
        "total_fees": strategy.state.total_fees_paid,
        "final_cash": strategy.state.cash,
        "final_eth": strategy.state.eth_holdings,
        "final_value": final_value
    }


def plot_strategy_performance(strategy: GridTradingStrategy, save_path: str = None):
    """Create comprehensive visualization of strategy performance"""
    if not strategy.portfolio_history:
        print("No portfolio history to plot")
        return
    
    # Convert to DataFrame for plotting
    portfolio_df = pl.DataFrame(strategy.portfolio_history)
    trades_df = pl.DataFrame(strategy.trade_history) if strategy.trade_history else None
    
    # Convert dates for plotting
    dates = [datetime.strptime(str(d), "%Y-%m-%d") for d in portfolio_df["date"]]
    prices = portfolio_df["price"].to_list()
    portfolio_values = portfolio_df["total_value"].to_list()
    cash_values = portfolio_df["cash"].to_list()
    eth_values = portfolio_df["eth_value"].to_list()
    allocations = portfolio_df["eth_allocation"].to_list()
    
    # Create subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])
    
    # 1. Price chart with trades
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, prices, 'b-', linewidth=1, label='ETH Price', alpha=0.7)
    
    # Add buy/sell markers
    if trades_df is not None and len(trades_df) > 0:
        buy_trades = trades_df.filter(pl.col("action") == "buy")
        sell_trades = trades_df.filter(pl.col("action") == "sell")
        
        if len(buy_trades) > 0:
            buy_prices = buy_trades["price"].to_list()
            # For plotting, we'll just use the trade prices as y-coordinates
            ax1.scatter([None] * len(buy_prices), buy_prices, 
                       c='green', marker='^', s=30, alpha=0.6, label='Buy')
        
        if len(sell_trades) > 0:
            sell_prices = sell_trades["price"].to_list()
            ax1.scatter([None] * len(sell_prices), sell_prices, 
                       c='red', marker='v', s=30, alpha=0.6, label='Sell')
    
    # Add grid lines for price range
    ax1.axhline(y=strategy.config.price_range_low, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=strategy.config.price_range_high, color='gray', linestyle='--', alpha=0.5)
    
    ax1.set_title('ETH Price and Trading Activity', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Portfolio value comparison
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(dates, portfolio_values, 'g-', linewidth=2, label='Strategy Portfolio')
    
    # Calculate buy & hold performance
    initial_eth = strategy.config.initial_capital / prices[0]
    buy_hold_values = [initial_eth * price for price in prices]
    ax2.plot(dates, buy_hold_values, 'r--', linewidth=2, label='Buy & Hold')
    
    ax2.set_title('Portfolio Value Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cash vs ETH allocation
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(dates, cash_values, 'b-', linewidth=1, label='Cash')
    ax3.plot(dates, eth_values, 'orange', linewidth=1, label='ETH Value')
    ax3.set_title('Portfolio Allocation', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Value ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. ETH allocation percentage
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(dates, [a * 100 for a in allocations], 'purple', linewidth=2)
    ax4.axhline(y=strategy.config.target_allocation * 100, color='gray', 
                linestyle='--', alpha=0.7, label=f'Target ({strategy.config.target_allocation:.0%})')
    ax4.set_title('ETH Allocation Percentage', fontsize=12, fontweight='bold')
    ax4.set_ylabel('ETH Allocation (%)')
    ax4.set_xlabel('Date')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    else:
        plt.show()


def print_performance_report(metrics: dict, config: GridTradingConfig):
    """Print a detailed performance report"""
    print("=" * 60)
    print("GRID TRADING STRATEGY PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Initial Cash: ${config.initial_capital:,.2f}")
    if config.initial_eth_holdings > 0:
        print(f"Initial ETH Holdings: {config.initial_eth_holdings:.4f} ETH")
    print(f"Price Range: ${config.price_range_low:,.0f} - ${config.price_range_high:,.0f}")
    print(f"Grid Levels: {config.num_grids}")
    print(f"Volatility Threshold: {config.volatility_threshold:.1%}")
    print(f"Target Allocation: {config.target_allocation:.1%} ETH")
    print("-" * 60)
    print(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
    print(f"Final Cash: ${metrics['final_cash']:,.2f}")
    print(f"Final ETH Holdings: {metrics['final_eth']:,.4f}")
    print(f"Total Return: {metrics['total_return']:,.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:,.2%}")
    print(f"Buy & Hold Return: {metrics['buy_hold_return']:,.2%}")
    print(f"Excess Return vs B&H: {metrics['excess_return']:,.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:,.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Total Fees Paid: ${metrics['total_fees']:,.2f}")
    print("=" * 60)


def run_parameter_sensitivity_analysis(base_config: GridTradingConfig, start_date: str = "2020-01-01"):
    """Run sensitivity analysis on key parameters"""
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Test different volatility thresholds
    volatility_thresholds = [0.03, 0.05, 0.07, 0.10]
    print("\nVolatility Threshold Analysis:")
    print("Threshold | Total Return | Sharpe | Trades | Excess Return")
    print("-" * 55)
    
    for thresh in volatility_thresholds:
        config = GridTradingConfig(
            initial_capital=base_config.initial_capital,
            price_range_low=base_config.price_range_low,
            price_range_high=base_config.price_range_high,
            volatility_threshold=thresh,
            target_allocation=base_config.target_allocation
        )
        strategy = run_backtest(config, start_date=start_date)
        metrics = calculate_performance_metrics(strategy)
        
        print(f"{thresh:>8.1%} | {metrics['total_return']:>10.1%} | {metrics['sharpe_ratio']:>6.2f} | "
              f"{metrics['total_trades']:>6d} | {metrics['excess_return']:>10.1%}")
    
    # Test different target allocations
    target_allocations = [0.3, 0.5, 0.7]
    print("\nTarget Allocation Analysis:")
    print("Target ETH | Total Return | Sharpe | Trades | Excess Return")
    print("-" * 55)
    
    for alloc in target_allocations:
        config = GridTradingConfig(
            initial_capital=base_config.initial_capital,
            price_range_low=base_config.price_range_low,
            price_range_high=base_config.price_range_high,
            volatility_threshold=base_config.volatility_threshold,
            target_allocation=alloc
        )
        strategy = run_backtest(config, start_date=start_date)
        metrics = calculate_performance_metrics(strategy)
        
        print(f"{alloc:>8.1%} | {metrics['total_return']:>10.1%} | {metrics['sharpe_ratio']:>6.2f} | "
              f"{metrics['total_trades']:>6d} | {metrics['excess_return']:>10.1%}")


def export_results_to_csv(strategy: GridTradingStrategy, file_path: str):
    """Export strategy results to CSV for further analysis"""
    if not strategy.portfolio_history:
        print("No data to export")
        return
    
    # Export portfolio history
    portfolio_df = pl.DataFrame(strategy.portfolio_history)
    portfolio_df.write_csv(f"{file_path}_portfolio.csv")
    
    # Export trade history
    if strategy.trade_history:
        trades_df = pl.DataFrame(strategy.trade_history)
        trades_df.write_csv(f"{file_path}_trades.csv")
    
    print(f"Results exported to {file_path}_portfolio.csv and {file_path}_trades.csv")


if __name__ == "__main__":
    # Example configuration - you can modify these parameters
    config = GridTradingConfig(
        initial_capital=5000.0,
        initial_eth_holdings=2.0,   # Start with 2 ETH
        price_range_low=3000.0,
        price_range_high=10000.0,
        num_grids=20,
        volatility_threshold=0.05,  # 5% price change triggers action
        max_position_size=0.10,     # Max 10% of cash per trade
        min_position_size=0.01,     # Min 1% of cash per trade
        target_allocation=0.5,      # Target 50% ETH, 50% cash
        rebalance_threshold=0.1,    # Rebalance when allocation deviates by 10%
        transaction_fee=0.001       # 0.1% fee per trade
    )
    
    # Run backtest - modify these dates to test different periods
    print("Running backtest...")
    strategy = run_backtest(config, start_date="2018-01-01", end_date="2020-12-31")  # Crypto Winter period
    
    # Calculate and display results
    metrics = calculate_performance_metrics(strategy)
    print_performance_report(metrics, config)
    
    # Show recent trades
    if strategy.trade_history:
        print("\nRecent Trades (last 10):")
        print("-" * 80)
        for trade in strategy.trade_history[-10:]:
            action = trade["action"]
            amount = trade["amount"]
            price = trade["price"]
            reason = trade["reason"]
            print(f"{action.upper()}: {amount:.4f} ETH @ ${price:,.2f} - {reason}")
    
    # Run parameter sensitivity analysis
    run_parameter_sensitivity_analysis(config, start_date="2018-01-01")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_strategy_performance(strategy, save_path="strategy_performance.png")
    
    # Export results for further analysis
    export_results_to_csv(strategy, "strategy_results")
    
    print("\nAnalysis complete! Check the generated files:")
    print("- strategy_performance.png: Performance charts")
    print("- strategy_results_portfolio.csv: Portfolio history")
    print("- strategy_results_trades.csv: Trade history")
