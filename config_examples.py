"""
Configuration examples for the grid trading strategy.

This file contains different configuration presets you can use to experiment with 
various trading strategies. Each configuration targets a different approach to
grid trading and volatility harvesting.
"""

# Import directly from the main script
exec(open('method-analysis.py').read())


def conservative_config():
    """Conservative strategy with lower volatility threshold and smaller position sizes"""
    return GridTradingConfig(
        initial_capital=5000.0,
        price_range_low=2000.0,
        price_range_high=12000.0,
        num_grids=30,
        volatility_threshold=0.03,    # 3% threshold - more sensitive
        max_position_size=0.05,       # Max 5% per trade - conservative
        min_position_size=0.01,       # Min 1% per trade
        target_allocation=0.4,        # 40% ETH, 60% cash - conservative
        rebalance_threshold=0.15,     # 15% deviation before rebalancing
        transaction_fee=0.001
    )


def aggressive_config():
    """Aggressive strategy with higher volatility threshold and larger position sizes"""
    return GridTradingConfig(
        initial_capital=5000.0,
        price_range_low=1500.0,
        price_range_high=15000.0,
        num_grids=15,
        volatility_threshold=0.08,    # 8% threshold - less sensitive
        max_position_size=0.20,       # Max 20% per trade - aggressive
        min_position_size=0.02,       # Min 2% per trade
        target_allocation=0.6,        # 60% ETH, 40% cash - aggressive
        rebalance_threshold=0.05,     # 5% deviation before rebalancing
        transaction_fee=0.001
    )


def high_frequency_config():
    """High frequency trading with very low volatility threshold"""
    return GridTradingConfig(
        initial_capital=5000.0,
        price_range_low=3000.0,
        price_range_high=10000.0,
        num_grids=50,
        volatility_threshold=0.02,    # 2% threshold - very sensitive
        max_position_size=0.08,       # Max 8% per trade
        min_position_size=0.005,      # Min 0.5% per trade
        target_allocation=0.5,        # 50% ETH, 50% cash
        rebalance_threshold=0.08,     # 8% deviation before rebalancing
        transaction_fee=0.001
    )


def large_range_config():
    """Strategy for capturing large price movements with wider range"""
    return GridTradingConfig(
        initial_capital=5000.0,
        price_range_low=500.0,        # Very wide range
        price_range_high=20000.0,
        num_grids=25,
        volatility_threshold=0.07,    # 7% threshold
        max_position_size=0.15,       # Max 15% per trade
        min_position_size=0.01,       # Min 1% per trade
        target_allocation=0.5,        # 50% ETH, 50% cash
        rebalance_threshold=0.12,     # 12% deviation before rebalancing
        transaction_fee=0.001
    )


def crypto_winter_config():
    """Strategy optimized for bear markets with lower price ranges"""
    return GridTradingConfig(
        initial_capital=5000.0,
        price_range_low=800.0,        # Lower range for bear market
        price_range_high=5000.0,
        num_grids=20,
        volatility_threshold=0.06,    # 6% threshold
        max_position_size=0.12,       # Max 12% per trade
        min_position_size=0.015,      # Min 1.5% per trade
        target_allocation=0.3,        # 30% ETH, 70% cash - very conservative
        rebalance_threshold=0.20,     # 20% deviation before rebalancing
        transaction_fee=0.001
    )


def run_config_comparison():
    """Compare all configuration strategies"""
    configs = {
        "Conservative": conservative_config(),
        "Aggressive": aggressive_config(),
        "High Frequency": high_frequency_config(),
        "Large Range": large_range_config(),
        "Crypto Winter": crypto_winter_config()
    }
    
    print("=" * 80)
    print("CONFIGURATION COMPARISON ANALYSIS")
    print("=" * 80)
    print("Strategy       | Total Return | Sharpe | Max DD | Trades | Excess Return")
    print("-" * 75)
    
    results = {}
    
    for name, config in configs.items():
        strategy = run_backtest(config, start_date="2020-01-01")
        metrics = calculate_performance_metrics(strategy)
        results[name] = metrics
        
        print(f"{name:<14} | {metrics['total_return']:>10.1%} | {metrics['sharpe_ratio']:>6.2f} | "
              f"{metrics['max_drawdown']:>6.1%} | {metrics['total_trades']:>6d} | {metrics['excess_return']:>10.1%}")
    
    # Find best performing strategy
    best_return = max(results.keys(), key=lambda x: results[x]['total_return'])
    best_sharpe = max(results.keys(), key=lambda x: results[x]['sharpe_ratio'])
    best_excess = max(results.keys(), key=lambda x: results[x]['excess_return'])
    
    print("\n" + "=" * 80)
    print("BEST PERFORMERS:")
    print(f"Highest Total Return: {best_return} ({results[best_return]['total_return']:.1%})")
    print(f"Best Sharpe Ratio: {best_sharpe} ({results[best_sharpe]['sharpe_ratio']:.2f})")
    print(f"Best Excess Return vs B&H: {best_excess} ({results[best_excess]['excess_return']:.1%})")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Run comparison of all configurations
    run_config_comparison()
    
    # You can also test individual configurations:
    print("\n\nDetailed analysis of Conservative configuration:")
    config = conservative_config()
    strategy = run_backtest(config, start_date="2020-01-01")
    metrics = calculate_performance_metrics(strategy)
    print_performance_report(metrics, config)
