"""
Quick script to test any date range you want.
Just modify the start_date and end_date below and run: uv run quick_test.py
"""

# Import directly from the main script
exec(open('method-analysis.py').read())

if __name__ == "__main__":
    # *** MODIFY THESE DATES TO TEST DIFFERENT PERIODS ***
    START_DATE = "2022-01-01"  # Change this
    END_DATE = "2024-12-31"    # Change this (or set to None for all available data)
    
    # Use default config or customize it
    config = GridTradingConfig(
        initial_capital=5000.0,
        initial_eth_holdings=2.0,   # *** CHANGE THIS to test different starting ETH amounts ***
        price_range_low=3000.0,
        price_range_high=10000.0,
        volatility_threshold=0.05,
        target_allocation=0.5,
        transaction_fee=0.001
    )
    
    print(f"Testing period: {START_DATE} to {END_DATE or 'present'}")
    print("=" * 60)
    
    # Run the backtest
    strategy = run_backtest(config, start_date=START_DATE, end_date=END_DATE)
    metrics = calculate_performance_metrics(strategy)
    
    # Show results
    print_performance_report(metrics, config)
    
    # Quick summary
    if metrics['excess_return'] > 0:
        print(f"\n🎯 GRID TRADING WINS! {metrics['excess_return']:+.1%} excess return")
    else:
        print(f"\n📈 Buy & Hold wins by {-metrics['excess_return']:.1%}")
        
    print(f"Strategy Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
