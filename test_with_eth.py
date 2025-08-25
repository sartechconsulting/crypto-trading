"""
Test grid trading strategy starting with different amounts of ETH holdings.
This gives a more realistic comparison since many people already own some ETH.
"""

# Import directly from the main script
exec(open('method-analysis.py').read())

def test_different_starting_eth():
    """Test strategy with different starting ETH amounts"""
    
    # Test periods where grid trading performed well
    test_periods = [
        ("Crypto Winter 2018-2020", "2018-01-01", "2020-12-31"),
        ("Recent Volatility 2022-2024", "2022-01-01", "2024-12-31"),
        ("2022 Bear Market", "2021-11-01", "2022-12-31"),
    ]
    
    # Different starting ETH amounts to test
    eth_amounts = [0.0, 1.0, 2.0, 5.0]
    
    for period_name, start_date, end_date in test_periods:
        print("=" * 80)
        print(f"TESTING PERIOD: {period_name}")
        print("=" * 80)
        print("ETH Start | Total Return | Buy&Hold | Excess | Portfolio Start | Portfolio End")
        print("-" * 75)
        
        for eth_amount in eth_amounts:
            config = GridTradingConfig(
                initial_capital=5000.0,
                initial_eth_holdings=eth_amount,
                price_range_low=3000.0,
                price_range_high=10000.0,
                volatility_threshold=0.05,
                target_allocation=0.5,
                transaction_fee=0.001
            )
            
            try:
                strategy = run_backtest(config, start_date=start_date, end_date=end_date)
                metrics = calculate_performance_metrics(strategy)
                
                # Get starting portfolio value
                portfolio_df = pl.DataFrame(strategy.portfolio_history)
                start_value = portfolio_df["total_value"][0]
                
                print(f"{eth_amount:>8.1f} | {metrics['total_return']:>10.1%} | "
                      f"{metrics['buy_hold_return']:>7.1%} | {metrics['excess_return']:>6.1%} | "
                      f"${start_value:>12,.0f} | ${metrics['final_value']:>11,.0f}")
                
            except Exception as e:
                print(f"{eth_amount:>8.1f} | ERROR: {str(e)}")
        
        print()


def compare_2_eth_vs_all_cash():
    """Compare starting with 2 ETH vs starting with all cash"""
    
    print("=" * 90)
    print("COMPARISON: Starting with 2 ETH vs Starting with All Cash")
    print("=" * 90)
    
    # Test on recent volatile period
    start_date = "2022-01-01"
    end_date = "2024-12-31"
    
    # Configuration with 2 ETH start
    config_with_eth = GridTradingConfig(
        initial_capital=5000.0,
        initial_eth_holdings=2.0,
        price_range_low=3000.0,
        price_range_high=10000.0,
        volatility_threshold=0.05,
        target_allocation=0.5,
        transaction_fee=0.001
    )
    
    # Configuration with all cash
    config_all_cash = GridTradingConfig(
        initial_capital=5000.0,
        initial_eth_holdings=0.0,
        price_range_low=3000.0,
        price_range_high=10000.0,
        volatility_threshold=0.05,
        target_allocation=0.5,
        transaction_fee=0.001
    )
    
    print(f"Testing period: {start_date} to {end_date}")
    print()
    
    # Run both strategies
    strategy_with_eth = run_backtest(config_with_eth, start_date=start_date, end_date=end_date)
    strategy_all_cash = run_backtest(config_all_cash, start_date=start_date, end_date=end_date)
    
    metrics_with_eth = calculate_performance_metrics(strategy_with_eth)
    metrics_all_cash = calculate_performance_metrics(strategy_all_cash)
    
    # Get starting values
    portfolio_df_eth = pl.DataFrame(strategy_with_eth.portfolio_history)
    portfolio_df_cash = pl.DataFrame(strategy_all_cash.portfolio_history)
    start_value_eth = portfolio_df_eth["total_value"][0]
    start_value_cash = portfolio_df_cash["total_value"][0]
    
    print("Strategy Comparison:")
    print("-" * 60)
    print(f"{'Metric':<25} | {'2 ETH Start':<15} | {'All Cash Start':<15}")
    print("-" * 60)
    print(f"{'Starting Portfolio':<25} | ${start_value_eth:<14,.0f} | ${start_value_cash:<14,.0f}")
    print(f"{'Final Portfolio':<25} | ${metrics_with_eth['final_value']:<14,.0f} | ${metrics_all_cash['final_value']:<14,.0f}")
    print(f"{'Total Return':<25} | {metrics_with_eth['total_return']:<14.1%} | {metrics_all_cash['total_return']:<14.1%}")
    print(f"{'Buy & Hold Return':<25} | {metrics_with_eth['buy_hold_return']:<14.1%} | {metrics_all_cash['buy_hold_return']:<14.1%}")
    print(f"{'Excess Return':<25} | {metrics_with_eth['excess_return']:<14.1%} | {metrics_all_cash['excess_return']:<14.1%}")
    print(f"{'Sharpe Ratio':<25} | {metrics_with_eth['sharpe_ratio']:<14.2f} | {metrics_all_cash['sharpe_ratio']:<14.2f}")
    print(f"{'Total Trades':<25} | {metrics_with_eth['total_trades']:<14d} | {metrics_all_cash['total_trades']:<14d}")
    print("-" * 60)
    
    # Determine winner
    if metrics_with_eth['total_return'] > metrics_all_cash['total_return']:
        winner = "Starting with 2 ETH"
        advantage = metrics_with_eth['total_return'] - metrics_all_cash['total_return']
    else:
        winner = "Starting with all cash"
        advantage = metrics_all_cash['total_return'] - metrics_with_eth['total_return']
    
    print(f"\n🏆 Winner: {winner} (by {advantage:+.1%})")
    
    return strategy_with_eth, strategy_all_cash


def test_optimal_starting_eth():
    """Find the optimal starting ETH amount for recent period"""
    
    print("\n" + "=" * 80)
    print("FINDING OPTIMAL STARTING ETH AMOUNT (2022-2024)")
    print("=" * 80)
    
    eth_amounts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    results = []
    
    for eth_amount in eth_amounts:
        config = GridTradingConfig(
            initial_capital=5000.0,
            initial_eth_holdings=eth_amount,
            price_range_low=3000.0,
            price_range_high=10000.0,
            volatility_threshold=0.05,
            target_allocation=0.5,
            transaction_fee=0.001
        )
        
        strategy = run_backtest(config, start_date="2022-01-01", end_date="2024-12-31")
        metrics = calculate_performance_metrics(strategy)
        
        results.append({
            'eth_amount': eth_amount,
            'total_return': metrics['total_return'],
            'excess_return': metrics['excess_return'],
            'sharpe_ratio': metrics['sharpe_ratio']
        })
    
    print("ETH Start | Total Return | Excess Return | Sharpe Ratio")
    print("-" * 55)
    for result in results:
        print(f"{result['eth_amount']:>8.1f} | {result['total_return']:>10.1%} | "
              f"{result['excess_return']:>11.1%} | {result['sharpe_ratio']:>10.2f}")
    
    # Find optimal amounts
    best_return = max(results, key=lambda x: x['total_return'])
    best_excess = max(results, key=lambda x: x['excess_return'])
    best_sharpe = max(results, key=lambda x: x['sharpe_ratio'])
    
    print(f"\n🎯 Best Total Return: {best_return['eth_amount']} ETH ({best_return['total_return']:.1%})")
    print(f"🎯 Best Excess Return: {best_excess['eth_amount']} ETH ({best_excess['excess_return']:+.1%})")
    print(f"🎯 Best Sharpe Ratio: {best_sharpe['eth_amount']} ETH ({best_sharpe['sharpe_ratio']:.2f})")


if __name__ == "__main__":
    # Test different starting ETH amounts across various periods
    test_different_starting_eth()
    
    # Compare 2 ETH vs all cash in detail
    print("\n")
    strategy_eth, strategy_cash = compare_2_eth_vs_all_cash()
    
    # Find optimal starting ETH amount
    test_optimal_starting_eth()
    
    print("\n" + "=" * 80)
    print("HOW TO USE STARTING ETH HOLDINGS:")
    print("=" * 80)
    print("1. Modify GridTradingConfig:")
    print("   config = GridTradingConfig(")
    print("       initial_capital=5000.0,")
    print("       initial_eth_holdings=2.0,  # Start with 2 ETH")
    print("       # ... other parameters")
    print("   )")
    print("")
    print("2. The strategy will:")
    print("   - Start with your specified ETH + cash")
    print("   - Calculate initial portfolio value correctly")
    print("   - Compare performance vs buying all ETH at start")
    print("   - Show realistic results for existing ETH holders")
