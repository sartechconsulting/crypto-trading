"""
Test the grid trading strategy across different time periods to see 
how it performs in various market conditions.
"""

# Import directly from the main script
exec(open('method-analysis.py').read())

def test_different_periods():
    """Test strategy across different market conditions"""
    
    config = GridTradingConfig(
        initial_capital=5000.0,
        price_range_low=3000.0,
        price_range_high=10000.0,
        volatility_threshold=0.05,
        target_allocation=0.5,
        transaction_fee=0.001
    )
    
    # Define test periods with different market characteristics
    test_periods = [
        {
            "name": "Early ETH (2015-2018)",
            "start": "2015-08-01", 
            "end": "2018-12-31",
            "description": "ETH launch through first major bubble and crash"
        },
        {
            "name": "Crypto Winter (2018-2020)", 
            "start": "2018-01-01", 
            "end": "2020-03-01",
            "description": "Bear market and consolidation period"
        },
        {
            "name": "COVID Bull Run (2020-2022)",
            "start": "2020-03-01", 
            "end": "2022-01-01", 
            "description": "Massive bull market from COVID lows"
        },
        {
            "name": "Recent Bear/Recovery (2022-2024)",
            "start": "2022-01-01", 
            "end": "2024-01-01",
            "description": "Bear market and recovery"
        },
        {
            "name": "Full History (2015-2025)",
            "start": "2015-08-01", 
            "end": "2025-08-24",
            "description": "Complete ETH trading history"
        },
        {
            "name": "Last 3 Years (2022-2025)",
            "start": "2022-01-01", 
            "end": "2025-08-24",
            "description": "Recent volatile period"
        }
    ]
    
    print("=" * 100)
    print("GRID TRADING STRATEGY - MULTI-PERIOD ANALYSIS")
    print("=" * 100)
    print("Period                    | Start Price | End Price | Strategy | Buy&Hold | Excess | Sharpe | Trades")
    print("-" * 95)
    
    results = {}
    
    for period in test_periods:
        try:
            strategy = run_backtest(config, start_date=period["start"], end_date=period["end"])
            metrics = calculate_performance_metrics(strategy)
            
            # Get price data for the period to show start/end prices
            df = load_eth_data("/Users/jesse/repos/crypto-trading/data/eth-prices.csv")
            start_date_obj = datetime.strptime(period["start"], "%Y-%m-%d").date()
            end_date_obj = datetime.strptime(period["end"], "%Y-%m-%d").date()
            period_df = df.filter(
                (pl.col("date") >= start_date_obj) & 
                (pl.col("date") <= end_date_obj)
            )
            
            if len(period_df) > 0:
                start_price = period_df["price"][0]
                end_price = period_df["price"][-1]
            else:
                start_price = end_price = 0
            
            results[period["name"]] = {
                "metrics": metrics,
                "start_price": start_price,
                "end_price": end_price,
                "description": period["description"]
            }
            
            print(f"{period['name']:<25} | ${start_price:>9.2f} | ${end_price:>8.2f} | "
                  f"{metrics['total_return']:>7.1%} | {metrics['buy_hold_return']:>7.1%} | "
                  f"{metrics['excess_return']:>6.1%} | {metrics['sharpe_ratio']:>5.2f} | {metrics['total_trades']:>6d}")
            
        except Exception as e:
            print(f"{period['name']:<25} | ERROR: {str(e)}")
    
    # Analysis of results
    print("\n" + "=" * 100)
    print("KEY INSIGHTS:")
    print("=" * 100)
    
    # Find periods where strategy outperformed
    outperformed = {name: data for name, data in results.items() 
                   if data["metrics"]["excess_return"] > 0}
    
    if outperformed:
        print("\n🎯 PERIODS WHERE GRID TRADING OUTPERFORMED BUY & HOLD:")
        for name, data in outperformed.items():
            print(f"  • {name}: {data['metrics']['excess_return']:+.1%} excess return")
            print(f"    {data['description']}")
    else:
        print("\n📊 Grid trading underperformed buy & hold in all tested periods.")
        print("   This is expected in strong bull markets where prices trend strongly upward.")
    
    # Best performing periods for strategy
    best_return = max(results.keys(), key=lambda x: results[x]["metrics"]["total_return"])
    best_sharpe = max(results.keys(), key=lambda x: results[x]["metrics"]["sharpe_ratio"])
    
    print(f"\n🏆 BEST STRATEGY PERFORMANCE:")
    print(f"  • Highest Return: {best_return} ({results[best_return]['metrics']['total_return']:.1%})")
    print(f"  • Best Risk-Adjusted: {best_sharpe} (Sharpe: {results[best_sharpe]['metrics']['sharpe_ratio']:.2f})")
    
    # Market condition analysis
    print(f"\n📈 MARKET CONDITION ANALYSIS:")
    for name, data in results.items():
        start_p = data["start_price"]
        end_p = data["end_price"]
        if start_p > 0 and end_p > 0:
            price_change = (end_p - start_p) / start_p
            market_type = "Bull" if price_change > 0.5 else "Bear" if price_change < -0.2 else "Sideways"
            print(f"  • {name}: {market_type} market ({price_change:+.1%} price change)")
    
    return results


def test_bear_market_focus():
    """Focus on bear market periods where grid trading might excel"""
    
    print("\n" + "=" * 80)
    print("BEAR MARKET FOCUSED ANALYSIS")
    print("=" * 80)
    
    # Test with crypto winter optimized config
    bear_config = GridTradingConfig(
        initial_capital=5000.0,
        price_range_low=1000.0,        # Lower range for bear market
        price_range_high=6000.0,
        volatility_threshold=0.06,     # 6% threshold
        target_allocation=0.3,         # Conservative 30% ETH
        max_position_size=0.15,
        transaction_fee=0.001
    )
    
    bear_periods = [
        ("Crypto Winter 2018", "2018-01-01", "2019-12-31"),
        ("COVID Crash", "2020-02-01", "2020-05-01"), 
        ("2022 Bear Market", "2021-11-01", "2022-12-31"),
    ]
    
    for name, start, end in bear_periods:
        strategy = run_backtest(bear_config, start_date=start, end_date=end)
        metrics = calculate_performance_metrics(strategy)
        
        print(f"\n{name}:")
        print(f"  Strategy Return: {metrics['total_return']:>8.1%}")
        print(f"  Buy & Hold Return: {metrics['buy_hold_return']:>6.1%}") 
        print(f"  Excess Return: {metrics['excess_return']:>10.1%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:>12.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:>10.1%}")


if __name__ == "__main__":
    results = test_different_periods()
    test_bear_market_focus()
    
    print("\n" + "=" * 80)
    print("HOW TO CUSTOMIZE DATE RANGES:")
    print("=" * 80)
    print("1. Edit method-analysis.py lines 532 and 550:")
    print('   strategy = run_backtest(config, start_date="2018-01-01", end_date="2022-12-31")')
    print("")
    print("2. Or create custom test:")
    print("   strategy = run_backtest(config, start_date='2022-01-01')")
    print("   # No end_date = runs through all available data")
    print("")
    print("3. Available date range in your data: 2015-08-07 to 2025-08-24")
