"""
Demo script showing different dashboard configurations.
Run this to see sample configurations for the Streamlit dashboard.
"""

def show_demo_configs():
    """Display different configuration examples for the dashboard"""
    
    print("🎯 Grid Trading Strategy Dashboard - Demo Configurations")
    print("=" * 70)
    
    configs = {
        "Conservative Bear Market Strategy": {
            "description": "Low risk approach for volatile markets",
            "settings": {
                "Initial Cash": "$5,000",
                "Initial ETH": "1.0 ETH",
                "Price Range": "$2,000 - $8,000",
                "Volatility Threshold": "3%",
                "Max Position Size": "5%",
                "Target ETH Allocation": "30%",
                "Date Range": "Crypto Winter (2018-2020)"
            },
            "expected": "Should outperform buy & hold significantly in bear markets"
        },
        
        "Aggressive Bull Market Strategy": {
            "description": "Higher risk approach for trending markets",
            "settings": {
                "Initial Cash": "$10,000", 
                "Initial ETH": "2.0 ETH",
                "Price Range": "$3,000 - $15,000",
                "Volatility Threshold": "7%",
                "Max Position Size": "15%",
                "Target ETH Allocation": "70%",
                "Date Range": "Recent Volatility (2022-2024)"
            },
            "expected": "Higher returns but more volatile performance"
        },
        
        "High Frequency Scalping": {
            "description": "Frequent small trades to capture volatility",
            "settings": {
                "Initial Cash": "$5,000",
                "Initial ETH": "0.5 ETH", 
                "Price Range": "$3,500 - $6,000",
                "Volatility Threshold": "2%",
                "Max Position Size": "8%",
                "Target ETH Allocation": "50%",
                "Date Range": "2022 Bear Market"
            },
            "expected": "Many trades, steady accumulation of small gains"
        },
        
        "Realistic Holder Strategy": {
            "description": "For someone who already owns ETH and wants to optimize",
            "settings": {
                "Initial Cash": "$5,000",
                "Initial ETH": "3.0 ETH",
                "Price Range": "$3,000 - $10,000", 
                "Volatility Threshold": "5%",
                "Max Position Size": "10%",
                "Target ETH Allocation": "60%",
                "Date Range": "Recent Volatility (2022-2024)"
            },
            "expected": "Realistic performance for existing ETH holders"
        }
    }
    
    for name, config in configs.items():
        print(f"\n📊 {name}")
        print("-" * len(name))
        print(f"💡 {config['description']}")
        print("\n⚙️ Settings:")
        for setting, value in config['settings'].items():
            print(f"   • {setting}: {value}")
        print(f"\n🎯 Expected: {config['expected']}")
        print()
    
    print("🚀 To try these configurations:")
    print("1. Run: streamlit run streamlit_app.py")
    print("2. Adjust the sidebar settings to match the above")
    print("3. Click 'Run Strategy' to see results")
    print("4. Compare different configurations side by side")
    
    print("\n💡 Pro Tips for the Dashboard:")
    print("• Use 'Crypto Winter (2018-2020)' to see grid trading at its best")
    print("• Try different starting ETH amounts (0, 1, 2, 5) to see the impact")
    print("• Lower volatility thresholds = more frequent trading")
    print("• Higher target ETH allocation = more aggressive strategy")
    print("• Watch the excess return metric - positive means beating buy & hold!")

if __name__ == "__main__":
    show_demo_configs()
