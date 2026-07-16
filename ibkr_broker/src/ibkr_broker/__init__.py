"""
ibkr_broker — Simple wrapper for Interactive Brokers TWS paper/live trading.

Usage:
    from ibkr_broker import Broker, OrderInfo, Position

    b = Broker()
    b.connect()
    b.buy("AAPL", 100)
    b.disconnect()
"""

from ibkr_broker.broker import Broker, OrderInfo, Position

__all__ = ["Broker", "OrderInfo", "Position"]
