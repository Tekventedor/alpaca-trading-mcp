#!/usr/bin/env python3
"""
Alpaca Trading MCP Server

This server provides trading and portfolio management capabilities
through the Alpaca API using the official MCP Python SDK.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

# Load environment variables
load_dotenv()

# Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER_TRADE = os.getenv("ALPACA_PAPER_TRADE", "True").lower() == "true"

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables are required")


class AlpacaMCPServer:
    """MCP Server for Alpaca Trading API"""

    def __init__(self):
        self.server = Server(
            name="alpaca-trading",
            version="1.0.0"
        )

        # Initialize Alpaca clients
        self.trading_client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=ALPACA_PAPER_TRADE
        )

        self.data_client = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY
        )

        self._register_handlers()

    def _register_handlers(self):
        """Register tool handlers with the MCP server"""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List all available tools"""
            return [
                Tool(
                    name="get_account_info",
                    description="Get account information including balance and buying power",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_positions",
                    description="Get all current positions in the portfolio",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_orders",
                    description="Get orders with optional status filter",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "description": "Filter orders by status (open, closed, all)",
                                "enum": ["open", "closed", "all"],
                                "default": "open"
                            }
                        }
                    }
                ),
                Tool(
                    name="place_market_order",
                    description="Place a market order to buy or sell a stock",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol"
                            },
                            "quantity": {
                                "type": "number",
                                "description": "Number of shares to trade"
                            },
                            "side": {
                                "type": "string",
                                "description": "Order side (buy or sell)",
                                "enum": ["buy", "sell"]
                            }
                        },
                        "required": ["symbol", "quantity", "side"]
                    }
                ),
                Tool(
                    name="place_limit_order",
                    description="Place a limit order with a specific price",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol"
                            },
                            "quantity": {
                                "type": "number",
                                "description": "Number of shares to trade"
                            },
                            "side": {
                                "type": "string",
                                "description": "Order side (buy or sell)",
                                "enum": ["buy", "sell"]
                            },
                            "limit_price": {
                                "type": "number",
                                "description": "Limit price for the order"
                            }
                        },
                        "required": ["symbol", "quantity", "side", "limit_price"]
                    }
                ),
                Tool(
                    name="cancel_order",
                    description="Cancel an open order by its ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "order_id": {
                                "type": "string",
                                "description": "The ID of the order to cancel"
                            }
                        },
                        "required": ["order_id"]
                    }
                ),
                Tool(
                    name="get_stock_quote",
                    description="Get the latest quote for a stock",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol"
                            }
                        },
                        "required": ["symbol"]
                    }
                ),
                Tool(
                    name="close_position",
                    description="Close a position for a specific symbol",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol to close position for"
                            }
                        },
                        "required": ["symbol"]
                    }
                ),
                Tool(
                    name="close_all_positions",
                    description="Close all open positions in the account",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "cancel_orders": {
                                "type": "boolean",
                                "description": "Also cancel all open orders",
                                "default": False
                            }
                        }
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Execute a tool with the given arguments"""

            try:
                # Run synchronous Alpaca operations in executor to avoid blocking
                loop = asyncio.get_event_loop()

                if name == "get_account_info":
                    result = await loop.run_in_executor(None, self._get_account_info)
                elif name == "get_positions":
                    result = await loop.run_in_executor(None, self._get_positions)
                elif name == "get_orders":
                    result = await loop.run_in_executor(
                        None,
                        self._get_orders,
                        arguments.get("status", "open")
                    )
                elif name == "place_market_order":
                    result = await loop.run_in_executor(
                        None,
                        self._place_market_order,
                        arguments["symbol"],
                        arguments["quantity"],
                        arguments["side"]
                    )
                elif name == "place_limit_order":
                    result = await loop.run_in_executor(
                        None,
                        self._place_limit_order,
                        arguments["symbol"],
                        arguments["quantity"],
                        arguments["side"],
                        arguments["limit_price"]
                    )
                elif name == "cancel_order":
                    result = await loop.run_in_executor(
                        None,
                        self._cancel_order,
                        arguments["order_id"]
                    )
                elif name == "get_stock_quote":
                    result = await loop.run_in_executor(
                        None,
                        self._get_stock_quote,
                        arguments["symbol"]
                    )
                elif name == "close_position":
                    result = await loop.run_in_executor(
                        None,
                        self._close_position,
                        arguments["symbol"]
                    )
                elif name == "close_all_positions":
                    result = await loop.run_in_executor(
                        None,
                        self._close_all_positions,
                        arguments.get("cancel_orders", False)
                    )
                else:
                    result = {"error": f"Unknown tool: {name}"}

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}, indent=2)
                )]

    def _get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            account = self.trading_client.get_account()
            return {
                "account_number": account.account_number,
                "status": account.status,
                "currency": account.currency,
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "transfers_blocked": account.transfers_blocked,
                "account_blocked": account.account_blocked,
                "trade_suspended_by_user": account.trade_suspended_by_user,
                "daytrade_count": account.daytrade_count,
                "daytrading_buying_power": float(account.daytrading_buying_power) if account.daytrading_buying_power else 0
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions"""
        try:
            positions = self.trading_client.get_all_positions()
            result = []
            for position in positions:
                result.append({
                    "symbol": position.symbol,
                    "quantity": float(position.qty),
                    "avg_entry_price": float(position.avg_entry_price),
                    "market_value": float(position.market_value),
                    "cost_basis": float(position.cost_basis),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                    "current_price": float(position.current_price) if position.current_price else None,
                    "lastday_price": float(position.lastday_price) if position.lastday_price else None,
                    "change_today": float(position.change_today) if position.change_today else None,
                    "side": position.side
                })
            return result
        except Exception as e:
            return [{"error": str(e)}]

    def _get_orders(self, status: str = "open") -> List[Dict[str, Any]]:
        """Get orders with optional status filter"""
        try:
            request_params = GetOrdersRequest()
            if status == "open":
                request_params.status = QueryOrderStatus.OPEN
            elif status == "closed":
                request_params.status = QueryOrderStatus.CLOSED

            orders = self.trading_client.get_orders(filter=request_params)
            result = []
            for order in orders:
                result.append({
                    "id": order.id,
                    "symbol": order.symbol,
                    "quantity": float(order.qty),
                    "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                    "side": order.side,
                    "order_type": order.order_type,
                    "time_in_force": order.time_in_force,
                    "limit_price": float(order.limit_price) if order.limit_price else None,
                    "stop_price": float(order.stop_price) if order.stop_price else None,
                    "status": order.status,
                    "created_at": str(order.created_at),
                    "updated_at": str(order.updated_at),
                    "filled_at": str(order.filled_at) if order.filled_at else None,
                    "expired_at": str(order.expired_at) if order.expired_at else None,
                    "canceled_at": str(order.canceled_at) if order.canceled_at else None,
                    "failed_at": str(order.failed_at) if order.failed_at else None,
                    "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None
                })
            return result
        except Exception as e:
            return [{"error": str(e)}]

    def _place_market_order(self, symbol: str, quantity: float, side: str) -> Dict[str, Any]:
        """Place a market order"""
        try:
            market_order_data = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )

            order = self.trading_client.submit_order(order_data=market_order_data)

            return {
                "success": True,
                "order_id": order.id,
                "symbol": order.symbol,
                "quantity": float(order.qty),
                "side": order.side,
                "order_type": order.order_type,
                "time_in_force": order.time_in_force,
                "status": order.status,
                "created_at": str(order.created_at)
            }
        except Exception as e:
            return {"error": str(e)}

    def _place_limit_order(self, symbol: str, quantity: float, side: str, limit_price: float) -> Dict[str, Any]:
        """Place a limit order"""
        try:
            limit_order_data = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price
            )

            order = self.trading_client.submit_order(order_data=limit_order_data)

            return {
                "success": True,
                "order_id": order.id,
                "symbol": order.symbol,
                "quantity": float(order.qty),
                "side": order.side,
                "order_type": order.order_type,
                "time_in_force": order.time_in_force,
                "limit_price": float(order.limit_price),
                "status": order.status,
                "created_at": str(order.created_at)
            }
        except Exception as e:
            return {"error": str(e)}

    def _cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return {
                "success": True,
                "message": f"Order {order_id} cancelled successfully"
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """Get latest stock quote"""
        try:
            request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request_params)
            quote = quotes[symbol]

            return {
                "symbol": symbol,
                "ask_price": float(quote.ask_price),
                "ask_size": int(quote.ask_size),
                "bid_price": float(quote.bid_price),
                "bid_size": int(quote.bid_size),
                "timestamp": str(quote.timestamp)
            }
        except Exception as e:
            return {"error": str(e)}

    def _close_position(self, symbol: str) -> Dict[str, Any]:
        """Close a position for a specific symbol"""
        try:
            self.trading_client.close_position(symbol)
            return {
                "success": True,
                "message": f"Position for {symbol} closed successfully"
            }
        except Exception as e:
            return {"error": str(e)}

    def _close_all_positions(self, cancel_orders: bool = False) -> Dict[str, Any]:
        """Close all positions"""
        try:
            self.trading_client.close_all_positions(cancel_orders)
            return {
                "success": True,
                "message": "All positions closed successfully",
                "orders_cancelled": cancel_orders
            }
        except Exception as e:
            return {"error": str(e)}

    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point"""
    server = AlpacaMCPServer()
    await server.run()


def run():
    """Synchronous entry point"""
    asyncio.run(main())


if __name__ == "__main__":
    run()