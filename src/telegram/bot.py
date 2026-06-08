import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.config.settings import settings
from src.trading.engine import TradingEngine
from src.utils.redis_client import get_redis_client
from src.database import set_telegram_chat_id, get_telegram_chat_id, get_news_for_symbol
from src.llm.prompts import _format_news_for_prompt
from src.llm.cache import get_cached_llm_response

logger = logging.getLogger(__name__)

class TelegramBot:
    _log_lock = threading.Lock()
    MAX_LOG_SIZE = 512 * 1024   # 512 KB
    MAX_LOG_BACKUPS = 10

    def __init__(self, engine: TradingEngine):
        self.engine = engine
        self.redis = get_redis_client()
        # Allowed chat ID – bot will only respond to this chat
        self.allowed_chat_id = None
        if settings.TELEGRAM_CHAT_ID:
            try:
                self.allowed_chat_id = int(settings.TELEGRAM_CHAT_ID)
            except ValueError:
                logger.error("TELEGRAM_CHAT_ID must be a valid integer")
        else:
            logger.warning("TELEGRAM_CHAT_ID not set. Bot will not respond to any chat.")
        self.app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
        self._register_handlers()
        self.keyboard = ReplyKeyboardMarkup(
            [
                [KeyboardButton("📊 Status"), KeyboardButton("📈 Trades")],
                [KeyboardButton("💰 Profit"), KeyboardButton("🚀 Performance")],
                [KeyboardButton("📰 News"), KeyboardButton("⚠️ Risk")],
                [KeyboardButton("⏸️ Pause"), KeyboardButton("▶️ Resume")],
                [KeyboardButton("🔄 Reload"), KeyboardButton("💸 Sell All")],
            ],
            resize_keyboard=True,
        )

    def _is_authorized(self, update: Update) -> bool:
        """Return True if the update comes from the allowed chat ID."""
        if self.allowed_chat_id is None:
            return False
        return update.effective_chat.id == self.allowed_chat_id

    def _register_handlers(self):
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("menu", self.cmd_menu))
        self.app.add_handler(CommandHandler("pause", self.cmd_pause))
        self.app.add_handler(CommandHandler("resume", self.cmd_resume))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("trades", self.cmd_trades))
        self.app.add_handler(CommandHandler("profit", self.cmd_profit))
        self.app.add_handler(CommandHandler("performance", self.cmd_performance))
        self.app.add_handler(CommandHandler("news", self.cmd_news_search))
        self.app.add_handler(CommandHandler("news_status", self.cmd_news_status))
        self.app.add_handler(CommandHandler("risk", self.cmd_risk))
        self.app.add_handler(CommandHandler("reload", self.cmd_reload))
        self.app.add_handler(CommandHandler("sell", self.cmd_sell))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_button))

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        chat_id = update.effective_chat.id
        await asyncio.to_thread(set_telegram_chat_id, chat_id)
        await update.message.reply_text(
            "Bot started! You will receive trade notifications here.\nUse the buttons below or type /menu to see them again.",
            reply_markup=self.keyboard,
        )

    async def cmd_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        await update.message.reply_text("Choose an option:", reply_markup=self.keyboard)

    async def handle_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        text = update.message.text
        logger.debug(f"Received button text: {text}")
        if text == "📊 Status":
            await self.cmd_status(update, context)
        elif text == "📈 Trades":
            await self.cmd_trades(update, context)
        elif text == "💰 Profit":
            await self.cmd_profit(update, context)
        elif text == "🚀 Performance":
            await self.cmd_performance(update, context)
        elif text == "⏸️ Pause":
            await self.cmd_pause(update, context)
        elif text == "▶️ Resume":
            await self.cmd_resume(update, context)
        elif text == "📰 News":
            await self.cmd_news(update, context)
        elif text == "⚠️ Risk":
            await self.cmd_risk(update, context)
        elif text == "🔄 Reload":
            await self.cmd_reload(update, context)
        elif text == "💸 Sell All":
            await self.cmd_sell(update, context)
        else:
            # Any other text (e.g., first message "hi") shows the keyboard
            await update.message.reply_text(
                "Use the buttons below to interact with the bot.",
                reply_markup=self.keyboard,
            )

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        await asyncio.to_thread(self.redis.set, "trading:paused", "1")
        await update.message.reply_text("Trading paused.", reply_markup=self.keyboard)

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        await asyncio.to_thread(self.redis.delete, "trading:paused")
        await update.message.reply_text("Trading resumed.", reply_markup=self.keyboard)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        try:
            def get_status():
                coins = self.engine.current_coins
                positions = self.engine.positions
                balance = self.engine.trader.fetch_balance()
                return coins, positions, balance
            coins, positions, balance = await asyncio.to_thread(get_status)
        except Exception as e:
            logger.error(f"Failed to get status: {e}", exc_info=True)
            await update.message.reply_text("⚠️ Could not retrieve status.", reply_markup=self.keyboard)
            return

        msg = "<b>📊 Current Status</b>\n\n"
        llm_model = settings.OLLAMA_MODEL if settings.LLM_PROVIDER == "ollama" else settings.OPENAI_MODEL
        msg += f"<b>🧠 LLM:</b> {settings.LLM_PROVIDER} / {llm_model}\n\n"
        coin_list = []
        for entry in coins:
            symbol = entry["symbol"]
            tf = entry["timeframe"]
            coin_list.append(f"{symbol} ({tf})")
        msg += f"<b>🪙 Tracked Coins:</b> {', '.join(coin_list) if coin_list else 'None'}\n\n"

        if positions:
            msg += "<b>📈 Open Positions:</b>\n"
            for sym, pos in positions.items():
                msg += (
                    f"  • <code>{sym}</code>\n"
                    f"    Amount: {pos['amount']:.6f}\n"
                    f"    Entry: {pos['price']:.4f}\n"
                    f"    SL: {pos['stop_loss']:.4f}  TP: {pos['take_profit']:.4f}\n"
                )
        else:
            msg += "<b>📈 Open Positions:</b> None\n"

        msg += "\n<b>💰 Balances:</b>\n"
        non_zero = {k: v for k, v in balance.items() if v > 0}
        if non_zero:
            for cur, amt in non_zero.items():
                msg += f"  • {cur}: {amt:.6f}\n"
        else:
            msg += "  No balances\n"

        await update.message.reply_text(msg, parse_mode='HTML', reply_markup=self.keyboard)

    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        try:
            open_trades = await asyncio.to_thread(self.engine.get_open_trades)
        except Exception as e:
            logger.error(f"Failed to get open trades: {e}", exc_info=True)
            await update.message.reply_text("⚠️ Could not retrieve open trades.", reply_markup=self.keyboard)
            return

        if not open_trades:
            await update.message.reply_text("📈 No open trades.", reply_markup=self.keyboard)
            return

        msg = "<b>📈 Open Trades</b>\n\n"
        for idx, t in enumerate(open_trades, start=1):
            sym = t['symbol']
            amt = t['amount']
            price = t['price']
            fee = t.get('fee', {})
            fee_cost = fee.get('cost', 0) or 0
            fee_currency = fee.get('currency', '')
            fee_str = f"{fee_cost:.6f} {fee_currency}" if fee_cost else "—"

            ts = datetime.fromtimestamp(t['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')

            # Fetch current price
            current_price = None
            try:
                ticker = await asyncio.to_thread(self.engine.exchange.fetch_ticker, sym)
                current_price = ticker.get('last') if ticker else None
            except Exception as e:
                logger.warning(f"Could not fetch current price for {sym}: {e}")

            line = f"<b>#{idx}</b> 🟢 <b>BUY</b> <code>{sym}</code>\n"
            line += f"   🕒 {ts}\n"
            line += f"   Amount: {amt:.6f}  Entry: {price:.4f}"
            if current_price is not None:
                line += f"  Current: {current_price:.4f}"
            line += "\n"
            line += f"   Fee: {fee_str}\n"

            pnl = t['unrealized_pnl']
            pnl_pct = t['unrealized_pnl_pct']
            pnl_sign = "+" if pnl >= 0 else ""
            pnl_pct_sign = "+" if pnl_pct >= 0 else ""
            line += f"   Unrealized P&L: {pnl_sign}{pnl:.4f} ({pnl_pct_sign}{pnl_pct:.2f}%)"

            msg += line + "\n\n"

        await update.message.reply_text(msg, parse_mode='HTML', reply_markup=self.keyboard)

    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        """Show performance summary grouped by coin and timeframe."""
        # Check if there are any closed sell trades at all
        closed_sells = [t for t in self.engine.trade_history if t.get("side") == "sell"]
        if not closed_sells:
            await update.message.reply_text(
                "🚀 No closed sell trades yet.", reply_markup=self.keyboard
            )
            return

        try:
            perf = await asyncio.to_thread(self.engine.get_performance_summary)
            rows = perf.get("rows", [])
            total = perf.get("total", {})

            if not rows:
                await update.message.reply_text(
                    "🚀 No closed sell trades yet.", reply_markup=self.keyboard
                )
                return

            msg = "<b>🚀 Performance by Coin</b>\n\n"
            for r in rows:
                symbol = r["symbol"]
                tf = r.get("timeframe") or "—"
                trades = r["trade_count"]
                profit = r["profit"]
                profit_pct = r["profit_pct"]
                win_rate = r["win_rate"]

                profit_emoji = "📈" if profit >= 0 else "📉"
                profit_sign = "+" if profit >= 0 else ""
                msg += (
                    f"<b>{symbol}</b> ({tf})\n"
                    f"  Trades: {trades}  |  {profit_emoji} {profit_sign}{profit:.4f} ({profit_sign}{profit_pct:.2f}%)\n"
                    f"  Win Rate: {win_rate:.1f}%\n\n"
                )

            if total:
                t = total
                t_profit = t["profit"]
                t_sign = "+" if t_profit >= 0 else ""
                t_emoji = "📈" if t_profit >= 0 else "📉"
                msg += (
                    f"<b>── TOTAL ──</b>\n"
                    f"  Trades: {t['trade_count']}  |  {t_emoji} {t_sign}{t_profit:.4f} ({t_sign}{t['profit_pct']:.2f}%)\n"
                    f"  Win Rate: {t['win_rate']:.1f}%"
                )
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}", exc_info=True)
            msg = "⚠️ Could not retrieve performance summary. Please try again later."

        await update.message.reply_text(msg, parse_mode='HTML', reply_markup=self.keyboard)

    async def cmd_news_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        """Show recent news for a specific coin (e.g., /news BTC)."""
        if not context.args:
            await update.message.reply_text(
                "Usage: /news <coin>\nExample: /news BTC",
                reply_markup=self.keyboard,
            )
            return

        coin = context.args[0].upper()
        # Remove any trailing "/USDT" if user typed a pair
        if "/" in coin:
            coin = coin.split("/")[0]

        articles = get_news_for_symbol(coin, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
        if not articles:
            await update.message.reply_text(f"No recent news for {coin}.", reply_markup=self.keyboard)
            return

        formatted = _format_news_for_prompt(articles)
        msg = f"*{coin}*\n{formatted}"
        # Send as plain text to avoid Markdown parsing errors
        await update.message.reply_text(msg, parse_mode=None, reply_markup=self.keyboard)

    async def cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        try:
            metrics = self.engine.get_risk_metrics()
        except Exception as e:
            logger.error(f"Failed to get risk metrics: {e}", exc_info=True)
            await update.message.reply_text("⚠️ Could not retrieve risk metrics.", reply_markup=self.keyboard)
            return

        pf = metrics['profit_factor']
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        msg = (
            f"<b>⚠️ Risk Metrics</b>\n\n"
            f"<b>Portfolio</b>\n"
            f"💰 Balance: {metrics['current_balance']:.2f} {metrics['base_currency']}\n"
            f"🏦 Initial: {metrics['initial_balance']:.2f} {metrics['base_currency']}\n"
            f"📊 P&L: {metrics['total_pnl']:.2f} ({metrics['total_pnl_pct']:.2f}%)\n"
            f"📉 Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n\n"
            f"<b>Positions</b>\n"
            f"📈 Open: {metrics['open_positions_count']}\n"
            f"💼 Exposure: {metrics['total_exposure']:.2f} {metrics['base_currency']}\n"
            f"🔝 Largest Position: {metrics['largest_position_exposure_pct']:.1f}% of portfolio\n"
            f"⛔ Total Stop Risk: {metrics['total_stop_loss_risk']:.2f} {metrics['base_currency']}\n\n"
            f"<b>Trade Stats</b>\n"
            f"📋 Total Trades: {metrics['total_trades']}\n"
            f"🏆 Win Rate: {metrics['win_rate']:.1f}%\n"
            f"📊 Profit Factor: {pf_str}\n"
            f"🟢 Avg Win: {metrics['avg_win']:.2f}  🔴 Avg Loss: {metrics['avg_loss']:.2f}"
        )
        await update.message.reply_text(msg, parse_mode='HTML', reply_markup=self.keyboard)

    async def cmd_news(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        """Show LLM-generated news summaries for all tracked coins (same as web card)."""
        try:
            coins = self.engine.current_coins
            if not coins:
                await update.message.reply_text("No coins currently tracked.")
                return

            await update.message.reply_text("Generating news summaries...")
            messages = []
            for entry in coins:
                symbol = entry["symbol"]
                base_coin = symbol.split("/")[0] if "/" in symbol else symbol
                articles = get_news_for_symbol(base_coin, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
                if not articles:
                    summary = "No recent news."
                else:
                    try:
                        formatted = _format_news_for_prompt(articles)
                        prompt = (
                            f"Here are recent news headlines and summaries for {base_coin}:\n\n"
                            f"{formatted}\n\n"
                            "Based on these articles, write a single very short sentence (max 15 words) "
                            "that explains the overall sentiment and the main reason for it. "
                            "Do not include any other text."
                        )
                        summary = await asyncio.to_thread(get_cached_llm_response, prompt, "", ttl=300)
                        summary = summary.strip()
                        if len(summary) > 120:
                            summary = summary[:117] + "..."
                    except Exception:
                        summary = "Could not generate summary."

                messages.append(f"<b>{symbol}</b>\n{summary}")

            full_text = "\n\n".join(messages)
            # Split if too long for Telegram
            if len(full_text) > 4000:
                for i in range(0, len(full_text), 4000):
                    await update.message.reply_text(full_text[i:i+4000], parse_mode='HTML')
            else:
                await update.message.reply_text(full_text, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Failed to generate news summaries: {e}", exc_info=True)
            await update.message.reply_text("⚠️ Could not retrieve news.", reply_markup=self.keyboard)

    async def cmd_news_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        """Show news article counts for tracked coins."""
        try:
            coins = self.engine.current_coins
            if not coins:
                await update.message.reply_text("No coins currently tracked.")
                return

            msg = "<b>📰 News Article Counts</b>\n\n"
            for entry in coins:
                symbol = entry["symbol"]
                base_coin = symbol.split("/")[0] if "/" in symbol else symbol
                articles = get_news_for_symbol(base_coin, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
                msg += f"<b>{symbol}</b>: {len(articles)} articles\n"
            await update.message.reply_text(msg, parse_mode='HTML', reply_markup=self.keyboard)
        except Exception as e:
            logger.error(f"Failed to get news status: {e}", exc_info=True)
            await update.message.reply_text("⚠️ Could not retrieve news status.", reply_markup=self.keyboard)

    async def cmd_reload(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        """Hot-reload the .env file."""
        try:
            settings.reload()
            await update.message.reply_text("✅ Settings reloaded from .env file.", reply_markup=self.keyboard)
        except Exception as e:
            logger.error(f"Failed to reload settings: {e}", exc_info=True)
            await update.message.reply_text(f"⚠️ Failed to reload settings: {e}", reply_markup=self.keyboard)

    async def cmd_sell(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        """Sell all open positions, or a specific one by trade ID (e.g., /sell 2)."""
        try:
            open_trades = await asyncio.to_thread(self.engine.get_open_trades)
        except Exception as e:
            logger.error(f"Failed to get open trades: {e}", exc_info=True)
            await update.message.reply_text("⚠️ Could not retrieve open trades.", reply_markup=self.keyboard)
            return

        if not open_trades:
            await update.message.reply_text("📈 No open trades to sell.", reply_markup=self.keyboard)
            return

        if context.args:
            # Sell a specific trade by its displayed ID
            try:
                trade_id = int(context.args[0])
            except ValueError:
                await update.message.reply_text("ℹ️ Usage: /sell <id>  (e.g., /sell 1)", reply_markup=self.keyboard)
                return

            if trade_id < 1 or trade_id > len(open_trades):
                await update.message.reply_text(f"❌ Invalid trade ID. Use a number between 1 and {len(open_trades)}.", reply_markup=self.keyboard)
                return

            symbol = open_trades[trade_id - 1]['symbol']
            await update.message.reply_text(f"🔄 Selling {symbol}...", reply_markup=self.keyboard)
            await self.engine.sell_position(symbol)
            await update.message.reply_text(f"✅ Sell order placed for {symbol}.", reply_markup=self.keyboard)
        else:
            # Sell all open positions
            count = len(open_trades)
            await update.message.reply_text(f"🔄 Selling all {count} open positions...", reply_markup=self.keyboard)
            await self.engine.sell_all_positions()
            await update.message.reply_text(f"✅ Sell orders placed for all {count} positions.", reply_markup=self.keyboard)

    async def cmd_profit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        try:
            summary = await asyncio.to_thread(self.engine.get_profit_summary)
            pnl = summary['total_pnl']
            pnl_pct = summary['pnl_percent']
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            pnl_sign = "+" if pnl >= 0 else ""

            msg = "<b>💰 Profit Summary</b>\n\n"
            msg += f"💵 Initial Balance:  {summary['initial_balance']:,.6f}\n"
            msg += f"🏦 Current Balance:  {summary['current_balance']:,.6f}\n"
            msg += f"📊 Open Positions:   {summary['open_value']:,.6f}\n"
            total_wallet = summary['current_balance'] + summary['open_value']
            msg += f"💼 Total Wallet:     {total_wallet:,.6f}\n"
            msg += f"🧾 Fees Paid:        {summary['total_fees']:,.6f}\n"
            msg += f"{pnl_emoji} Total P&L:         {pnl_sign}{pnl:,.6f}  ({pnl_sign}{pnl_pct:.2f}%)\n"
            wins = summary.get('wins', 0)
            losses = summary.get('losses', 0)
            win_rate = summary.get('win_rate', 0.0)
            msg += f"\n🏆 Wins: {wins}  💔 Losses: {losses}\n"
            msg += f"📊 Win Rate: {win_rate*100:.1f}%\n"
        except Exception as e:
            logger.error(f"Failed to get profit summary: {e}", exc_info=True)
            msg = "⚠️ Could not retrieve profit summary. Please try again later."

        await update.message.reply_text(msg, parse_mode='HTML', reply_markup=self.keyboard)

    def _write_notification_log(self, log_path: Path, summary: dict):
        """Write a summary dict as a JSON line to log_path, rotating if > MAX_LOG_SIZE."""
        with TelegramBot._log_lock:
            # Rotate if file exists and is too large
            if log_path.exists() and log_path.stat().st_size >= self.MAX_LOG_SIZE:
                # Remove oldest backup if it exists
                oldest = log_path.with_suffix(f".jsonl.{self.MAX_LOG_BACKUPS}")
                if oldest.exists():
                    oldest.unlink()
                # Shift existing backups
                for i in range(self.MAX_LOG_BACKUPS - 1, 0, -1):
                    src = log_path.with_suffix(f".jsonl.{i}")
                    dst = log_path.with_suffix(f".jsonl.{i+1}")
                    if src.exists():
                        src.rename(dst)
                # Rename current log to .1
                log_path.rename(log_path.with_suffix(".jsonl.1"))
            # Write the new entry
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    @staticmethod
    def _compact_summary(summary: dict) -> dict:
        """Return a minimal version of the summary dict to keep the notification log small."""
        # Allowed keys – only these will be kept
        allowed_keys = {
            "symbol", "action", "confidence", "reason",
            "price", "amount", "realized_pnl", "exit_reason", "mode",
            "coins", "daily_pnl", "target_amount", "strategy_type",
            "sentiment", "backtest",
        }
        compact = {}
        for key in allowed_keys:
            if key in summary:
                value = summary[key]
                # Truncate reason to 50 characters
                if key == "reason" and isinstance(value, str):
                    if len(value) > 50:
                        value = value[:47] + "..."
                # If coins is a list of dicts, keep only the symbols
                if key == "coins" and isinstance(value, list):
                    if value and isinstance(value[0], dict):
                        value = [c.get("symbol", c) for c in value]
                # Compact sentiment to just avg_compound and total_articles
                if key == "sentiment" and isinstance(value, dict):
                    value = {
                        "avg_compound": value.get("avg_compound"),
                        "total_articles": value.get("total_articles"),
                    }
                # Truncate backtest summary to 50 characters
                if key == "backtest" and isinstance(value, str):
                    if len(value) > 50:
                        value = value[:47] + "..."
                compact[key] = value
        return compact

    async def send_notification(self, message: str, summary: dict = None):
        """Send a notification to the stored chat ID and optionally log a summary."""
        chat_id = await asyncio.to_thread(get_telegram_chat_id)
        logger.info(f"send_notification called, chat_id={chat_id}, message={message[:50]}...")
        if not chat_id:
            logger.warning("No chat_id stored – cannot send notification. Use /start first.")
            return
        try:
            await self.app.bot.send_message(chat_id=int(chat_id), text=message)
            logger.info("Notification sent successfully.")
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}", exc_info=True)

        # --- Log summary to JSONL file (if enabled) ---
        if summary is not None and settings.NOTIFICATION_LOG_ENABLED:
            data_dir = Path(settings.DATA_DIR)
            data_dir.mkdir(parents=True, exist_ok=True)
            log_path = data_dir / "notifications.jsonl"

            # Ensure a UTC timestamp is present
            if "timestamp" not in summary:
                summary["timestamp"] = datetime.now(timezone.utc).isoformat()

            # Compact the summary to keep the log small
            summary = self._compact_summary(summary)

            await asyncio.to_thread(self._write_notification_log, log_path, summary)

    async def start(self):
        """Start the bot (initialize, start polling, start application)."""
        await self.app.initialize()
        await self.app.updater.start_polling()
        await self.app.start()
        logger.info("Telegram bot started and polling.")
        # Notify the user about the trading mode
        mode = settings.TRADING_MODE.upper()
        await self.send_notification(
            f"🤖 Bot started in {mode} mode.",
            summary={
                "action": "INFO",
                "reason": "Bot started",
                "mode": mode,
            }
        )

    async def stop(self):
        """Stop the bot gracefully."""
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()
