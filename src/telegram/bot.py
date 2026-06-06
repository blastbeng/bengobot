import asyncio
import logging
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.config.settings import settings
from src.trading.engine import TradingEngine
from src.utils.redis_client import get_redis_client
from src.database import set_telegram_chat_id, get_telegram_chat_id, get_news_for_symbol
from src.llm.prompts import _format_news_for_prompt

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, engine: TradingEngine):
        self.engine = engine
        self.redis = get_redis_client()
        self.app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
        self._register_handlers()
        self.keyboard = ReplyKeyboardMarkup(
            [
                [KeyboardButton("📊 Status"), KeyboardButton("📈 Trades")],
                [KeyboardButton("💰 Profit"), KeyboardButton("📊 Performance")],
                [KeyboardButton("📰 News"), KeyboardButton("⚠️ Risk")],
                [KeyboardButton("⏸️ Pause"), KeyboardButton("▶️ Resume")],
            ],
            resize_keyboard=True,
        )

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
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_button))

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        await asyncio.to_thread(set_telegram_chat_id, chat_id)
        await update.message.reply_text(
            "Bot started! You will receive trade notifications here.\nUse the buttons below or type /menu to see them again.",
            reply_markup=self.keyboard,
        )

    async def cmd_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Choose an option:", reply_markup=self.keyboard)

    async def handle_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
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

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await asyncio.to_thread(self.redis.set, "trading:paused", "1")
        await update.message.reply_text("Trading paused.", reply_markup=self.keyboard)

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await asyncio.to_thread(self.redis.delete, "trading:paused")
        await update.message.reply_text("Trading resumed.", reply_markup=self.keyboard)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        def get_status():
            coins = self.engine.current_coins
            positions = self.engine.positions
            balance = self.engine.trader.fetch_balance()
            return coins, positions, balance
        coins, positions, balance = await asyncio.to_thread(get_status)

        msg = "<b>📊 Current Status</b>\n\n"
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
        open_trades = await asyncio.to_thread(self.engine.get_open_trades)
        if not open_trades:
            await update.message.reply_text("No open trades.", reply_markup=self.keyboard)
            return

        msg = "<b>📈 Open Trades</b>\n\n"
        for t in open_trades:
            sym = t['symbol']
            amt = t['amount']
            price = t['price']
            fee = t.get('fee', {})
            fee_cost = fee.get('cost', 0) or 0
            fee_currency = fee.get('currency', '')
            fee_str = f"{fee_cost:.6f} {fee_currency}" if fee_cost else "—"

            ts = datetime.fromtimestamp(t['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')

            line = f"🟢 <b>BUY</b> <code>{sym}</code>\n"
            line += f"   🕒 {ts}\n"
            line += f"   Amount: {amt:.6f}  Entry: {price:.4f}\n"
            line += f"   Fee: {fee_str}\n"

            pnl = t['unrealized_pnl']
            pnl_pct = t['unrealized_pnl_pct']
            pnl_sign = "+" if pnl >= 0 else ""
            pnl_pct_sign = "+" if pnl_pct >= 0 else ""
            line += f"   Unrealized P&L: {pnl_sign}{pnl:.4f} ({pnl_pct_sign}{pnl_pct:.2f}%)"

            msg += line + "\n\n"

        await update.message.reply_text(msg, parse_mode='HTML', reply_markup=self.keyboard)

    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show performance summary grouped by coin and timeframe."""
        try:
            perf = await asyncio.to_thread(self.engine.get_performance_summary)
            rows = perf.get("rows", [])
            total = perf.get("total", {})

            if not rows:
                await update.message.reply_text(
                    "📊 No closed sell trades yet.", reply_markup=self.keyboard
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
        if len(msg) > 4000:
            for i in range(0, len(msg), 4000):
                await update.message.reply_text(msg[i:i+4000], parse_mode="Markdown")
        else:
            await update.message.reply_text(msg, parse_mode="Markdown")

    async def cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        metrics = self.engine.get_risk_metrics()
        pf = metrics['profit_factor']
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        msg = (
            f"<b>⚠️ Risk Metrics</b>\n\n"
            f"<b>Portfolio</b>\n"
            f"Balance: {metrics['current_balance']:.2f} {metrics['base_currency']}\n"
            f"Initial: {metrics['initial_balance']:.2f} {metrics['base_currency']}\n"
            f"P&L: {metrics['total_pnl']:.2f} ({metrics['total_pnl_pct']:.2f}%)\n"
            f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n\n"
            f"<b>Positions</b>\n"
            f"Open: {metrics['open_positions_count']}\n"
            f"Exposure: {metrics['total_exposure']:.2f} {metrics['base_currency']}\n"
            f"Largest Position: {metrics['largest_position_exposure_pct']:.1f}% of portfolio\n"
            f"Total Stop Risk: {metrics['total_stop_loss_risk']:.2f} {metrics['base_currency']}\n\n"
            f"<b>Trade Stats</b>\n"
            f"Total Trades: {metrics['total_trades']}\n"
            f"Win Rate: {metrics['win_rate']:.1f}%\n"
            f"Profit Factor: {pf_str}\n"
            f"Avg Win: {metrics['avg_win']:.2f}  Avg Loss: {metrics['avg_loss']:.2f}"
        )
        await update.message.reply_text(msg, parse_mode='HTML', reply_markup=self.keyboard)

    async def cmd_news(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show recent news for all currently tracked coins."""
        coins = self.engine.current_coins
        if not coins:
            await update.message.reply_text("No coins currently tracked.")
            return

        await update.message.reply_text("Fetching latest news...")
        messages = []
        for entry in coins:
            symbol = entry["symbol"]
            base_coin = symbol.split("/")[0] if "/" in symbol else symbol
            articles = get_news_for_symbol(base_coin, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
            if articles:
                formatted = _format_news_for_prompt(articles)
                messages.append(f"*{base_coin}*\n{formatted}")
            else:
                messages.append(f"*{base_coin}*\nNo recent news.")

        full_text = "\n\n".join(messages)
        # Telegram messages have a 4096 character limit; split if needed
        if len(full_text) > 4000:
            for i in range(0, len(full_text), 4000):
                await update.message.reply_text(full_text[i:i+4000], parse_mode="Markdown")
        else:
            await update.message.reply_text(full_text, parse_mode="Markdown")

    async def cmd_news_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show news article counts for tracked coins."""
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

    async def cmd_profit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            summary = await asyncio.to_thread(self.engine.get_profit_summary)
            pnl = summary['total_pnl']
            pnl_pct = summary['pnl_percent']
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            pnl_sign = "+" if pnl >= 0 else ""

            msg = "<b>💰 Profit Summary</b>\n\n"
            msg += f"💵 Initial Balance:  {summary['initial_balance']:,.2f}\n"
            msg += f"🏦 Current Balance:  {summary['current_balance']:,.2f}\n"
            msg += f"📊 Open Positions:   {summary['open_value']:,.2f}\n"
            total_wallet = summary['current_balance'] + summary['open_value']
            msg += f"💼 Total Wallet:     {total_wallet:,.2f}\n"
            msg += f"🧾 Fees Paid:        {summary['total_fees']:,.2f}\n"
            msg += f"{pnl_emoji} Total P&L:         {pnl_sign}{pnl:,.2f}  ({pnl_sign}{pnl_pct:.2f}%)\n"
            wins = summary.get('wins', 0)
            losses = summary.get('losses', 0)
            win_rate = summary.get('win_rate', 0.0)
            msg += f"\n🏆 Wins: {wins}  💔 Losses: {losses}\n"
            msg += f"📊 Win Rate: {win_rate*100:.1f}%\n"
        except Exception as e:
            logger.error(f"Failed to get profit summary: {e}", exc_info=True)
            msg = "⚠️ Could not retrieve profit summary. Please try again later."

        await update.message.reply_text(msg, parse_mode='HTML', reply_markup=self.keyboard)

    async def send_notification(self, message: str):
        """Send a notification to the stored chat ID."""
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

    async def start(self):
        """Start the bot (initialize, start polling, start application)."""
        await self.app.initialize()
        await self.app.updater.start_polling()
        await self.app.start()
        logger.info("Telegram bot started and polling.")

    async def stop(self):
        """Stop the bot gracefully."""
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()
