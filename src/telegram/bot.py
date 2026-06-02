import asyncio
import logging
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.config.settings import settings
from src.trading.engine import TradingEngine
from src.utils.redis_client import get_redis_client
from src.database import set_telegram_chat_id, get_telegram_chat_id

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
                [KeyboardButton("💰 Profit"), KeyboardButton("⏸️ Pause"), KeyboardButton("▶️ Resume")],
            ],
            resize_keyboard=True,
        )
        self._initialized = False

    def _register_handlers(self):
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("menu", self.cmd_menu))
        self.app.add_handler(CommandHandler("pause", self.cmd_pause))
        self.app.add_handler(CommandHandler("resume", self.cmd_resume))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("trades", self.cmd_trades))
        self.app.add_handler(CommandHandler("profit", self.cmd_profit))
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
        if text == "📊 Status":
            await self.cmd_status(update, context)
        elif text == "📈 Trades":
            await self.cmd_trades(update, context)
        elif text == "💰 Profit":
            await self.cmd_profit(update, context)
        elif text == "⏸️ Pause":
            await self.cmd_pause(update, context)
        elif text == "▶️ Resume":
            await self.cmd_resume(update, context)

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
        msg += f"<b>🪙 Tracked Coins:</b> {', '.join(coins) if coins else 'None'}\n\n"

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
        trades = await asyncio.to_thread(lambda: self.engine.trade_history[-10:])
        if not trades:
            await update.message.reply_text("No trades yet.", reply_markup=self.keyboard)
            return

        msg = "<b>📜 Recent Trades (last 10)</b>\n\n"
        for t in trades:
            side = t['side'].upper()
            sym = t['symbol']
            amt = t['amount']
            price = t['price']
            fee = t.get('fee', {})
            fee_cost = fee.get('cost', 0) or 0
            fee_currency = fee.get('currency', '')
            fee_str = f"{fee_cost:.6f} {fee_currency}" if fee_cost else "—"

            emoji = "🟢" if side == "BUY" else "🔴"
            line = f"{emoji} <b>{side}</b> <code>{sym}</code>\n"
            line += f"   Amount: {amt:.6f}  Price: {price:.4f}\n"
            line += f"   Fee: {fee_str}"

            if t['side'] == 'sell' and 'realized_pnl' in t:
                pnl = t['realized_pnl']
                pnl_sign = "+" if pnl >= 0 else ""
                line += f"  P&L: {pnl_sign}{pnl:.4f}"

            msg += line + "\n\n"

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
            msg += f"🧾 Fees Paid:        {summary['total_fees']:,.2f}\n"
            msg += f"{pnl_emoji} Total P&L:         {pnl_sign}{pnl:,.2f}  ({pnl_sign}{pnl_pct:.2f}%)\n"
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
        # Ensure keyboard is never None (race condition safety)
        if self.keyboard is None:
            self.keyboard = ReplyKeyboardMarkup(
                [
                    [KeyboardButton("📊 Status"), KeyboardButton("📈 Trades")],
                    [KeyboardButton("💰 Profit"), KeyboardButton("⏸️ Pause"), KeyboardButton("▶️ Resume")],
                ],
                resize_keyboard=True,
            )
        try:
            await self.app.bot.send_message(chat_id=int(chat_id), text=message, reply_markup=self.keyboard)
            logger.info("Notification sent successfully.")
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}", exc_info=True)

    async def initialize(self):
        """Initialize and start the bot application (idempotent)."""
        if not self._initialized:
            await self.app.initialize()
            await self.app.start()
            self._initialized = True

    async def run(self):
        """Start polling for updates."""
        await self.initialize()
        await self.app.updater.start_polling()
        # Keep the task alive
        while True:
            await asyncio.sleep(3600)
