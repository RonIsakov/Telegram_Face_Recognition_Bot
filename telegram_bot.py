import os
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")

# Define the button layout
button_labels = ["Hello", "World", "Telegram", "Bot"]
keyboard = ReplyKeyboardMarkup.from_row(button_labels, resize_keyboard=True)

# Handle /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Choose an option:", reply_markup=keyboard)

# Handle button responses
async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    if user_input in button_labels:
        await update.message.reply_text(user_input, reply_markup=keyboard)
    else:
        await update.message.reply_text("Please choose a valid option.", reply_markup=keyboard)

# Main entry point
if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond))

    print("Bot is running... Press Ctrl+C to stop.")
    app.run_polling()
