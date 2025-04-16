import os
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import face_recognition
import io
from PIL import Image, ImageDraw
import numpy as np

# Load .env variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")

# Define reply keyboard
menu_keyboard = [['Add face'], ['Recognize faces'], ['Reset faces']]
markup = ReplyKeyboardMarkup(menu_keyboard, resize_keyboard=True)

# In-memory database of faces
known_faces = []  # list of dicts: {name, encoding}

# Track conversation state per user
user_states = {}

# Start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome! What would you like to do?", reply_markup=markup)

# Handle button clicks and messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text

    if text == 'Add face':
        user_states[user_id] = {'state': 'awaiting_face_image'}
        await update.message.reply_text("Upload an image with a single face.")
    elif text == 'Recognize faces':
        user_states[user_id] = {'state': 'awaiting_recognition_image'}
        await update.message.reply_text("Upload an image with at least one face and I will recognize who is in it.")
    elif text == 'Reset faces':
        known_faces.clear()
        await update.message.reply_text("All known faces have been reset.", reply_markup=markup)
    elif user_id in user_states and user_states[user_id].get('state') == 'awaiting_name':
        # Save name + last uploaded encoding
        name = text.strip()
        encoding = user_states[user_id]['last_encoding']
        known_faces.append({'name': name, 'encoding': encoding})
        user_states.pop(user_id)
        await update.message.reply_text("Great. I will now remember this face.", reply_markup=markup)
    else:
        await update.message.reply_text("Please choose one of the options using the buttons below.", reply_markup=markup)

# Handle incoming photos
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id not in user_states:
        await update.message.reply_text("Please choose one of the options first.", reply_markup=markup)
        return

    state = user_states[user_id]['state']
    photo_file = await update.message.photo[-1].get_file()
    byte_stream = io.BytesIO()
    await photo_file.download_to_memory(out=byte_stream)
    byte_stream.seek(0)
    image = face_recognition.load_image_file(byte_stream)

    if state == 'awaiting_face_image':
        encodings = face_recognition.face_encodings(image)
        if len(encodings) != 1:
            await update.message.reply_text("Please upload an image with exactly one face.")
            return
        user_states[user_id]['last_encoding'] = encodings[0]
        user_states[user_id]['state'] = 'awaiting_name'
        await update.message.reply_text("Great. What's the name of the person in this image?")
    elif state == 'awaiting_recognition_image':
        face_locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, face_locations)
        names = []

        for face_encoding in encodings:
            matches = face_recognition.compare_faces(
                [entry['encoding'] for entry in known_faces],
                face_encoding,
                tolerance=0.5
            )
            name = "Unknown"
            if True in matches:
                index = matches.index(True)
                name = known_faces[index]['name']
            names.append(name)

        # Draw results
        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img)
        for (top, right, bottom, left), name in zip(face_locations, names):
            draw.rectangle(((left, top), (right, bottom)), outline="blue", width=2)
            draw.text((left + 6, bottom - 10), name, fill="blue")

        # Send back result
        output = io.BytesIO()
        pil_img.save(output, format='PNG')
        output.seek(0)
        await update.message.reply_photo(photo=InputFile(output), caption=f"I found {len(names)} faces: {', '.join(names)}", reply_markup=markup)

        user_states.pop(user_id)

# Main entry
if __name__ == '__main__':
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("Bot is running... Press Ctrl+C to stop.")
    app.run_polling()
