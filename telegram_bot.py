import os
import io
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")

# Main menu
menu_keyboard = [
    ['Add face'],
    ['Recognize faces'],
    ['Reset faces'],
    ['Similar celebs'],
    ['map']
]
markup = ReplyKeyboardMarkup(menu_keyboard, resize_keyboard=True)

# In‑memory storage
known_faces = []       # [{'name':str, 'encoding':np.array, 'pil_image': PIL.Image}]
user_states = {}       # {user_id: {'state':str, ...}}
celeb_encodings = []   # [{'name':str, 'image_path':str, 'encoding':np.array}]

def load_celeb_faces(folder_name='celebs'):
    base = os.path.join(os.path.dirname(__file__), folder_name)
    if not os.path.isdir(base):
        print(f"No celeb folder found at {base}")
        return

    for celeb_name in os.listdir(base):
        celeb_dir = os.path.join(base, celeb_name)
        if not os.path.isdir(celeb_dir):
            continue

        for fname in os.listdir(celeb_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(celeb_dir, fname)
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    celeb_encodings.append({
                        'name': celeb_name,
                        'image_path': path,
                        'encoding': encs[0]
                    })
            except Exception as e:
                print(f"Failed to load celeb image {path}: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome! Choose an option:", reply_markup=markup)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text.strip()

    if text == 'Add face':
        user_states[user_id] = {'state': 'awaiting_face_image'}
        await update.message.reply_text("Upload an image with a single face.")
    elif text == 'Recognize faces':
        user_states[user_id] = {'state': 'awaiting_recognition_image'}
        await update.message.reply_text("Upload an image with faces; I'll identify anyone I know.")
    elif text == 'Reset faces':
        known_faces.clear()
        await update.message.reply_text("Reset complete—no faces remembered.", reply_markup=markup)
    elif text == 'Similar celebs':
        user_states[user_id] = {'state': 'awaiting_similar_celeb_image'}
        await update.message.reply_text("Upload a single-face photo; I'll find your celeb lookalike.")
    elif text == 'map':
        await update.message.reply_text("Generating face similarity map, one moment...")

        all_faces = []
        seen = set()
        for k in known_faces:
            all_faces.append({
                'name': k['name'],
                'encoding': k['encoding'],
                'image_path': None,
                'pil_image': k['pil_image']
            })
            
        for c in celeb_encodings:
            if c['name'] not in seen:
                all_faces.append({
                    'name': c['name'],
                    'encoding': c['encoding'],
                    'image_path': c['image_path'],
                    'pil_image': None
                })
                seen.add(c['name'])

        if len(all_faces) < 2:
            await update.message.reply_text("Not enough faces to generate a map.")
            return

        X = np.stack([f['encoding'] for f in all_faces])
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(X)-1))
        coords = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 10))
        for (x, y), item in zip(coords, all_faces):
            try:
                img = Image.open(item['image_path']).convert("RGB") if item['image_path'] else item['pil_image']
                img = img.resize((40, 40))
                im = OffsetImage(img, zoom=1)
                ab = AnnotationBbox(im, (x, y), frameon=False, box_alignment=(0.5, 1))
                ax.add_artist(ab)
                ax.text(x, y - 13, item['name'], fontsize=6, ha='center', va='top', color='black')
            except Exception as e:
                print(f"⚠️ Failed to plot {item['name']}: {e}")

        for spine in ax.spines.values():
            spine.set_visible(False)

        #ax.set_title("t-SNE of Celeb + Uploaded Faces")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        coords_x, coords_y = zip(*coords)
        ax.set_xlim(min(coords_x) - 10, max(coords_x) + 10)
        ax.set_ylim(min(coords_y) - 10, max(coords_y) + 10)

        buf = io.BytesIO()
        plt.savefig(buf, format='PNG')
        plt.close(fig)
        buf.seek(0)

        await update.message.reply_photo(photo=buf, caption="🗺️ Here's your face similarity map!", reply_markup=markup)

    elif user_id in user_states and user_states[user_id].get('state') == 'awaiting_name':
        enc = user_states[user_id]['last_encoding']
        img = user_states[user_id]['last_image']
        known_faces.append({'name': text, 'encoding': enc, 'pil_image': img})
        user_states.pop(user_id)
        await update.message.reply_text("Face saved! What next?", reply_markup=markup)
    else:
        await update.message.reply_text("Please pick from the menu below.", reply_markup=markup)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id not in user_states:
        await update.message.reply_text("Start by choosing an option from the menu.", reply_markup=markup)
        return

    state = user_states[user_id]['state']
    photo = await update.message.photo[-1].get_file()
    buf = io.BytesIO()
    await photo.download_to_memory(out=buf)
    buf.seek(0)
    img = face_recognition.load_image_file(buf)

    if state == 'awaiting_face_image':
        encs = face_recognition.face_encodings(img)
        if len(encs) != 1:
            await update.message.reply_text("Please send exactly one face.")
            return
        user_states[user_id] = {
            'state': 'awaiting_name',
            'last_encoding': encs[0],
            'last_image': Image.fromarray(img).copy()
        }
        await update.message.reply_text("Great—what’s this person’s name?")

    elif state == 'awaiting_recognition_image':
        locs = face_recognition.face_locations(img)
        encs = face_recognition.face_encodings(img, locs)
        names = []
        for e in encs:
            matches = face_recognition.compare_faces([kf['encoding'] for kf in known_faces], e, tolerance=0.5)
            names.append(known_faces[matches.index(True)]['name'] if True in matches else "Unknown")

        if all(n == "Unknown" for n in names):
            await update.message.reply_text("I don't recognize anyone here.", reply_markup=markup)
            user_states.pop(user_id)
            return

        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        for (top, right, bottom, left), n in zip(locs, names):
            draw.rectangle(((left, top), (right, bottom)), outline="blue", width=2)
            draw.text((left, bottom + 4), n, fill="blue")
        out = io.BytesIO(); pil.save(out, format='PNG'); out.seek(0)
        await update.message.reply_photo(photo=out, caption=f"I found {len(names)}: {', '.join(names)}", reply_markup=markup)
        user_states.pop(user_id)

    elif state == 'awaiting_similar_celeb_image':
        encs = face_recognition.face_encodings(img)
        if len(encs) != 1:
            await update.message.reply_text("Please send exactly one face.")
            return
        query = encs[0]
        best, best_dist = None, float('inf')
        for celeb in celeb_encodings:
            d = np.linalg.norm(celeb['encoding'] - query)
            if d < best_dist:
                best, best_dist = celeb, d

        if best:
            pil_img = Image.open(best['image_path'])
            buf2 = io.BytesIO()
            pil_img.save(buf2, format='JPEG')
            buf2.seek(0)
            await update.message.reply_photo(photo=buf2, caption=f"The celeb most similar is {best['name']}", reply_markup=markup)
        else:
            await update.message.reply_text("No celeb match found.", reply_markup=markup)

        user_states.pop(user_id)

if __name__ == '__main__':
    print("Loading celeb database…")
    load_celeb_faces()
    print(f"Loaded {len(celeb_encodings)} celeb faces.")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("Bot running—press Ctrl+C to stop.")
    app.run_polling()