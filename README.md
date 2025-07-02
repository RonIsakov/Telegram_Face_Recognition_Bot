#  Telegram Face Recognition Bot

This project is a smart and interactive Telegram bot that performs real-time face recognition. Users can upload photos, teach the bot to recognize new faces, match them to known celebrities, and even view a visual similarity map of all known faces.

##  Features

- **Add Face**: Upload a photo and name the person. The bot learns and stores the face.
- **Recognize Faces**: Upload a group photo and the bot identifies known individuals using bounding boxes.
- **Reset Faces**: Clears the bot’s memory of previously added faces.
- **Similar Celebs**: Upload a face and the bot returns the most similar celebrity.
- **Face Similarity Map**: Generates a 2D TSNE map showing visual proximity of known faces (users and celebrities).
-  **Face history**: Shows all the faces recognized and what time and date they were recognized at

## 🛠 Technologies Used

- Python
- [face_recognition](https://github.com/ageitgey/face_recognition)
- Telegram Bot API (`python-telegram-bot`)
- OpenCV
- Matplotlib (for plotting TSNE)
- scikit-learn (for TSNE)
- `python-dotenv` (for token management)
- Git & GitHub (version control)

## 🧪 Setup Instructions

1. **Clone the repository and create a virtual environment:**
   ```bash
   git clone https://github.com/your-username/telegram-face-bot.git
   cd telegram-face-bot
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your `.env` file:**
   ```
   TELEGRAM_TOKEN=your_telegram_bot_token_here
   ```

4. **Run the bot:**
   ```bash
   python telegram_bot.py
   ```

## 🖼️ Dataset Structure

- **User Faces**: Saved dynamically on upload.
- **Celeb Dataset**: Preloaded directory structured as can use your own or [download](https://drive.google.com/file/d/19PAxe94UbuHeSGw8ySkXT_BDB1neACix/view?usp=sharing):
  ```
  celebs/
    ├── Emma_Watson/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── Tom_Hanks/
        ├── image1.jpg
        └── image2.jpg
  ```

## 🧠 How It Works

- Faces are encoded using `face_recognition` into 128-dim vectors.
- Matching is done using Euclidean distance with a configurable similarity threshold.
- TSNE projects all face encodings to 2D and `pyplot` visualizes the cluster.
