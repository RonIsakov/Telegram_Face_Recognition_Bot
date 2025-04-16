import os
import face_recognition
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# --- Load Celeb Encodings ---
project_root = os.getcwd()
celebs_dir = os.path.join(project_root, 'celebs')
celeb_encodings = []

for celeb_name in os.listdir(celebs_dir):
    celeb_dir = os.path.join(celebs_dir, celeb_name)
    if not os.path.isdir(celeb_dir):
        continue

    for fname in os.listdir(celeb_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(celeb_dir, fname)
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    celeb_encodings.append({
                        'name': celeb_name,
                        'encoding': encs[0],
                        'image_path': path
                    })
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
            break  # only take one image per celeb

if not celeb_encodings:
    raise RuntimeError("🚨 No encodings found!")

print(f"✅ Loaded {len(celeb_encodings)} celeb faces")

# --- Run t-SNE ---
X = np.stack([c['encoding'] for c in celeb_encodings])
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
coords = tsne.fit_transform(X)

print(f"[DEBUG] Total celeb encodings loaded: {len(celeb_encodings)}")
print(f"[DEBUG] t-SNE coordinates shape: {coords.shape}")
print(f"[DEBUG] First few t-SNE coords: {coords[:3]}")
print(f"[DEBUG] First few celeb names: {[c['name'] for c in celeb_encodings[:3]]}")


# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 10))
for (x, y), item in zip(coords, celeb_encodings):
    try:
        img = Image.open(item['image_path']).convert("RGB").resize((40, 40))
        im = OffsetImage(img, zoom=1)
        ab = AnnotationBbox(im, (x, y), frameon=False, box_alignment=(0.5, 1))  # box_alignment puts bottom at point
        ax.add_artist(ab)

        ax.text(x, y - 13, item['name'], fontsize=6, ha='center', va='top', color='black')
    except Exception as e:
        print(f"⚠️ Failed to process image {item['image_path']}: {e}")

ax.set_title("t-SNE of Celeb Faces")
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
coords_x, coords_y = zip(*coords)
ax.set_xlim(min(coords_x) - 10, max(coords_x) + 10)
ax.set_ylim(min(coords_y) - 10, max(coords_y) + 10)

plt.show()
