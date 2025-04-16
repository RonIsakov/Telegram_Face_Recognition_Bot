import face_recognition
from numpy.linalg import norm

# Corrected file paths
image1 = face_recognition.load_image_file("faces/face1.png")
image2 = face_recognition.load_image_file("faces/face2.png")
image3 = face_recognition.load_image_file("faces/face3.png")

# Get face encodings (assumes one face per image)
encoding1 = face_recognition.face_encodings(image1)[0]
encoding2 = face_recognition.face_encodings(image2)[0]
encoding3 = face_recognition.face_encodings(image3)[0]

# Compare similarities
sim_12 = 1 - norm(encoding1 - encoding2)
sim_13 = 1 - norm(encoding1 - encoding3)

print(f"Similarity between face1 and face2: {sim_12:.4f}")
print(f"Similarity between face1 and face3: {sim_13:.4f}")

if sim_12 > sim_13:
    print("face1 is more similar to face2")
else:
    print("face1 is more similar to face3")
