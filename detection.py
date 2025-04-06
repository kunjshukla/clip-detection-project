import os
import torch
import clip
from PIL import Image
import cv2
import numpy as np

dataset_path = "../archive(3)/dataset_images/"

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define categories
categories = ["human", "cow", "goat", "lion", "dog", "cat", "horse", "bird", "elephant", "monkey", "butterfly", "frog", "snake"]
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(device)

# Dataset directory
DATASET_PATH = "dataset_images/"

def detect_objects(image):
    """ Placeholder for object detection model """
    h, w, _ = image.shape
    boxes = [(w//4, h//4, w//2, h//2)]  # Simulated bounding box
    return boxes

def classify_objects(images, model, preprocess):
    """ Classify cropped objects using CLIP """

    if len(images) == 0:
        return []

    preprocessed_images = torch.stack([preprocess(Image.fromarray(img)) for img in images]).to(device)    
    
    with torch.no_grad():
        text_features = model.encode_image(preprocessed_images)
        image_features = model.encode_image(preprocessed_images)
        similarities = (image_features @ model.encode_text(text_inputs).T).softmax(dim=-1)
    
    best_matches = similarities.argmax(dim=-1).tolist()
    return [categories[idx] for idx in best_matches]

def process_image(image_path):
    """ Process an image for detection """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {image_path}")
        return

    boxes = detect_objects(image)
    cropped_images = [image[y:y+h, x:x+w] for x, y, w, h in boxes]
    
    if not cropped_images:
        print(f"No objects detected in {image_path}")
        return
    
    labels = classify_objects(cropped_images, model, preprocess)
    
    for (x, y, w, h), label in zip(boxes, labels):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"Detected: {label} in {image_path}")
    
    cv2.imshow("Detection", image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

def process_video(video_path):
    """ Process a video for detection """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error loading the video.")
    frame_skip = 20  # Process every 5th frame
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        boxes = detect_objects(frame)
        cropped_images = [frame[y:y+h, x:x+w] for x, y, w, h in boxes]
        labels = classify_objects(cropped_images, model, preprocess)
        
        for (x, y, w, h), label in zip(boxes, labels):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Detected: {label}")
        
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_dataset(dataset_path):
    """ Process all images in dataset directory """
    image_files = [f for f in os.listdir(dataset_path) if f.endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        print("No images found in dataset directory.")
        return
    
    for img_file in image_files:
        img_path = os.path.join(dataset_path, img_file)
        process_image(img_path)



def test_accuracy_on_dataset(test_path):
    correct = 0
    total = 0
    for category in os.listdir(test_path):
        category_path = os.path.join(test_path, category)
        for image_file in os.listdir(category_path):
            image_path = os.path.join(category_path, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            boxes = [(0, 0, image.shape[1], image.shape[0])]  # Full image as single box
            cropped_images = [image[y:y+h, x:x+w] for x, y, w, h in boxes]
            labels = classify_objects(cropped_images, model, preprocess)
            predicted = labels[0]
            total += 1
            if predicted == category:
                correct += 1
            print(f"Image: {image_file}, Actual: {category}, Predicted: {predicted}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nFinal Accuracy: {accuracy*100:.2f}%")


# Run detection
# process_video("animal2.mp4")
# process_dataset(DATASET_PATH)
test_accuracy_on_dataset("archive(4)/test_dataset/")
