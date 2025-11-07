# 🦟 AI-Driven Mosquito Species Classification for Vector Control

An **AI-based mosquito species recognition system** designed for **vector surveillance and public health monitoring**.  
The project uses **deep learning** and **computer vision** to automatically identify mosquito species from both **images** and **videos**, enabling faster, scalable, and more accurate mosquito detection.

---

## 🚀 Features

- 🧬 **Dual Model Approach** — Custom CNN and Transfer Learning using **ResNet50 (ImageNet pretrained)**  
- 🎥 **Video Support** — Integrated **YOLOv8 + OpenCV** pipeline for frame extraction and classification  
- 📊 **High Accuracy** —  
  - 99.48% test accuracy on the *Smashed Condition* dataset  
  - 92.07% test accuracy on the *Complete Combined Dataset*  
- 🧩 **Four mosquito species classified:**  
  *Aedes aegypti*, *Aedes albopictus*, *Anopheles stephensi*, *Culex quinquefasciatus*  
- 🔁 **Extensible** — Model generalizes to unseen mosquito videos and supports real-time inference (~45 FPS)

---

## 🧰 Tech Stack

| Category | Tools / Frameworks |
|-----------|-------------------|
| **Language** | Python 3 |
| **Deep Learning** | TensorFlow, Keras |
| **Transfer Learning** | ResNet50 |
| **Computer Vision** | OpenCV, YOLOv8 |
| **Data Handling** | NumPy, Pandas |
| **Visualization** | Matplotlib |
| **Environment** | Google Colab (GPU) |

---

## 🧪 Methodology

1. **Data Loading & Preprocessing**
   - Datasets: *Vector Mosquito Images* (Pise et al., 2022) and *Mosquito-on-Human-Skin* (Ong, 2022)
   - Loaded using `tf.keras.preprocessing.image_dataset_from_directory`
   - Resized to 256×256, normalized, and augmented (flip, rotation, color jitter)

2. **Model 1 – Custom CNN**
   - 6× `Conv2D` + `MaxPooling2D` layers
   - Dense(64, ReLU) → Dense(n_classes, Softmax)
   - Optimizer: Adam, Loss: SparseCategoricalCrossentropy

3. **Model 2 – Transfer Learning (ResNet50)**
   - Base: `ResNet50(weights='imagenet', include_top=False)`
   - Added: GlobalAveragePooling2D → Dense(64, ReLU) → Dense(n_classes, Softmax)
   - Trained for 50 epochs on augmented datasets

4. **Video Classification**
   - Extracted frames using **OpenCV**
   - Applied **YOLOv8** for mosquito detection (confidence > 0.5)
   - Classified detected frames via trained **ResNet50 model**

---

## 📊 Results Summary

| Dataset | Model | Test Accuracy |
|----------|--------|---------------|
| Landing Condition | ResNet50 | 95.96 % |
| Smashed Condition | ResNet50 | **99.48 %** |
| Vector Dataset | ResNet50 | 97.38 % |
| Complete Dataset | ResNet50 | **92.07 %** |

- ResNet50 consistently outperformed the custom CNN across all datasets  
- Video pipeline achieved real-time frame classification (~45 FPS)

---

## 📽️ Example Output

Detected mosquito frames from video, classified as *Aedes aegypti* with high confidence.

