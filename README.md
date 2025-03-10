## 🎧 Speech Emotion Recognition (SER) using Transformer-Based Model

### 📌 Project Overview
This project classifies emotions from speech audio using a Transformer-based model. It employs Wav2Vec2 for feature extraction and trains a deep learning classifier using the RAVDESS and TESS datasets.

---

### 👥 Installation

#### ✅ Prerequisites
- Python 3.8+
- `pip` package manager

#### 🚀 Step 1: Clone the Repository
```bash
git clone https://github.com/didar-ali-deed/Vocal-Sentiments-Transformer-Based-SER.git
cd your-repo-folder
```

#### 🏢 Step 2: Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

#### 🛂 Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 👤 Dataset Download
Since datasets are ignored in Git, download them manually from Kaggle:

#### 💽 RAVDESS Dataset 
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
#### 💽 TESS Dataset
https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

#### 📂 Organize the Data
Extract the datasets into the `data/` folder as follows:
```
data/
 ├── ravdess/          # Extract RAVDESS dataset here
 ├── tess/             # Extract TESS dataset here
```

---

### 🔄 Data Preprocessing
Run the preprocessing script to clean the audio data, extract metadata, 
```bash
python scripts/preprocess.py --tess_path data/tess/ --ravdess_path data/ravdess/ --output_dir Preprocessed_Data/ 
```
**Outputs:**
- `Preprocessed_Data/combined_data.csv`
- Graphs for emotion distribution and audio duration

---

### 💖 Feature Extraction with Wav2Vec2
Extract deep speech features using Wav2Vec2:
```bash
python scripts/wav2vec_feature_extraction.py --input_path Preprocessed_Data/combined_data.csv --output_path Extracted_Features/combined_wav2vec_features.csv --batch_size 32
```
**Outputs:**
- `Extracted_Features/combined_wav2vec_features.csv`
- Feature distribution plot (`feature_distribution.png`)

---

### 🏋️ Model Training
Train the emotion classifier using the extracted features:
```bash
python scripts/train_emotion_classifier.py
```
**Outputs:**
- Model saved at `models/emotion_classifier/emotion_classifier.pth`
- Training loss and accuracy plots in `results/`

---

### 📊 Model Evaluation
Test the trained model on unseen data:
```bash
python scripts/test_emotion_classifier.py
```
**Outputs:**
- ✅ Accuracy, Precision, Recall, F1-score
- 🎮 Confusion Matrix (`confusion_matrix.png`)
- 📊 Class-wise Performance (`class_metrics.png`)
- 🏆 Overall Accuracy Pie Chart (`accuracy_pie_chart.png`)

---

### 📚 Project Structure
```
SER_Project/
 ├── data/                           # Contains RAVDESS & TESS datasets (not included in Git)
 ├── Preprocessed_Data/               # Preprocessed dataset (CSV)
 ├── Extracted_Features/              # Wav2Vec2 extracted features (CSV)
 ├── models/                          # Trained emotion classifier
 ├── results/                         # Evaluation metrics and visualizations
 ├── scripts/
 │   ├── preprocess.py                # Preprocessing script
 │   ├── wav2vec_feature_extraction.py # Feature extraction script
 │   ├── train_emotion_classifier.py   # Model training script
 │   └── test_emotion_classifier.py    # Model evaluation script
 ├── requirements.txt                  # Dependencies
 ├── README.md                         # Project documentation
 └── .gitignore                         # Ignored files (datasets, venv, etc.)
```

---

### 🔧 Dependencies
If `pip install -r requirements.txt` fails, install manually:
```bash
pip install pandas librosa torch torchvision torchaudio transformers matplotlib seaborn tqdm scikit-learn argparse
```

---

### 📊 Results & Observations
- The model successfully classifies emotions with high accuracy.
- Wav2Vec2 extracts meaningful speech features without manual feature engineering.
- Confusion matrix analysis identifies potential misclassifications.

---

### 🚀 Future Improvements
- Use larger datasets like IEMOCAP, CREMA-D
- Implement real-time emotion detection using Flask + WebRTC
- Optimize model hyperparameters for better generalization

---

### 🙌 Credits
- **Datasets:** RAVDESS, TESS
- **Feature Extraction:** Facebook’s Wav2Vec2
- **Deep Learning Framework:** PyTorch

---

### 📩 Contact
If you have any questions or suggestions, feel free to open an issue or contact me.

💡 *If you find this project helpful, please ⭐ star the repository!* 🚀
