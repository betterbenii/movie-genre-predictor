import kagglehub
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Download the Netflix Movies and TV Shows Dataset from Kaggle
path = kagglehub.dataset_download("anandshaw2001/netflix-movies-and-tv-shows")

# Debug: List files in the downloaded dataset directory
print("Downloaded dataset files:", os.listdir(path))

# Automatically detect the correct dataset file
dataset_files = os.listdir(path)
dataset_path = f"{path}/{dataset_files[0]}"  # Select first file dynamically
print("Using dataset file:", dataset_path)

# Step 2: Load the Dataset
df = pd.read_csv(dataset_path)
print("Dataset Columns:", df.columns)  # Debugging step

# Step 3: Preprocess the Data
# Selecting text and category columns
df = df[['description', 'listed_in']].copy()
df.dropna(inplace=True)
df['description'] = df['description'].astype(str).str.lower()

# Splitting multiple genres into separate rows
df = df.assign(listed_in=df['listed_in'].str.split(', ')).explode('listed_in')

# Filter out rare categories (less than 3 occurrences)
category_counts = df['listed_in'].value_counts()
valid_categories = category_counts[category_counts >= 3].index
df = df[df['listed_in'].isin(valid_categories)]

# Convert categories to numerical labels
label_encoder = LabelEncoder()
y_full = label_encoder.fit_transform(df['listed_in'])

# Step 4: Convert Text to Numerical Features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['description'])
y = y_full

# Step 5: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Train the Model
classifier = LogisticRegression(max_iter=300, solver='saga', multi_class='multinomial', tol=0.01, random_state=42)
print("Training model...")
classifier.fit(X_train, y_train)
print("Model training completed!")

# Step 7: Evaluate the Model
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_[:len(set(y_test))]))

# Step 8: Classify New Text
def classify_new_text(text):
    text_transformed = vectorizer.transform([text])
    prediction = classifier.predict(text_transformed)
    return label_encoder.inverse_transform(prediction)[0]

# Example Usage
new_text = "A thrilling mystery about a detective solving a high-profile murder case."
print("Predicted Genre:", classify_new_text(new_text))


# Output:

# Classification Report:
#                                precision    recall  f1-score   support

#           Action & Adventure       0.15      0.13      0.14       172
#               Anime Features       0.00      0.00      0.00        14
#                 Anime Series       0.00      0.00      0.00        35
#             British TV Shows       0.00      0.00      0.00        51
#     Children & Family Movies       0.27      0.18      0.22       128
#            Classic & Cult TV       0.00      0.00      0.00         6
#               Classic Movies       0.00      0.00      0.00        23
#                     Comedies       0.11      0.15      0.13       335
#               Crime TV Shows       0.10      0.04      0.06        94
#                  Cult Movies       0.00      0.00      0.00        14
#                Documentaries       0.38      0.41      0.39       174
#                   Docuseries       0.26      0.22      0.23        79
#                       Dramas       0.12      0.25      0.16       486
#         Faith & Spirituality       0.00      0.00      0.00        13
#                Horror Movies       0.22      0.08      0.12        71
#           Independent Movies       0.00      0.00      0.00       151
#         International Movies       0.10      0.22      0.13       551
#       International TV Shows       0.08      0.09      0.08       270
#                     Kids' TV       0.44      0.36      0.39        90
#              Korean TV Shows       0.00      0.00      0.00        30
#                 LGBTQ Movies       0.00      0.00      0.00        20
#                       Movies       0.00      0.00      0.00        11
#             Music & Musicals       0.05      0.01      0.02        75
#                   Reality TV       0.47      0.16      0.24        51
#              Romantic Movies       0.00      0.00      0.00       123
#            Romantic TV Shows       0.00      0.00      0.00        74
#             Sci-Fi & Fantasy       0.00      0.00      0.00        49
#          Science & Nature TV       0.00      0.00      0.00        18
#    Spanish-Language TV Shows       0.00      0.00      0.00        35
#                Sports Movies       0.11      0.02      0.04        44
#              Stand-Up Comedy       0.82      0.65      0.73        69
# Stand-Up Comedy & Talk Shows       0.00      0.00      0.00        11
#        TV Action & Adventure       0.00      0.00      0.00        34
#                  TV Comedies       0.23      0.05      0.08       116
#                    TV Dramas       0.02      0.01      0.01       153
#                    TV Horror       0.00      0.00      0.00        15
#                 TV Mysteries       0.00      0.00      0.00        20
#          TV Sci-Fi & Fantasy       0.00      0.00      0.00        17
#                     TV Shows       0.00      0.00      0.00         3
#                 TV Thrillers       0.00      0.00      0.00        11
#                Teen TV Shows       0.00      0.00      0.00        14
#                    Thrillers       0.07      0.01      0.02       115

#                     accuracy                           0.14      3865
#                    macro avg       0.10      0.07      0.08      3865
#                 weighted avg       0.13      0.14      0.13      3865

# Predicted Genre: Crime TV Shows