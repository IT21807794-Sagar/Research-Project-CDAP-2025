from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

file_path = 'Dataset/Dataset.csv'
df = pd.read_csv(file_path)

# Separate features from identifiers
student_ids = df['student_id']
X = df.drop('student_id', axis=1)

# Define which columns are numerical and which are categorical
numerical_cols = ['age', 'reading_level', 'math_level', 'art_interest',
                 'music_interest', 'sports_interest', 'science_interest',
                 'storytelling_interest', 'attention_span', 'social_interaction']
categorical_cols = ['preferred_activity', 'learning_style', 'energy_level',
                   'special_needs', 'friendship_groups']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
