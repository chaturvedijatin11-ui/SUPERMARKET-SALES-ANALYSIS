import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

def run_phase_1_pipeline(file_path, sample_size=50000):
    print(" Starting Phase 1: Heavy Data Analysis & Feature Engineering...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing {file_path}. Place Reviews.csv in data/raw/.")
        
    # 1. Load Data (Optimized for Large Files)
    print(" Loading dataset chunks...")
    chunks = []
    # Read in chunks to prevent system RAM crashes on massive data files
    for chunk in pd.read_csv(file_path, chunksize=20000):
        chunks.append(chunk)
    df = pd.concat(chunks, axis=0)
    print(f" Total rows successfully loaded: {df.shape[0]}")
    
    # 2. Downsample for rapid iteration during company-grade prototyping
    if sample_size and sample_size < len(df):
        print(f" Downsampling to {sample_size} records for clean execution performance...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
    # 3. Clean Missing Fields
    print("🧹 Cleaning missing textual values...")
    df['Summary'] = df['Summary'].fillna('')
    df['Text'] = df['Text'].fillna('')
    
    # 4. Heavy Feature Engineering 
    print(" Generating engineered feature matrices...")
    
    # Textual length extraction
    df['Text_Length'] = df['Text'].apply(len)
    df['Summary_Length'] = df['Summary'].apply(len)
    df['Word_Count'] = df['Text'].apply(lambda x: len(x.split()))
    
    # Helpfulness Ratio (Avoid division by zero bugs)
    df['Helpful_Ratio'] = np.where(
        df['HelpfulnessDenominator'] > 0, 
        df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'], 
        0.0
    )
    
    # Quick Sentiment Extraction using TextBlob 
    print(" Performing Sentiment Mining on review copies...")
    df['Sentiment_Polarity'] = df['Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['Sentiment_Subjectivity'] = df['Text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    
    # 5. Missing Value & Outlier Check Summary
    print("\n Dataset Metadata Profile Summary:")
    print(df[['Text_Length', 'Helpful_Ratio', 'Sentiment_Polarity', 'Score']].describe().T)
    
    # 6. Save Cleaned Training Set
    os.makedirs("data/processed", exist_ok=True)
    output_path = os.path.join("data", "processed", "cleaned_features.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\n Phase 1 Matrix saved successfully as: '{output_path}'")
    
    # 7. Generate Statistical Visuals for the Dashboard later
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Score', palette='viridis')
    plt.title('Distribution of Target Class (Customer Star Ratings)')
    os.makedirs("assets", exist_ok=True)
    plt.savefig(os.path.join("assets", "target_distribution.png"))
    plt.close()
    print(" Diagnostic metric visualization exported to 'assets/target_distribution.png'")

if __name__ == "__main__":
    # Point this to your actual file location
    run_phase_1_pipeline(os.path.join("data", "raw", "Reviews.csv"), sample_size=50000)
