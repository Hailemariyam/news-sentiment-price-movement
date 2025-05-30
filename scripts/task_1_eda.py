# 📦 Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 📁 Load the Dataset
df = pd.read_csv("raw_analyst_ratings.csv")

# 🧹 Preprocess 'date' column
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# ➕ Feature Engineering
df['headline_length'] = df['headline'].apply(len)
df['publish_date'] = df['date'].dt.date
df['publish_hour'] = df['date'].dt.hour

# -------------------------------------------
# 📊 1. DESCRIPTIVE STATISTICS
# -------------------------------------------

# 1.1 Headline Length Statistics
print("📏 Headline Length Stats:")
print(df['headline_length'].describe())

# 1.2 Articles Per Publisher
print("\n🏢 Top Publishers:")
top_publishers = df['publisher'].value_counts().head(10)
print(top_publishers)

# 1.3 Articles Per Day
daily_counts = df.groupby('publish_date').size()
plt.figure(figsize=(14, 5))
daily_counts.plot()
plt.title("🗓 Articles Published Per Day")
plt.xlabel("Date")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------
# 🧠 2. TOPIC MODELING (LDA)
# -------------------------------------------

# Drop NaNs for topic modeling
df_nlp = df.dropna(subset=['headline'])

vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=10)
X = vectorizer.fit_transform(df_nlp['headline'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

print("\n🧾 Top 10 Words per Topic:")
for idx, topic in enumerate(lda.components_):
    print(f"\nTopic {idx + 1}:")
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])

# -------------------------------------------
# ⏰ 3. TIME SERIES ANALYSIS
# -------------------------------------------

# 3.1 Publishing Hour Distribution
plt.figure(figsize=(10, 5))
df['publish_hour'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("⏰ Publishing Hour Distribution")
plt.xlabel("Hour of Day (UTC-4)")
plt.ylabel("Number of Articles")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------
# 📰 4. PUBLISHER ANALYSIS
# -------------------------------------------

# 4.1 Publisher Bar Chart
plt.figure(figsize=(8, 5))
top_publishers.sort_values().plot(kind='barh', color='lightgreen')
plt.title("🏢 Top 10 Publishers")
plt.xlabel("Article Count")
plt.tight_layout()
plt.show()

# 4.2 Publisher Domain Analysis (if email format is used)
df['publisher_domain'] = df['publisher'].str.extract(r'@(\S+)')
top_domains = df['publisher_domain'].value_counts().head(10)
print("\n📨 Top Publisher Email Domains:")
print(top_domains)
