# 🎬 Movie Similarity & Recommendation System
Graduate NLP Project | IMDB Dataset | Text Analytics | Machine Learning

### 🧩 TLDR

I used the four methods below to build a standardized movie search engine that also re-evaluates maturity ratings (G, PG, etc.), with the goal of normalizing content standards across countries and time periods.

1.   Data Wrangling and NLP Pipeline
2.   Text Similarity
3.   Text Summarization
4.   Document Clustering 

## ✨ Why I Built This

As I explored international films, I noticed two things:

- **Cultural standards differ.** Some countries (e.g. France, South Korea) allow more nudity or violence in youth-rated films than the US does.
- **Standards shift over time.** US movies from the 1980s often had more profanity in children's media than what would be accepted today.

This project began as an NLP assignment, but I expanded it to evaluate how movie content ratings vary by culture and era, and to build a tool that recommends similar films based on both genre and plot. It is the most technically comprehensive project I’ve done to date.

## 🎯 Project Objectives

- Recommend movies based on content and genre similarity.
- Generate coherent summaries from plot synopses.
- Predict content maturity levels (violence, nudity, profanity) using summary text.
- Reclassify movies into modern rating categories (G, PG, PG-13, R) using ML models.
- Surface inconsistencies in content standards across regions and decades.


##### 🧹 Data Wrangling & NLP Pipeline

- Kept only the **primary production country** for each movie.
- Cleaned and standardized **maturity columns**:
  - Fixed mislabeled values (e.g. movies like *Snow White* were marked "No Rating" for profanity, when they should be "None").
  - Movies with no rating data remained as "Not Rated."
- Standardized the **Name** column to improve search accuracy.
- Created a **Summary** column:
  - Used synopsis as the primary source.
  - If synopsis was missing, fallback to plot description to avoid nulls.

**NLP Preprocessing:**
- Applied regex cleaning, case folding, and stripping.
- Tokenized at the word level.
- Removed stopwords and lemmatized text.
- Generated a **normalized corpus**, stored in a column called `Summary_Cleaned`.

This cleaned summary text powers both **text similarity** and **document clustering** steps.


##### 🔍 Text Similarity

- One-hot encoded **genre** data.
- Computed **cosine similarity** matrices for:
  - One-hot encoded genres.
  - TF-IDF vectorized `Summary_Cleaned` text.
- Built a function that:
  - Takes a movie title, a list of movie names, and both similarity matrices.
  - Converts the title to its index and calculates similarity scores across both genre and summary.
  - Combines the scores and returns a ranked list of similar movies.
- Result: A single function call returns content- and genre-based movie recommendations.

##### 📝 Text Summarization

The summarization process was divided into two sections:

**Section 1 – Sentence-Level Normalization:**
- Applied a new regex pattern and standard NLP preprocessing.
- Tokenized summaries into individual sentences.
- Cleaned each sentence (stopword removal, lemmatization).
- Stored cleaned sentences in a list per movie in a new column: `Normalized_Summary`.

**Section 2 – Extractive Summarization:**
- Calculated cosine similarity between all normalized sentences.
- Ranked and selected the **top 10 most representative** sentences.
- Cleaned original sentences to fix formatting and punctuation issues.
- Mapped normalized sentences back to their original, coherent versions.
- Reordered the top 10 sentences based on original sequence (not similarity rank).
- Final summaries stored in a new column: `Summarized_Text`.


##### 📊 Document Clustering – Part 1: Predicting Maturity Categories

- Created a new `movies_cluster` dataframe with only relevant columns.
- Filtered to include only:
  - U.S.-produced films.
  - Movies with a valid certificate (G, PG, etc.).
  - Movies with at least one populated maturity category.
- Converted maturity ratings (e.g. "None", "Moderate") into ordinal values (0–4).
- Plotted the distribution of certificates across the dataset.
- Vectorized `Summary_Cleaned` using **TF-IDF**.
- Trained a **Random Forest** model to predict the five maturity categories:
  - Strong results for **Frightening**, **Violence**, and **Nudity**.
  - Poor accuracy on **Profanity**, moderate on **Alcohol**.
- Applied the trained model to a separate `foreign_movies` dataset and appended the predicted outputs.


##### 📈 Document Clustering – Part 2: Predicting Certificate Ratings

- Further trimmed the dataset for model training.
- Removed rows with unhelpful labels (e.g. "No Rate", "Not Rated").
- Converted certificate labels to ordinal values for modeling.
- Trained a **Random Forest** classifier to predict movie certificate (G, PG, PG-13, R).
- Evaluated performance using a confusion matrix—results were strong.
- Added predicted certificate labels to the dataframe.
- Merged predictions with the rest of the dataset (including movies excluded from training).
- Reorganized the final dataframe and exported the output.


