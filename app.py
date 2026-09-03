import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="AI Hashtag Generator",
    page_icon="🔖",
    layout="centered"
)


# --------------------------------------------------
# DATASET
# --------------------------------------------------

DATA = [
    ("Beautiful sunset at Marine Drive Mumbai",
     "#sunset #mumbai #marinedrive #travel"),

    ("Amazing cricket match with friends today",
     "#cricket #match #friends #sports"),

    ("Delicious pizza with my friends",
     "#pizza #food #friends #foodie"),

    ("Exploring beautiful places in Goa",
     "#goa #travel #explore #nature"),

    ("Best birthday celebration with family",
     "#birthday #celebration #family #memories"),

    ("Morning workout at the gym",
     "#workout #gym #fitness #health"),

    ("Coding a new Python project today",
     "#python #coding #programming #technology"),

    ("Beautiful mountains and peaceful nature",
     "#mountains #nature #travel #peace"),

    ("College friends enjoying a fun day",
     "#college #friends #fun #memories"),

    ("Photography of a beautiful sunset",
     "#photography #sunset #nature #photo"),

    ("Football game with our team",
     "#football #sports #team #game"),

    ("Delicious Indian food for dinner",
     "#food #indianfood #dinner #foodie"),

    ("Weekend trip to a beautiful beach",
     "#beach #travel #weekend #vacation"),

    ("New graphic design project completed",
     "#graphicdesign #design #creative #project"),

    ("Learning artificial intelligence and machine learning",
     "#ai #machinelearning #technology #learning"),

    ("Peaceful evening with beautiful flowers",
     "#flowers #nature #peace #evening"),

    ("Traveling with friends and exploring new places",
     "#travel #friends #explore #adventure"),

    ("Winning the cricket tournament with our team",
     "#cricket #tournament #winner #team"),

    ("Healthy vegetarian food and fitness lifestyle",
     "#vegetarian #food #fitness #health"),

    ("Working on a college mini project",
     "#college #project #student #technology"),

    ("Beautiful Mumbai city lights at night",
     "#mumbai #citylights #night #photography"),

    ("Enjoying coffee on a rainy morning",
     "#coffee #rain #morning #lifestyle"),

    ("Creative poster design for a birthday",
     "#posterdesign #birthday #design #creative"),

    ("Coding and learning new technologies",
     "#coding #technology #learning #programming"),

    ("A relaxing day at the beach",
     "#beach #relax #travel #vacation")
]


df = pd.DataFrame(
    DATA,
    columns=["text", "hashtags"]
)


# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------

def preprocess(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove special characters and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Split words
    words = text.split()

    # Remove stop words
    words = [
        word
        for word in words
        if word not in ENGLISH_STOP_WORDS
        and len(word) > 2
    ]

    return " ".join(words)


clean_texts = df["text"].apply(preprocess)


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=500
)

tfidf_matrix = vectorizer.fit_transform(
    clean_texts
)


# --------------------------------------------------
# CATEGORY DETECTION
# --------------------------------------------------

def detect_category(text):

    text = text.lower()

    categories = {

        "🏏 Sports": [
            "cricket",
            "football",
            "match",
            "sports",
            "game",
            "player",
            "team",
            "tournament"
        ],

        "✈️ Travel": [
            "travel",
            "trip",
            "goa",
            "mumbai",
            "beach",
            "vacation",
            "tour",
            "explore",
            "journey"
        ],

        "🍕 Food": [
            "food",
            "pizza",
            "burger",
            "restaurant",
            "dinner",
            "lunch",
            "breakfast",
            "cooking"
        ],

        "💻 Technology": [
            "python",
            "coding",
            "technology",
            "computer",
            "programming",
            "software",
            "artificial intelligence",
            "machine learning"
        ],

        "🎂 Birthday": [
            "birthday",
            "cake",
            "celebration",
            "party",
            "gift"
        ],

        "🌿 Nature": [
            "nature",
            "mountain",
            "mountains",
            "flowers",
            "sunset",
            "forest",
            "river",
            "garden"
        ],

        "💪 Fitness": [
            "gym",
            "workout",
            "fitness",
            "exercise",
            "health",
            "running",
            "training"
        ]
    }

    category_scores = {}

    for category, keywords in categories.items():

        score = 0

        for keyword in keywords:

            if keyword in text:
                score += 1

        category_scores[category] = score

    best_category = max(
        category_scores,
        key=category_scores.get
    )

    if category_scores[best_category] == 0:
        return "📌 General"

    return best_category


# --------------------------------------------------
# HASHTAG GENERATION
# --------------------------------------------------

def generate_hashtags(
    user_text,
    number_of_hashtags
):

    cleaned_text = preprocess(user_text)

    if not cleaned_text:
        return [], [], 0

    # Convert user caption into TF-IDF vector
    user_vector = vectorizer.transform(
        [cleaned_text]
    )

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    )[0]

    # Get top 5 similar captions
    best_indices = similarity_scores.argsort()[::-1][:5]

    hashtag_scores = {}

    # --------------------------------------------------
    # Similarity based hashtag scoring
    # --------------------------------------------------

    for index in best_indices:

        similarity = float(
            similarity_scores[index]
        )

        if similarity <= 0:
            continue

        hashtags = df.iloc[index]["hashtags"].split()

        for hashtag in hashtags:

            hashtag = hashtag.lower()

            hashtag_scores[hashtag] = (
                hashtag_scores.get(hashtag, 0)
                + similarity
            )

    # --------------------------------------------------
    # Keyword matching bonus
    # --------------------------------------------------

    input_words = set(
        cleaned_text.split()
    )

    for index in best_indices:

        caption_words = set(
            preprocess(
                df.iloc[index]["text"]
            ).split()
        )

        common_words = (
            input_words.intersection(
                caption_words
            )
        )

        hashtags = df.iloc[index]["hashtags"].split()

        for hashtag in hashtags:

            hashtag = hashtag.lower()

            hashtag_scores[hashtag] = (
                hashtag_scores.get(hashtag, 0)
                + len(common_words) * 0.3
            )

    # --------------------------------------------------
    # Rank hashtags
    # --------------------------------------------------

    ranked_hashtags = sorted(
        hashtag_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    final_hashtags = []

    for hashtag, score in ranked_hashtags:

        if hashtag not in final_hashtags:
            final_hashtags.append(hashtag)

        if len(final_hashtags) >= number_of_hashtags:
            break

    # --------------------------------------------------
    # Important TF-IDF keywords
    # --------------------------------------------------

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    tfidf_values = user_vector.toarray()[0]

    keyword_scores = []

    for i, value in enumerate(tfidf_values):

        if value > 0:

            keyword_scores.append(
                (feature_names[i], value)
            )

    keyword_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    important_keywords = [
        word
        for word, score in keyword_scores[:5]
    ]

    # --------------------------------------------------
    # Similarity percentage
    # --------------------------------------------------

    valid_scores = [
        similarity_scores[i]
        for i in best_indices
        if similarity_scores[i] > 0
    ]

    if valid_scores:

        similarity_percentage = round(
            max(valid_scores) * 100,
            2
        )

    else:

        similarity_percentage = 0

    return (
        final_hashtags,
        important_keywords,
        similarity_percentage
    )


# --------------------------------------------------
# USER INTERFACE
# --------------------------------------------------

st.title(
    "🔖 AI Automatic Hashtag Generator"
)

st.write(
    "Generate relevant hashtags using "
    "NLP, TF-IDF and Cosine Similarity."
)

st.divider()


# Caption input

caption = st.text_area(
    "📝 Enter your caption",

    placeholder=(
        "Example: Amazing cricket match "
        "with my friends today"
    ),

    height=120
)


# Number of hashtags

number = st.slider(
    "🔢 Number of hashtags",

    min_value=3,
    max_value=10,

    value=6
)


# Generate button

if st.button(
    "✨ Generate Hashtags",
    use_container_width=True
):

    if caption.strip() == "":

        st.warning(
            "⚠️ Please enter a caption."
        )

    else:

        hashtags, keywords, similarity = (
            generate_hashtags(
                caption,
                number
            )
        )

        if hashtags:

            st.success(
                "✅ Hashtags Generated Successfully!"
            )

            # --------------------------------------------------
            # CATEGORY
            # --------------------------------------------------

            category = detect_category(
                caption
            )

            st.subheader(
                "📂 Detected Category"
            )

            st.info(
                f"Category: {category}"
            )

            # --------------------------------------------------
            # HASHTAGS
            # --------------------------------------------------

            st.subheader(
                "🏷️ Generated Hashtags"
            )

            hashtag_text = " ".join(
                hashtags
            )

            st.code(
                hashtag_text,
                language="text"
            )

            st.caption(
                "📋 Use the copy icon on the "
                "hashtag box to copy all hashtags."
            )

            # --------------------------------------------------
            # IMPORTANT KEYWORDS
            # --------------------------------------------------

            st.subheader(
                "🔑 Important Keywords"
            )

            if keywords:

                st.write(
                    ", ".join(keywords)
                )

            else:

                st.write(
                    "No important keywords found."
                )

            # --------------------------------------------------
            # SIMILARITY SCORE
            # --------------------------------------------------

            st.subheader(
                "📊 Similarity Score"
            )

            st.progress(
                min(similarity / 100, 1.0)
            )

            st.write(
                f"Similarity with dataset: "
                f"**{similarity}%**"
            )

            # --------------------------------------------------
            # SHARE RESULT
            # --------------------------------------------------

            st.subheader(
                "🔗 Share Result"
            )

            share_text = (
                f"Caption: {caption}\n\n"
                f"Category: {category}\n\n"
                f"Hashtags: {hashtag_text}"
            )

            st.code(
                share_text,
                language="text"
            )

            st.caption(
                "Use the copy icon in this box "
                "to copy and share the result."
            )

        else:

            st.info(
                "ℹ️ No relevant hashtags found. "
                "Try a longer or more specific caption."
            )


# --------------------------------------------------
# HOW PROJECT WORKS
# --------------------------------------------------

st.divider()

with st.expander(
    "🧠 How does this project work?"
):

    st.write(
        """
1. User enters a caption.
2. Text preprocessing cleans the caption.
3. TF-IDF identifies important words.
4. Cosine Similarity compares the caption with captions in the dataset.
5. Similar captions are selected.
6. Relevant hashtags are ranked.
7. Category Detection identifies the caption category.
8. Final hashtags are displayed.
"""
    )


# --------------------------------------------------
# ALGORITHMS USED
# --------------------------------------------------

with st.expander(
    "📚 Algorithms Used"
):

    st.write(
        """
**TF-IDF**

Term Frequency - Inverse Document Frequency
is used to identify important words in the caption.

**Cosine Similarity**

Cosine Similarity measures how similar the
user's caption is to captions available in
the dataset.

**Keyword Matching**

Common keywords between the user caption
and dataset captions improve hashtag ranking.

**Category Detection**

Keyword-based classification detects
Sports, Travel, Food, Technology, Birthday,
Nature and Fitness.
"""
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "🤖 AI Automatic Hashtag Generator | "
    "NLP • TF-IDF • Cosine Similarity"
)