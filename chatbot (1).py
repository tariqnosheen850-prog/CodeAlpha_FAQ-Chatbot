"""
FAQ Chatbot
-----------
Task 2 - Artificial Intelligence Tasks

Pipeline:
1. Load FAQs (question, answer) from faqs.json
2. Preprocess text with NLTK (tokenize, lowercase, remove stopwords/punctuation, lemmatize)
3. Vectorize preprocessed FAQ questions with TF-IDF
4. On each user query: preprocess -> vectorize -> cosine similarity vs all FAQ questions
5. Return the answer of the best-matching FAQ (if similarity is above a threshold)
"""

import json
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# One-time NLTK downloads (safe to call every run; no-op if already present)
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Similarity below this value = "I don't understand" fallback
CONFIDENCE_THRESHOLD = 0.25


def preprocess(text: str) -> str:
    """Tokenize, lowercase, strip punctuation/stopwords, lemmatize."""
    tokens = word_tokenize(text.lower())
    cleaned = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in string.punctuation and tok not in STOP_WORDS and tok.isalpha()
    ]
    return " ".join(cleaned)


class FAQChatbot:
    def __init__(self, faq_path: str):
        with open(faq_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        self.questions = [item["question"] for item in self.faqs]
        self.answers = [item["answer"] for item in self.faqs]

        # Preprocess all FAQ questions once, up front
        self.processed_questions = [preprocess(q) for q in self.questions]

        # Fit TF-IDF on the FAQ question corpus
        self.vectorizer = TfidfVectorizer()
        self.faq_vectors = self.vectorizer.fit_transform(self.processed_questions)

    def get_response(self, user_query: str):
        """Return (answer, matched_question, similarity_score)."""
        processed_query = preprocess(user_query)
        query_vector = self.vectorizer.transform([processed_query])

        similarities = cosine_similarity(query_vector, self.faq_vectors).flatten()
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        if best_score < CONFIDENCE_THRESHOLD:
            return (
                "Sorry, I don't have an answer for that. Could you rephrase your question "
                "or contact support@example.com?",
                None,
                best_score,
            )

        return self.answers[best_idx], self.questions[best_idx], best_score


def run_cli(faq_path="faqs.json"):
    bot = FAQChatbot(faq_path)
    print("FAQ Chatbot ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Bot: Goodbye!")
            break
        if not user_input:
            continue

        answer, matched_q, score = bot.get_response(user_input)
        print(f"Bot: {answer}")
        if matched_q:
            print(f"     (matched: \"{matched_q}\" | similarity: {score:.2f})\n")
        else:
            print()


if __name__ == "__main__":
    run_cli()
