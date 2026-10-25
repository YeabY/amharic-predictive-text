
import streamlit as st
import joblib
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "amharic_ngram_model.joblib"

st.set_page_config(
    page_title="Amharic Predictive Text",
   # page_icon="✍️",
    layout="centered",
)

# --------------------------------------------------
# Load model only once
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Run the model-saving cell in your notebook first."
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the language model: {exc}")
    st.stop()

unigram_counts = model["unigram_counts"]
context_counts = model["context_counts"]
next_word_counts = model["next_word_counts"]
total_tokens = model["total_tokens"]
vocab_size = model["vocab_size"]


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def suggest_next_words(text, top_k=5):
    tokens = text.strip().split()

    # Trigram: use the last two words.
    if len(tokens) >= 2:
        context = tuple(tokens[-2:])
        counts = next_word_counts[2].get(context)

        if counts:
            denominator = context_counts[2][context] + vocab_size

            return [
                (word, (count + 1) / denominator)
                for word, count in counts.most_common(top_k)
            ]

    # Bigram: use the last word.
    if tokens:
        context = (tokens[-1],)
        counts = next_word_counts[1].get(context)

        if counts:
            denominator = context_counts[1][context] + vocab_size

            return [
                (word, (count + 1) / denominator)
                for word, count in counts.most_common(top_k)
            ]

    # Unigram fallback.
    denominator = total_tokens + vocab_size

    return [
        (word, (count + 1) / denominator)
        for word, count in unigram_counts.most_common(top_k)
    ]


# --------------------------------------------------
# Session state and button callbacks
# --------------------------------------------------

if "sentence" not in st.session_state:
    st.session_state.sentence = "ኢትዮጵያ ውስጥ የቡና"


def append_word(word):
    current = st.session_state.sentence.strip()

    if current:
        st.session_state.sentence = current + " " + word
    else:
        st.session_state.sentence = word


def clear_sentence():
    st.session_state.sentence = ""


# --------------------------------------------------
# User interface
# --------------------------------------------------

st.title(" Amharic Predictive Text")
st.caption(
    "Write in Amharic and get next-word suggestions "
    "powered by your GPAC n-gram language model."
)

st.markdown(
    """
    <style>
    .suggestion-note {
        color: #64748b;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.subheader("Compose your sentence")

st.text_area(
    "Amharic text",
    key="sentence",
    height=150,
    placeholder="እዚህ የአማርኛ ዓረፍተ ነገር ይጻፉ...",
)

left, right = st.columns([1, 1])

with left:
    st.button(
        "Clear text",
        on_click=clear_sentence,
        use_container_width=True,
    )

with right:
    st.button(
        "Refresh suggestions",
        use_container_width=True,
    )

sentence = st.session_state.sentence.strip()

st.divider()
st.subheader("Next-word suggestions")

if not sentence:
    st.info("Enter some Amharic text to get suggestions.")
else:
    suggestions = suggest_next_words(sentence, top_k=5)

    if suggestions:
        st.caption(
            "Select a word to append it to your sentence."
        )

        for index, (word, probability) in enumerate(
            suggestions, start=1
        ):
            col_word, col_prob, col_button = st.columns(
                [2.2, 1.5, 1.2],
                vertical_alignment="center",
            )

            with col_word:
                st.markdown(f"**{index}. {word}**")

            with col_prob:
                st.caption(f"P = {probability:.6g}")

            with col_button:
                st.button(
                    "Use word",
                    key=f"suggestion_{index}_{word}",
                    on_click=append_word,
                    args=(word,),
                    use_container_width=True,
                )
    else:
        st.warning("No suggestions found for this context.")

st.divider()

st.caption(
    "Model: GPAC | Trigram → Bigram → Unigram fallback"
)
