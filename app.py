
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import build_inverted_index, evaluate_boolean_query, generate_pdf_report

st.set_page_config(page_title="Inverted Index Virtual Lab", page_icon="🔎", layout="wide")

DATA_PATH = Path(__file__).parent / "data.json"


@st.cache_data
def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


DATA = load_data()
STOPWORDS = set(DATA["stopwords"])
CORPORA = DATA["corpora"]
QUIZ = DATA["quiz"]

if "trials" not in st.session_state:
    st.session_state.trials = []
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

st.sidebar.title("🔎 Inverted Index Lab")
section = st.sidebar.radio(
    "Navigate",
    ["Theory", "Simulation", "Quiz", "Report Generation"],
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# THEORY
# ---------------------------------------------------------------------------
if section == "Theory":
    st.title("Construction of an Inverted Index")

    st.header("Aim")
    st.write(
        "To create an inverted index that maps terms to the documents (and their "
        "occurrence counts) in which they appear, forming a basic searchable index "
        "that supports efficient keyword retrieval."
    )

    st.header("Background Theory")
    st.markdown(
        """
An **inverted index** is the core data structure behind nearly every modern search
engine. Rather than storing, for each document, the list of terms it contains (a
*forward index*), an inverted index stores, for each **term**, the list of
**documents** in which it occurs — along with how many times it occurs there
(the *term frequency*), and optionally at which positions.

This inversion is what makes keyword search fast: given a query term, the search
engine does not need to scan every document in the collection. It simply looks
up the term in the index and retrieves its **postings list** — the set of
documents (and occurrence data) already associated with that term.

**Index construction pipeline:**

1. **Document collection** — gather the set of documents to be indexed, each
   assigned a unique document ID.
2. **Tokenization** — split the raw text of each document into individual
   terms (tokens), typically on whitespace and punctuation.
3. **Normalization** — apply case-folding (lowercasing) and strip punctuation
   so that different surface forms of the same token are treated identically.
4. **Stopword removal (optional)** — discard very frequent, low-information
   words (e.g. "the", "is", "and") to shrink the index and reduce noise.
5. **Stemming / Lemmatization (optional)** — reduce inflected or derived
   words to a common root (e.g. "running" → "run") so that related forms are
   indexed under a single term.
6. **Postings construction** — for every remaining token in every document,
   record an entry `(term, doc_id)` and accumulate term frequency.
7. **Dictionary/vocabulary assembly** — collect the set of unique terms
   (the vocabulary) and associate each with its postings list, typically
   sorted alphabetically for efficient lookup (e.g. via a B-tree or hash
   table in a real system).
8. **Query processing** — at search time, look up each query term's postings
   list and combine multiple lists using set operations for boolean queries
   (AND = intersection, OR = union, NOT = complement).
        """
    )

    st.header("Key Terminology")
    terms = {
        "Term / Token": "A single word (or word-like unit) extracted from a document during tokenization.",
        "Document": "One unit of text in the collection, identified by a unique document ID.",
        "Vocabulary (Dictionary)": "The set of all distinct terms across the entire document collection.",
        "Postings List": "For a given term, the list of documents (and occurrence data) in which it appears.",
        "Term Frequency (TF)": "The number of times a term occurs within one specific document.",
        "Document Frequency (DF)": "The number of documents in the collection that contain a term at least once.",
        "Stopwords": "Very common words (e.g. 'the', 'a', 'is') that are often excluded from the index.",
        "Stemming": "Reducing a word to an approximate root form by stripping suffixes.",
        "Boolean Query": "A query combining terms with AND, OR, and NOT operators over postings lists.",
    }
    for term, definition in terms.items():
        st.markdown(f"- **{term}** — {definition}")

    st.header("Step-by-Step Procedure")
    st.markdown(
        """
1. Select or enter a document collection.
2. Choose preprocessing options: lowercasing, stopword removal, stemming.
3. The lab tokenizes and normalizes every document automatically.
4. For each surviving token, the term is added to the vocabulary and its
   posting `(doc_id, frequency)` is recorded or updated.
5. Inspect the resulting inverted index: vocabulary, postings lists, and
   document frequencies.
6. Issue a boolean query (e.g. `brutus AND caesar`, `caesar OR antony`,
   `NOT caesar`) and observe which documents are retrieved by intersecting,
   unioning, or complementing postings lists.
7. Record trials to include in your final report.
        """
    )

    st.header("References")
    st.markdown(
        """
1. C. D. Manning, P. Raghavan, and H. Schütze, *Introduction to Information
   Retrieval*, Cambridge University Press, 2008.
2. R. Baeza-Yates and B. Ribeiro-Neto, *Modern Information Retrieval*,
   2nd Edition, Addison-Wesley, 2011.
3. G. Salton and M. J. McGill, *Introduction to Modern Information Retrieval*,
   McGraw-Hill, 1983.
        """
    )

# ---------------------------------------------------------------------------
# SIMULATION
# ---------------------------------------------------------------------------
elif section == "Simulation":
    st.title("Simulation: Build an Inverted Index")

    left, right = st.columns([1, 2])

    with left:
        st.subheader("1. Choose a Corpus")
        corpus_options = {v["label"]: k for k, v in CORPORA.items()}
        corpus_choice_label = st.selectbox(
            "Document corpus", list(corpus_options.keys()) + ["Custom input"]
        )

        if corpus_choice_label == "Custom input":
            st.caption("Enter one document per line.")
            custom_text = st.text_area(
                "Custom documents",
                value=(
                    "The cat sat on the mat.\n"
                    "The dog chased the cat around the yard.\n"
                    "Cats and dogs can be great pets."
                ),
                height=150,
            )
            lines = [line.strip() for line in custom_text.split("\n") if line.strip()]
            documents = {f"Doc{i+1}": line for i, line in enumerate(lines)}
            corpus_label = "Custom input"
        else:
            corpus_key = corpus_options[corpus_choice_label]
            documents = CORPORA[corpus_key]["documents"]
            corpus_label = corpus_choice_label

        st.subheader("2. Preprocessing Options")
        lowercase = st.checkbox("Convert to lowercase", value=True)
        remove_stopwords = st.checkbox("Remove stopwords", value=False)
        apply_stemming = st.checkbox("Apply simple stemming", value=False)

        options_summary = (
            f"lowercase={lowercase}, remove_stopwords={remove_stopwords}, "
            f"stemming={apply_stemming}"
        )

    if not documents:
        st.warning("Add at least one document to build an index.")
        st.stop()

    index, doc_tokens = build_inverted_index(
        documents,
        lowercase=lowercase,
        remove_stopwords=remove_stopwords,
        apply_stemming=apply_stemming,
        stopwords=STOPWORDS,
    )

    with right:
        st.subheader("Document Collection")
        doc_df = pd.DataFrame(
            [{"Doc ID": k, "Text": v, "Token Count": len(doc_tokens[k])} for k, v in documents.items()]
        )
        st.dataframe(doc_df, width="stretch", hide_index=True)

        st.subheader(f"Constructed Inverted Index ({len(index)} terms)")
        index_rows = []
        for term in sorted(index.keys()):
            postings = index[term]
            postings_str = ", ".join(f"{doc}:{freq}" for doc, freq in sorted(postings.items()))
            index_rows.append(
                {
                    "Term": term,
                    "Document Frequency": len(postings),
                    "Postings (doc:freq)": postings_str,
                }
            )
        index_df = pd.DataFrame(index_rows)
        st.dataframe(index_df, width="stretch", hide_index=True, height=350)

    st.divider()
    st.subheader("3. Boolean Query Simulation")
    st.caption(
        "Use terms combined with AND / OR / NOT, e.g. `brutus AND caesar`, "
        "`caesar OR antony`, `NOT caesar`. Terms are matched against the index above."
    )
    query = st.text_input("Query", value="")

    all_doc_ids = set(documents.keys())
    matches, trace = ([], [])
    if query.strip():
        matches, trace = evaluate_boolean_query(
            query, index, all_doc_ids, lowercase=lowercase, apply_stemming=apply_stemming
        )
        st.write("**Evaluation trace:**")
        for step in trace:
            st.text(step)
        st.write("**Matching documents:**")
        if matches:
            for doc_id in sorted(matches):
                st.success(f"{doc_id}: {documents[doc_id]}")
        else:
            st.info("No documents matched this query.")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("➕ Record this trial", type="primary"):
            st.session_state.trials.append(
                {
                    "corpus_label": corpus_label,
                    "options_summary": options_summary,
                    "vocab_size": len(index),
                    "query": query.strip(),
                    "matches": sorted(matches) if query.strip() else [],
                }
            )
            st.toast("Trial recorded! View it in Report Generation.", icon="✅")
    with col_b:
        st.caption(f"Trials recorded so far: {len(st.session_state.trials)}")

# ---------------------------------------------------------------------------
# QUIZ
# ---------------------------------------------------------------------------
elif section == "Quiz":
    st.title("Quiz: Inverted Index Concepts")
    st.caption("Answer each question below. Feedback is shown instantly.")

    correct_count = 0
    answered_count = 0

    for i, q in enumerate(QUIZ):
        st.markdown(f"**Q{i+1}. {q['question']}**")
        choice = st.radio(
            f"q_{i}",
            options=list(range(len(q["options"]))),
            format_func=lambda idx, opts=q["options"]: opts[idx],
            index=None,
            key=f"quiz_radio_{i}",
            label_visibility="collapsed",
        )
        if choice is not None:
            answered_count += 1
            st.session_state.quiz_answers[i] = choice
            if choice == q["answer"]:
                correct_count += 1
                st.success(f"Correct! {q['explanation']}")
            else:
                st.error(
                    f"Not quite. Correct answer: **{q['options'][q['answer']]}**. "
                    f"{q['explanation']}"
                )
        st.divider()

    st.subheader("Score")
    if answered_count > 0:
        pct = 100 * correct_count / answered_count
        st.metric("Correct", f"{correct_count} / {answered_count}", f"{pct:.1f}%")
        st.session_state.quiz_score = {
            "score": correct_count,
            "total": answered_count,
            "percent": pct,
        }
    else:
        st.info("Answer at least one question to see your score.")

# ---------------------------------------------------------------------------
# REPORT GENERATION
# ---------------------------------------------------------------------------
elif section == "Report Generation":
    st.title("Report Generation")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Student Name")
    with col2:
        roll = st.text_input("Roll Number")

    st.subheader("Recorded Simulation Trials")
    if not st.session_state.trials:
        st.info("No trials recorded yet. Go to the Simulation section and click 'Record this trial'.")
    else:
        trials_df = pd.DataFrame(
            [
                {
                    "#": i + 1,
                    "Corpus": t["corpus_label"],
                    "Options": t["options_summary"],
                    "Vocab Size": t["vocab_size"],
                    "Query": t["query"] or "-",
                    "Matches": ", ".join(t["matches"]) if t["matches"] else "-",
                }
                for i, t in enumerate(st.session_state.trials)
            ]
        )
        st.dataframe(trials_df, width="stretch", hide_index=True)

        if st.button("Clear all recorded trials"):
            st.session_state.trials = []
            st.rerun()

    quiz_result = st.session_state.get("quiz_score")
    if quiz_result:
        st.subheader("Quiz Result")
        st.write(f"Score: {quiz_result['score']} / {quiz_result['total']} ({quiz_result['percent']:.1f}%)")
    else:
        st.caption("No quiz attempted yet — complete the Quiz section to include your score in the report.")

    st.divider()
    if st.button("📄 Generate PDF Report", type="primary", disabled=not name or not roll):
        pdf_bytes = generate_pdf_report(name, roll, st.session_state.trials, quiz_result)
        st.download_button(
            "⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"inverted_index_report_{roll or 'student'}.pdf",
            mime="application/pdf",
        )
    if not name or not roll:
        st.caption("Enter your name and roll number to enable report generation.")