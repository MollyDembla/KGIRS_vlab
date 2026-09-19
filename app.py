
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
PRETEST = QUIZ[:5]
POSTTEST = QUIZ[5:]

if "trials" not in st.session_state:
    st.session_state.trials = []
if "pretest_answers" not in st.session_state:
    st.session_state.pretest_answers = {}
if "posttest_answers" not in st.session_state:
    st.session_state.posttest_answers = {}

st.sidebar.title("🔎 Inverted Index Lab")
section = st.sidebar.radio(
    "Navigate",
    ["Aim", "Theory", "Procedure", "Simulation", "Pretest", "Posttest", "Report Generation"],
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# AIM
# ---------------------------------------------------------------------------
if section == "Aim":
    st.title("Construction of an Inverted Index")

    st.header("Aim")
    st.write(
        "To create an inverted index that maps terms to the documents (and their "
        "occurrence counts) in which they appear, forming a basic searchable index "
        "that supports efficient keyword retrieval."
    )

    st.header("Objectives")
    objectives = [
        "Understand the concept and structure of an inverted index.",
        "Tokenize documents into individual terms.",
        "Apply text normalization: case-folding, stopword removal, and stemming.",
        "Map each term to the documents in which it occurs and record occurrence frequency.",
        "Construct and inspect postings lists and the vocabulary (dictionary).",
        "Perform keyword searches using boolean queries (AND / OR / NOT) over the constructed index.",
        "Observe how an inverted index supports efficient document retrieval.",
    ]
    for i, obj in enumerate(objectives, 1):
        st.write(f"**Objective {i}:** {obj}")

    st.header("Expected Outcome")
    st.success(
        "A basic searchable index supporting efficient keyword retrieval, with the "
        "ability to execute boolean queries and retrieve matching documents."
    )

# ---------------------------------------------------------------------------
# THEORY
# ---------------------------------------------------------------------------
elif section == "Theory":
    st.title("Theory")

    st.header("Background")
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

### Example

Suppose we have three documents:

- **D1:** apple banana orange
- **D2:** banana mango apple
- **D3:** orange mango banana

The inverted index contains:

| Term | Posting List |
|---|---|
| apple | D1(1), D2(1) |
| banana | D1(1), D2(1), D3(1) |
| mango | D2(1), D3(1) |
| orange | D1(1), D3(1) |

### Index construction pipeline

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
    terms_df = pd.DataFrame(
        list(terms.items()), columns=["Term", "Definition"]
    )
    st.table(terms_df)

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
# PROCEDURE
# ---------------------------------------------------------------------------
elif section == "Procedure":
    st.title("Experimental Procedure")

    steps = [
        "Step 1: Select or enter a document collection (corpus).",
        "Step 2: Choose preprocessing options: lowercasing, stopword removal, stemming.",
        "Step 3: Tokenize each document into individual terms.",
        "Step 4: Convert terms to lowercase and remove unnecessary punctuation.",
        "Step 5: Optionally remove stopwords and apply stemming to reduce terms to root forms.",
        "Step 6: Count the occurrence of every term in each document.",
        "Step 7: Construct the inverted index by mapping every term to its postings (doc_id, frequency).",
        "Step 8: Observe the generated term dictionary and posting lists.",
        "Step 9: Issue a boolean query (e.g. `brutus AND caesar`, `caesar OR antony`, `NOT caesar`).",
        "Step 10: Retrieve the documents matching the query by intersecting, unioning, or complementing postings lists.",
        "Step 11: Record the trial and observe the retrieval results.",
        "Step 12: Complete the Pretest and Posttest quizzes.",
        "Step 13: Generate and download the experiment report.",
    ]
    for step in steps:
        st.write(f"- {step}")

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
    query = st.text_input("Query", value="", key="query_input")

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
# PRETEST
# ---------------------------------------------------------------------------
elif section == "Pretest":
    st.title("Pretest: Inverted Index Concepts")
    st.caption("Answer each question below to test your prior understanding. Feedback is shown instantly.")

    correct_count = 0
    answered_count = 0

    for i, q in enumerate(PRETEST):
        st.markdown(f"**Q{i+1}. {q['question']}**")
        choice = st.radio(
            f"pretest_q_{i}",
            options=list(range(len(q["options"]))),
            format_func=lambda idx, opts=q["options"]: opts[idx],
            index=None,
            key=f"pretest_radio_{i}",
            label_visibility="collapsed",
        )
        if choice is not None:
            answered_count += 1
            st.session_state.pretest_answers[i] = choice
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
        st.session_state.pretest_score = {
            "score": correct_count,
            "total": answered_count,
            "percent": pct,
        }
    else:
        st.info("Answer at least one question to see your score.")

# ---------------------------------------------------------------------------
# POSTTEST
# ---------------------------------------------------------------------------
elif section == "Posttest":
    st.title("Posttest: Inverted Index Concepts")
    st.caption("Answer each question below to test what you learned from the simulation. Feedback is shown instantly.")

    correct_count = 0
    answered_count = 0

    for i, q in enumerate(POSTTEST):
        st.markdown(f"**Q{i+1}. {q['question']}**")
        choice = st.radio(
            f"posttest_q_{i}",
            options=list(range(len(q["options"]))),
            format_func=lambda idx, opts=q["options"]: opts[idx],
            index=None,
            key=f"posttest_radio_{i}",
            label_visibility="collapsed",
        )
        if choice is not None:
            answered_count += 1
            st.session_state.posttest_answers[i] = choice
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
        st.session_state.posttest_score = {
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

    pretest_result = st.session_state.get("pretest_score")
    posttest_result = st.session_state.get("posttest_score")

    st.subheader("Quiz Results")
    if pretest_result:
        st.write(f"**Pretest:** {pretest_result['score']} / {pretest_result['total']} ({pretest_result['percent']:.1f}%)")
    else:
        st.caption("Pretest: Not attempted yet")

    if posttest_result:
        st.write(f"**Posttest:** {posttest_result['score']} / {posttest_result['total']} ({posttest_result['percent']:.1f}%)")
    else:
        st.caption("Posttest: Not attempted yet")

    # Combine for PDF
    quiz_result = None
    if pretest_result or posttest_result:
        quiz_result = {
            "pretest": pretest_result,
            "posttest": posttest_result,
        }

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
