
"""Core logic for the Inverted Index virtual lab: preprocessing,
index construction, boolean query evaluation, and PDF report generation."""

import re
from datetime import datetime

from fpdf import FPDF
from fpdf.enums import XPos, YPos

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

# A tiny suffix-stripping stemmer for demonstration purposes.
_STEM_SUFFIXES = ["ing", "edly", "ed", "ly", "ies", "es", "s"]


def simple_stem(word: str) -> str:
    for suf in _STEM_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            if suf == "ies":
                return word[: -len(suf)] + "y"
            return word[: -len(suf)]
    return word


def tokenize(text: str, lowercase: bool) -> list:
    text = text.lower() if lowercase else text
    return TOKEN_RE.findall(text)


def preprocess(
    text: str,
    lowercase: bool = True,
    remove_stopwords: bool = False,
    apply_stemming: bool = False,
    stopwords: set = None,
) -> list:
    tokens = tokenize(text, lowercase)
    if remove_stopwords and stopwords:
        tokens = [t for t in tokens if t.lower() not in stopwords]
    if apply_stemming:
        tokens = [simple_stem(t) for t in tokens]
    return tokens


def build_inverted_index(documents: dict, **preprocess_kwargs):
    """documents: dict of {doc_id: raw_text}
    Returns (index, doc_tokens) where:
      index: {term: {doc_id: term_frequency}}
      doc_tokens: {doc_id: [tokens...]}
    """
    index = {}
    doc_tokens = {}
    for doc_id, text in documents.items():
        tokens = preprocess(text, **preprocess_kwargs)
        doc_tokens[doc_id] = tokens
        for term in tokens:
            postings = index.setdefault(term, {})
            postings[doc_id] = postings.get(doc_id, 0) + 1
    return index, doc_tokens


def evaluate_boolean_query(query: str, index: dict, all_doc_ids: set, lowercase: bool, apply_stemming: bool):
    """Very small left-to-right boolean evaluator supporting AND / OR / NOT."""
    raw_tokens = query.strip().split()
    if not raw_tokens:
        return set(), []

    def normalize_term(term: str) -> str:
        t = term.lower() if lowercase else term
        if apply_stemming:
            t = simple_stem(t)
        return t

    def get_docs(term: str) -> set:
        return set(index.get(normalize_term(term), {}).keys())

    result = None
    pending_op = None
    trace = []
    i = 0
    n = len(raw_tokens)
    while i < n:
        tok = raw_tokens[i]
        upper = tok.upper()
        if upper in ("AND", "OR"):
            pending_op = upper
            i += 1
            continue
        if upper == "NOT":
            i += 1
            if i >= n:
                break
            term = raw_tokens[i]
            docs = all_doc_ids - get_docs(term)
            trace.append(f"NOT {term} -> {sorted(docs)}")
            i += 1
        else:
            docs = get_docs(tok)
            trace.append(f"{tok} -> {sorted(docs)}")
            i += 1

        if result is None:
            result = docs
        elif pending_op == "AND":
            result = result & docs
            pending_op = None
        elif pending_op == "OR":
            result = result | docs
            pending_op = None
        else:
            # No explicit operator between terms: default to AND
            result = result & docs

    return (result if result is not None else set()), trace


class LabReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(
            0, 10, "Virtual Lab Report: Construction of an Inverted Index",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C",
        )
        self.set_font("Helvetica", "", 9)
        self.set_text_color(110, 110, 110)
        self.cell(
            0, 6, "Information Retrieval Virtual Laboratory",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C",
        )
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(230, 236, 245)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.ln(1)

    def body_text(self, text: str, size: int = 10):
        self.set_font("Helvetica", "", size)
        self.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def generate_pdf_report(name: str, roll: str, trials: list, quiz_result: dict = None) -> bytes:
    pdf = LabReportPDF()
    pdf.add_page()

    pdf.section_title("Student Details")
    pdf.body_text(f"Name: {name or '-'}")
    pdf.body_text(f"Roll Number: {roll or '-'}")
    pdf.body_text(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.ln(2)

    pdf.section_title("Aim")
    pdf.body_text(
        "To understand and implement the construction of an inverted index for a "
        "collection of text documents, enabling efficient boolean keyword retrieval."
    )
    pdf.ln(2)

    pdf.section_title(f"Recorded Simulation Trials ({len(trials)})")
    if not trials:
        pdf.body_text("No trials were recorded during this session.")
    else:
        for i, trial in enumerate(trials, start=1):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, f"Trial {i}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.body_text(f"Corpus: {trial['corpus_label']}")
            pdf.body_text(f"Preprocessing options: {trial['options_summary']}")
            pdf.body_text(f"Vocabulary size: {trial['vocab_size']} terms")
            if trial.get("query"):
                pdf.body_text(f"Query: {trial['query']}")
                pdf.body_text(f"Matching documents: {', '.join(trial['matches']) or 'None'}")
            pdf.ln(2)

    if quiz_result is not None:
        pdf.section_title("Quiz Result")
        pdf.body_text(
            f"Score: {quiz_result['score']} / {quiz_result['total']} "
            f"({quiz_result['percent']:.1f}%)"
        )
        pdf.ln(2)

    pdf.section_title("References")
    pdf.body_text(
        "1. C. D. Manning, P. Raghavan, and H. Schutze, Introduction to Information "
        "Retrieval, Cambridge University Press, 2008.\n"
        "2. R. Baeza-Yates and B. Ribeiro-Neto, Modern Information Retrieval, "
        "2nd Edition, Addison-Wesley, 2011.\n"
        "3. G. Salton and M. J. McGill, Introduction to Modern Information Retrieval, "
        "McGraw-Hill, 1983."
    )

    return bytes(pdf.output())