# ============================================================
# Streamlit web UI: Chat (RAG + image, memory, citations, save),
# Flashcards, Quiz, My notes, Export.
# ============================================================

import os
from pathlib import Path

import streamlit as st
from PIL import Image

import rag_engine
import store
import flashcards
import quiz
import export_notes

store.init_db()

CHROMA_DIR = rag_engine.CHROMA_DIR


@st.cache_resource
def load_models():
    if not os.path.isdir(CHROMA_DIR):
        raise FileNotFoundError(f"Run build_rag_index.py first. Missing {CHROMA_DIR}")
    embedder, coll = rag_engine.load_retriever()
    processor, model = rag_engine.load_medgemma()
    return embedder, coll, processor, model


def chat_tab(embedder, coll, processor, model):

    if "conversation" not in st.session_state:
        st.session_state.conversation = [
            {
                "question": None,
                "answer": (
                    "Hi, I’m the MedGemma Learning Assistant. "
                    "I help you explore medical topics, interpret images, "
                    "and learn through flashcards and quizzes. "
                    "Ask me anything when you're ready."
                ),
            }
        ]

    if "last_question" not in st.session_state:
        st.session_state.last_question = None
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []

    st.subheader("Chat (RAG + image)")
    uploaded = st.file_uploader("Upload image (optional)", type=["png", "jpg", "jpeg", "webp"], key="chat_image")
    analyze_clicked = st.button("Analyze uploaded image") if uploaded else False

    for h in st.session_state.conversation:
        with st.chat_message("user"):
            st.write(h["question"])
        with st.chat_message("assistant"):
            st.write(h["answer"])

    question = st.chat_input("Ask a medical question")
    if question or (analyze_clicked and uploaded is not None):
        question = question or "Describe the findings in this medical image."
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            q = question or "Describe the findings in this medical image."
            with st.spinner("Generating..."):
                answer, sources = rag_engine.ask_image(image, q, embedder, coll, processor, model)
            st.session_state.last_question = q
        else:
            history = [{"question": h["question"], "answer": h["answer"]} for h in st.session_state.conversation[-6:]]
            with st.spinner("Generating..."):
                answer, sources = rag_engine.ask_text(question, embedder, coll, processor, model, conversation_history=history)
            st.session_state.last_question = question

        st.session_state.last_answer = answer
        st.session_state.last_sources = sources or []
        st.session_state.conversation.append({
            "question": st.session_state.last_question,
            "answer": answer,
        })
        st.rerun()

    if st.session_state.last_answer is not None:
        if st.session_state.last_sources:
            with st.expander("Based on (sources)"):
                for i, s in enumerate(st.session_state.last_sources[:5], 1):
                    st.caption(f"[{i}] {s[:300]}..." if len(s) > 300 else f"[{i}] {s}")
        if st.button("Save last Q&A to My notes"):
            store.save_note(st.session_state.last_question, st.session_state.last_answer, st.session_state.last_sources)
            st.success("Saved.")
            st.rerun()


def flashcards_tab(embedder, coll, processor, model):
    st.subheader("Flashcards")
    sub = st.radio("Mode", ["Generate new", "Review deck"], horizontal=True)

    if sub == "Generate new":
        topic = st.text_input("Topic (e.g. corticosteroids and ARDS)")
        from_last = st.checkbox("Or generate from last chat Q&A (if you used Chat first)", value=False)
        if st.button("Generate flashcards"):
            if from_last and "last_question" in st.session_state and st.session_state.get("last_question"):
                cards = flashcards.generate_flashcards_from_qa(
                    st.session_state.last_question, st.session_state.get("last_answer", ""), processor, model
                )
                topic_label = "last_qa"
            elif topic:
                cards = flashcards.generate_flashcards_from_topic(topic, embedder, coll, processor, model)
                topic_label = topic
            else:
                st.warning("Enter a topic or use last Q&A after chatting.")
                return
            if not cards:
                st.warning("No flashcards generated.")
                return
            n = flashcards.save_generated_flashcards(cards, topic=topic_label)
            st.success(f"Saved {n} flashcards.")
        st.caption("Generated cards can be reviewed in Review deck.")
    else:
        cards = flashcards.get_review_deck(limit=30)
        if not cards:
            st.info("No flashcards to review. Generate some first.")
            return
        if "fc_index" not in st.session_state:
            st.session_state.fc_index = 0
        if "fc_show_back" not in st.session_state:
            st.session_state.fc_show_back = False

        idx = st.session_state.fc_index % len(cards)
        c = cards[idx]
        st.write("**Front:**", c["front"])
        if st.session_state.fc_show_back:
            st.write("**Back:**", c["back"])
            if st.button("Next"):
                st.session_state.fc_index += 1
                st.session_state.fc_show_back = False
                st.rerun()
        else:
            if st.button("Show back"):
                st.session_state.fc_show_back = True
                st.rerun()
        st.caption(f"Card {idx + 1} of {len(cards)}")


def quiz_tab(embedder, coll, processor, model):
    st.subheader("Quiz")
    topic = st.text_input("Topic for quiz", value="corticosteroids and respiratory disease")
    if st.button("Start quiz"):
        questions = quiz.generate_quiz(topic, embedder, coll, processor, model, num_questions=3)
        if not questions:
            st.warning("No questions generated. Try another topic.")
            return
        st.session_state.quiz = questions
        st.session_state.quiz_idx = 0
        st.session_state.quiz_done = False
        st.rerun()

    if "quiz" not in st.session_state or not st.session_state.quiz:
        return

    qs = st.session_state.quiz
    idx = st.session_state.quiz_idx
    if idx >= len(qs):
        st.success("Quiz complete!")
        return

    q = qs[idx]
    st.write("**Question:**", q["question"])
    choice = st.radio("Choose:", q["options"], key=f"quiz_opt_{idx}")
    if st.button("Submit"):
        user_idx = q["options"].index(choice)
        correct = user_idx == q["correct_index"]
        if correct:
            st.success("Correct!")
        else:
            st.error(f"Wrong. Correct is {q['options'][q['correct_index']]}")
        with st.spinner("Explanation..."):
            expl = quiz.get_explanation(q["question"], q["options"], q["correct_index"], user_idx, processor, model)
        st.write("**Explanation:**", expl)
        st.session_state.quiz_idx = idx + 1
        st.rerun()


def notes_tab():
    st.subheader("My notes")
    notes = store.get_all_notes()
    if not notes:
        st.info("No saved notes. Use Chat and click 'Save last Q&A to My notes'.")
        return
    for n in notes:
        with st.expander(n["question"][:60] + "..." if len(n["question"]) > 60 else n["question"]):
            st.write("**A:**", n["answer"])
            if n.get("sources"):
                st.caption("Sources: " + " | ".join(s[:80] + "..." for s in n["sources"][:2]))


def export_tab():
    st.subheader("Export notes")
    notes = store.get_all_notes()
    if not notes:
        st.info("No notes to export.")
        return
    if st.button("Export to Markdown"):
        path = export_notes.export_to_markdown()
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        st.download_button("Download .md file", data=data, file_name=Path(path).name, mime="text/markdown")


def main():
    st.set_page_config(page_title="MedGemma Learning", layout="wide")
    st.title("MedGemma Learning Assistant")

    try:
        embedder, coll, processor, model = load_models()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chat", "Flashcards", "Quiz", "My notes", "Export"])
    with tab1:
        chat_tab(embedder, coll, processor, model)
    with tab2:
        flashcards_tab(embedder, coll, processor, model)
    with tab3:
        quiz_tab(embedder, coll, processor, model)
    with tab4:
        notes_tab()
    with tab5:
        export_tab()


if __name__ == "__main__":
    main()
