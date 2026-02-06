import streamlit as st
from pathlib import Path
import pandas as pd

from predict import html_to_lines, load_model, separate
from ner_infer import load_ner, extract_structured  # NER (qty/unit/form/item)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "tfidf_svm.joblib"

@st.cache_resource
def _load_cls():
    return load_model(MODEL_PATH)

@st.cache_resource
def _load_ner():
    try:
        return load_ner()  # returns (tokenizer, model)
    except Exception:
        return None, None

def main():
    st.set_page_config(page_title="🍳 Recipe Ingredient & Step Extractor", layout="wide")
    st.title("🍳 Recipe Ingredient & Step Extractor (Blog → Structured)")

    st.markdown(
        "Paste any cooking blog **text or HTML**. "
        "The app separates **Ingredients** and **Instructions**, and (optionally) parses ingredients into "
        "**qty / unit / form / item** for CSV export."
    )

    # Inputs
    text = st.text_area("Paste blog HTML or plain text", height=260, placeholder="Paste content here…")
    uploaded = st.file_uploader("…or upload .html / .txt", type=["html", "txt"])
    do_ner = st.checkbox("Parse structured ingredients (NER: qty / unit / form / item)")

    if uploaded and not text:
        text = uploaded.read().decode("utf-8", errors="ignore")

    # Load models
    cls_model = _load_cls()
    tok, ner_model = _load_ner()

    if st.button("Extract"):
        if not text or len(text.strip()) < 5:
            st.warning("Please paste some content.")
            return

        # Step-1: split + classify lines
        lines = html_to_lines(text)
        ingredients, instructions = separate(lines, cls_model)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Ingredients")
            if ingredients:
                # plain list
                st.write("\n".join(f"• {x}" for x in ingredients))

                # NER table + CSV
                if do_ner:
                    if tok is None or ner_model is None:
                        st.error("NER model not found. Train it first:\n\n"
                                 "`python src/build_ner_data.py`\n"
                                 "`python src/train_ner_bert.py`")
                    else:
                        rows = [extract_structured(x, tok, ner_model) for x in ingredients]
                        df = pd.DataFrame(rows, columns=["qty", "unit", "item"])
                        st.markdown("**Structured (qty / unit / item)**")
                        st.dataframe(df, use_container_width=True)

                        # CSV download
                        csv_bytes = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download structured_ingredients.csv",
                            data=csv_bytes,
                            file_name="structured_ingredients.csv",
                            mime="text/csv",
                        )

                # Raw ingredients.txt download
                st.download_button(
                    "⬇️ Download ingredients.txt",
                    data="\n".join(ingredients),
                    file_name="ingredients.txt",
                    mime="text/plain",
                )
            else:
                st.info("No ingredients detected.")

        with c2:
            st.subheader("Instructions")
            if instructions:
                st.write("\n".join(f"{i+1}. {x}" for i, x in enumerate(instructions)))
                st.download_button(
                    "⬇️ Download instructions.txt",
                    data="\n".join(instructions),
                    file_name="instructions.txt",
                    mime="text/plain",
                )
            else:
                st.info("No instructions detected.")

if __name__ == "__main__":
    main()
