import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import google.generativeai as genai
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import Optional, List
import fitz
import os

# embiddings
class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return [self.model.encode(t).tolist() for t in texts]

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# gemini api 
class GeminiLLM(LLM):
    model: str = "gemini-1.5-flash"
    api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set or not loaded")
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        resp = model.generate_content(prompt)
        return resp.text

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "gemini"

# streamlit ui
st.title("AI Diagnosis Assistant (Educational Prototype)")
st.warning("This is not a substitute for professional medical advice. Always consult a doctor.")

# load vectorstore
embeddings = HuggingFaceEmbeddings()
try:
    vectorstore = FAISS.load_local(
        "harrison_vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    st.error(f"Error loading knowledge base: {e}. Ensure 'harrison_vectorstore' exists.")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# gemini llm
try:
    
    if  os.getenv("GEMINI_API_KEY"):
        llm = GeminiLLM()
        
    else:
        raise ValueError("No API keys found")
except Exception as e:
    st.warning(f"No cloud API available: {e}. Falling back to local model...")
    generator = pipeline("text-generation", model="distilbert-base-uncased", max_new_tokens=200)
    llm = HuggingFacePipeline(pipeline=generator)
    st.success("✅ Using local DistilBERT model as fallback (very limited)")

# promt 
prompt_tmpl = PromptTemplate(
    template=(
        "You are a careful medical assistant using Harrison's Principles of Internal Medicine.\n"
        "Use the context to reason about the user's symptoms and uploaded report.\n"
        "Return likely differential diagnoses, key red flags, and initial management steps.\n"
        "Always include: 'Disclaimer: Consult a doctor.'\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
    input_variables=["context", "question"],
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def rag_answer(question: str):
    # 1) retrieve
    docs = retriever.get_relevant_documents(question)
    # 2) build prompt
    context = format_docs(docs)
    prompt = prompt_tmpl.format(context=context, question=question)
    # 3) call LLM
    try:
        text = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
    except TypeError:
        # Some LLMs only implement __call__
        text = llm(prompt)
    # Normalize to string
    text = str(text)
    return text, docs

# inputs
symptoms = st.text_area("Describe your symptoms:")
uploaded_pdf = st.file_uploader("Upload medical report (PDF):", type="pdf")

if st.button("Get Diagnosis"):
    pdf_text = ""
    if uploaded_pdf:
        try:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            with fitz.open("temp.pdf") as pdf:
                for page in pdf:
                    try:
                        t = page.get_text("text")
                        if t:
                            pdf_text += t + " "
                        else:
                            st.warning(f"Skipping page {page.number + 1}: no text extracted")
                    except Exception as e:
                        st.warning(f"Error reading page {page.number + 1}: {e}")
            os.remove("temp.pdf")
        except Exception as e:
            st.error(f"Error processing uploaded PDF: {e}")

    if symptoms or pdf_text:
        question = f"Symptoms: {symptoms}\nMedical report: {pdf_text}"
        with st.spinner("Generating response..."):
            try:
                answer, src_docs = rag_answer(question)
                pages = [d.metadata.get("page", "Unknown") for d in src_docs]
                st.markdown("**Response:**")
                st.write(answer)
                st.markdown("**Source Pages:** " + ", ".join(map(str, pages)) if pages else "No pages found")
            except Exception as e:
                st.error(f"Error generating response: {e}")
    else:
        st.error("Provide symptoms or a PDF.")
