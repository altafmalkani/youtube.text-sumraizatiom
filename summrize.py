## AI Summarizer - Deep dive 
'''
Project : OmniBrief (AI Summarizer)

Goal: Summarize content from a URL (Youtube , website, PDF ) or any uploaded pdf

What this will teach you:
- Streamlit advance quick UI
- Loading real world content (Youtube , website, PDF ) 
- Chunking long text and running a map-reduce summarization chain
- using Groq LLMs via Langchain in a safe way 
'''

# Import
import os, re, json, tempfile
from urllib.parse import urlparse

# Network and validation
import requests # to fetch web/pdf/caption files
import validators # to validate URL inputs

# UI Framework
import streamlit as st 

# langchin core pieces
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# loaders
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader, PyPDFLoader, UnstructuredURLLoader

# LLM
from langchain_groq import ChatGroq

# Youtube caption edge case and fallbacks
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from yt_dlp import YoutubeDL

# Minimal Page setup
st.set_page_config(page_title="OmniBrief - AI Summarizer", page_icon="🧠", layout="wide")
st.title("🧠 OmniBrief - Summarizer")
st.caption("Build with Streamlit + Langchain + Groq")

# Sidebar
# LLM Model , Temperature , target , target length , etc.
with st.sidebar:
    st.subheader("🔑 API & Model")
    groq_api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    
    # pick a GROQ model you have access to
    model = st.selectbox(
        "Groq Model",
        ["gemma2-9b-it", "llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        index=0,
        help="If you get 'model not found', update this ID to a valid Groq model."
    )
    custom_model = st.text_input("Custom Model (optional)", help="Overrides selection above if filled.")

    st.subheader("🧠 Generation")
    temperature = st.slider("Temperature (creativity)", 0.0, 1.0, 0.2, 0.05)
    out_len = st.slider("Target summary length (words)", 90, 800, 300, 20)

    st.subheader("✍🏻 Style")
    out_style = st.selectbox("Output Style", ["Bullets", "Paragraph", "Both"])
    tone = st.selectbox("Tone", ["Neutral", "Formal", "Casual", "Executive Brief"])
    out_lang = st.selectbox("Language", ["English", "Urdu", "Roman Urdu", "Auto"])

    st.subheader("⚙️ Processing")
    chain_mode = st.radio("Chain Type", ["Auto", "Stuff", "Map-Reduce"], index=0)
    chunk_size = st.slider("Chunk Size (characters)", 500, 4000, 1600, 100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 800, 150, 10)
    max_map_chunks = st.slider("Max chunks (for combine step)", 9, 64, 28, 1)

    st.subheader("👀 Extras")
    show_preview = st.checkbox("Show source preview", value=True)
    want_outline = st.checkbox("Also produce an outline", value=True)
    want_keywords = st.checkbox("Also extract keywords and hashtags", value=True)

# main Input
left, right = st.columns([2, 1])
with left:
    url = st.text_input("Paste URL (website, YouTube, or direct PDF link)", 
                       placeholder="https://youtube.com/watch?v=... or https://example.com")
with right:
    uploaded = st.file_uploader("... or upload a PDF", type=["pdf"])

# Helper functions
def is_youtube(u: str) -> bool:
    try:
        netloc = urlparse(u).netloc.lower()
        return any(host in netloc for host in ["youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com"])
    except Exception:
        return False

def extract_video_id(url):
    # Extract YouTube video ID from URL
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def head_content_type(u: str, timeout=12) -> str | None:
    try:
        r = requests.head(u, allow_redirects=True, timeout=timeout, 
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        return (r.headers.get("Content-Type") or "").lower()
    except Exception:
        return None

def clean_caption_text(text: str) -> str:
    text = re.sub(r"\[(?:music|applause|laughter|.*?)\]", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_llm(groq_api_key: str, model: str, temperature: float):
    chosen = custom_model.strip() if custom_model else model
    return ChatGroq(model=chosen, groq_api_key=groq_api_key, temperature=temperature)

def build_prompts(out_len: int, out_style: str, tone: str, want_outline: bool, want_keywords: bool, out_lang: str):
    map_template = """
    Summarize the following text in 3–6 crisp bullet points, maximum 80 words total.
    Keep only the core facts/claims.

    TEXT:
    {text}
    """
    map_prompt = PromptTemplate(template=map_template, input_variables=["text"])

    style_map = {
        "Bullets": "Return crisp bullet points only",
        "Paragraph": "Return one cohesive paragraph only",
        "Both": "Start with 6–10 concise bullet points, then a cohesive paragraph",
    }

    tone_map = {
        "Neutral": "neutral, information-dense",
        "Formal": "formal and precise",
        "Casual": "casual and friendly",
        "Executive Brief": "executive, top-down, action-oriented",
    }
    
    lang = "Match the user's language." if out_lang == "Auto" else f"Write in {out_lang}."

    extras = []
    if want_outline:
        extras.append("Provide a short outline with top 3–6 sections.")
    if want_keywords:
        extras.append("Extract 8–12 keywords and 5–8 suggested hashtags.")
    extras_text = ("\n- " + "\n- ".join(extras)) if extras else ""

    combine_template = f"""
    You will receive multiple mini-summaries of different parts of the same source.
    Combine them into a single, faithful summary.

    Constraints and style:
    - Target length = {out_len} words.
    - Output Style: {style_map[out_style]}
    - Tone: {tone_map[tone]}
    - {lang}
    - Be faithful to the source; do not invent facts.
    - If the content is opinionated, label opinions as opinions.
    - Avoid repetition.
    {extras_text}

    Return only the summary (and requested sections); no preambles.

    INPUT_SUMMARIES:
    {{text}}
    """
    combine_prompt = PromptTemplate(template=combine_template, input_variables=["text"])
    return map_prompt, combine_prompt

def choose_chain_type(chain_mode: str, docs: list) -> str:
    if chain_mode != "Auto":
        return chain_mode.lower().replace("-", "_")
    total_chars = sum(len(d.page_content or "") for d in docs)
    return "map_reduce" if total_chars > 15000 else "stuff"

def even_sample(docs, k: int):
    n = len(docs)
    if k >= n:
        return docs
    idxs = [round(i * (n-1) / (k-1)) for i in range(k)]
    return [docs[i] for i in idxs]

def load_youtube_docs(url: str):
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Could not extract video ID from YouTube URL")
    
    # Try to get transcript using youtube-transcript-api
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB', 'ur'])
        text = " ".join([entry['text'] for entry in transcript])
        
        from langchain.schema import Document
        return [Document(page_content=text, metadata={"source": url, "title": "YouTube Video"})], {"type": "youtube"}
    
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        st.warning(f"Could not get transcript: {str(e)}. Trying fallback method...")
        
        # Fallback: Use yt-dlp to get video info
        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US', 'en-GB'],
        }
        
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'YouTube Video')
                
                # Try to get automatic captions
                automatic_captions = info.get('automatic_captions', {})
                subtitles = info.get('subtitles', {})
                
                # Check for English captions
                captions = None
                for lang in ['en', 'en-US', 'en-GB']:
                    if lang in automatic_captions:
                        captions = automatic_captions[lang]
                        break
                    if lang in subtitles:
                        captions = subtitles[lang]
                        break
                
                if captions:
                    # Get the first caption format (usually vtt or srt)
                    caption_url = captions[0]['url']
                    response = requests.get(caption_url)
                    caption_text = response.text
                    
                    # Simple VTT parsing (remove timestamps)
                    lines = caption_text.split('\n')
                    text_lines = []
                    for line in lines:
                        if '-->' not in line and not line.strip().isdigit() and line.strip() != '':
                            text_lines.append(line.strip())
                    
                    text = ' '.join(text_lines)
                    
                    from langchain.schema import Document
                    return [Document(page_content=text, metadata={"source": url, "title": title})], {"type": "youtube_fallback"}
                else:
                    raise RuntimeError("No captions available for this video")
                    
        except Exception as e:
            raise RuntimeError(f"Failed to get YouTube content: {str(e)}")

@st.cache_data(show_spinner=False, ttl=300)
def fetch_and_load(url: str, chunk_size: int, chunk_overlap: int):
    meta = {"source": url, "type": "html", "title": None}

    if is_youtube(url):
        try:
            docs, yt_meta = load_youtube_docs(url)
            meta.update(yt_meta)
            return docs, meta
        except Exception as e:
            st.error(f"Error loading YouTube video: {str(e)}")
            # Fallback to WebBaseLoader for YouTube
            try:
                loader = WebBaseLoader([url])
                docs = loader.load()
                if docs and docs[0].metadata.get("title"):
                    meta["title"] = docs[0].metadata["title"]
                return docs, meta
            except Exception as e2:
                raise Exception(f"Both methods failed: {str(e)} and {str(e2)}")
    
    ctype = head_content_type(url) or ""
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        meta["type"] = "pdf"
        try:
            loader = UnstructuredURLLoader(urls=[url])
            docs = loader.load()
            return docs, meta
        except Exception as e:
            st.warning(f"Direct PDF loading failed: {str(e)}. Trying alternative method...")
            try:
                with requests.get(url, stream=True, timeout=30, headers={"User-Agent": "Mozilla/5.0"}) as r:
                    r.raise_for_status()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                tmp.write(chunk)
                        tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                os.unlink(tmp_path)
                return docs, meta
            except Exception as e2:
                raise Exception(f"PDF loading failed: {str(e2)}")
    
    # Webpage
    try:
        loader = WebBaseLoader([url])
        docs = loader.load()
        if docs and docs[0].metadata.get("title"):
            meta["title"] = docs[0].metadata["title"]
    except Exception as e:
        st.warning(f"WebBaseLoader failed: {str(e)}. Trying simple HTML extraction...")
        try:
            response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            from langchain.schema import Document
            docs = [Document(page_content=text, metadata={"source": url})]
        except Exception as e2:
            raise Exception(f"Both methods failed: {str(e)} and {str(e2)}")
    
    # Split documents if they're too long
    if docs and sum(len(d.page_content or "") for d in docs) > chunk_size * 1.5:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "?", "!", " "],
        )
        out = []
        for d in docs:
            out.extend(splitter.split_documents([d]))
        return out, meta
    
    return docs, meta

def load_pdf_from_upload(uploaded_file, chunk_size: int, chunk_overlap: int):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        if docs and sum(len(d.page_content or "") for d in docs) > chunk_size * 1.5:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", "?", "!", " "],
            )
            parts = []
            for d in docs:
                parts.extend(splitter.split_documents([d]))
            return parts
        
        return docs
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

# chain runner
def run_chain(llm, docs, map_prompt: PromptTemplate, combine_prompt: PromptTemplate, mode: str, max_map_chunks: int) -> str:
    mode = mode.lower().replace("-", "_")

    if mode == "stuff":
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=combine_prompt)
    else:
        if len(docs) > max_map_chunks:
            docs = even_sample(docs, max_map_chunks)
            st.info(f"Long source: sampled {max_map_chunks} chunks evenly to fit the context.")
        
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_promise=combine_prompt,
        )

    try:
        res = chain.invoke({"input_documents": docs})
        return res["output_text"] if isinstance(res, dict) and "output_text" in res else str(res)
    except Exception as e:
        st.error(f"Error in chain execution: {str(e)}")
        # Fallback to simple summarization
        try:
            combined_text = " ".join([doc.page_content for doc in docs[:5]])  # Use first 5 chunks
            prompt = f"Please summarize the following text: {combined_text[:3000]}"
            response = llm.invoke(prompt)
            return response.content
        except Exception as e2:
            return f"Error: Failed to generate summary. {str(e2)}"

# Main execution
st.markdown("### 🚀 Run")
go = st.button("Summarize")

if go:
    if not groq_api_key.strip():
        st.error("Please provide your Groq API Key in the sidebar")
        st.stop()
    
    docs, meta = [], {"type": None, "source": None, "title": None}

    try:
        stage = "loading source"
        with st.spinner("Loading source..."):
            if uploaded is not None:
                docs = load_pdf_from_upload(uploaded, chunk_size, chunk_overlap)
                meta.update({"type": "pdf", "source": uploaded.name})
            elif url.strip():
                if not validators.url(url):
                    st.error("Please enter a valid URL.")
                    st.stop()
                docs, meta = fetch_and_load(url, chunk_size, chunk_overlap)
            else:
                st.error("Provide a URL or upload a PDF.")
                st.stop()
            
            if not docs or not any((d.page_content or "").strip() for d in docs):
                st.error("Could not extract text from the source. See notes below.")
                st.stop()
        
        # quick preview for sanity
        if show_preview:
            with st.expander("🔍 source preview"):
                preview = "".join(d.page_content or "" for d in docs[:3])[:1200].strip()
                st.write(f"**Detected type:** `{meta.get('type')}`")
                if meta.get("title"): 
                    st.write(f"**Title:** {meta['title']}")
                st.text_area("First ~1200 characters", preview, height=150)
        
        # build LLM + Prompt
        stage = "initializing LLM"
        llm = build_llm(groq_api_key, model, temperature)
        
        stage = "building prompts"
        map_prompt, combine_prompt = build_prompts(out_len, out_style, tone, want_outline, want_keywords, out_lang)

        # pick chain type (auto/stuff/map_reduce)
        stage = "selecting chain"
        mode = choose_chain_type(chain_mode, docs)

        # run the chain and display
        stage = f"running chain ({mode})"
        with st.spinner(f"Summarizing via {(custom_model or model)} ({mode})..."):
            summary = run_chain(llm, docs, map_prompt, combine_prompt, mode, max_map_chunks)
        
        st.success("Done.")
        st.subheader("✅ Summary")
        st.write(summary)

        # export
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download .txt", data=summary, file_name="summary.txt", mime="text/plain")
        with col2:
            st.download_button("⬇️ Download .md", data=f"# Summary\n\n{summary}\n", file_name="summary.md", mime="text/markdown")
            
    except Exception as e:
        st.error(f"Failed during **{stage}** -> {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc())

with st.expander("🚨 Notes: What works vs. what to avoid"):
    st.markdown("""
- **Best:** Public webpages, YouTube Videos with captions (or auto-caption), direct PDF links, and uploaded PDFs.
- **Might Fail:** Login-only pages, heavy JS Pages, scanned PDFs with OCR, or sites that block scrapes (CORS Blockage)
- **Too long?** Lower Chunk Size / Max Chunks, or keep Map-Reduce on.

This avoids context-length errors.

**YouTube Specific Notes:**
- Works best with videos that have official captions
- Auto-generated captions might have lower accuracy
- Some videos may not have captions available
    """)
