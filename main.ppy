import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv

# ───────────────────────────────────────────────
# Load environment variables
# ───────────────────────────────────────────────
load_dotenv()
ASSEMBLY_KEY = os.getenv("ASSEMBLYAI_API_KEY")
HUGGING_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL = os.getenv("HF_SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")

# ───────────────────────────────────────────────
# Streamlit UI setup
# ───────────────────────────────────────────────
st.set_page_config(page_title="Meeting Summarizer", page_icon="🧠")
st.title("🧠 AI Meeting Summarizer")
st.caption("Upload a meeting recording. I’ll transcribe it with AssemblyAI and summarize it with Hugging Face.")

uploaded = st.file_uploader("🎧 Upload an audio file", type=["mp3", "wav", "m4a"])

# ───────────────────────────────────────────────
# Helper: call Hugging Face with retry
# ───────────────────────────────────────────────
def summarize_text_snippet(snippet: str, retries: int = 3) -> str:
    hf_headers = {"Authorization": f"Bearer {HUGGING_KEY}"}
    payload = {
        "inputs": snippet,
        "parameters": {"max_length": 250, "min_length": 60, "do_sample": False},
    }
    for i in range(retries):
        try:
            r = requests.post(
                f"https://api-inference.huggingface.co/models/{HF_MODEL}",
                headers=hf_headers,
                json=payload,
                timeout=90,
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and "summary_text" in data[0]:
                    return data[0]["summary_text"]
                if isinstance(data, dict) and "generated_text" in data:
                    return data["generated_text"]
            time.sleep(3)
        except Exception:
            time.sleep(2)
    # fallback
    return " ".join(snippet.split(". ")[:5]) + "..."

# ───────────────────────────────────────────────
# Chunk large transcripts and summarize each
# ───────────────────────────────────────────────
def summarize_in_chunks(full_text: str, chunk_size: int = 4000) -> str:
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    summaries = []
    progress = st.progress(0, text="Summarizing chunks...")
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        summary = summarize_text_snippet(chunk)
        summaries.append(summary)
        progress.progress((idx + 1) / total, text=f"Chunk {idx + 1}/{total} summarized")
    progress.empty()
    combined = " ".join(summaries)
    if len(chunks) > 1:
        # one more pass to blend the summaries together
        return summarize_text_snippet(combined)
    return combined

# ───────────────────────────────────────────────
# Main logic
# ───────────────────────────────────────────────
if uploaded:
    fname = uploaded.name
    with open(fname, "wb") as f:
        f.write(uploaded.read())

    # 1. Upload to AssemblyAI
    st.info("📤 Uploading to AssemblyAI…")
    headers = {"authorization": ASSEMBLY_KEY}
    with open(fname, "rb") as audio:
        up = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=audio)
    audio_url = up.json()["upload_url"]

    # 2. Create transcript request
    st.info("🧩 Starting transcription job…")
    endpoint = "https://api.assemblyai.com/v2/transcript"
    tr = requests.post(endpoint, json={"audio_url": audio_url}, headers=headers).json()
    tid = tr["id"]

    # 3. Poll for completion with spinner
    with st.spinner("🎙️ Transcribing audio (this can take a minute)…"):
        while True:
            poll = requests.get(f"{endpoint}/{tid}", headers=headers).json()
            if poll["status"] == "completed":
                break
            if poll["status"] == "error":
                st.error("❌ Transcription failed: " + poll["error"])
                st.stop()
            time.sleep(5)

    text = poll.get("text", "")
    if not text:
        st.error("Transcription returned empty text.")
        st.stop()

    st.success("✅ Transcription complete!")
    st.subheader("🗒️ Transcript Preview")
    st.text_area("Transcript", text[:3000] + ("…" if len(text) > 3000 else ""), height=200)

    # 4. Summarize
    st.info("✍️ Summarizing transcript...")
    with st.spinner("Generating AI summary..."):
        summary = summarize_in_chunks(text)

    # 5. Display results
    st.success("✅ Summarization complete!")
    st.subheader("📝 Meeting Summary")
    st.write(summary)

    st.download_button("⬇️ Download Summary", summary, "summary.txt")

else:
    st.caption("👆 Upload an audio file above to begin transcription and summarization.")
