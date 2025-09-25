
import subprocess
import re
import json
from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt

# -------- CONFIG --------
MODEL = "Phi-3-mini-4k-instruct-q4.gguf"
RUNNER = "./windows/llama-run.exe"   # update if different
CONTEXT = "2000"
THREADS = "2"

DATA_DIR = Path("data")
DATA_JSON = DATA_DIR / "data.json"
SUMMARIES_JSON = DATA_DIR / "summaries.json"
TABLE_CSV = DATA_DIR / "table.csv"
PIE_PNG = DATA_DIR / "topic_pie.png"

# -------- Utils: run local phi via subprocess (llama-run.exe) --------
def run_model(prompt: str, timeout: int = 30) -> str:
    """Run local model and return cleaned stdout. Returns an error string on failure."""
    cmd = [RUNNER, "--context-size", CONTEXT, "--threads", THREADS, MODEL, prompt]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
        out = proc.stdout or proc.stderr or ""
        # strip color codes and leading/trailing whitespace
        clean = re.sub(r"\x1b\[[0-9;]*m", "", out).strip()
        return clean
    except subprocess.CalledProcessError as e:
        return f"[Error: CalledProcessError] {e}"
    except subprocess.TimeoutExpired as e:
        return f"[Error: TimeoutExpired] {e}"
    except Exception as e:
        return f"[Error] {e}"

# -------- Prompts for Phi (short and strict) --------
UX_QUESTION_PROMPT = (
    "You are a concise assistant. Ask the user a single short question (one line) "
    "to choose an output format for scraped data. Present numbered choices clearly: "
    "1) summary  2) pie chart  3) table  4) distinguish  5) mix. "
    "Do NOT add anything else. Keep it <= 1 sentence."
)

SUMMARY_PROMPT_TEMPLATE = (
    "You are a concise summarizer. Produce a 2-3 sentence summary for the article content below. "
    "Output only the summary, nothing else.\n\nArticle content:\n{content}\n\nSummary:"
)

TOPIC_PROMPT_TEMPLATE = (
    "You are a topic classifier. From the article content below, return ONE short topic tag (1-3 words), "
    "lowercase, no punctuation. Output only the single tag.\n\nArticle content:\n{content}\n\nTag:"
)

# -------- Loading helper --------
def load_crawled_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the crawler first.")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Expecting list of dicts. If single dict wrap into list.
    if isinstance(data, dict):
        data = [data]
    return data

# -------- Extract best text to summarize ----------
def pick_text(record: dict) -> str:
    """Prefer markdown, then extracted_content, then html, then title."""
    for k in ("markdown", "extracted_content", "html", "title"):
        v = record.get(k)
        if v:
            # If extracted_content is dict, try to stringify
            if isinstance(v, dict):
                return json.dumps(v, ensure_ascii=False)
            return v
    return ""


# -------- Core operations --------
def generate_summaries(records):
    results = []
    for i, rec in enumerate(records, 1):
        txt = pick_text(rec)
        if not txt:
            summary = ""
            tag = "unknown"
        else:
            prompt = SUMMARY_PROMPT_TEMPLATE.format(content=txt[:6000])  # limit input length
            summary = run_model(prompt)
            # small safety: if model returns error string, keep empty
            if summary.startswith("[Error"):
                summary = ""
            # topic/classify
            tprompt = TOPIC_PROMPT_TEMPLATE.format(content=txt[:4000])
            tag = run_model(tprompt)
            if not tag or tag.startswith("[Error"):
                tag = "unknown"
            # sanitize tag: keep only word chars and spaces, lowercase
            tag = re.sub(r"[^\w\s\-]", "", tag).strip().lower()[:60] or "unknown"

        results.append({
            "url": rec.get("url"),
            "title": rec.get("title"),
            "summary": summary,
            "topic": tag
        })
        print(f"[{i}/{len(records)}] summary len={len(summary):4} topic={tag}")
        time.sleep(0.5)  # small polite pause between model calls

    return results

def save_table(records_summary, csv_path: Path):
    df = pd.DataFrame(records_summary)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved table to {csv_path} ({len(df)} rows)")

def save_summaries_json(records_summary, json_path: Path):
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records_summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summaries to {json_path}")

def make_topic_pie(records_summary, png_path: Path):
    df = pd.DataFrame(records_summary)
    counts = df["topic"].fillna("unknown").value_counts()
    if counts.empty:
        print("No topics to plot.")
        return
    # Single plot with matplotlib (no custom colors)
    plt.figure(figsize=(6, 6))
    counts.plot.pie(autopct="%1.1f%%", ylabel="")
    plt.title("Topic distribution")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    print(f"Saved pie chart to {png_path}")

# -------- Main interactive flow --------
def main():
    # load data
    try:
        records = load_crawled_data(DATA_JSON)
    except Exception as e:
        print("Error loading crawled data:", e)
        return

    # Ask Phi to produce the UX question, then ask user
    ux_question = run_model(UX_QUESTION_PROMPT)
    if ux_question.startswith("[Error"):
        # fallback
        ux_question = "Choose an output format (type number): 1) summary 2) pie chart 3) table 4) distinguish 5) mix"
    print()
    print("Phi asks:", ux_question)
    choice = input("Your choice (1-5 or name): ").strip().lower()

    # Interpret choice
    mapping = {
        "1": "summary",
        "2": "pie",
        "3": "table",
        "4": "distinguish",
        "5": "mix",
        "summary": "summary",
        "pie": "pie",
        "table": "table",
        "distinguish": "distinguish",
        "mix": "mix"
    }
    choice_key = mapping.get(choice, None)
    if not choice_key:
        print("Invalid choice. Exiting.")
        return

    # Generate summaries & topics when needed:
    do_summary = choice_key in ("summary", "mix", "table", "distinguish", "pie")
    if do_summary:
        print("Generating summaries and topic tags (this uses Phi for each page)...")
        rec_summaries = generate_summaries(records)
    else:
        rec_summaries = []

    # Do operations
    if choice_key in ("summary", "mix"):
        save_summaries_json(rec_summaries, SUMMARIES_JSON)

    if choice_key in ("table", "mix"):
        save_table(rec_summaries, TABLE_CSV)

    if choice_key in ("pie", "mix"):
        make_topic_pie(rec_summaries, PIE_PNG)

    if choice_key == "distinguish":
        # just save topic tags + summary
        save_summaries_json(rec_summaries, SUMMARIES_JSON)
        print("Distinguish finished. Use the JSON to filter or review by topic.")

    print("All requested work finished. Files are in the data/ folder.")

if __name__ == "__main__":
    main()