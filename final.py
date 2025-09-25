
import subprocess
import re
import requests
from bs4 import BeautifulSoup
import time
import asyncio
import json
from pathlib import Path
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

# crawl4ai async imports
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.async_configs import BrowserConfig

# -------- CONFIG --------
MODEL = "Phi-3-mini-4k-instruct-q4.gguf"
RUNNER = "./windows/llama-run.exe"  # update if needed
CONTEXT = "2000"
THREADS = "2"

QUERY_FILE = Path("query.txt")
LINKS_FILE = Path("links.txt")
DATA_DIR = Path("data")
DATA_JSON = DATA_DIR / "data.json"

SUMMARIES_JSON = DATA_DIR / "summaries.json"
TABLE_CSV = DATA_DIR / "table.csv"
PIE_PNG = DATA_DIR / "topic_pie.png"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/140.0.0.0 Safari/537.36"
}

# Crawl settings
CONCURRENT_TASKS = 4
DELAY_BETWEEN_CRAWLS = 1.0  # polite delay per worker

# Model prompts
SYSTEM_PROMPT = (
    "You are a search query generator. Your task is to take the given information "
    "and return ONE short, precise, optimized search query. Do not add extra words; "
    "output the final query only."
)

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

# Ensure data dir exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -------- Utilities --------
def run_model(prompt: str, timeout: int = 60000) -> str:
    """Run local Phi (llama-run.exe) and return cleaned stdout or an error string."""
    cmd = [RUNNER, "--context-size", CONTEXT, "--threads", THREADS, MODEL, prompt]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
        out = proc.stdout or proc.stderr or ""
        clean = re.sub(r"\x1b\[[0-9;]*m", "", out).strip()
        return clean
    except subprocess.TimeoutExpired as e:
        return f"[Error: TimeoutExpired] {e}"
    except subprocess.CalledProcessError as e:
        return f"[Error: CalledProcessError] {e}"
    except Exception as e:
        return f"[Error] {e}"

def generate_query_from_input(user_input: str) -> str:
    final_prompt = f"{SYSTEM_PROMPT}\nUser info: {user_input}\nQuery:"
    return run_model(final_prompt)

def duckduckgo_search_links(query: str, max_results: int = 10) -> List[str]:
    """Scrape DuckDuckGo html endpoint for results (returns raw hrefs)."""
    search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    try:
        resp = requests.post(search_url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.find_all("a", class_="result__a"):
            href = a.get("href")
            if href:
                links.append(href)
            if len(links) >= max_results:
                break
        return links
    except Exception as e:
        print("  [search error]", e)
        return []

# -------- Async crawler (crawl4ai) --------
async def crawl_url(crawler: AsyncWebCrawler, url: str, run_config=None) -> dict:
    try:
        result = await crawler.arun(url, config=run_config)
    except Exception as e:
        return {"url": url, "error": str(e)}
    rec = {
        "url": url,
        "title": getattr(result, "title", None),
        "markdown": getattr(result, "markdown", None),
        "html": getattr(result, "html", None) if hasattr(result, "html") else None,
        "extracted_content": getattr(result, "extracted_content", None),
    }
    return rec

async def worker(worker_id: int, crawler: AsyncWebCrawler, q: asyncio.Queue, out_list: List[dict], run_config=None):
    while True:
        url = await q.get()
        if url is None:
            q.task_done()
            break
        print(f"[worker {worker_id}] crawling: {url}")
        rec = await crawl_url(crawler, url, run_config=run_config)
        out_list.append(rec)
        # save markdown per page if provided (overwrite)
        md = rec.get("markdown")
        if md:
            safe_name = (url.replace("://", "_").replace("/", "_"))[:180]
            md_path = DATA_DIR / f"{safe_name}.md"
            try:
                md_path.write_text(md, encoding="utf-8")
            except Exception as e:
                print(f"  [warn] couldn't write md for {url}: {e}")
        await asyncio.sleep(DELAY_BETWEEN_CRAWLS)
        q.task_done()

async def run_crawl(urls: List[str]) -> List[dict]:
    out = []
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig()
    q = asyncio.Queue()
    for u in urls:
        q.put_nowait(u)
    for _ in range(CONCURRENT_TASKS):
        q.put_nowait(None)
    async with AsyncWebCrawler(config=browser_config) as crawler:
        tasks = [asyncio.create_task(worker(i+1, crawler, q, out, run_config=run_config)) for i in range(CONCURRENT_TASKS)]
        await q.join()
        await asyncio.gather(*tasks, return_exceptions=True)
    return out

# -------- Summarization & outputs --------
def pick_text_for_model(record: dict) -> str:
    for key in ("markdown", "extracted_content", "html", "title"):
        v = record.get(key)
        if v:
            if isinstance(v, dict):
                return json.dumps(v, ensure_ascii=False)
            return v
    return ""

def generate_summaries_and_topics(records: List[dict]) -> List[dict]:
    results = []
    for idx, rec in enumerate(records, 1):
        txt = pick_text_for_model(rec)
        if not txt:
            summary = ""
            tag = "unknown"
        else:
            # limit passed content to avoid context overflow
            prompt_sum = SUMMARY_PROMPT_TEMPLATE.format(content=txt[:6000])
            summary = run_model(prompt_sum)
            if summary.startswith("[Error"):
                summary = ""
            prompt_tag = TOPIC_PROMPT_TEMPLATE.format(content=txt[:4000])
            tag = run_model(prompt_tag)
            if not tag or tag.startswith("[Error"):
                tag = "unknown"
            tag = re.sub(r"[^\w\s\-]", "", tag).strip().lower()[:60] or "unknown"
        results.append({
            "url": rec.get("url"),
            "title": rec.get("title"),
            "summary": summary,
            "topic": tag
        })
        print(f"[{idx}/{len(records)}] summary_len={len(summary):3} topic={tag}")
        time.sleep(0.5)
    return results

def save_summaries_json(records_summary: List[dict]):
    with SUMMARIES_JSON.open("w", encoding="utf-8") as f:
        json.dump(records_summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summaries to {SUMMARIES_JSON}")

def save_table_csv(records_summary: List[dict]):
    df = pd.DataFrame(records_summary)
    df.to_csv(TABLE_CSV, index=False, encoding="utf-8")
    print(f"Saved table to {TABLE_CSV} ({len(df)} rows)")

def make_topic_pie(records_summary: List[dict]):
    df = pd.DataFrame(records_summary)
    if "topic" not in df.columns:
        print("No topic column found, skipping pie.")
        return
    counts = df["topic"].fillna("unknown").value_counts()
    if counts.empty:
        print("No topics to plot.")
        return
    plt.figure(figsize=(6,6))
    counts.plot.pie(autopct="%1.1f%%", ylabel="")
    plt.title("Topic distribution")
    plt.tight_layout()
    plt.savefig(PIE_PNG)
    plt.close()
    print(f"Saved pie chart to {PIE_PNG}")

# -------- Main interactive flow --------
def main():
    print("⚡ Warm-up model...")
    _ = run_model(SYSTEM_PROMPT)  # warm-up

    # 1) Get user input -> generate query -> write query.txt (w)
    user_input = input("You (describe what you want to search): ").strip()
    if not user_input:
        print("No input given. Exiting.")
        return
    print("Generating optimized query with Phi...")
    query = generate_query_from_input(user_input)
    if query.startswith("[Error"):
        print("Model error while generating query:", query)
        return
    QUERY_FILE.write_text(query + "\n", encoding="utf-8")  # overwrite each run
    print("Saved query to", QUERY_FILE)
    print("Query:", query)

    # 2) Search DuckDuckGo and save top links (overwrite links.txt)
    print("Searching DuckDuckGo for top links...")
    links = duckduckgo_search_links(query, max_results=10)
    LINKS_FILE.write_text("\n".join(links) + ("\n" if links else ""), encoding="utf-8")
    print(f"Saved {len(links)} links to {LINKS_FILE}")
    if not links:
        print("No links found. Exiting.")
        return

    # 3) Crawl links with crawl4ai (async) and write data.json (overwrite)
    print("Crawling links with crawl4ai (this may take a while)...")
    try:
        crawled = asyncio.run(run_crawl(links))
    except Exception as e:
        print("Crawl error:", e)
        crawled = []
    with DATA_JSON.open("w", encoding="utf-8") as f:
        json.dump(crawled, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(crawled)} records to {DATA_JSON}")

    # 4) Ask Phi to ask the user which output format they want
    ux_q = run_model(UX_QUESTION_PROMPT)
    if ux_q.startswith("[Error"):
        ux_q = "Choose an output format (type number): 1) summary 2) pie chart 3) table 4) distinguish 5) mix"
    print("\nPhi asks:", ux_q)
    choice = input("Your choice (1-5 or name): ").strip().lower()
    mapping = {"1":"summary","2":"pie","3":"table","4":"distinguish","5":"mix",
               "summary":"summary","pie":"pie","table":"table","distinguish":"distinguish","mix":"mix"}
    choice_key = mapping.get(choice)
    if not choice_key:
        print("Invalid choice. Exiting.")
        return

    # Determine whether to generate summaries & tags (most choices need it)
    if choice_key in ("summary","mix","table","distinguish","pie"):
        print("Generating summaries and topic tags (uses Phi per page)...")
        rec_summaries = generate_summaries_and_topics(crawled)
    else:
        rec_summaries = []

    # Save outputs according to choice (all using modes that overwrite)
    if choice_key in ("summary","mix"):
        save_summaries_json(rec_summaries)
    if choice_key in ("table","mix"):
        save_table_csv(rec_summaries)
    if choice_key in ("pie","mix"):
        make_topic_pie(rec_summaries)
    if choice_key == "distinguish":
        save_summaries_json(rec_summaries)
        print("Distinguish finished. Use the JSON to filter or review by topic.")

    print("All done. Files are in the 'data' folder (and query.txt / links.txt).")

if __name__ == "__main__":
    main()