import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI

ARXIV_FEED_URL = (
    "http://export.arxiv.org/api/query?"
    "search_query=cat:cs.AI&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending"
)

SYSTEM_PROMPT = (
    "你是一个科技资讯专家，负责将复杂的 AI 论文摘要转化为易读的简报。"
    "你必须严格以 JSON 对象格式返回结果。"
)

USER_PROMPT_TEMPLATE = (
    "请将以下论文标题和摘要翻译并总结为 100 字以内的中文。要求：语气专业、客观。"
    "同时，该条资讯的分类统一标记为'科研突破'。"
    "请严格返回一个 JSON 对象（json object），包含两个字段："
    "\"translated_title\"（中文标题）和 \"chinese_summary\"（中文总结），"
    "不得输出任何多余文字（包括解释、注释或代码块）。"
    "原文标题：{title}，原文摘要：{summary}"
)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def init_clients() -> Tuple[str, str, OpenAI, str]:
    load_dotenv()

    supabase_url = _require_env("SUPABASE_URL")
    supabase_service_key = _require_env("SUPABASE_SERVICE_KEY")
    # 优先使用豆包配置，其次回退到 OpenAI 官方配置
    doubao_api_key = os.getenv("DOUBAO_API_KEY")
    doubao_base_url = os.getenv("DOUBAO_BASE_URL")
    model = (
        os.getenv("DOUBAO_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    )

    if doubao_api_key and doubao_base_url:
        llm = OpenAI(api_key=doubao_api_key, base_url=doubao_base_url)
    else:
        openai_api_key = _require_env("OPENAI_API_KEY")
        openai_base_url = os.getenv("OPENAI_BASE_URL")  # 可选，自定义网关
        if openai_base_url:
            llm = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        else:
            llm = OpenAI(api_key=openai_api_key)

    return supabase_url, supabase_service_key, llm, model


def fetch_arxiv_feed(url: str = ARXIV_FEED_URL) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_arxiv_entries(feed_xml: str) -> list[Dict[str, Any]]:
    """
    Returns a list of dicts:
      - title: str
      - summary: str (raw abstract)
      - link: str (canonical arXiv url)
      - published: str (ISO-ish)
    """
    try:
        import feedparser  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: feedparser. Install it via `pip install feedparser`."
        ) from e

    feed = feedparser.parse(feed_xml)
    entries: list[Dict[str, Any]] = []
    for entry in getattr(feed, "entries", []) or []:
        title = (getattr(entry, "title", "") or "").strip()
        summary = (getattr(entry, "summary", "") or "").strip()
        link = (getattr(entry, "link", "") or "").strip()
        published = (getattr(entry, "published", "") or "").strip()
        if not (title and summary and link):
            continue
        entries.append(
            {
                "title": title,
                "summary": summary,
                "link": link,
                "published": published,
            }
        )
    return entries


def supabase_headers(service_key: str) -> Dict[str, str]:
    return {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def supabase_exists_by_source_url(
    supabase_url: str, service_key: str, source_url: str
) -> bool:
    url = f"{supabase_url.rstrip('/')}/rest/v1/ai_news"
    params = {
        "source_url": f"eq.{source_url}",
        "select": "id",
        "limit": 1,
    }
    resp = requests.get(
        url,
        headers=supabase_headers(service_key),
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json() or []
    return len(data) > 0


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Accepts:
      - raw JSON
      - JSON inside ```json ... ```
      - extra prose around a JSON object
    """
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    raise ValueError("Model output is not valid JSON.")


def llm_translate_and_summarize(
    llm: OpenAI, *, model: str, title: str, summary: str
) -> Tuple[str, str]:
    user_prompt = USER_PROMPT_TEMPLATE.format(title=title, summary=summary)

    resp = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    content = (resp.choices[0].message.content or "").strip()
    obj = _extract_json_from_text(content)

    translated_title = (obj.get("translated_title") or "").strip()
    chinese_summary = (obj.get("chinese_summary") or "").strip()

    if not translated_title or not chinese_summary:
        raise ValueError("JSON must contain non-empty translated_title and chinese_summary.")

    return translated_title, chinese_summary


def _parse_published_to_iso(published: str) -> Optional[str]:
    if not published:
        return None
    try:
        # arXiv: "2026-03-09T00:00:00Z"
        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return published


def insert_ai_news(
    supabase_url: str,
    service_key: str,
    *,
    title: str,
    summary: str,
    content_raw: str,
    source_name: str,
    source_url: str,
    publish_date: Optional[str],
) -> None:
    url = f"{supabase_url.rstrip('/')}/rest/v1/ai_news"
    payload = {
        "category": "科研突破",
        "title": title,
        "summary": summary,
        "content_raw": (content_raw or "")[:500],
        "source_name": source_name,
        "source_url": source_url,
        "publish_date": publish_date,
    }

    resp = requests.post(
        url,
        headers={**supabase_headers(service_key), "Prefer": "return=minimal"},
        json=payload,
        timeout=30,
    )
    # 409 也可能是因为唯一键冲突，这里交由上游查重逻辑避免；遇到则抛错看日志。
    resp.raise_for_status()


def main() -> None:
    print("开始抓取 arXiv...")
    supabase_url, supabase_service_key, llm, llm_model = init_clients()

    try:
        xml = fetch_arxiv_feed()
    except Exception as e:
        print(f"[错误] 抓取失败：{e}")
        raise

    try:
        items = parse_arxiv_entries(xml)
    except Exception as e:
        print(f"[错误] 解析 arXiv XML 失败：{e}")
        raise

    if not items:
        print("未解析到任何资讯，结束。")
        return

    new_items = []
    for it in items:
        url = it["link"]
        try:
            if supabase_exists_by_source_url(supabase_url, supabase_service_key, url):
                print(f"数据库已存在，跳过：{url}")
                continue
            new_items.append(it)
        except Exception as e:
            print(f"[警告] 查重失败，跳过该条：{url}，原因：{e}")

    if not new_items:
        print("没有新资讯需要处理。")
        return

    print(f"发现 {len(new_items)} 条新资讯，正在处理...")

    for idx, it in enumerate(new_items, start=1):
        raw_title = it["title"]
        raw_summary = it["summary"]
        source_url = it["link"]
        publish_date = _parse_published_to_iso(it.get("published", ""))

        print(f"[{idx}/{len(new_items)}] 处理：{source_url}")

        try:
            translated_title, chinese_summary = llm_translate_and_summarize(
                llm, model=llm_model, title=raw_title, summary=raw_summary
            )
        except Exception as e:
            print(f"[错误] LLM 处理失败，跳过：{source_url}，原因：{e}")
            time.sleep(1)
            continue

        try:
            insert_ai_news(
                supabase_url,
                supabase_service_key,
                title=translated_title,
                summary=chinese_summary,
                content_raw=raw_summary,
                source_name="arXiv",
                source_url=source_url,
                publish_date=publish_date,
            )
            print(f"已入库：{source_url}")
        except Exception as e:
            print(f"[错误] 入库失败：{source_url}，原因：{e}")

        time.sleep(1)

    print("完成。")


if __name__ == "__main__":
    main()

