import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI

GITHUB_TRENDING_URL = "https://github.com/trending/python?since=daily"

SYSTEM_PROMPT = (
    "你是一位资深产品经理，擅长挖掘开源项目的实际应用价值。"
    "你必须严格以 JSON 对象格式返回结果。"
)

USER_PROMPT_TEMPLATE = (
    "请分析这个 GitHub 项目：{repo_name}。"
    "基于其描述：{description}。"
    "用中文总结它的核心功能和对开发者的意义（100字以内），语气专业、客观。"
    "该条资讯的分类统一标记为'产品进展'。"
    "请严格返回一个 JSON 对象（json object），包含两个字段："
    "\"translated_title\"（项目名称+简要功能的中文标题）和 \"chinese_summary\"（中文总结），"
    "不得输出任何多余文字（包括解释、注释或代码块）。"
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
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        if openai_base_url:
            llm = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        else:
            llm = OpenAI(api_key=openai_api_key)

    return supabase_url, supabase_service_key, llm, model


def fetch_github_trending(url: str = GITHUB_TRENDING_URL) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AIPulseBot/1.0; +https://github.com/)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_github_trending(html: str) -> list[Dict[str, Any]]:
    """
    Returns list of:
      - repo_name: "owner/repo"
      - description: project description
      - repo_url: https://github.com/owner/repo
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: beautifulsoup4. Install it via `pip install beautifulsoup4`."
        ) from e

    soup = BeautifulSoup(html, "html.parser")
    items: list[Dict[str, Any]] = []

    for article in soup.select("article.Box-row"):
        title_a = article.select_one("h2 a")
        if not title_a:
            continue
        raw_name = title_a.get_text(strip=True).replace("\n", " ")
        # Format is usually "owner / repo"
        repo_name = " ".join(raw_name.split())
        href = title_a.get("href") or ""
        repo_url = f"https://github.com{href}".strip()

        desc_el = article.select_one("p")
        description = (desc_el.get_text(strip=True) if desc_el else "").strip()

        if not (repo_name and repo_url):
            continue

        items.append(
            {
                "repo_name": repo_name,
                "description": description,
                "repo_url": repo_url,
            }
        )

    return items


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
    llm: OpenAI, *, model: str, repo_name: str, description: str
) -> Tuple[str, str]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        repo_name=repo_name,
        description=description or "（原文暂无描述，请根据项目名称进行合理推断。）",
    )

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


def insert_ai_news(
    supabase_url: str,
    service_key: str,
    *,
    title: str,
    summary: str,
    content_raw: str,
    source_name: str,
    source_url: str,
) -> None:
    url = f"{supabase_url.rstrip('/')}/rest/v1/ai_news"
    # GitHub Trending 没有具体发布日期，用当前时间作为 publish_date
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = {
        "category": "产品进展",
        "title": title,
        "summary": summary,
        "content_raw": (content_raw or "")[:500],
        "source_name": source_name,
        "source_url": source_url,
        "publish_date": now_iso,
    }

    resp = requests.post(
        url,
        headers={**supabase_headers(service_key), "Prefer": "return=minimal"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()


def main() -> None:
    print("开始抓取 GitHub Trending AI 项目...")
    supabase_url, supabase_service_key, llm, llm_model = init_clients()

    try:
        html = fetch_github_trending()
    except Exception as e:
        print(f"[错误] 抓取失败：{e}")
        raise

    try:
        items = parse_github_trending(html)
    except Exception as e:
        print(f"[错误] 解析 GitHub Trending 页面失败：{e}")
        raise

    if not items:
        print("未解析到任何项目，结束。")
        return

    new_items = []
    for it in items:
        url = it["repo_url"]
        try:
            if supabase_exists_by_source_url(supabase_url, supabase_service_key, url):
                print(f"数据库已存在，跳过：{url}")
                continue
            new_items.append(it)
        except Exception as e:
            print(f"[警告] 查重失败，跳过该条：{url}，原因：{e}")

    if not new_items:
        print("没有新项目需要处理。")
        return

    print(f"发现 {len(new_items)} 个新项目，正在处理...")

    for idx, it in enumerate(new_items, start=1):
        repo_name = it["repo_name"]
        description = it["description"]
        source_url = it["repo_url"]

        print(f"[{idx}/{len(new_items)}] 处理：{source_url}")

        try:
            translated_title, chinese_summary = llm_translate_and_summarize(
                llm,
                model=llm_model,
                repo_name=repo_name,
                description=description,
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
                content_raw=description,
                source_name="GitHub Trending",
                source_url=source_url,
            )
            print(f"已入库：{source_url}")
        except Exception as e:
            print(f"[错误] 入库失败：{source_url}，原因：{e}")

        # 防止请求过快被限流
        time.sleep(1)

    print("完成。")


if __name__ == "__main__":
    main()

