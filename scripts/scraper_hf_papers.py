import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI

HF_PAPERS_URL = "https://huggingface.co/papers"

SYSTEM_PROMPT = (
    "你是一位 AI 领域专家，擅长结合社区信号评估论文的实际影响力。"
    "你必须严格以 JSON 对象格式返回结果。"
)

USER_PROMPT_TEMPLATE = (
    "这是一篇在 Hugging Face 社区中获得 {upvotes} 个点赞的论文。"
    "请基于下列论文标题，结合你对 AI 领域的理解，完成以下任务："
    "1）翻译并给出一个中文标题；"
    "2）用 100 字以内的中文说明：该研究解决了什么实际痛点？"
    "3）说明为什么开发者社区会对它高度关注；"
    "4）如果常见，请指出是否通常会配套开源模型或代码（允许根据经验合理推断）。"
    "该条资讯的分类统一标记为'科研突破'。"
    "即使标题信息有限，也不要说“无效”“信息不足”“无法判断”等字样，"
    "而是基于标题做出合理、概括性的推断。"
    "请严格返回一个 JSON 对象（json object），包含两个字段："
    "\"translated_title\"（中文标题）和 \"chinese_summary\"（上述分析的综合中文总结），"
    "不得输出任何多余文字（包括解释、注释或代码块）。"
    "论文标题：{title}"
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


def fetch_hf_papers(url: str = HF_PAPERS_URL) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AIPulseBot/1.0; +https://huggingface.co/)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_hf_papers(html: str) -> List[Dict[str, Any]]:
    """
    返回列表：
      - title: 论文标题
      - url: HF 论文页面链接
      - upvotes: 点赞数（int）
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: beautifulsoup4. Install it via `pip install beautifulsoup4`."
        ) from e

    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict[str, Any]] = []

    # 结构可能会变化，这里采用尽量稳健的选择器：
    # 每篇论文一般是一个包含链接和标题的 card。
    for card in soup.select("a[href^='/papers/']"):
        title_el = card.find("h3") or card.find("h4") or card
        title = (title_el.get_text(strip=True) if title_el else "").strip()
        href = card.get("href") or ""
        if not title or not href:
            continue
        url = href if href.startswith("http") else f"https://huggingface.co{href}"

        # 尝试在 card 内部查找点赞数
        upvotes = 0
        upvote_el = card.find(string=lambda s: s and "likes" in s.lower())
        if upvote_el:
            digits = "".join(ch for ch in upvote_el if ch.isdigit())
            if digits:
                try:
                    upvotes = int(digits)
                except Exception:
                    upvotes = 0

        results.append(
            {
                "title": title,
                "url": url,
                "upvotes": upvotes,
            }
        )

    return results


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
    llm: OpenAI,
    *,
    model: str,
    title: str,
    upvotes: int,
) -> Tuple[str, str]:
    user_prompt = USER_PROMPT_TEMPLATE.format(title=title, upvotes=upvotes)

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


def _is_suspicious_output(title: str, summary: str) -> bool:
    """
    简单过滤一些明显是“模型拒绝/无效”类的输出，避免写入这种内容。
    """
    bad_keywords = [
        "无效论文标题",
        "标题无效",
        "信息不足",
        "无法明确",
        "无法判断",
        "无法给出有效分析",
    ]
    combined = f"{title} {summary}"
    return any(k in combined for k in bad_keywords)


def _normalize_title(title: str) -> str:
    s = title.lower().strip()
    for ch in "\n\r\t":
        s = s.replace(ch, " ")
    # 去掉多余空格和常见标点
    for ch in [":", "：", "-", "—", "_", ".", ","]:
        s = s.replace(ch, " ")
    parts = [p for p in s.split(" ") if p]
    return " ".join(parts)


def fetch_recent_research_titles(
    supabase_url: str,
    service_key: str,
) -> List[Dict[str, Any]]:
    url = f"{supabase_url.rstrip('/')}/rest/v1/ai_news"
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    params = {
        "category": "eq.科研突破",
        "created_at": f"gte.{seven_days_ago.isoformat()}",
        "select": "id,title,source_name,source_url",
        "order": "created_at.desc",
    }
    resp = requests.get(
        url,
        headers=supabase_headers(service_key),
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json() or []
    return data


def find_title_conflict(
    new_title: str,
    recent_items: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    new_norm = _normalize_title(new_title)
    if not new_norm:
        return None

    for item in recent_items:
        old_title = (item.get("title") or "").strip()
        old_norm = _normalize_title(old_title)
        if not old_norm:
            continue
        if new_norm == old_norm:
            return item
        if new_norm in old_norm or old_norm in new_norm:
            return item
    return None


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
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = {
        "category": "科研突破",
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


def update_ai_news_with_hf(
    supabase_url: str,
    service_key: str,
    *,
    record_id: int,
    title: str,
    summary: str,
) -> None:
    url = f"{supabase_url.rstrip('/')}/rest/v1/ai_news"
    params = {
        "id": f"eq.{record_id}",
    }
    payload = {
        "title": title,
        "summary": summary,
        "source_name": "arXiv & Hugging Face",
    }

    resp = requests.patch(
        url,
        headers={**supabase_headers(service_key), "Prefer": "return=minimal"},
        params=params,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()


def main() -> None:
    print("开始抓取 Hugging Face Papers...")
    supabase_url, supabase_service_key, llm, llm_model = init_clients()

    try:
        html = fetch_hf_papers()
    except Exception as e:
        print(f"[错误] 抓取失败：{e}")
        raise

    try:
        papers = parse_hf_papers(html)
    except Exception as e:
        print(f"[错误] 解析 HF 页面失败：{e}")
        raise

    if not papers:
        print("未解析到任何论文，结束。")
        return

    try:
        recent_items = fetch_recent_research_titles(supabase_url, supabase_service_key)
    except Exception as e:
        print(f"[警告] 获取最近科研突破标题失败，将仅按 URL 去重。原因：{e}")
        recent_items = []

    inserted = 0
    updated = 0

    for idx, paper in enumerate(papers, start=1):
        title = paper["title"]
        url = paper["url"]
        upvotes = int(paper.get("upvotes") or 0)

        print(f"[{idx}/{len(papers)}] 处理：{url}")

        # 1. 先根据 URL 去重
        try:
            if supabase_exists_by_source_url(supabase_url, supabase_service_key, url):
                print(f"数据库已存在该 URL，跳过：{url}")
                continue
        except Exception as e:
            print(f"[警告] URL 查重失败，继续尝试标题匹配：{url}，原因：{e}")

        # 2. 标题冲突检查
        conflict = find_title_conflict(title, recent_items) if recent_items else None

        try:
            translated_title, chinese_summary = llm_translate_and_summarize(
                llm,
                model=llm_model,
                title=title,
                upvotes=upvotes,
            )
            if _is_suspicious_output(translated_title, chinese_summary):
                print(f"[警告] LLM 输出疑似无效内容，跳过写入：{url}")
                time.sleep(1)
                continue
        except Exception as e:
            print(f"[错误] LLM 处理失败，跳过：{url}，原因：{e}")
            time.sleep(1)
            continue

        # 3. 冲突：升级已有记录（UPDATE）
        if conflict is not None:
            record_id = conflict.get("id")
            if record_id is None:
                print(f"[警告] 冲突记录缺少 id，跳过升级：{conflict}")
            else:
                try:
                    update_ai_news_with_hf(
                        supabase_url,
                        supabase_service_key,
                        record_id=int(record_id),
                        title=translated_title,
                        summary=chinese_summary,
                    )
                    updated += 1
                    print(f"已升级现有记录（ID={record_id}）：{url}")
                except Exception as e:
                    print(f"[错误] 升级记录失败（ID={record_id}）：{e}")

        # 4. 无冲突：插入新记录
        else:
            try:
                insert_ai_news(
                    supabase_url,
                    supabase_service_key,
                    title=translated_title,
                    summary=chinese_summary,
                    content_raw=title,
                    source_name="Hugging Face",
                    source_url=url,
                )
                inserted += 1
                print(f"已入库新论文：{url}")
            except Exception as e:
                print(f"[错误] 新增记录失败：{url}，原因：{e}")

        time.sleep(1)

    print(f"[HF Sync] 新增 {inserted} 条，更新（升级） {updated} 条。")


if __name__ == "__main__":
    main()

