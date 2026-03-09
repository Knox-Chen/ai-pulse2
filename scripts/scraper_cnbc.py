import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI

CNBC_AI_URL = "https://www.cnbc.com/ai-artificial-intelligence/"

SYSTEM_PROMPT = (
    "你是一位专业的商业分析师，擅长解读科技公司的财报与战略决策。"
    "你必须严格以 JSON 对象格式返回结果。"
)

USER_PROMPT_TEMPLATE = (
    "CNBC 报道了以下一则关于人工智能领域的商业新闻。"
    "请用 100 字以内的中文，从专业商业分析视角，总结："
    "1）该事件的核心财务数据或关键战略决策是什么；"
    "2）它对整个 AI 行业竞争格局有何深远影响；"
    "3）重点标注涉及的上市公司名称及可能的股价或市值影响（如可推断）。"
    "该条资讯的分类统一标记为'商业资讯'。"
    "请严格返回一个 JSON 对象（json object），包含两个字段："
    "\"translated_title\"（干练不标题党的中文标题）和 \"chinese_summary\"（聚焦财务影响和行业格局的中文总结），"
    "不得输出任何多余文字（包括解释、注释或代码块）。"
    "原始新闻标题：{title}"
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


def fetch_cnbc_ai(url: str = CNBC_AI_URL) -> Optional[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AIPulseBot/1.0; +https://www.cnbc.com/)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code in (401, 403):
            print(f"[CNBC] 被拒绝访问（status={resp.status_code}），本次跳过。")
            return None
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[CNBC] 抓取异常，跳过本次：{e}")
        return None


def parse_cnbc_ai(html: str) -> List[Dict[str, Any]]:
    """
    返回列表：
      - title: 新闻标题
      - url: 新闻链接
      - published: 发布时间（字符串，尽量从页面中提取）
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: beautifulsoup4. Install it via `pip install beautifulsoup4`."
        ) from e

    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, Any]] = []

    # CNBC 结构较复杂，这里尽量捕获 AI 频道列表中的卡片
    for card in soup.select("a.Card-title, a[data-target='Card.title']"):
        title = (card.get_text(strip=True) or "").strip()
        href = card.get("href") or ""
        if not title or not href:
            continue

        if href.startswith("http"):
            url = href
        else:
            url = f"https://www.cnbc.com{href}"

        # 发布时间尝试在父级块中查找 time 元素
        published = ""
        parent = card.find_parent()
        if parent:
            time_el = parent.find("time")
            if time_el and time_el.has_attr("datetime"):
                published = (time_el.get("datetime") or "").strip()
            elif time_el:
                published = (time_el.get_text(strip=True) or "").strip()

        items.append(
            {
                "title": title,
                "url": url,
                "published": published,
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
    llm: OpenAI,
    *,
    model: str,
    title: str,
) -> Tuple[str, str]:
    user_prompt = USER_PROMPT_TEMPLATE.format(title=title)

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


def _normalize_title(title: str) -> str:
    s = title.lower().strip()
    for ch in "\n\r\t":
        s = s.replace(ch, " ")
    for ch in [":", "：", "-", "—", "_", ".", ","]:
        s = s.replace(ch, " ")
    parts = [p for p in s.split(" ") if p]
    return " ".join(parts)


def fetch_recent_business_news(
    supabase_url: str,
    service_key: str,
) -> List[Dict[str, Any]]:
    url = f"{supabase_url.rstrip('/')}/rest/v1/ai_news"
    since = datetime.now(timezone.utc) - timedelta(hours=48)
    params = {
        "category": "eq.商业资讯",
        "created_at": f"gte.{since.isoformat()}",
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


def _parse_published_to_iso(published: str) -> Optional[str]:
    if not published:
        return None
    try:
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
        "category": "商业资讯",
        "title": title,
        "summary": summary,
        "content_raw": (content_raw or "")[:500],
        "source_name": source_name,
        "source_url": source_url,
        "publish_date": publish_date or datetime.now(timezone.utc).isoformat(),
    }

    resp = requests.post(
        url,
        headers={**supabase_headers(service_key), "Prefer": "return=minimal"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()


def update_business_news_with_cnbc(
    supabase_url: str,
    service_key: str,
    *,
    record_id: int,
    title: str,
    summary: str,
) -> None:
    url = f"{supabase_url.rstrip('/')}/rest/v1/ai_news"
    params = {"id": f"eq.{record_id}"}
    payload = {
        "title": title,
        "summary": summary,
        "source_name": "TechCrunch & CNBC",
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
    print("开始抓取 CNBC AI 频道...")
    supabase_url, supabase_service_key, llm, llm_model = init_clients()

    html = fetch_cnbc_ai()
    if not html:
        print("[CNBC Sync] 本次未获取到页面内容，结束。")
        return

    try:
        articles = parse_cnbc_ai(html)
    except Exception as e:
        print(f"[错误] 解析 CNBC 页面失败：{e}")
        return

    if not articles:
        print("未解析到任何 CNBC AI 文章，结束。")
        return

    try:
        recent_business = fetch_recent_business_news(supabase_url, supabase_service_key)
    except Exception as e:
        print(f"[警告] 获取最近商业资讯失败，将仅按 URL 去重。原因：{e}")
        recent_business = []

    upgraded = 0
    inserted = 0

    print(f"[CNBC Sync] 发现 {len(articles)} 条候选深度报道，开始处理...")

    for idx, art in enumerate(articles, start=1):
        title = art["title"]
        url = art["url"]
        publish_date = _parse_published_to_iso(art.get("published", ""))

        print(f"[{idx}/{len(articles)}] 处理：{url}")

        # 1. URL 查重
        try:
            if supabase_exists_by_source_url(supabase_url, supabase_service_key, url):
                print(f"数据库已存在该 URL，跳过：{url}")
                time.sleep(2)
                continue
        except Exception as e:
            print(f"[警告] URL 查重失败，继续尝试标题匹配：{url}，原因：{e}")

        # 2. 标题冲突检查
        conflict = find_title_conflict(title, recent_business) if recent_business else None

        try:
            translated_title, chinese_summary = llm_translate_and_summarize(
                llm,
                model=llm_model,
                title=title,
            )
        except Exception as e:
            print(f"[错误] LLM 处理失败，跳过：{url}，原因：{e}")
            time.sleep(2)
            continue

        if conflict is not None:
            record_id = conflict.get("id")
            if record_id is None:
                print(f"[警告] 冲突记录缺少 id，跳过升级：{conflict}")
            else:
                try:
                    update_business_news_with_cnbc(
                        supabase_url,
                        supabase_service_key,
                        record_id=int(record_id),
                        title=translated_title,
                        summary=chinese_summary,
                    )
                    upgraded += 1
                    print(f"已升级商业资讯记录（ID={record_id}）：{url}")
                except Exception as e:
                    print(f"[错误] 升级记录失败（ID={record_id}）：{e}")
        else:
            try:
                insert_ai_news(
                    supabase_url,
                    supabase_service_key,
                    title=translated_title,
                    summary=chinese_summary,
                    content_raw=title,
                    source_name="CNBC",
                    source_url=url,
                    publish_date=publish_date,
                )
                inserted += 1
                print(f"已入库 CNBC 深度报道：{url}")
            except Exception as e:
                print(f"[错误] 新增记录失败：{url}，原因：{e}")

        time.sleep(2)

    print(f"[CNBC Sync] 发现 {len(articles)} 条深度报道，升级了 {upgraded} 条快讯记录，新增 {inserted} 条。")


if __name__ == "__main__":
    main()

