import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI

PH_AI_URL = "https://www.producthunt.com/topics/artificial-intelligence"

SYSTEM_PROMPT = (
    "你是一位顶尖的产品猎人（Product Hunter），擅长从用户视角评价产品价值与创新点。"
    "你必须严格以 JSON 对象格式返回结果。"
)

USER_PROMPT_TEMPLATE = (
    "这是一个在 Product Hunt 上获得 {votes} 个赞的 AI 产品。"
    "产品名称：{name}。"
    "产品短描述：{tagline}。"
    "请用 100 字以内的中文说明："
    "1）它解决了用户的什么具体痛点；"
    "2）它的交互或功能有什么独特创新；"
    "3）它适合哪类人群使用。"
    "该条资讯的分类统一标记为'产品进展'。"
    "请严格返回一个 JSON 对象（json object），包含两个字段："
    "\"translated_title\"（产品名+一句话核心功能的中文标题）和 \"chinese_summary\"（侧重用户价值与创新点的中文总结），"
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


def fetch_producthunt_ai(url: str = PH_AI_URL) -> Optional[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AIPulseBot/1.0; +https://www.producthunt.com/)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code in (401, 403):
            print(f"[PH] 被拒绝访问（status={resp.status_code}），本次跳过。")
            return None
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[PH] 抓取异常，跳过本次：{e}")
        return None


def parse_producthunt_ai(html: str) -> List[Dict[str, Any]]:
    """
    尽量解析当日热门 AI 产品：
      - name: 产品名称
      - tagline: 短描述
      - url: Product Hunt 页面链接
      - votes: 点赞数（int）
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: beautifulsoup4. Install it via `pip install beautifulsoup4`."
        ) from e

    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, Any]] = []

    # Product Hunt 页面是 React 渲染的，这里通过尽量宽松的选择器匹配产品卡片链接
    for card in soup.select("a[href^='/posts/']"):
        name = (card.get_text(strip=True) or "").strip()
        href = card.get("href") or ""
        if not name or not href:
            continue

        url = f"https://www.producthunt.com{href}" if href.startswith("/") else href

        # 在相邻或父元素中尝试找到短描述和票数
        tagline = ""
        votes = 0

        parent = card.find_parent()
        if parent:
            # 找短描述
            tagline_el = parent.find("p")
            if tagline_el:
                tagline = (tagline_el.get_text(strip=True) or "").strip()

            # 找票数（常见为带数值的 span/div）
            vote_el = None
            for span in parent.find_all(["span", "div"]):
                text = (span.get_text(strip=True) or "").strip()
                if text.isdigit():
                    vote_el = span
                    break
            if vote_el:
                try:
                    votes = int(vote_el.get_text(strip=True))
                except Exception:
                    votes = 0

        items.append(
            {
                "name": name,
                "tagline": tagline,
                "url": url,
                "votes": votes,
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
    name: str,
    tagline: str,
    votes: int,
) -> Tuple[str, str]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        name=name,
        tagline=tagline or "（暂无官方短描述，请基于产品名称推断。）",
        votes=votes,
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


def _normalize_title(title: str) -> str:
    s = title.lower().strip()
    for ch in "\n\r\t":
        s = s.replace(ch, " ")
    for ch in [":", "：", "-", "—", "_", ".", ","]:
        s = s.replace(ch, " ")
    parts = [p for p in s.split(" ") if p]
    return " ".join(parts)


def fetch_recent_product_news(
    supabase_url: str,
    service_key: str,
) -> List[Dict[str, Any]]:
    url = f"{supabase_url.rstrip('/')}/rest/v1/ai_news"
    since = datetime.now(timezone.utc) - timedelta(days=3)
    params = {
        "category": "eq.产品进展",
        "created_at": f"gte.{since.isoformat()}",
        "select": "id,title,summary,source_name,source_url",
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


def _parse_publish_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    payload = {
        "category": "产品进展",
        "title": title,
        "summary": summary,
        "content_raw": (content_raw or "")[:500],
        "source_name": source_name,
        "source_url": source_url,
        "publish_date": _parse_publish_now(),
    }

    resp = requests.post(
        url,
        headers={**supabase_headers(service_key), "Prefer": "return=minimal"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()


def update_product_with_ph(
    supabase_url: str,
    service_key: str,
    *,
    record_id: int,
    title: str,
    summary: str,
    votes: int,
) -> None:
    url = f"{supabase_url.rstrip('/')}/rest/v1/ai_news"
    params = {"id": f"eq.{record_id}"}

    extra_line = f"该项目在 Product Hunt 获得 {votes} 赞，市场反馈热烈。"
    merged_summary = summary.strip()
    if extra_line not in merged_summary:
        merged_summary = (merged_summary + "\n\n" + extra_line).strip()

    payload = {
        "title": title,
        "summary": merged_summary,
        "source_name": "GitHub & Product Hunt",
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
    print("开始抓取 Product Hunt AI 热门产品...")
    supabase_url, supabase_service_key, llm, llm_model = init_clients()

    html = fetch_producthunt_ai()
    if not html:
        print("[PH Sync] 本次未获取到页面内容，结束。")
        return

    try:
        products = parse_producthunt_ai(html)
    except Exception as e:
        print(f"[错误] 解析 Product Hunt 页面失败：{e}")
        return

    if not products:
        print("未解析到任何 Product Hunt AI 产品，结束。")
        return

    try:
        recent_products = fetch_recent_product_news(supabase_url, supabase_service_key)
    except Exception as e:
        print(f"[警告] 获取最近产品进展失败，将仅按 URL 去重。原因：{e}")
        recent_products = []

    updated_cross = 0
    total = len(products)

    print(f"[PH Sync] 发现 {total} 个候选热门产品，开始处理...")

    for idx, prod in enumerate(products, start=1):
        name = prod["name"]
        tagline = prod["tagline"]
        url = prod["url"]
        votes = int(prod.get("votes") or 0)

        print(f"[{idx}/{total}] 处理：{url}")

        # 1. URL 查重
        try:
            if supabase_exists_by_source_url(supabase_url, supabase_service_key, url):
                print(f"数据库已存在该 URL，跳过：{url}")
                time.sleep(1)
                continue
        except Exception as e:
            print(f"[警告] URL 查重失败，继续尝试标题匹配：{url}，原因：{e}")

        # 2. 标题冲突检查
        conflict = find_title_conflict(name, recent_products) if recent_products else None

        try:
            translated_title, chinese_summary = llm_translate_and_summarize(
                llm,
                model=llm_model,
                name=name,
                tagline=tagline,
                votes=votes,
            )
        except Exception as e:
            print(f"[错误] LLM 处理失败，跳过：{url}，原因：{e}")
            time.sleep(1)
            continue

        if conflict is not None:
            record_id = conflict.get("id")
            old_summary = (conflict.get("summary") or "").strip()
            if record_id is None:
                print(f"[警告] 冲突记录缺少 id，跳过升级：{conflict}")
            else:
                try:
                    # 使用旧 summary，加上 Product Hunt 权重说明
                    update_product_with_ph(
                        supabase_url,
                        supabase_service_key,
                        record_id=int(record_id),
                        title=translated_title,
                        summary=old_summary or chinese_summary,
                        votes=votes,
                    )
                    updated_cross += 1
                    print(f"已升级跨界项目记录（ID={record_id}）：{url}")
                except Exception as e:
                    print(f"[错误] 升级记录失败（ID={record_id}）：{e}")
        else:
            try:
                insert_ai_news(
                    supabase_url,
                    supabase_service_key,
                    title=translated_title,
                    summary=chinese_summary,
                    content_raw=f"{name} - {tagline}",
                    source_name="Product Hunt",
                    source_url=url,
                )
                print(f"已入库 Product Hunt 产品：{url}")
            except Exception as e:
                print(f"[错误] 新增记录失败：{url}，原因：{e}")

        time.sleep(1)

    print(f"[PH Sync] 发现 {total} 个热门产品，更新 {updated_cross} 个跨界项目。")


if __name__ == "__main__":
    main()

