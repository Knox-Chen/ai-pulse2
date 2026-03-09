"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabaseClient";
import type { AINewsCategory, AINewsItem } from "@/types/AINewsItem";

const CATEGORIES: { label: string; value: AINewsCategory }[] = [
  { label: "科研突破", value: "科研突破" },
  { label: "商业资讯", value: "商业资讯" },
  { label: "产品进展", value: "产品进展" },
];

type DateRange = "today" | "week" | "month" | "custom";

function formatDate(publishIso: string | null, createdIso: string | null): string {
  const iso = publishIso || createdIso;
  if (!iso) return "时间未知";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "时间未知";
  return d.toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function HomePage() {
  const [category, setCategory] = useState<AINewsCategory>("科研突破");
  const [news, setNews] = useState<AINewsItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [range, setRange] = useState<DateRange>("week");
  const [customStart, setCustomStart] = useState<string>("");
  const [customEnd, setCustomEnd] = useState<string>("");

  useEffect(() => {
    let cancelled = false;

    async function fetchNews() {
      setLoading(true);
      try {
        const now = new Date();
        let from: string | null = null;
        let to: string | null = null;

        if (range === "today") {
          const start = new Date(
            now.getFullYear(),
            now.getMonth(),
            now.getDate(),
            0,
            0,
            0,
            0,
          );
          from = start.toISOString();
        } else if (range === "week") {
          const start = new Date(now);
          start.setDate(start.getDate() - 7);
          from = start.toISOString();
        } else if (range === "month") {
          const start = new Date(now);
          start.setMonth(start.getMonth() - 1);
          from = start.toISOString();
        } else if (range === "custom") {
          if (customStart) {
            const start = new Date(customStart);
            if (!Number.isNaN(start.getTime())) {
              from = start.toISOString();
            }
          }
          if (customEnd) {
            const end = new Date(customEnd);
            if (!Number.isNaN(end.getTime())) {
              end.setHours(23, 59, 59, 999);
              to = end.toISOString();
            }
          }
        }

        let query = supabase
          .from("ai_news")
          .select("*")
          .eq("category", category)
          .order("publish_date", { ascending: false })
          .limit(20);

        if (from) {
          query = query.gte("publish_date", from);
        }
        if (to) {
          query = query.lte("publish_date", to);
        }

        const { data, error } = await query;

        if (error) {
          console.error("Failed to fetch ai_news:", error);
          if (!cancelled) setNews([]);
          return;
        }

        if (!cancelled) {
          setNews(data as AINewsItem[]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchNews();

    return () => {
      cancelled = true;
    };
  }, [category, range, customStart, customEnd]);

  return (
    <div className="min-h-screen bg-zinc-50 font-sans text-zinc-900">
      <main className="mx-auto flex min-h-screen max-w-4xl flex-col px-4 pb-16 pt-10 sm:px-6 lg:px-8">
        {/* Header */}
        <header className="mb-8 flex flex-col items-center gap-3 text-center">
          <div className="inline-flex items-center gap-2 rounded-full bg-zinc-900 px-4 py-1 text-sm font-medium text-white shadow-sm">
            <span className="inline-flex h-2 w-2 animate-pulse rounded-full bg-yellow-400" />
            <span>AI Pulse · 实时 AI 行业脉动</span>
          </div>
          <h1 className="flex items-center gap-3 text-3xl font-semibold tracking-tight sm:text-4xl">
            <span className="inline-flex h-9 w-9 items-center justify-center rounded-full bg-yellow-400 text-xl shadow-md">
              ⚡
            </span>
            <span>AI Pulse</span>
          </h1>
          <p className="mt-1 max-w-xl text-sm text-zinc-500 sm:text-base">
            自动汇总前沿 AI 资讯，按科研突破、商业动态与产品进展分类，一页洞察行业节奏。
          </p>
        </header>

        {/* Tabs + Date Range */}
        <div className="mb-6 flex flex-col items-center gap-3 sm:flex-row sm:justify-between">
          {/* Category tabs */}
          <div className="flex justify-center">
            <div className="inline-flex rounded-full bg-zinc-100 p-1 shadow-inner">
              {CATEGORIES.map((cat) => {
                const active = cat.value === category;
                return (
                  <button
                    key={cat.value}
                    type="button"
                    onClick={() => setCategory(cat.value)}
                    className={[
                      "relative mx-0.5 rounded-full px-4 py-1.5 text-sm font-medium transition-all",
                      active
                        ? "bg-zinc-900 text-white shadow-sm"
                        : "bg-transparent text-zinc-600 hover:bg-white hover:text-zinc-900",
                    ].join(" ")}
                  >
                    {cat.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Date range selector */}
          <div className="flex flex-col items-center gap-2 text-xs text-zinc-500 sm:flex-row sm:text-sm">
            <span className="mr-1 hidden sm:inline">时间范围：</span>
            <div className="inline-flex rounded-full bg-zinc-100 p-0.5 shadow-inner">
              {[
                { label: "今日", value: "today" as DateRange },
                { label: "近一周", value: "week" as DateRange },
                { label: "近一月", value: "month" as DateRange },
                { label: "自定义", value: "custom" as DateRange },
              ].map((r) => {
                const active = r.value === range;
                return (
                  <button
                    key={r.value}
                    type="button"
                    onClick={() => setRange(r.value)}
                    className={[
                      "rounded-full px-3 py-1 text-xs font-medium transition-all",
                      active
                        ? "bg-zinc-900 text-white shadow-sm"
                        : "bg-transparent text-zinc-600 hover:bg-white hover:text-zinc-900",
                    ].join(" ")}
                  >
                    {r.label}
                  </button>
                );
              })}
            </div>

            {range === "custom" && (
              <div className="flex flex-wrap items-center gap-1 sm:gap-2">
                <input
                  type="date"
                  value={customStart}
                  onChange={(e) => setCustomStart(e.target.value)}
                  className="h-7 rounded-md border border-zinc-200 bg-white px-2 text-xs text-zinc-700 shadow-sm focus:border-zinc-400 focus:outline-none"
                />
                <span>—</span>
                <input
                  type="date"
                  value={customEnd}
                  onChange={(e) => setCustomEnd(e.target.value)}
                  className="h-7 rounded-md border border-zinc-200 bg-white px-2 text-xs text-zinc-700 shadow-sm focus:border-zinc-400 focus:outline-none"
                />
              </div>
            )}
          </div>
        </div>

        {/* Content */}
        <section className="flex-1">
          {loading ? (
            <div className="flex flex-col items-center justify-center gap-3 py-16 text-zinc-500">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-zinc-300 border-t-zinc-900" />
              <p className="text-sm">AI 正在为你扫描 {category} 相关最新资讯...</p>
            </div>
          ) : news.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-3 py-16 text-zinc-500">
              <p className="text-sm font-medium">AI 正在该领域紧急搜寻中...</p>
              <p className="text-xs text-zinc-400">
                暂无 {category} 相关资讯，请稍后再来看看。
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {news.map((item) => (
                <article
                  key={item.id}
                  className="group rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm transition hover:-translate-y-0.5 hover:border-zinc-300 hover:shadow-md"
                >
                  <div className="mb-2 flex items-center justify-between gap-3 text-xs text-zinc-500">
                    <span className="inline-flex items-center gap-1 rounded-full bg-zinc-100 px-2 py-0.5 font-medium text-zinc-700">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                      {item.category}
                    </span>
                    <span>{formatDate(item.publish_date, item.created_at)}</span>
                  </div>

                  <h2 className="mb-1 text-base font-semibold leading-snug text-zinc-900">
                    <a
                      href={item.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 transition-colors hover:text-zinc-950 group-hover:text-zinc-950"
                    >
                      <span>{item.title}</span>
                      <span className="text-xs text-zinc-400 group-hover:text-zinc-500">
                        ↗
                      </span>
                    </a>
                  </h2>

                  {item.summary && (
                    <p className="mb-3 text-sm leading-relaxed text-zinc-600">
                      {item.summary}
                    </p>
                  )}

                  <div className="flex items-center justify-between text-xs text-zinc-400">
                    <span>{item.source_name || "未知来源"}</span>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
