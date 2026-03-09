export type AINewsCategory = "科研突破" | "商业资讯" | "产品进展";

export interface AINewsItem {
  id: number;
  category: AINewsCategory;
  title: string;
  summary: string | null;
  content_raw: string | null;
  source_name: string | null;
  source_url: string;
  publish_date: string | null;
  created_at: string;
}

