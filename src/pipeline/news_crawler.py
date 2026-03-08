import feedparser
from datetime import datetime

# RSS feeds từ các nguồn tin uy tín
NEWS_FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "CNN": "http://rss.cnn.com/rss/edition.rss",
    "Reuters": "https://www.reutersagency.com/feed/?best-topics=business"
}


def crawl_news(limit=10):
    """
    Crawl tin tức từ các RSS feed
    """

    articles = []

    for source, url in NEWS_FEEDS.items():

        try:

            feed = feedparser.parse(url)

            for entry in feed.entries[:limit]:

                articles.append({

                    "source": source,
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", ""),
                    "time": datetime.now().strftime("%H:%M")

                })

        except Exception as e:
            print(f"Error crawling {source}: {e}")

    return articles


# dùng khi test file riêng
if __name__ == "__main__":

    news = crawl_news()

    for article in news[:5]:
        print(article)