import networkx as nx
from pyvis.network import Network


def build_graph(articles):
    """
    Tạo graph từ danh sách articles
    """

    G = nx.Graph()

    for article in articles:

        source = article["source"]
        title = article["title"]

        G.add_node(source, type="source")
        G.add_node(title, type="article")

        G.add_edge(source, title)

    return G


def visualize_graph(graph, output_path="static/news_graph.html"):
    """
    Xuất graph ra HTML interactive
    """

    net = Network(
        height="700px",
        width="100%",
        bgcolor="#111111",
        font_color="white"
    )

    for node, data in graph.nodes(data=True):

        if data["type"] == "source":
            net.add_node(node, label=node, color="#00ffcc", size=30)
        else:
            net.add_node(node, label=node, color="#ffcc00", size=15)

    for edge in graph.edges():
        net.add_edge(edge[0], edge[1])

    net.save_graph(output_path)

    print(f"Graph saved to {output_path}")


if __name__ == "__main__":

    sample_articles = [
        {"source": "BBC", "title": "AI revolution"},
        {"source": "CNN", "title": "Stock market rises"},
        {"source": "Reuters", "title": "Global trade news"}
    ]

    graph = build_graph(sample_articles)

    visualize_graph(graph)