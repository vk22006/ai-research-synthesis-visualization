import requests

def fetch_arxiv(query):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results=10"
    response = requests.get(url)
    return response.text