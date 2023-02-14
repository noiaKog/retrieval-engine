# retrieval-engine


SearchEngine
Welcome to our search engine over the whole wikipedia corpus (over 8 million documents)

How To Start? open new python file and send requests to the engine

import request
response = request.get(url="http://34.72.166.196/search",params={"query":"hello world"})
Key Component of the Engine
Our engine contains several logic units:

search_forntend - flask application the reveal 6 endpoints:
search - the main search method of the engine, combine results from number of sub-searches.
search_body - search function over the body index
search_title - search function over the title index
search_anchor - search function over the anchor index
get_pagerank - return the pagerank of a given wiki id article (based on internal links)
get_pageviews - return the page views number of a given wiki article id
search_backend - Engine backend
search - multithreaded implementation for the search endpoint
search_body - logical implemantation of the search body endpoint
search_title - logical implemantation of the search title endpoint
search_anchor - logical implemantation of the search anchor endpoint
