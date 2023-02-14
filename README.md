# retrieval-engine


SearchEngine - search engine over the whole wikipedia corpus (over 8 million documents)

How To Start? open new python file and send requests to the engine

import request
response = request.get(url="http://35.223.4.123:8080/search",params={"query":"hello world"})
Key Component of the Engine
The engine contains several logic units:

**search_forntend - flask application the reveal 6 endpoints:

*search - the main search method of the engine, combine results from number of sub-searches.

*search_body - Search over Wikipedia only by the body of the documents. The documents relevancy is ranked by Cosine-Similarity   score using TF-IDF.

*search_title - Search over Wikipedia only by the titles of the documents. The documents relevancy is ranked by binary search.

*search_anchor - Search over Wikipedia only by the anchor text related to the documents. The documents relevancy is ranked by      binary search.

*get_pagerank - return the pagerank of a given wiki id article (based on internal links)

*get_pageviews - return the page views number of a given wiki article id

# similarity - Engine backend
search - logical implementation for the search endpoint

search_body - logical implemantation of the search body endpoint

search_title - logical implemantation of the search title endpoint

search_anchor - logical implemantation of the search anchor endpoint




File directory:
search_frontend.py

Loading the indexes, page rank and page view files, implemnet all the function input

beckend.py

Implemnet all the measure - BM25, cosin similarity, binary anchor, binary title, search body, and merge function that combine the measure by wight.

index_code.ipynb

Load 6M doc from Wiki.
Preproccecing.
Build the Index
Save the index to bucket's cloud
inverted_index_gcp.py

Implemnet tha InvertedIndex, MultiFileReader, MultiFileWriter Class.

bucket file

Text filr for each bucket content.
