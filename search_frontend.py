import pickle
# from pathlib import Path
print('check1')
from flask import Flask, request, jsonify
from google.cloud import storage
import similarity

client = storage.Client()
bucket = client.get_bucket('207653312_313689804')

blob = bucket.blob('Body_index.pkl')
pkl = blob.download_as_string()
Body_index = pickle.loads(pkl)

blob = bucket.blob('doc_norm.pkl')
pkl = blob.download_as_string()
doc_norm = pickle.loads(pkl)

blob = bucket.blob('Title_index.pkl')
pkl = blob.download_as_string()
Title_index = pickle.loads(pkl)

blob = bucket.blob('Anchor_index.pkl')
pkl = blob.download_as_string()
Anchor_index = pickle.loads(pkl)

blob = bucket.blob('Anchor_index_new.pkl')
pkl = blob.download_as_string()
Anchor_index_new = pickle.loads(pkl)

blob = bucket.blob('doc_title.pkl')
pkl = blob.download_as_string()
doc_title = pickle.loads(pkl)
#
blob = bucket.blob('body_dl.pkl')
pkl = blob.download_as_string()
body_dl = pickle.loads(pkl)
#
blob = bucket.blob('body_idf.pkl')
pkl = blob.download_as_string()
body_idf = pickle.loads(pkl)

blob = bucket.blob('pagerank.pkl')
pkl = blob.download_as_string()
pagerank = pickle.loads(pkl)

blob = bucket.blob('pageview.pkl')
pkl = blob.download_as_string()
pageview = pickle.loads(pkl)
print('check2')
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        print('check3')
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

print('check4')
@app.route("/search")
def search():
    print('check5')
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
        # BEGIN SOLUTION
    query_vector = similarity.tokenize(query)
    candidate_body = similarity.get_candidate_documents_and_scores_search(query_vector, Body_index, 'body')
    candidate_title = similarity.get_candidate_documents_and_scores_search(query_vector, Title_index, 'title')
    # candidate_anchor = similarity.get_candidate_documents_and_scores_search(query_vector, Anchor_index, 'anchor')
    candidate_body.extend(candidate_title)
    # candidate_body.extend(candidate_anchor)
    res_doc = similarity.binary_search(candidate_body)[:10]
    res = []
    for doc_id in res_doc:
        res.append((doc_id, doc_title.get(doc_id, "")))
        # if doc_id in doc_title:
        #     res.append((doc_id,doc_title[doc_id]))
        # else:
        #     res.append((doc_id,""))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query_vector = similarity.tokenize(query)
    query_tfidf = similarity.tfidf_query(query_vector)
    candidate = similarity.get_candidate_documents_and_scores(list(query_tfidf.keys()), Body_index, 'body')
    tfidf_candid = similarity.tfidf_candidates(candidate, body_dl, body_idf)
    res_doc = similarity.cosine_similarity(query_tfidf, tfidf_candid, doc_norm)[:100]
    res = []
    for doc_id in res_doc:
        res.append((doc_id, doc_title.get(doc_id, "")))
        # if doc_id in doc_title:
        #     res.append((doc_id, doc_title[doc_id]))
        # else:
        #     res.append((doc_id, ""))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query_vector = similarity.tokenize(query)
    candidate = similarity.get_candidate_documents_and_scores_search(query_vector, Title_index, 'title')
    res_doc = similarity.binary_search(candidate)
    res = []
    for doc_id in res_doc:
        res.append((doc_id, doc_title.get(doc_id, "")))
        # if doc_id in doc_title:
        #     res.append((doc_id,doc_title[doc_id]))
        # else:
        #     res.append((doc_id,""))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query_vector = similarity.tokenize(query)
    candidate = similarity.get_candidate_documents_and_scores_search(query_vector, Anchor_index_new, 'anchor_new')
    res_doc = similarity.binary_search(candidate)
    res = []
    for doc_id in res_doc:
        res.append((doc_id, doc_title.get(doc_id, "")))
        # if doc_id in doc_title:
        #     res.append((doc_id,doc_title[doc_id]))
        # else:
        #     res.append((doc_id,""))
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for page in wiki_ids:
        if page in pagerank.keys():
            res.append(pagerank[page])
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for page in wiki_ids:
        if page in pageview.keys():
            res.append(pageview[page])
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
