# Information Retrieval

Using the TF-IDF algorithm, this code retrieves text data. The Google Cloud storage library is utilized to access a Google Cloud Storage bucket, while the NLTK library performs tokenization and stopword removal.

Initially, the code defines a function that leverages the storage library to gain access to a bucket on Google Cloud Storage. Additionally, NLTK's stopword list is employed to eliminate common words that lack significant meaning during the tokenization process.

The code proceeds to define several text retrieval functions: tokenization function that tokenizes a given text and removes stopwords, get_candidate_documents_and_scores function that retrieves candidate documents and their scores, and binary_search function that determines the similarity between the query and documents.

To rank candidate documents and queries, the code employs the tfidf_candidates and tfidf_query functions to calculate their respective tf-idf scores.

Finally, the binary_search function is used again to determine the similarity between the query and the documents.
