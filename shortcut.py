import itertools
import jsonlines
from tqdm import tqdm

# stop_words and punctuations used for filtering extracted n-gram patterns
import nltk
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
stop_words.append('uh')
import string
puncs = string.punctuation

def ngram_extraction(prediction_files, tokenizer):
    '''
    Extract all ngrams from input reviews as features.
    
    INPUT: 
      - prediction_files: file path for all predictions
      - tokenizer: tokenizer used for tokenization
    
    OUTPUT: 
      - ngrams: a list of dicts, each dict with a type of n-grams (n=1,2,3 or 4) as keys, and predicted label counts as values.
    '''
    ngrams = [{}, {}, {}, {}]
    label_to_id = {"positive": 0, "negative": 1}
    
    for pred_file in prediction_files:
        with jsonlines.open(pred_file, "r") as reader:
            preds = [pr for pr in reader.iter()]
        
        for pred in tqdm(preds):
            #################################################################
            #         TODO: construct n-gram patterns as dictionary         # 
            #################################################################
            
            review_words = [word.strip("Ġ") for word in tokenizer.tokenize(pred["review"].lower()) if word.strip("Ġ")]
            pred_id = label_to_id[pred["prediction"]]
            
            # Replace "..." statement with your code
            for n in range(1, 5):  # we consider 1/2/3/4-gram

                for i in range(len(review_words) - n + 1):

                    n_gram_tokens = review_words[i:i+n]

                    if any(token in stop_words or any(char in puncs for char in token) for token in n_gram_tokens):
                        continue  # Skip n-grams with stop words or punctuation

                    n_gram = ' '.join(n_gram_tokens)

                    # Add or update the n-gram in the dictionary
                    if n_gram in ngrams[n-1]:
                        ngrams[n-1][n_gram][pred_id] += 1
                    else:
                        ngrams[n-1][n_gram] = [0, 0]
                        ngrams[n-1][n_gram][pred_id] += 1
                        #####################################################
            #                   END OF YOUR CODE                #
            #####################################################

    return ngrams
