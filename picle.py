import pickle
import os
import re
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.corpus import stopwords


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticon = re.findall()

if __name__ == '__main__':
    clf = SGDClassifier(loss='log', random_state=1, n_iter_no_change=1)
    stop = stopwords.words('english')
    cur_dir = os.path.dirname(__file__)

    dest = os.path.join('klasyfikator_filmowy', 'pkl_objects')
    if not os.path.exists(dest):
        os.makedirs(dest)
    pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'),
                protocol=4)
    pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'),
                protocol=4)

    pickle.load(open(os.path.join(cur_dir, 'pkl_objects',
                                  'stopwords.pkl'), 'rb'))