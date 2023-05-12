from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling import LabelingFunction

# For clarity, we define constants to represent the class labels for positive, negative, and abstaining.
ABSTAIN =-1
NEG = 0
POS = 1
  
#generic labeling functions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
@labeling_function()
def vader(x):
  sid_obj = SentimentIntensityAnalyzer()
  sentiment_dict = sid_obj.polarity_scores(x.review_text)
  if sentiment_dict['compound'] >= 0.75 :
    return POS
  elif sentiment_dict['compound'] <= - 0.75 :
    return NEG
  else :
    return ABSTAIN

from snorkel.preprocess import preprocessor
from textblob import TextBlob
@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.review_text)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    return x
@labeling_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return POS if x.polarity > 0.9 else ABSTAIN
@labeling_function(pre=[textblob_sentiment])
def textblob_subjectivity(x):
    return POS if x.subjectivity >= 0.7 else ABSTAIN


@labeling_function()
def lf_contains_positive_word(x):
    return POS if "great" in x["review_text"].lower() or "excellent" in x["review_text"].lower() else ABSTAIN

@labeling_function()
def lf_contains_negative_word(x):
    return NEG if "poor" in x["review_text"].lower() or "bad" in x["review_text"].lower() else ABSTAIN

@labeling_function()
def lf_contains_recommendation(x):
    return POS if "recommend" in x["review_text"].lower() else ABSTAIN

#mod cloth specific labeling functions
@labeling_function()
def fit_val(x):
  if(x.fit=='fit'):
    return POS
  else:
    return NEG
@labeling_function()
def rating(x):
  if(x.rating>8):
    return POS
  elif x.rating<=5:
    return NEG
  else:
    return ABSTAIN

# amazon specific labeling functions
@labeling_function()
def lf_positive(x):
    return POS if x['overall'] >= 4 else ABSTAIN

@labeling_function()
def lf_negative(x):
    return NEG if x['overall'] <= 2 else ABSTAIN

