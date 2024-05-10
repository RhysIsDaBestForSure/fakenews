from flask import Flask, request, render_template
from joblib import load
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from urllib.request import urlopen
from bs4 import BeautifulSoup
import nltk
from flask import abort
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

clf = load('model.joblib')
vectorizer = load('vectorizer.joblib')

def preprocess_text(text):
    try:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
        text = text.replace("reuters", "").replace("Reuters", "")
        words = word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        words = [stemmer.stem(word) for word in words]
        text = ' '.join(words)
        return text
    except:
        abort(404)

app = Flask(__name__)

@app.route('/')
def home():
    user_query = ''  # Initialize user_query here
    if request.method == 'POST':
        user_query = request.form['query']
        
        top_headlines = newsapi.get_top_headlines(q=user_query, language='en', country='us')
        top_headlines = top_headlines['articles'][:10]

        all_articles = newsapi.get_everything(q=user_query, language='en', sort_by='relevancy')
        all_articles = all_articles['articles'][:10]
    return render_template('home.html', user_query=user_query, top_headlines=top_headlines, all_articles=all_articles)

@app.errorhandler(404)
def page_not_found(error):
    error="Hmm. I don't know what you did but it broke something. Did you try and input a carrot again?"
    return render_template('errors.html', error=error), 404

@app.route('/predict', methods=['POST'])
def predict():
    try: 
        url = request.form['text']
        html = urlopen(url).read()
        soup = BeautifulSoup(html, features="html.parser")
    
        for script in soup(["script", "style"]):
            script.extract()
    
        for tag in soup.find_all(["nav", "footer", "header", "aside"]):
            tag.extract()
    
        text = soup.get_text(separator='\n', strip=True)
    
        unwanted_texts = ["Do you want to enable cookies", "this video cannot be played","BBC News Image source","Getty Images"," EPA Image caption"]
        for unwanted_text in unwanted_texts:
            if unwanted_text in text:
                text = text.replace(unwanted_text, '')
    
        preprocessed_text = preprocess_text(text)
        X = vectorizer.transform([preprocessed_text])
        y_pred = clf.predict(X)
        if y_pred[0]== 1:
            result = 'Real'
        else:
            result = 'Fake'
        analyzer = SentimentIntensityAnalyzer()
        sentiment_dict = analyzer.polarity_scores(text)
        negative = sentiment_dict['neg']* 100
        neutral = sentiment_dict['neu']*100
        positive = sentiment_dict['pos']*100
    
        if sentiment_dict['compound'] >= 0.05 :
            compound = "Positive"
        elif sentiment_dict['compound'] <= - 0.05 :
            compound = "Negative"
        else :
            compound = "Neutral"
        return render_template('result.html', result=result, text=text, positive=positive, negative=negative,neutral=neutral,compound=compound)
    except:
        abort(404)

if __name__ == '__main__':
    app.run()
