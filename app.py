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
        # Lowercase the text
        text = text.lower()
    
        # Remove punctuation and digits
        text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    
        # Remove the specific word "Reuters"
        text = text.replace("reuters", "")
        text = text.replace("Reuters", "")
    
    
        # Tokenize the text
        words = word_tokenize(text)
    
        # Remove stop words
        words = [word for word in words if word not in stop_words]
    
        # Stem or lemmatize the words
        words = [stemmer.stem(word) for word in words]
       
            # Join the words back into a string
        text = ' '.join(words)
    
        return text
    except:
        abort(404)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.errorhandler(404)
def page_not_found(error):
    error=["Hmm. I don't know what you did but it broke something. Did you try and input a carrot again?"]
    return render_template('errors.html', error=error), 404

@app.route('/predict', methods=['POST'])
def predict():
    try: 
        url = request.form['text']
        # download and parse article
        html = urlopen(url).read()
        soup = BeautifulSoup(html, features="html.parser")
    
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
    
        # Remove specific elements by tag name
        for tag in soup.find_all(["nav", "footer", "header", "aside"]):
            tag.extract()
    
        # Get text
        text = soup.get_text(separator='\n', strip=True)
    
        # Remove specific unwanted text patterns
        unwanted_texts = ["Do you want to enable cookies", "this video cannot be played","BBC News Image source","Getty Images"," EPA Image caption"]
        for unwanted_text in unwanted_texts:
            if unwanted_text in text:
                text = text.replace(unwanted_text, '')
    
        print(text)
        preprocessed_text = preprocess_text(text)
        X = vectorizer.transform([preprocessed_text])
        y_pred = clf.predict(X)
        if y_pred[0]== 1:
            result = 'Real'
        else:
            result = 'Fake'
        # Analyze the sentiment of the article using the VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        sentiment_dict = analyzer.polarity_scores(text)
        negative = sentiment_dict['neg']* 100
        neutral = sentiment_dict['neu']*100
        positive = sentiment_dict['pos']*100
    
        # decide sentiment as positive, negative and neutral
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
