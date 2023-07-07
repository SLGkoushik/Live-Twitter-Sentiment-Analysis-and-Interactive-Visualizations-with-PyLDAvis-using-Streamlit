# Twitter Sentiment Analysis
This application performs sentiment analysis on Twitter data related to a specific topic. It uses the streamlit library for the user interface and incorporates various natural language processing techniques for text preprocessing and sentiment classification.
### Link to website : https://slgkoushik-live-twitter-sentiment-analysis--streamlitapp-rmyrce.streamlit.app/
# Setup
To run the application, you need to install the required dependencies. You can do this by running the following command:
<pre>
    <code>
       pip install streamlit numpy pandas matplotlib seaborn snscrape nltk vaderSentiment
    </code>
</pre>
You also need to download additional resources for the NLTK library. Open a Python terminal and run the following commands:
<pre>
    <code>
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    </code>
</pre>
# Usage
To start the application, run the following command:
<pre>
    <code>
        streamlit run your_script_name.py
    </code>
</pre>
Replace your_script_name.py with the filename of the Python script containing the code provided.
# Application Features
### The application provides the following features:

- **Input**: Enter the topic you are interested in and specify the number of tweets to extract.
- **Date Range**: Select the start and end dates to limit the tweet extraction.
- **Sentiment Analysis**: The application performs sentiment analysis on the extracted tweets and displays the sentiment distribution as well as the total counts of positive, negative, and neutral tweets.
- **Pie Chart**: A pie chart is generated to visualize the sentiment distribution.
- **Extracted Data**: Click the button to see the extracted tweet data in a table.
- **Year Wise Sentiments**: Click the button to view the sentiment distribution for each year within the specified date range.
- **Wordcloud**: Click the button to generate a word cloud based on the most common words in the extracted tweets.
- **Top 5 Topics**: Click the button to perform topic modeling on the cleaned tweets and visualize the top 5 topics.
# License
This project is licensed under the MIT License. Feel free to use and modify the code according to your needs.






