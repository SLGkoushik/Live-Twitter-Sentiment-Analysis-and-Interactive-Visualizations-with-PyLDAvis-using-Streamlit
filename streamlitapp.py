import streamlit as st

def main():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import snscrape.modules.twitter as sntwitter
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    import matplotlib
    matplotlib.use('Agg')
    import warnings
    warnings.filterwarnings("ignore")


    def get_tweets(topic, count):
        query = "(from:{}) until:2022-12-03 since:2015-08-03".format(topic)
        tweets = []
        limit = count

        for tweet in sntwitter.TwitterSearchScraper(query).get_items():

            if len(tweets) == limit:
                break
            else:
                tweets.append([tweet.date, tweet.user.username, tweet.content])
        df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
        return df

    html_temp = """
    	<div style="background-color:tomato;"><p style="color:white;font-size:40px;padding:9px">Live twitter Sentiment analysis</p></div>
    	"""
    st.markdown(html_temp, unsafe_allow_html=True)
    Topic = str()
    Topic = str(st.text_input("Enter the topic you are interested in"))
    Topi = int()
    Topi = int(st.number_input("Number of Tweets ", min_value=1, max_value=1000, value=50, step=1))
    if len(Topic) > 0:
        # Call the function to extract the data. pass the topic and filename you want the data to be stored in.
        with st.spinner("Please wait"):
            Topic = "#" + Topic
            df = get_tweets(Topic, Topi)
        st.success('Tweets have been Extracted !!!!')
        df.drop(['Date', 'User'], inplace=True, axis=1)
        df = df.rename(columns={"Tweet": "tweets"})
        data = df
        data.drop_duplicates(inplace=True)
        print(data)

        def cleaningalldata(data):
            import re
            import nltk
            import string
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')

            from nltk.corpus import stopwords

            stop_words = stopwords.words('english')

            def Tweetcleaning(tweet):
                clean_tweet = re.sub(r"@[a-zA-Z0-9]*", "", tweet)
                clean_tweet = re.sub(r"#[a-zA-Z0-9\s]*", "", clean_tweet)
                clean_tweet = ' '.join(word for word in clean_tweet.split() if word not in stop_words)
                return clean_tweet

            data['cleanedTweets'] = data['tweets'].apply(Tweetcleaning)

            data['cleanedTweets'] = data['cleanedTweets'].apply(lambda x: x.lower())

            data['cleanedTweets'] = data['cleanedTweets'].apply(
                lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

            def remove_nums(text_object):
                clean_tweet = ' '.join(word for word in text_object.split() if word.isalpha())
                return clean_tweet

            data['cleanedTweets'] = data['cleanedTweets'].apply(remove_nums)

            data.head(3)

            from nltk import word_tokenize
            from nltk.stem import WordNetLemmatizer
            def tokenize_text(df_text):
                wnl = WordNetLemmatizer()
                tokenized = [wnl.lemmatize(word) for word in df_text.split()]
                return tokenized

            data['cleanedTweets'] = data['cleanedTweets'].apply(tokenize_text)
            data.head(3)

            from nltk.stem import WordNetLemmatizer
            def lemmatize_text(df_text):
                wnl = WordNetLemmatizer()
                lemmatized = [wnl.lemmatize(word) for word in df_text]
                return lemmatized

            data['cleanedTweets'] = data['cleanedTweets'].apply(lemmatize_text)
            data.head()

            def to_string(text_object):
                clean_tweet = ' '.join(word for word in text_object)
                return clean_tweet

            data['cleanedTweets'] = data['cleanedTweets'].apply(to_string)

            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            data['compound'] = [analyzer.polarity_scores(v)['compound'] for v in data['cleanedTweets']]
            data['neg'] = [analyzer.polarity_scores(v)['neg'] for v in data['cleanedTweets']]
            data['neu'] = [analyzer.polarity_scores(v)['neu'] for v in data['cleanedTweets']]
            data['pos'] = [analyzer.polarity_scores(v)['pos'] for v in data['cleanedTweets']]

            def format_VADER_output(compound):

                polarity = "Neutral"

                if (compound >= 0.05):
                    polarity = "Positive"

                elif (compound <= -0.05):
                    polarity = "Negative"

                return polarity

            data['v_segmentation'] = data['compound'].apply(format_VADER_output)
            return data

        data = cleaningalldata(data)
        st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic, len(data.tweets)))
        st.write("Total Positive Tweets are : {}".format(len(data[data["v_segmentation"] == "Positive"])))
        st.write("Total Negative Tweets are : {}".format(len(data[data["v_segmentation"] == "Negative"])))
        st.write("Total Neutral Tweets are : {}".format(len(data[data["v_segmentation"] == "Neutral"])))
        if len(data[data["v_segmentation"] == "Positive"]) >= len(data[data["v_segmentation"] == "Negative"]):
            st.write("""The Topic has Positive opinion in Public""")
        else:
            st.write("""The Topic has Negative opinion in Public""")

        # See the Extracted Data :
        if st.button("See the Extracted Data"):
            # st.markdown(html_temp, unsafe_allow_html=True)
            st.success("Below is the Extracted Data :")
            st.write(data.head(50))

        # Piechart
        if st.button("Get Pie Chart for Different Sentiments"):
            st.success("Generating A Pie Chart")
            a = len(data[data["v_segmentation"] == "Positive"])
            b = len(data[data["v_segmentation"] == "Negative"])
            c = len(data[data["v_segmentation"] == "Neutral"])
            d = np.array([a, b, c])
            explode = (0.1, 0.0, 0.1)
            st.write(
                plt.pie(d, shadow=True, explode=explode, labels=["Positive", "Negative", "Neutral"], autopct='%1.2f%%'))
            st.pyplot()

        def topicmodeling(data):
            all_words = [word for tweet in data['cleanedTweets'] for word in tweet]
            tweet_lengths = [len(tokens) for tokens in data['cleanedTweets']]
            vocab = sorted(list(set(all_words)))
            print('{} words total, with a vocabulary size of {}'.format(len(all_words), len(vocab)))
            print('Max tweet length is {}'.format(max(tweet_lengths)))

            from nltk import word_tokenize
            from nltk.stem import WordNetLemmatizer
            def tokenize_text(df_text):
                wnl = WordNetLemmatizer()
                tokenized = [wnl.lemmatize(word) for word in df_text.split()]
                return tokenized

            data['cleanedTweets'] = data['cleanedTweets'].apply(tokenize_text)

            def lemmatize_text(df_text):
                wnl = WordNetLemmatizer()
                lemmatized = [wnl.lemmatize(word) for word in df_text]
                return lemmatized

            data['cleanedTweets'] = data['cleanedTweets'].apply(lemmatize_text)
            from nltk.probability import FreqDist

            # iterate through each tweet, then each token in each tweet, and store in one list
            flat_words = [item for sublist in data['cleanedTweets'] for item in sublist]

            word_freq = FreqDist(flat_words)

            most_common_count = [x[1] for x in word_freq.most_common(30)]

            most_common_word = [x[0] for x in word_freq.most_common(30)]

            top_30_dictionary = dict(zip(most_common_word, most_common_count))
            from gensim.corpora import Dictionary

            text_dict = Dictionary(data.cleanedTweets)

            tweets_bow = [text_dict.doc2bow(tweet) for tweet in data['cleanedTweets']]
            from gensim.models.ldamodel import LdaModel
            k = 5
            tweets_lda = LdaModel(tweets_bow,
                                  num_topics=k,
                                  id2word=text_dict,
                                  random_state=1,
                                  passes=10)

            import pyLDAvis
            import pyLDAvis.gensim_models as gensimvis
            #pyLDAvis.enable_notebook()
            tweets_bow = [text_dict.doc2bow(tweet) for tweet in data['cleanedTweets'][:1000]]
            lda_viz = gensimvis.prepare(tweets_lda, tweets_bow, dictionary=tweets_lda.id2word)
            html_string = pyLDAvis.prepared_data_to_html(lda_viz)
            from streamlit import components
            components.v1.html(html_string, width=1300, height=800)

        def wordcloud(data):
            all_words = [word for tweet in data['cleanedTweets'] for word in tweet]
            tweet_lengths = [len(tokens) for tokens in data['cleanedTweets']]
            vocab = sorted(list(set(all_words)))
            print('{} words total, with a vocabulary size of {}'.format(len(all_words), len(vocab)))
            print('Max tweet length is {}'.format(max(tweet_lengths)))

            from nltk import word_tokenize
            from nltk.stem import WordNetLemmatizer
            def tokenize_text(df_text):
                wnl = WordNetLemmatizer()
                tokenized = [wnl.lemmatize(word) for word in df_text.split()]
                return tokenized

            data['cleanedTweets'] = data['cleanedTweets'].apply(tokenize_text)

            def lemmatize_text(df_text):
                wnl = WordNetLemmatizer()
                lemmatized = [wnl.lemmatize(word) for word in df_text]
                return lemmatized

            data['cleanedTweets'] = data['cleanedTweets'].apply(lemmatize_text)
            from nltk.probability import FreqDist

            # iterate through each tweet, then each token in each tweet, and store in one list
            flat_words = [item for sublist in data['cleanedTweets'] for item in sublist]

            word_freq = FreqDist(flat_words)

            most_common_count = [x[1] for x in word_freq.most_common(30)]

            most_common_word = [x[0] for x in word_freq.most_common(30)]

            top_30_dictionary = dict(zip(most_common_word, most_common_count))
            import warnings

            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from wordcloud import WordCloud

            wordcloud = WordCloud(colormap='Accent', background_color='black').generate_from_frequencies(
                top_30_dictionary)

            plt.figure(figsize=(10, 8))
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            plt.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot()
        if st.button("Wordcloud"):
            # st.markdown(html_temp, unsafe_allow_html=True)
            wordcloud(data)
        if st.button("Frequently repeated topics dashboard"):
            # st.markdown(html_temp, unsafe_allow_html=True)
            topicmodeling(data)

if __name__ == '__main__':
    main()
