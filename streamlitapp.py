def main():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import snscrape.modules.twitter as sntwitter
    import re
    import nltk
    import string
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import matplotlib
    import datetime
    matplotlib.use('Agg')
    import warnings
    warnings.filterwarnings("ignore")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(
        page_title="Twitter Sentiment Analysis",
        page_icon=":bar_chart:",
        layout="wide",

        )

    # Create a container to hold your content
    container = st.container()
    # Define theme colors
    primary_color = "#0077c2"
    secondary_color = "#b3d9ff"


    # Set theme style
    st.markdown(
        f"""
        <style>
            
            .reportview-container .main .block-container {{
            max-width: 1200px;
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }}
        .streamlit-button {{
            background-color: #0077c2;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            font-size: 18px;
            padding: 0.5rem 1rem;
            margin-right: 1rem;
        }}
        .streamlit-form button[type="submit"] {{
            font-size: 30px;
            font-weight: bold;
            background-color: #b3d9ff;
            color: black;
            border-radius: 10px;
            padding: 0.5rem 1.5rem;
        }}
        
            .reportview-container .main .block-container {{
                max-width: 1200px;
                padding-top: 1rem;
                padding-right: 1rem;
                padding-left: 1rem;
                padding-bottom: 1rem;
            }}
            .css-17eq0hr {{
                font-size: 18px;
                color: {{primary_color}};
                font-weight: bold;
            }}
            .css-1t42vg8 {{
                font-size: 16px;
                color: {{primary_color}};
                font-weight: bold;
            }}
            .button-container {{
    display: flex;
    justify-content: center;
}}

.button {{
    margin: 10px;
}}
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Add your content to the container

    with st.container():
        def get_tweets(topic, count,year1,year2):
            query = "(from:#{}) until:{} since:{}".format(topic,year2,year1)
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
        	<div style="display: flex; align-items: center; background-color: #f5f5f5; padding: 10px; margin-bottom:20px;">
  <div style="flex: 1;">
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQprocCFOXmsPhna-VHQLXcBaJQS_bt9tzXFg&usqp=CAU" style="width: 100%; max-width: 50px;">
  </div>
  <div style="flex: 2;">
    <h1 style="font-family: 'Arial Black', sans-serif; font-size: 30px; color: #4c4c4c; margin-top: 0;">Twitter Sentiment Visualization</h1>
  </div>
</div>

        	"""
        st.markdown(html_temp, unsafe_allow_html=True)
        st.empty()
        col1, col2 = st.columns(2)
        with col1:
            Topic = str(st.text_input("Enter the topic you are interested in"))
        with col2:
            Topi = int(st.number_input("Number of Tweets", min_value=1, max_value=10000, value=100, step=1))
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", datetime.date(2015, 1, 1))
        with col2:
            end_date = st.date_input("End date", datetime.date(2023, 1, 1))
        if start_date > end_date:
            st.error("Error: End date must be after start date.")
        if len(Topic) > 0:
            # Call the function to extract the data. pass the topic and filename you want the data to be stored in.
            with st.spinner("Please wait"):
                df = get_tweets(Topic, Topi,start_date,end_date)
            st.success('Tweets have been Extracted !!!!')
            df.drop(['Date', 'User'], inplace=True, axis=1)
            df = df.rename(columns={"Tweet": "tweets"})
            data = df
            data.drop_duplicates(inplace=True)

            def cleaningalldata(data):

                def Tweetcleaning(tweet):
                    clean_tweet = re.sub(r"@[a-zA-Z0-9]*", "", tweet)
                    clean_tweet = re.sub(r"#[a-zA-Z0-9\s]*", "", clean_tweet)
                    return clean_tweet

                def remove_nums(text_object):
                    clean_tweet = ' '.join(word for word in text_object.split() if word.isalpha())
                    return clean_tweet

                def clean_stop_words(tweet):
                    stop_words = set(stopwords.words('english'))
                    filtered_sentence = []
                    for w in tweet.split():
                        if w not in stop_words:
                            filtered_sentence.append(w)
                    clean_tweet = ' '.join([word for word in filtered_sentence if len(word) > 3])
                    return clean_tweet

                def tokenize_text(df_text):
                    wnl = WordNetLemmatizer()
                    tokenized = [wnl.lemmatize(word) for word in df_text.split()]
                    return tokenized

                def lemmatize_text(df_text):
                    wnl = WordNetLemmatizer()
                    lemmatized = [wnl.lemmatize(word) for word in df_text]
                    return lemmatized

                def to_string(text_object):
                    clean_tweet = ' '.join(word for word in text_object)
                    return clean_tweet

                data['cleanedTweets'] = data['tweets'].apply(Tweetcleaning)

                data['cleanedTweets'] = data['cleanedTweets'].apply(lambda x: x.lower())

                data['cleanedTweets'] = data['cleanedTweets'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

                data['cleanedTweets'] = data['cleanedTweets'].apply(remove_nums)

                data['cleanedTweets'] = data['cleanedTweets'].apply(clean_stop_words)

                data['cleanedTweets'] = data['cleanedTweets'].apply(tokenize_text)

                data['cleanedTweets'] = data['cleanedTweets'].apply(lemmatize_text)

                data['cleanedTweets'] = data['cleanedTweets'].apply(to_string)

                #Vader Sentiment Analysis

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

            col1, col2 = st.columns([600,400])
            col2_height= 500
            col1_height = 500
            with col1:
                if len(data[data["v_segmentation"] == "Positive"]) >= len(data[data["v_segmentation"] == "Negative"]):
                    st.write('<div style="color: green; font-size: 35px; font-family: Arial, sans-serif;">The Topic has Positive opinion in Public</div>', unsafe_allow_html=True)
                else:
                    st.write('<div style="color: red; font-size: 35px; font-family: Arial, sans-serif;">The Topic has Negative opinion in Public</div>', unsafe_allow_html=True)


                st.write("<span style='font-size: 30px;'>Total Tweets Extracted for Topic '{}' are : {}</span>".format(
                        Topic,len(data.tweets)),unsafe_allow_html=True)
                st.write("<span style='font-size: 25px;'>Total Positive Tweets are : {}</span>".format(
                    len(data[data["v_segmentation"] == "Positive"])), unsafe_allow_html=True)
                st.write("<span style='font-size: 25px;'>Total Negative Tweets are : {}</span>".format(
                    len(data[data["v_segmentation"] == "Negative"])), unsafe_allow_html=True)
                st.write("<span style='font-size: 25px;'>Total Neutral Tweets are : {}</span>".format(
                    len(data[data["v_segmentation"] == "Neutral"])), unsafe_allow_html=True)

            with col2:
                a = len(data[data["v_segmentation"] == "Positive"])
                b = len(data[data["v_segmentation"] == "Negative"])
                c = len(data[data["v_segmentation"] == "Neutral"])
                d = np.array([a, b, c])
                explode = (0.1, 0.0, 0.1)
                fig, ax = plt.subplots(figsize=(3,3))
                ax.pie(d, shadow=True, explode=explode, labels=["Positive", "Negative", "Neutral"],
                            autopct='%1.2f%%',textprops={'fontsize': 8})
                st.pyplot(fig)

            st.write("")
            st.write("Click on button to generate:")
            col1, col2, col3, col4 = st.columns(4)
            # See the Extracted Data :
            button_style = """
                                <style>
                                .stButton > button {
                                display: inline;
                                margin: 0 auto;
                                background-color: #008CBA;
                                padding: 10px 20px;
                                border-radius: 4px;
                                font-size: 16px;
                                width:30%;
                                
                                }
                                </style>
                                """
            st.markdown(button_style, unsafe_allow_html=True)
            if st.button("See the Extracted Data"):
                # st.markdown(html_temp, unsafe_allow_html=True)
                st.success("Below is the Extracted Data :")
                st.write(data.head(50), width=-1)


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
                # pyLDAvis.enable_notebook()
                tweets_bow = [text_dict.doc2bow(tweet) for tweet in data['cleanedTweets'][:1000]]
                lda_viz = gensimvis.prepare(tweets_lda, tweets_bow, dictionary=tweets_lda.id2word)
                html_string = pyLDAvis.prepared_data_to_html(lda_viz)
                from streamlit import components
                components.v1.html(html_string, width=1200, height=800)

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
            if st.button("Year wise sentiments"):
                def get_tweet(topic, count, year1, year2):
                    query = "(from:#{}) until:{}-01-01 since:{}-01-01".format(topic, year2, year1)
                    tweets = []
                    limit = count

                    for tweet in sntwitter.TwitterSearchScraper(query).get_items():

                        if len(tweets) == limit:
                            break
                        else:
                            tweets.append([tweet.date, tweet.user.username, tweet.content])
                    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
                    return df
                y1=str(start_date)[:4]
                y2=str(end_date)[:4]
                y1=int(y1)
                y2=int(y2)
                year,po,ne,neu=[],[],[],[]
                for i in range(y1,y2+1):
                    d = get_tweet(Topic,50,i,i+1)
                    d.drop(['Date', 'User'], inplace=True, axis=1)
                    d = d.rename(columns={"Tweet": "tweets"})
                    d.drop_duplicates(inplace=True)
                    d=cleaningalldata(d)
                    p=len(d[d["v_segmentation"] == "Positive"])
                    n=len(d[d["v_segmentation"] == "Negative"])
                    y=len(d[d["v_segmentation"] == "Neutral"])
                    year.append(i)
                    po.append(p)
                    ne.append(n)
                    neu.append(y)
                data = {
                    'Year': year,
                    'Positive': po,
                    'Negative': ne,
                    'Neutral': neu
                }
                df = pd.DataFrame(data)

                # Create a bar chart
                fig, ax = plt.subplots()
                fig.set_size_inches(6, 4)
                ax.bar(df['Year'] - 0.2, df['Positive'], width=0.2, color='blue', align='center')
                ax.bar(df['Year'], df['Negative'], width=0.2, color='tomato', align='center')
                ax.bar(df['Year'] + 0.2, df['Neutral'], width=0.2, color='grey', align='center')
                ax.set_xticks(df['Year'])
                ax.set_xticklabels(df['Year'])
                ax.set_xlabel('Year')
                ax.set_ylabel('Sentiment Count')
                ax.legend(['Positive', 'Negative', 'Neutral'])
                st.pyplot(fig)

            if st.button("Wordcloud"):
                # st.markdown(html_temp, unsafe_allow_html=True)
                wordcloud(data)

            if st.button("See Top 5 Topics"):
                # st.markdown(html_temp, unsafe_allow_html=True)
                topicmodeling(data)


if __name__ == '__main__':
    main()
