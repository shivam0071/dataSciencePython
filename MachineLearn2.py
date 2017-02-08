#Twitter Sentiments app
import tweepy   #Twitter API ..

from textblob import TextBlob # For Sentiment Analysis

  # From Twitter Website
  

#first copy these keys from twitter website
#omitted here
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)   #this method takes 2 args

auth.set_access_token(access_token,access_token_secret)         #this method finally cretes auhenticaltion variable which can now be use to authenticate yourself and create an object o use twitter API

api = tweepy.API(auth)

public_tweets= api.search('Trump')  #other methods create delete find

for tweet in public_tweets:
    print(tweet.text)
    analysis=TextBlob(tweet.text)
    #print(analysis.sentences)
    print(analysis.sentiment) #polarity and subjectivity ..polarity tells negativity and positivity and Subjectivity tells
    #how much of and opinion it is vs how factual
    print('******************************************************************************')

