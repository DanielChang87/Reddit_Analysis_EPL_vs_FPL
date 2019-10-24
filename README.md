# Reddit_Analysis_EPL_vs_FPL
An Analysis of the EPL and FPL Subreddits

<h><b>INTRODUCTION</b></h>
<b>What is the Premier League?</b>

The Premier League (often referred to as the English Premier League or the EPL outside England) is the top level of the English football league system. In 2019, as Manchester City and Liverpool contested a thrilling title race, a cumulative global audience of 3.2 billion for all programming watched the action, <a href="https://www.premierleague.com/news/1280062">an increase of six per cent on the previous season.</a> This rise in viewership numbers, combined with an increasingly engaged fanbase, has translated into a greater interest in Premier League related games and interactive content.

<b>What is the Fantasy Premier League?</b>

Fantasy football is a game in which participants assemble an imaginary team of real life footballers and score points based on those players' actual statistical performance or their perceived contribution on the field of play. The way the game works is simple: you pick eleven players, and whenever these players perform well in the live matches, they get points. A goal for a striker, for example, is worth 3 points, and a clean sheet (no goals conceded) is worth 6 points.

The Fantasy Premier League (abbreviated to FPL), in particular, is the world's largest fantasy league with over 6 million players. With attractive prizes (top FPL players in each region earn a ticket to watch their favourite football team, while the overall first place wins a cash prize), the FPL fanbase are often more engaged and fanatical about football.

<b>Reddit and the Premier League</b>

Reddit is a social news aggregation, web content rating, and discussion website. Posts are organized by subject into user-created boards called "subreddits", which cover a variety of topics including news, science, movies, and of course, football.

/r/PremierLeague/ is one such subreddit, where fans of the English Premier League aggregate and discuss the game. The subreddit is 8 years old and has 108,0000 members as of 2019.

/r/FantasyPL/ is a subreddit dedicated to the fantasy version of the EPL. The subreddit is also 8 years old, and surprisingly has more members than /r/PremierLeague, at 177,000.

<b>Preamble:</b>

<u>This project will be addressed from the perspective from the marketing team of the Fantasy Premier League (FantasyPL) App.</u>

The goals of this project will be outlined in the following section.


<h><b>PROBLEM STATEMENT</b></h>
As managers of the FPL app, we want to evaluate: 
<li>What kind of people surf the FPL reddit, and what kind of people surf the EPL reddit? Are there overlaps (same username) who surf both reddits, and what's the extent of this overlap?
<li> Overall sentiment for EPL and FPL reddits, is one more positive than the other?
<li> What type of content are posted in the EPL and FPL reddits respectively? What features differentiates an EPL post from an FPL post?
<br>
<br>
To frame the problem from a data science perspective, this is both an inference problem and a prediction problem:
<li> First, we need to create a classification model with sufficient predictive power that differentiates between the two reddits. We will apply a few different models here to evaluate differences in features selected and predictive power. Intuitively, the higher the predictive power, the better the model is at distinguishing between the two subreddits, and therefore the features it selects are better indicators of difference.
<li> Second, we need to find what features are characteristic of each reddit.
<li> Third, are there differences in what teams each subreddit discuss? E.g., is Manchester United popular in EPL, but unpopular in FPL? Are there differences in the type of comments made in each subreddit?</li>
    
Using this information, we should gather information that supports a targeted marketing campaign - i.e. how to appeal to the broader spectrum of people who watch EPL but don't play FPL. <b>If I want to create a targeted marketing campaign on r/EPL, how should I go about doing it?</b>


<b>Data Dictionary</b>

There are over 100 columns in the data dump the Reddit API provides.
The dictionary below covers the relevant fields extracted from the API, and the engineered fields which will be covered in later sections.

|Feature|Type|Description|Analysis|
|---|---|---|---|
|subreddit_name_prefixed|string/object|Name of the subreddit| Used as target for classification|
|author|string/object|User name of who made the main post| |
|title|string/object|Title of the post| To be included in bag of words|
|selftext|string/object|A self post is a text post, instead of a link post. A link post directs to an external link. A self post is nothing but the text you enter.| To be included in bag of words|
|domain|string/object|For link posts, domain captures the website the link belongs to. For self posts, the domain captures the subreddit the post was made in. |Used to differentiate between link post and self post. Self posts have a domain beginning with 'self'|
|link_flair_text|string/object|What category the post falls under - differs from subreddit to subreddit|Can run a barplot to get a sense of categories in the subreddit|
|created|float|When the post was made|Will be mapped to return date only; remove time data|
|media|string/object|For link posts, the type of link attached (e.g. Twitter, Facebook, news site)|Can extract tweet content from this field using a regex|
|media_embed|string/object|Description of link post| |
|url|string/object|External link, for link posts|Run through OCR if image|
|permalink|string/object|Permanent link to post|Used to loop through post for comments in crawler|
|num_comments|int64|Number of comments made on the post||
|score|int64|Number of upvotes - number of downvotes||
|ups|int64|Number of upvotes| |
|comments|string/object|List of up to 10 comments extracted from the comments made in the post| Engineered field - To be included in bag of words|

<b> Selection of Models </b>

In this notebook we will be applying four different models to our dataset, namely:
<li> Naive Bayes (multinomial) approach - this will be our baseline model
<li> Logistic Regression approach
<li> Linear SVM approach </li>
<li> Decision Tree approach </li>

As explained in the problem statement, the reason for running multiple models is to evaluate differences in features selected and predictive power. Furthermore, this lends itself to a bagging ensemble approach should we choose to use the model for predictive purposes rather than inferential purposes in the future.

As for why these four models are chosen, all four support some form of feature importance method (either via .coefs_ or .get_feature_importances_), so we can evaluate the strength of features (words). Other classification models like KNearest Neighbors or Random Forests are not conducive to feature importance evaluations, and as such, will not be included in this project.

<br>
<br>
<b> Datasets </b>

Each model will be run on 4 different subsets of our master data:
<li> Title only
<li> Selftext only    
<li> Comments only
<li> Combined (Title + Selftext + Comments) </li>

The purpose of running multiple subsets is to investigate how much 'text' from the post is needed before the model can make a good prediction (accuracy > 80%). Can we make a good model using only the title text? How about only the self text? At what point does adding more content/text to the model have diminishing returns? Are there differences in the type of comments made in both subreddits?
<br>
<br>
<b> Interpreting text data </b>

Two vectorizers will also be tested: A count vectorizer vs a tfidf vectorizer.

CountVectorizer just counts the word frequencies, while the TFIDFVectorizer the value increases proportionally to count, but is offset by the frequency of the word in the corpus (the inverse document frequency). This helps to adjust for the fact that some words appear more frequently. The purpose of testing both is to see the effect on the top features selected - does the countvectorizer affect the top n predictive features?

<b> Metrics for Evaluating Model </b>

We will be using accuracy (correct predictions / total predictions) as the main metric for scoring our models, in combination with the AUC score (does the model do a good job of distinguishing the positive and the negative values?). The better the model is at making correct predictions (high accuracy), the better the selected features are at explaining the difference between the two.
<b>Conclusion</b>
From our look into the defining features of the FPL and EPL subreddit, we make the following findings:

<br>
<li> <b> FPL is interested in players; EPL is interested in teams </b> : FPL is a game that focuses on the performance of individual players, so it is perhaps not surprising to find that player names are mentioned far more often in the FPL subreddit. EPL redditors, on the other hand, are interested in teams - they are excited when a team does well or poorly, and place a lower impetus on individual player performance.</li>
<br> 
<li> <b> FPL is full of FPL jargon </b>: FPL players have their own language! They use technical terms like XG (expected goals), XA (expected assists), 'clean sheet', 'blank', 'captaining'..</li>
<br>
<li> <b> Most popular teams </b> : The most popular teams on r/EPL, are, unsurprisingly, the big 6 teams. Conversely, the less popular teams are frequently discussed on r/FPL, since these are generally players with lower prices, greater value, and providing some return on points. An interesting strategy might be advertising to the fans of the less popular teams like Norwich and Bournemouth, since they might be surprised to find that their players are doing well on the FPl platform.
