# Music Genre Classification Using Song Lyrics and kNN
Joseph Wright  
7/2/2020  

## Abstract
This project uses a custom kNN algorithm in an attempt to classify the genre of a song based on the text of the lyrics. Given a text file of song lyrics, we compare attributes of this song to a set of example songs from popular music genres. The main idea is that since music genres share motifs in their lyrical content, that we could therefore classify the genre of a song by comparing the words in their lyrics between the example and other classified songs. We show a technique to isolate keywords in a song using word stemming and how to measure the 'closeness' of two songs using word vectors, among other metrics.

## Training Data
The number of genres to use, the specific genre, and the genre examples were all arbitrarily selected from song lyric websites like genius.com or lyrics.com that already classified the genre. The sample size is very small, with n=10 per genre. Each sub-folder name in this directory is the 'label' attached to each song. Each song is stored in a text file with its track title and artist name. Summary statistics are calculated about each genre using the files in the genre folders.

## Algorithm
This kNN implementation utilizes the Euclidean distance between a song and points in n-dimensional space with a word count penalty. 
By providing a genre label and example lyrics, we build a training lexicon by extracting the individual words that make up all of the training songs. Furthermore, we capture a glimpse of a genres lyrical style by counting the word occurrences in each song in our lexicon. In this process, we've represented our training songs and their genres as points in a vector space with the number of training songs we've provided as dimensions. Songs that use similar vocabularies will therefore be plotted closer to each other in the n-dimensional space than songs that use different words.

We repeat the process of tokenizing the words of a song we want to test and counting the frequency of the words used in it. In the field of computing, function words ("stop words") are sometimes
ignored because they express grammatical relationships and do not tell you much about the lexical meaning of sentences. To improve the performance of the algorithm, we've stripped function words
from the lexicon. Not only does this improve the processing speed, but by removing function words we are now only measuring words that contribute more to what a song is about. This helps us make better conclusions about the relationships between two songs considering only their lyrical text. Theoretically, a song could contain only function words, or no words, and still have neighbors in the vector space. This is practically true, music genres don't need lyrics, genres are defined by their sound. For our purposes, it is sufficient to ignore them because we've 'pre-qualified' our training data.

Another interesting topic in the computing field is the idea of word-stemming. Word-stemming works by reducing the grammatical inflections of a word down to its stem. For instance, 'connected',
'connecting', 'connection', all derive from the root 'connect'. This is a useful technique in many information processing applications, but in this instance, we have applied word-stemming because
it allows us to reduce the amount of words in our overall lexicon, effectively allowing us to find more word matches between songs, improve the processing speed, and measure the relationships between songs more accurately. There are many word-stemming algorithms. We've chosen to use the 'Porter Stemmer', which is built into the nltk Python package. The original Porter Stemmer algorithm was written
by Martin Porter in 1980. It works by removing defined lists of suffixes from words while following certain conditions. We chose this algorithm because it's simple and effective compared to others.

With pre-processing done, we plot the stemmed, stop-word free version of our test songs into the vector space and perform the kNN calculation. We tested several k values less than 10 and chose 5 for simplicity. Our sample sizes are so small that the k value did not appear to really increase the algorithms accuracy. We calculate the distance between two songs by taking the Euclidean distance between
the classified point containing the entire training lexicon and the test song word frequencies. We created a distance penalty for songs by subtracting the genres average word count from the test songs word count and squaring it. This will push songs away from each other if there is a significant miss match in song length. This results in a list of distances between each classified point and the test song and the classified points genre label. The kNN 'vote' is now performed by sorting this list by the distance results and filtering the list to k records. The majority vote is calculated by tallying the resulting labels in the list. The label of the majority vote winner is placed as the genre of the test song.

## Performance
With much room for improvement (and ignoring statistical bias or musical opinions), the results were okay, correctly classifying 75% of the testing songs. It was surprising to me how poorly the algorithm worked before adding a control for the length of songs. With a very small sample size per training label, there wasn't enough information to capture the essence of a genre based solely off of the word vectors (true for stemmed and unstemmed tests). The length of a song acted as a good supporting indicator of which style of music a song is. Interestingly, it was not good alone, I found equally poor results focusing only on song length. The Euclidean distance and word length penalty supported each other in the calculation.

We had to make several attempts find an appropriate way to weigh these factors in the calculation. One problem was that genres with very long or very short word counts would scale the distance too far, or not far enough, when multiplied by the word distance. For example, the hip-hop training songs have significantly more words than any other genre, so song length ratios including hip-hop songs were heavily skewed by this. Simply adding the squared difference in word length was effective at tuning the calculation.

## Improvements
While entertaining, it is clear that this is an overly simplistic approach to music genre classification. This is a non-trivial topic. There are problems with how I collected the data 
and with how I selected songs to classify. Essentially, I inserted my music opinions into the solution by selecting songs I felt represented genres and picking which songs I thought would be classifiable. A scalable, controlled solution to a problem of this nature would require statistical guidance to increase the accuracy of the classification and control the bias in data collection and testing.

Simply adding more training data would also help us make more accurate classifications. There are public data sets or scraping techniques that could be used to gather
a large quantity of data.  Companies like Spotify have robust data on their songs, even going into specifics about characteristics of the songs actual sound. We could use more descriptive information besides the lyrics of the song to classify it. This is all publicly available via their API. Certainly, there are also other public databases that contain large sums of song lyrics.

We ignored an extensive opportunity to see if statistical methods would make a better prediction than kNN alone. It would be interesting to implement a Naive Bayes classification against this kNN implementation. We could have also tested other calculations than the Euclidean distance, such as the cosine similarity, or used statistics to predict the nearest neighbors.

There is so much opportunity to use other language processing tools to make more meaningful vector spaces. For instance, we could compare pairs of words used in songs rather individual words. 
Is there a way we could detect rhyming in hip-hop, for example? We could take a different approach to our vector space and try a completely different model. Namely, GloVe, Word2vec, or fastText
are all very interesting and could be used to create a mapping of the relationships between song lyrics.

## References
[1] "Dropping common terms: stop words." https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html  
[2] Ahmed Bou-Rabee, Keegan Go, Karanveer Mohan. 
    "Classifying the Subjective: Determining Genre of Music from Lyrics". 
    http://cs229.stanford.edu/proj2012/BourabeeGoMohan-ClassifyingTheSubjectiveDeterminingGenreOfMusicFromLyrics.pdf  
[3] "Porter stemmer". http://people.scs.carleton.ca/~armyunis/projects/KAPI/porter.pdf  
[4] "Stemming". https://en.wikipedia.org/wiki/Stemming  
[5] Nathan Thomas. "Using k-nearest neighbors to predict the genre of Spotify tracks". 
    https://towardsdatascience.com/using-k-nearest-neighbours-to-predict-the-genre-of-spotify-tracks-796bbbad619f  
[6] Thushan Ganegedata. "Intuitive Guide to Understanding GloVe Embeddings" 
    https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010  
[7] Sunil Ray. "6 Easy Steps to Learn Naive Bayes Algorithm with codes in Python and R". 
    https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/  
[8] "GloVe (machine learning)." https://en.wikipedia.org/wiki/GloVe_(machine_learning)
