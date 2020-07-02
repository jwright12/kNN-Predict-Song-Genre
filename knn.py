import copy
import math
import regex
import string
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from operator import itemgetter 
from os import listdir
from os.path import isfile, join

k               = 5
country_folder     = './Country/'
hip_hop_folder     = './Hip_hop/'
pop_folder     = './Pop/'
rock_folder     = './Rock/'
test_folder     = './testing/'

country_avg_len     = 0
hip_hop_avg_len     = 0
pop_avg_len     =     0
rock_avg_len     =    0

def remove_punctuation_stop_words_and_stem(text):
    """ Remove punctuation from all text, remove stop-words, reduce inflected words to their stem """

    text = regex.sub(r'\p{P}+', "", text)
    text = text.lower()

    stop_words_set = set(stopwords.words('english'))
    stop_words = []

    for s in stop_words_set: 
        s = regex.sub(r'[^\w\s]', '',s)
        stop_words.append(s)

    song_words = text.split() 
    song_no_stop_words = [word for word in song_words if word not in stop_words]

    porter = PorterStemmer() # PorterStemmer avoids over-stemming words relative to other stemming algorithms
    stemmed_word_list = [porter.stem(word) for word in song_no_stop_words]
    text= ' '.join(stemmed_word_list)

    return text
    
def file_list(folder):
    return [f for f in listdir(folder) if isfile(join(folder, f))]

def count_words(file):
    f = open(file, 'r')
    read = f.read()
    return len(read.split())

def all_file_list():
    """ Produces list of files for reading in data, it's convenient to calculate genre length averages here before pre-processing """

    global country_avg_len
    global hip_hop_avg_len
    global pop_avg_len
    global rock_avg_len

    country_files = file_list(country_folder)
    country_word_count = 0
    for i in range(len(country_files)):
        country_word_count += count_words(country_folder+country_files[i])
        country_files[i] = country_folder + country_files[i]

    country_avg_len = country_word_count / len(country_files)

    hip_hop_files = file_list(hip_hop_folder)
    hip_hop_word_count = 0
    for i in range(len(hip_hop_files)):
        hip_hop_word_count += count_words(hip_hop_folder+hip_hop_files[i])
        hip_hop_files[i] = hip_hop_folder + hip_hop_files[i]

    hip_hop_avg_len = hip_hop_word_count / len(hip_hop_files)

    pop_files = file_list(pop_folder)
    pop_word_count = 0
    for i in range(len(pop_files)):
        pop_word_count += count_words(pop_folder+pop_files[i])
        pop_files[i] = pop_folder + pop_files[i]

    pop_avg_len = pop_word_count / len(pop_files)

    rock_files = file_list(rock_folder)
    rock_word_count = 0
    for i in range(len(rock_files)):
        rock_word_count += count_words(rock_folder+rock_files[i])
        rock_files[i] = rock_folder + rock_files[i]

    rock_avg_len = rock_word_count / len(rock_files)

    return country_files + hip_hop_files + pop_files + rock_files

def get_genre_avg_length(tag):
    switch = {
        'Country': country_avg_len,
        'Hip_hop': hip_hop_avg_len,
        'Pop':     pop_avg_len,
        'Rock':    rock_avg_len
    }
    return switch[tag]

def file_to_stemmed_word_list(f):
    fr        = open(f, 'r')
    text_read = fr.read() 
    text      = remove_punctuation_stop_words_and_stem(text_read)
    
    return text.split()

def get_vocabularies(all_files):
    voc = {}
    for f in all_files:
        words = file_to_stemmed_word_list(f)
        for w in words:
            voc[w] = 0

    return voc

def load_training_data():
    all_files = all_file_list()
    voc = get_vocabularies(all_files) 
     
    training_data = []

    for f in all_files:
        tag   = f.split('/')[1]
        point = copy.deepcopy(voc) 
        words = file_to_stemmed_word_list(f) 
        genre_average = get_genre_avg_length(tag)
 
        for w in words:
            point[w] += 1 

        d = {'tag':tag, 'point':point, 'genre_avg':genre_average}  
        training_data.append(d)

    return training_data 

def get_distance(p1, p2):
    sq_sum = 0

    for w in p1:
        if w in p2:
            sq_sum += (p1[w]-p2[w])*(p1[w]-p2[w])
        else:
            sq_sum += p1[w]*p1[w]

    return math.sqrt(sq_sum)

def test(training_data, txt_file):
    dist_list = []
    txt       = {}
    item      = {}
    max_i     = 0

    words = file_to_stemmed_word_list(txt_file) 
    unstemmed_test_song_word_count = count_words(txt_file)
   
    for w in words:
        if w in txt:
            txt[w] += 1
        else:
            txt[w]  = 1

    for pt in training_data:
        item['tag'] = pt['tag']
        item['genre_avg'] = pt['genre_avg']
        item['distance'] = get_distance(pt['point'], txt) + (unstemmed_test_song_word_count - item['genre_avg'])**2 # Penalizes songs that are too long or short for the genre
        
        dist_list.append(copy.deepcopy(item))

    dist_list = sorted(dist_list, key=itemgetter('distance'))

    neighbors = dist_list[:k]
    
    vote_result = {}
    for d in dist_list:
        if d['tag'] in vote_result:
            vote_result[d['tag']] += 1
        else:
            vote_result[d['tag']]  = 1
   
    result = dist_list[0]['tag']
    for vote in vote_result: # Calculates the majority vote
        if vote_result[vote] > vote_result[result]:
            result = vote

    return result
        
def main():
    td = load_training_data()
    test_files = file_list(test_folder)

    for file in test_files:
        print(file)
        print('    Category: ' + test(td, test_folder+file))

if __name__ == '__main__':
    main()