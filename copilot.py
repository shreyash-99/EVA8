# create a function to count the number of times a word appears in a string
def count_words(text, word):    
    text = text.lower()
    word = word.lower()
    match_count = 0
    for match in re.finditer(word, text):
        match_count += 1
    return match_count

# create a function to remove punctuation from a string
def remove_punctuation(text):   
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

# create a function to tokenize a string
def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

# create a python function which can figure out the least value from the numpy array and return the standard deviation of the other values from this value
def std_deviation(values):
    least = np.min(values)
    return np.std(values[values != least])  


