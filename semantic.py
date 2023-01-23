# Import spacy.
import spacy

# Load 'en_core_web_md' model.
nlp = spacy.load('en_core_web_md')

# Similarity example
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print("\n-----------Similarity cat, monkey, banana--------------")
print(f"{word1} {word2} = {word1.similarity(word2)}")
print(f"{word3} {word2} = {word3.similarity(word2)}")
print(f"{word3} {word1} = {word3.similarity(word1)}")
print("---------------------------------------------------------\n")

# Use nested for loops to undertake a comparison of the words.
tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

"""
NOTE 1:
What I noticed was that cat and monkey had a higher similarity than monkey and banana. Perhaps this is because both cat 
and monkey are animals.
It is also interesting to note that monkey and banana have a higher similarity than cat and banana. This is perhaps
because monkeys eat banana, hence monkey may be more associated with banana than cat with banana. 
"""

# my example: Similarity of apple, red and purple
word1 = nlp("apple")
word2 = nlp("red")
word3 = nlp("purple")
print("\n-----------Similarity apple, red, purple----------------")
print(f"{word1} {word2} = {word1.similarity(word2)}")
print(f"{word3} {word2} = {word3.similarity(word2)}")
print(f"{word3} {word1} = {word3.similarity(word1)}")
print("---------------------------------------------------------\n")

# Use nested for loops to undertake a comparison of the words.
tokens = nlp('carrot red purple apple ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Similarity between longer sentences.
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my cat on my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)

# Use for loop to display the similarity of the sentences.
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

"""
NOTE 2: Running example file with simpler language model 'en_core_web_sm' and advanced language model 'en_core_web_md'.
The main difference I saw between the sm and md model was that, 'en_core_web_md' generated a much higher similarity than
en_core_web_sm.
Also, whilst running the en_core_web_sm model, you get a warning sign that says small language model has no word vectors
loaded and similarity is based on tagger, parser and NER.
"""