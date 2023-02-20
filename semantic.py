import spacy
nlp = spacy.load('en_core_web_md')

"""Calculates and prints the similarity scores between tokens"""
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


"""Calculates and prints the similarity scores between a model sentence and a list of other sentences"""
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

"""Calculates and prints the similarity scores between tokens in a second object"""
tokens2 = nlp('pen desk book school')
for token1 in tokens2:
    for token2 in tokens2:
        print(token1.text, token2.text, token1.similarity(token2))

# Write a note about what you found interesting about the similarities
# between cat, monkey and banana and think of an example of your own.
'''I found it interesting that "cat", "monkey", and "banana" have higher similarity scores with each other than with "apple". 
This suggests that they might be related or used together in certain contexts, like in a jungle or zoo. 
For example, we often think of cats and monkeys as animals found in the jungle and bananas as the food monkeys eat.'''

# Run the example file with the simpler language model ‘en_core_web_sm’
# and write a note on what you notice is different from the model 'en_core_web_md'.
'''The key difference between the two programs is that the first program generates similarity scores without any warnings, 
while the second program generates similarity scores with user warnings that the model does not have word vectors loaded, 
and the results may not be very useful. This suggests that the en_core_web_md model might be a better choice 
for generating useful similarity scores between text tokens than the en_core_web_sm model, especially if word vectors are needed.'''