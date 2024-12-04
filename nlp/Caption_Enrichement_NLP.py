import re

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet

taxonomy=['smile', 'wave', 'talk', 'sleep', 'sit', 'laught', 'jump', 'wear a mask']


def synonym_antonym_extractor(phrase):
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(phrase):
        for l in syn.lemmas():
            print(l)
            synonyms.append(l.name())
            print(syn.definition())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
            token3 = l.name

    print(set(synonyms))
    print(set(antonyms))
    return set(synonyms), set(antonyms), token3


def dissimilar(nlp, taxonomy, word1):
    sim = {}
    word = nlp(word1)
    token = ''
    for w in word:
        token = w
    for t in taxonomy:
        t = nlp(t)
        sim_ = token.similarity(t)
        print("Similarity:", sim_)
        sim[t] = sim_
    sorted_ = sorted(sim.items(), key=lambda x: x[1])

    print(sorted_)
    dissimilarity = min(sim, key=sim.get)
    print(dissimilarity)

    return dissimilarity


def augment_caption(nlp, taxonomy, sentence):
    doc = nlp(sentence)
    wordtoreplace = {}
    for token in doc:
        if token.pos_ == 'VERB':
            verbtoreplace = token.text
            wordtoreplace["verb"] = verbtoreplace
    for word in wordtoreplace:
        print(wordtoreplace[word])

        rep = dissimilar(nlp, taxonomy, wordtoreplace[word])
        ini_ = wordtoreplace[word]
        rep = rep.text
        newsentence = sentence.replace(ini_, rep)

    return newsentence


def change_caption(nlp, caption, category, subcategory, rep):
    sentence = str(caption)
    sentence = sentence.lower()
    category = category.lower()
    doc = nlp(sentence)

    wordtoreplace = {}
    newcaption = ''
    for token in doc:
        if token.text == category:
            toreplace = token.text
            wordtoreplace[category] = toreplace
        elif token.text == subcategory:
            toreplace = token.text
            wordtoreplace[category] = toreplace

    for word in wordtoreplace:
        ini_ = wordtoreplace[word]
        rep = str(rep)
        newcaption = sentence.replace(ini_, rep)
    if newcaption == '':
        newcaption = 'high-fidelity image of ' + sentence
    else:
        new = 'high-fidelity image of ' + newcaption
    print(newcaption)
    return newcaption


def caption_category(caption, category, rep):
    sentence = str(caption)
    sentence = sentence.lower()
    category = category.lower()

    if sentence != category:
        newcaption = category + rep
    if newcaption == '':
        newcaption = 'high-fidelity image of ' + category
    else:
        newcaption = 'high-fidelity image of ' + newcaption

    return newcaption

    
if __name__ == "__main__":
    import spacy

    nlp = spacy.load("en_core_web_sm")
    taxonomy = ['smiling', 'waving', 'talking', 'sleeping', 'siting', 'laughting', 'jumping', 'wearing a mask']
    sent = augment_caption(nlp, taxonomy, "the nurse is eating")
    print(sent)