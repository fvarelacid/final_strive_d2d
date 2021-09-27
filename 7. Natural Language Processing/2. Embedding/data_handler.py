import numpy as np

def preprocess(pth):

    with open(pth, 'r') as f:

        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace(".","").replace(",","").replace(":","").replace(";","").replace("?","").replace("!","").replace("-","").replace("_","").replace("\"","").replace("\'","")

            with open("processed.txt", "w") as p:
                p.writelines(lines)

def get_sentences(pth):

    f = open(pth, 'r')

    lines = f.readlines()

    sentences = [line.lower().split() for line in lines]

    f.close()

    return sentences


def clean_sentences(sentences):

    i = 0

    while i < len(sentences):

        if sentences[i] == []:
            sentences.pop(i)
        
        else:
            i += 1

    return sentences


def get_dicts(sentences):

    vocab = []

    for sentence in sentences:
        for token in sentence:

            if token not in vocab:
                vocab.append(token)

    w2i = { w: i for (i, w) in enumerate(vocab) }
    i2w = { i: w for (i, w) in enumerate(vocab) }

    return w2i, i2w, len(vocab)


def get_pairs(sentences, w2i, r):

    pairs = []

    for sentence in sentences:

        tokens = [ w2i[word] for word in sentence ]

        for center in range(len(tokens)):

            for context in range(-r, r+1):

                context_word = center + context

                if context_word < 0 or context_word >= len(tokens) or context_word == center:
                    continue
                else:
                    pairs.append( (tokens[center], tokens[context_word]) )
    
    return np.array(pairs)

def get_dataset():

    sentences = get_sentences("processed.txt")
    clean_sents = clean_sentences(sentences)

    w2i, i2w, vocal_len = get_dicts(clean_sents)
    pairs = get_pairs(clean_sents, w2i, 4)

    return pairs, vocal_len

get_dataset()