def preprocess(pth):

    with open(pth, 'r') as f:

        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace(".","").replace(",","").replace(":","").replace(";","").replace("?","").replace("!","").replace("-","").replace("_","").replace("\"","").replace("\'","")

            with open("preprocessed.txt", "w") as p:
                p.writelines(lines)

def get_sentences(pth):

    f = open(pth, 'r')

    lines = f.readlines()

    sentences = [line.split() for line in lines]

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
