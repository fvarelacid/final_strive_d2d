def preprocess(pth):

    with open(pth, 'r') as f:

        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace(".","").replace(",","").replace(":","").replace(";","").replace("?","").replace("!","").replace("-","").replace("_","").replace("\"","").replace("\'","")

            with open("preprocessed.txt", "w") as p:
                p.writelines(lines)
