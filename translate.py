#Student_ID= 0417312030
#run this file, it'll automatically train the data and model, then it'll show the output of test data in console.

from trainer import *

main('./data/train.json', outfile='./data/model.json', verbose=True)

with open('./data/model.json', 'r', encoding='utf-8') as f:
    model = json.load(f)
testCorpus = getCorpus('./data/test.json')
wordsInTestData = getWords(testCorpus)
inputWordsInTestData = wordsInTestData['A']
print(inputWordsInTestData)

for word in inputWordsInTestData:
    if word in model:
        print(word + " = " + model[word])
    else:
        print(word)

    print("\n")
