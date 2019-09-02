from spacy.tokenizer import Tokenizer
import nltk
nltk.download('words')
nltk.download('ieer')
nltk.download('maxent_ne_chunker')
import re
import spacy
from spacy import displacy
from collections import Counter
from nltk.tag import StanfordNERTagger
import en_core_web_sm
sent='  Mohamed was born in Cairo. Ahmed was born in Alex. Ahmed in ZewaiCity is  in Giza.'
spacydict={}
nlp = en_core_web_sm.load()
#########################
spacy_nlp = spacy.load('en')
document = spacy_nlp(sent)
for element in document.ents:
    print('ennnnnnnnnnnnnType: %s, Value: %s' % (element.label_, element))
    spacydict.update({str(element):str(element.label_)})
#print (spacydict)
print('Mohamed in Cairo')
print('%s in %s'% (spacydict['Mohamed'] , spacydict['Cairo']))

#####################
##from spacy.en import English
##nlp = English()
##tagger = nlp.create_pipe("tagger")
##ner = nlp.create_pipe("ner")
##processed = ner(doc)
sent_tg = nltk.corpus.treebank.tagged_sents()[21]
#print(nltk.ne_chunk(sent_tg))


###stanford
mdict={}

stanford_ner_tagger = StanfordNERTagger(
    'stanford-ner-2018-10-16/' + 'classifiers/english.muc.7class.distsim.crf.ser.gz',
    'stanford-ner-2018-10-16/' + 'stanford-ner-3.9.2.jar')

tokens = nltk.word_tokenize(sent)
taged = stanford_ner_tagger.tag(sent.split())
for idx ,tag in enumerate(taged):
    tag_value = tag[0]
    tag_type = tag[1]
    if tag_type != 'O':
        print('yohooo Type: %s, Value: %s' % (tag_type, tag_value))
        mdict.update({tag_value:tag_type})

#for i,tk in enumerate(sent.split()):
#    if(tk in mdict.keys()):
        
#        print('%s in %s',tk, mdict['LOCATION'] )
print('Ahmed in Alex') 
print('%s in %s'% (mdict["Ahmed"] , mdict["Alex."])) 
print('ZewailCity in Giza')
print('%s in %s'% (mdict["ZewaiCity"] , mdict["Giza."]))
print('Mohamed in Cairo')
#print('%s in %s'% (mdict["Mohamed"] , mdict["Cairo."]))
#######

mdict2={}
doc=nlp(sent)
print([(ent.text, ent.label_) for ent in doc.ents])
[mdict2.update({ent.text:ent.label_}) for ent in doc.ents]
[print(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
            [child for child in token.children])

##########################################
tokens = nltk.word_tokenize(sent)
print(tokens)
def get_ner():
 
        for i in tokens:
            
            tagged = nltk.pos_tag(i)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
#            chunked.draw()

            print(chunked)

####################  

get_ner()
### itried alot to change the order of giza and how it come but always as person with spacy , stanford fixed it but messed up by stanford but stanford missed the names 
