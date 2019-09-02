from spacy.tokenizer import Tokenizer
import nltk
from nltk import *
nltk.download('words')
nltk.download('ieer')
from nltk.tag import StanfordNERTagger

nltk.download('maxent_ne_chunker')
import re
stanford_ner_tagger = StanfordNERTagger(
    'stanford-ner-2018-10-16/' + 'classifiers/english.muc.7class.distsim.crf.ser.gz',
    'stanford-ner-2018-10-16/' + 'stanford-ner-3.9.2.jar'
)

IN = re.compile(r'.*\bin\b(?!\b.+ing)')

sent='  Mohamed was born in Cairo. Ahmed was born in Alex. Zewail City is in Giza.   '
sentences = nltk.sent_tokenize(sent)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
mdict={}
taged = stanford_ner_tagger.tag(sent.split())
for tag in taged:
    tag_value = tag[0]
    tag_type = tag[1]
    if tag_type != 'O':
        print('Type: %s, Value: %s' % (tag_type, tag_value))
        mdict.update({tag_value:tag_type})


#doc.text=[""]
#for sents in (tagged_sentences):
#    print(sents)
ne_tree=[nltk.ne_chunk(sentence) for sentence in tagged_sentences]
doc = nltk.corpus.reader.ieer.IEERDocument(text = ne_tree,  headline='my text')
   
for rel in nltk.sem.extract_rels('LOC', 'PERSON', doc, corpus='ieer', pattern=IN):
    print(rel,'yohoooooooo' )
    print(nltk.sem.show_raw_rtuple(rel),'yohoooooooo' )

         

#for doc in chunked_sentences:
#    for rel in nltk.sem.extract_rels('ORG', 'PERSON', doc,corpus='ieer', pattern=IN):
#        print( nltk.sem.show_raw_rtuple(rel))
