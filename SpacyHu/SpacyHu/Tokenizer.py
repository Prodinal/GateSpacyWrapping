import spacy
from spacy.tokens import Doc
import urllib
import xml.etree.ElementTree as ET
from SpacyHu.BaseSpacyHuComponent import BaseSpacyHuComponent


class HuTokenizer(BaseSpacyHuComponent):
    def __init__(self, vocab, url='http://localhost:8000/process?run='):
        necessary_modules = ['QT']
        self.url = url + ','.join(necessary_modules) + '&text='
        self.vocab = vocab

    def __call__(self, input):
        text = urllib.parse.quote_plus(input)

        result = urllib.request.urlopen(self.url + text).read()
        annotations = ET.fromstring(result).find('AnnotationSet')

        words = [element.getchildren()[1].find('Value').text
                 for element in annotations.getchildren()
                 if element.get('Type') == 'Token']
        return Doc(self.vocab, words=words)


if __name__ == "__main__":
    nlp = spacy.blank("en")
    nlp.tokenizer = HuTokenizer(nlp.vocab)
    doc = nlp('Jó, hogy ez az alma piros, mert az olyan almákat szeretem.')
    for token in doc:
        print('Token is: ' + str(token))
