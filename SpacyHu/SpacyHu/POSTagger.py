import spacy
from spacy.tokens import Doc, Span, Token
import urllib
import xml.etree.ElementTree as ET
import re
from SpacyHu.BaseSpacyHuComponent import BaseSpacyHuComponent


class HuPOSTagger(BaseSpacyHuComponent):
    def __init__(self,
                 nlp,
                 label='POS',
                 url='http://localhost:8000/process?run='):
        necessary_modules = ['QT', 'ML3-PosLem-hfstcode']
        super().__init__(nlp, label, url, necessary_modules)
        Token.set_extension('pos', default='')

    def get_word_from_annotation(self, annotation):
        for feature in annotation.getchildren():
            if feature.find('Name').text == 'string':
                return feature.find('Value').text

    def get_token_by_idx(self, idx, doc):
        for token in doc:
            if token.idx == idx:
                return token

    def __call__(self, doc):
        text = urllib.parse.quote_plus(doc.text)
        result = urllib.request.urlopen(self.url + text).read()
        annotationset = ET.fromstring(result).find('AnnotationSet')
        for annotation in annotationset.getchildren():
            if annotation.get('Type') != 'Token':
                continue

            word_index = int(annotation.get('StartNode'))
            word = self.get_word_from_annotation(annotation)
            token = self.get_token_by_idx(word_index, doc)
            for feature in annotation.getchildren():
                if feature.find('Name').text == 'hfstana':
                    hfstana = (feature.find('Value').text
                               if feature.find('Value').text is not None
                               else '')
                    token.tag_ = hfstana
                if feature.find('Name').text == 'pos':
                    pos = (feature.find('Value').text
                           if feature.find('Value').text is not None
                           else '')
                    token._.pos = pos
                    # token.pos_ = pos
                if feature.find('Name').text == 'lemma':
                    lemma = (feature.find('Value').text
                             if feature.find('Value').text is not None
                             else '')
                    token.lemma_ = lemma
                    break
        return doc


if __name__ == "__main__":
    from Tokenizer import HuTokenizer

    remote_url = 'http://hlt.bme.hu/chatbot/gate/process?run='
    debug_text = 'A kastély nem vár senkire, csak akkor ha annyi az idő'
    nlp = spacy.blank("en")
    nlp.tokenizer = HuTokenizer(nlp.vocab, url=remote_url)
    POS_analyzer = HuPOSTagger(nlp, url=remote_url)
    nlp.add_pipe(POS_analyzer, last=True)

    doc = nlp(debug_text)
    for token in doc:
        print('Token is: ' + token.text)
        print('tag is: ' + token.tag_)
        print('pos is: ' + token._.pos)
        print('lemma is: ' + token.lemma_ + '\n')
