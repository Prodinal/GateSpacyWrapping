import spacy
from spacy.tokens import Doc, Span, Token
import urllib
import xml.etree.ElementTree as ET
from SpacyHu.BaseSpacyHuComponent import BaseSpacyHuComponent


class PreverbIdentifier(BaseSpacyHuComponent):

    def __init__(self,
                 nlp,
                 label='PreverbIdentifier',
                 url='http://localhost:8000/process?run='):
        necessary_modules = ['QT', 'ML3-PosLem-hfstcode', 'ML3-Dep,Preverb']
        super().__init__(nlp, label, url, necessary_modules)
        Token.set_extension('preverb', default='')
        Token.set_extension('lemmaWithPreverb', default='')

    def get_token_by_idx(self, idx, doc):
        for token in doc:
            if token.idx == idx:
                return token

    def get_value_from_annotation(self, annotation, attr_name):
        for child in annotation.getchildren():
            if child.find('Name').text == attr_name:
                return child.find('Value').text

    def __call__(self, doc):
        text = urllib.parse.quote_plus(doc.text)
        result = urllib.request.urlopen(self.url + text).read()
        annotationset = ET.fromstring(result).find('AnnotationSet')
        for annotation in annotationset.getchildren():
            if annotation.get('Type') != 'Token':
                continue

            word_index = int(annotation.get('StartNode'))
            token = self.get_token_by_idx(word_index, doc)
            preverb = self.get_value_from_annotation(annotation, 'preverb')
            if preverb is not None:
                token._.preverb = preverb
            lemma_with = (
                self.get_value_from_annotation(annotation, 'lemmaWithPreverb'))
            if lemma_with is not None:
                token._.lemmaWithPreverb = lemma_with

        return doc

if __name__ == '__main__':
    from Tokenizer import HuTokenizer

    remote_url = 'http://hlt.bme.hu/chatbot/gate/process?run='
    debug_text = u'El is áztam amikor került elő a repcsi'
    nlp = spacy.blank('en')
    nlp.tokenizer = HuTokenizer(nlp.vocab, url=remote_url)
    preverb_identifier = PreverbIdentifier(nlp, url=remote_url)
    nlp.add_pipe(preverb_identifier, last=True)

    doc = nlp(debug_text)
    for token in doc:
        print('Token is: ' + token.text)
        if token._.preverb != '':
            print('Preverb is: ' + token._.preverb)
        if token._.lemmaWithPreverb != '':
            print('lemmaWithPreverb is: ' + token._.lemmaWithPreverb)
        print()
