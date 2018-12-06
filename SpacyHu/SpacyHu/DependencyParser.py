import spacy
from spacy.tokens import Doc, Span, Token
import urllib
import xml.etree.ElementTree as ET
import re
from SpacyHu.BaseSpacyHuComponent import BaseSpacyHuComponent


class DependencyParser(BaseSpacyHuComponent):

    def __init__(self,
                 nlp,
                 label='DepParser',
                 url='http://localhost:8000/process?run='):
        necessary_modules = ['QT', 'ML3-PosLem-hfstcode', 'ML3-Dep']
        super().__init__(nlp, label, url, necessary_modules)
        Token.set_extension('dep_type', default='')

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

            # Setting head
            deptarget = self.get_value_from_annotation(annotation, 'depTarget')
            if deptarget is None:
                continue

            target_token_idx = None
            for i in annotationset.getchildren():
                if i.get('Id') == deptarget:
                    target_token_idx = int(i.get('StartNode'))
                    break

            # Setting depType
            deptype = self.get_value_from_annotation(annotation, 'depType')
            if deptype is None:
                raise Exception('This should not have happened, if'
                                'deptarget is present so should depType be')

            token.head = self.get_token_by_idx(target_token_idx, doc)
            token._.dep_type = deptype
            # token.dep = deptype # needs to conver string to int
            # https://github.com/explosion/spaCy/blob/master/spacy/symbols.pyx

        return doc


if __name__ == '__main__':
    from Tokenizer import HuTokenizer

    debug_text = u'Autonóm autók hárítják a biztosítás terhét gyártók felé'
    # debug_text = 'megszentségteleníthetetlenségeitekért meghalnak'
    remote_url = 'http://hlt.bme.hu/chatbot/gate/process?run='
    nlp = spacy.blank('en')
    nlp.tokenizer = HuTokenizer(nlp.vocab, url=remote_url)
    dependeny_parser = DependencyParser(nlp, url=remote_url)
    nlp.add_pipe(dependeny_parser, last=True)

    doc = nlp(debug_text)
    for token in doc:
        print('Token is: ' + token.text)
        print('Head is: ' + token.head.text)
        print('DepType is: ' + token._.dep_type)
        print()
