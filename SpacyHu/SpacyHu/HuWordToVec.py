import numpy as np


class HUWordToVec(object):

    def __init__(self, file_path=('/home/mrengineer/School/DipTerv/'
                                  'aut_chatbots/SpacyHu/SpacyHu/wiki.hu.vec')):
        self.file_path = file_path
        self.lines = None

    def __call__(self, doc):
        doc.user_hooks['vector'] = self.get_doc_vector
        doc.user_token_hooks['vector'] = self.get_token_vector

        return doc

    def get_token_vector(self, token):
        # Implement actual logic here
        # Spacy uses 300dimensional GloVe vectors by default

        # with open(self.file_path, 'r') as f:
        #     for line in f:
        #         if token.text.lower() in line:
        #             # each line starts with the word and ends with a new line
        #             coords = line.split(' ')[1:]
        #             return np.array([float(x) for x in coords if x != '\n'])
        #     print('Word not found in dictionary: ' + token.text)
        #     return np.zeros(300)

        if self.lines is None:
            print("Reading word vectors into memory")
            with open(self.file_path, 'r') as f:
                self.lines = f.readlines()
        for line in self.lines:
            if token.text.lower() in line:
                # each line starts with the word and ends with a new line
                coords = line.split(' ')[1:]
                return np.array([float(x) for x in coords if x != '\n'])
        print('Word not found in dictionary: ' + token.text)
        return np.zeros(300)

    def get_doc_vector(self, doc):
        return np.average([token.vector for token in doc], axis=0)

if __name__ == '__main__':
    import spacy
    from Tokenizer import HuTokenizer

    class Container(object):
        pass

    comp = HUWordToVec()
    obj = Container()
    obj.text = 'alma'
    vec1 = comp.get_token_vector(obj)
    print(vec1)
    print(len(vec1))  # expected: 300

    nlp = spacy.blank('en')
    nlp.tokenizer = HuTokenizer(nlp.vocab)
    nlp.add_pipe(HUWordToVec(), last=True)
    doc = nlp('Alma körte barack')
    for token in doc:
        print('Token is: ' + token.text)
        print(token.vector[0])
    print(doc.vector)
