import spacy
import re


class VerbTypeChecker:
    """
    The class contains methods that determine the type of the verb - result, means, or indirect engagement.
    """
    result_exclude = {'appointed', 'selected', 'tasked', 'intend', 'want', 'need', 'love', 'like', 'ask', 'hope',
                      'ensure', 'plan', 'exist'}
    result_verbs = {'help', 'increase', 'maximize', 'drive', 'develop', 'enhance', 'enforce', 'transform'}
    verbs_with_ccomps = {'help', 'allow', 'let', 'discover', 'enable'}
    PREP_RESULT = {'to', 'for'}
    PREP_MEANS = {'via', 'by', 'through'}
    ROLES = {'leader', 'specialist', 'professional', 'strategist', 'manager', 'coordinator', 'intern', 'admin',
             'consultant', 'director', 'marketer', 'officer', 'apprentice', 'associate', 'assistant', 'expert'}
    means_verbs = {'use', 'leverage', 'navigate', 'visit', 'follow', 'subscribe'}

    def __init__(self, doc: spacy.tokens.doc.Doc):
        self.doc = doc

    def isResultVerb(self, verb: spacy.tokens.token.Token) -> str | None:
        if verb.dep_ != 'ROOT':
            if ((self.doc[verb.i - 1].text in self.PREP_RESULT)
                    and (self.doc[verb.i - 2].text not in self.result_exclude)
                    and (verb.head.text not in self.result_exclude)):
                return 'result'
            if verb.lemma_ in self.result_verbs:
                return 'result'
            if (verb.head.text in self.verbs_with_ccomps and verb.dep_ == 'ccomp'
                    and verb.i > verb.head.i
                    and verb.text != 'using'):
                return 'result'
            # if verb.dep_ in ('xcomp', 'ccomp') and verb.head.pos_ == 'VERB' and verb.i > verb.head.i and verb.text != 'using' \
            #         and verb.head.text not in result_exclude:
            #     return True
        if len(self.doc) >= verb.i + 2:
            if (verb.text + ' ' + self.doc[verb.i + 1].text) in ('led to', 'leading to', 'resulted in', 'resulting in'):
                return 'result'

    def isMeansVerb(self, verb: spacy.tokens.token.Token) -> str | None:
        if verb.head.text in self.PREP_MEANS or self.doc[verb.i - 1].text in self.PREP_MEANS:
            return 'means'
        if verb.lemma_ in self.means_verbs:
            return 'means'
        if str(self.doc[verb.i - 4:verb.i]) in ('with the use of', 'with the help of'):
            return 'means'

    def isIndirectEngagement(self, verb: spacy.tokens.token.Token) -> str | None:
        if (verb.head.pos_ in ('NOUN', 'PROPN')
                and verb.i > verb.head.i
                and verb.head.text not in self.ROLES
                and verb.dep_ not in ['ROOT', 'conj']):
            if verb.dep_ == 'relcl':
                return 'indirect engagement'
            if ((bool(re.search(r'ing\b', verb.text))
                 or bool(re.search(r'ed\b', verb.text)))
                    and verb.dep_ != 'amod'
                    and verb.text != 'using'
                    and self.doc[verb.i + 1].pos_ != 'PUNCT'):
                return 'indirect engagement'
