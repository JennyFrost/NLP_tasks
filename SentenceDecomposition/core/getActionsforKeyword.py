from auxiliary_functions import *
from ConjunctsHandler import ConjunctsHandler
from typing import Tuple, List


def main_tok_from_indices(doc: spacy.tokens.doc.Doc,
                          indices: List[int]) -> Tuple[spacy.tokens.token.Token, spacy.tokens.span.Span]:
    token_ = doc[indices[-1]]
    kw_span = doc[indices[0]:indices[1] + 1]
    main_tok = token_
    if kw_span[0] == token_.head:
        main_tok = kw_span[0]
    elif kw_span[-1] != token_ and kw_span[-1] == token_.head and kw_span[-1] == kw_span[0].head:
        main_tok = kw_span[-1]
    return main_tok, kw_span


def get_benefactive(doc: spacy.tokens.doc.Doc,
                    main_tok: spacy.tokens.token.Token,
                    kw_span: spacy.tokens.span.Span) -> List[Tuple]:
    verbs = []
    benef_candidates = [main_tok.head, doc[main_tok.i-1]]
    if list(main_tok.lefts):
        benef_candidates.append(doc[list(main_tok.lefts)[0].i-1])
    for cand in benef_candidates:
        if cand.text.lower() == 'for' and cand.i != 0:
            flag = 'benefactive'
            other_verbs = get_other_verbs(doc, cand, kw_span)
            if other_verbs:
              verbs = other_verbs
            verbs += [(main_tok.i, '', '', '', flag)]
            return verbs


def verb_parent_condition(doc: spacy.tokens.doc.Doc,
                          main_tok: spacy.tokens.token.Token,
                          kw_span: spacy.tokens.span.Span) -> bool:
    if (main_tok.head.pos_ in ('VERB', 'AUX')
        and main_tok.head not in kw_span
        and not (re.search(r'ed\b', main_tok.head.text)
        and ((main_tok.head.head.text in ROLES
        or (doc[main_tok.head.i - 1].text == ','
        and doc[main_tok.head.i - 2].pos_ == 'ADJ')
        or doc[main_tok.head.i - 1].pos_ == 'ADJ')
        or (main_tok.head.dep_ in ('conj', 'appos')
        and not re.search(r'ed\b', main_tok.head.head.text))))):
        return True


def means_condition(doc: spacy.tokens.doc.Doc,
                    main_tok: spacy.tokens.token.Token,
                    kw_span: spacy.tokens.span.Span) -> bool:
    if (main_tok.head.text.lower() in ('by', 'through', 'via')
        or doc[kw_span[0].i - 1].text.lower() in ('by', 'through', 'via')
        or (len(doc) > 4 and str(doc[main_tok.i - 4:main_tok.i]).lower() in ('with the use of', 'with the help of'))):
        if not ((main_tok.head.text.lower() == 'by'
                and main_tok.head.dep_ == 'agent')
                and not (doc[kw_span[0].i - 1].text.lower() == 'by'
                and doc[kw_span[0].i - 1].dep_ == 'agent')):
            return True


def get_subject_verbs(doc: spacy.tokens.doc.Doc,
                        main_tok: spacy.tokens.token.Token,
                        kw: spacy.tokens.token.Token,
                        kw_span: spacy.tokens.span.Span) -> List[Tuple]:
    flag = 'subject'
    if main_tok.head.pos_ in ('VERB', 'AUX') and main_tok.head not in kw_span:
        verb = main_tok.head
        if verb.text in verbs_stoplist:
            nearest_verb = get_nearest_verb(doc, main_tok, kw_span, go_left=False)
            if nearest_verb:
                verb = nearest_verb
            else:
                return [(main_tok.i, '', '', '', flag)]
    nearest_verb = get_nearest_verb(doc, main_tok, kw_span, go_left=False)
    if nearest_verb:
        verb = nearest_verb
    else:
        return [(main_tok.i, '', '', '', flag)]
    verbs = get_action_verb_tuples(doc, main_tok, kw, verb, kw_span)
    return verbs


def getActionsforKeyword(doc: spacy.tokens.doc.Doc,
                         indices: List[int]) -> List[Tuple]:
    """

    Args:
        doc: the sentence processed with spacy
        indices:
            - the index of the first token in the keyword;
            - the index of the KW token before the main (rightmost) token of the KW;
            - the index of the main (rightmost) token of the KW

    Returns the list of tuples of 2 types:
    - a tuple with the main token of the keyword and its flag:
        - benefactive
        - means (without a verb)
    - a tuple with a verb, its object and flag:
        - action
        - subject
        - state
        - means (with a verb)
        - result
        - indirect engagement

    The tuple comprises:
    - the index of the verb/main token in the sentence Doc;
    - the text (string) of the verb token (or empty string);
    - the text (string) of the verb object (or empty string);
    - the text (string) of the preposition after the verb/before the keyword (or empty string);
    - the flag.
    For each pair (verb, object) there is a separate tuple.

    """
    main_tok, kw_span = main_tok_from_indices(doc, indices)
    verbs = []
    flag = 'action'
    prep = ''
    obj = ''
    kw = main_tok
    cj = ConjunctsHandler(doc)
    if main_tok.pos_ == 'VERB':
        cj.get_main_verb_token(main_tok)
        main_tok = cj.main_verb
    else:
        main_tok = cj.get_main_token(main_tok, kw_span)
    if main_tok.head.lemma_ == 'include':
        main_tok = main_tok.head
        if main_tok.dep_ == 'ROOT':
          flag = 'enum'
          return [(main_tok.i, '', obj, prep, flag)]

    if main_tok.head.text.lower()=='of' and main_tok.head.head.text.lower() in objects_exclude:
        main_tok = main_tok.head.head

    if main_tok.dep_ == 'nsubj' or (main_tok.dep_ == 'ROOT' and main_tok.pos_ != 'VERB'):
        return get_subject_verbs(doc, main_tok, kw, kw_span)

    if means_condition(doc, main_tok, kw_span):
        flag = 'means'
        other_verbs = get_other_verbs(doc, main_tok, kw_span)
        if other_verbs:
          verbs = other_verbs
        verbs += [(main_tok.i, '', '', '', flag)]
        return verbs

    verbs = get_benefactive(doc, main_tok, kw_span)
    if verbs:
        return verbs

    if verb_parent_condition(doc, main_tok, kw_span) and main_tok.dep_ not in ('ROOT', 'conj'):
        verb = main_tok.head
        if verb.i > main_tok.i or verb.text in verbs_stoplist:
            nearest_verb = get_nearest_verb(doc, main_tok, kw_span)
            if nearest_verb:
                verb = nearest_verb
            else:
                return []
        verbs = get_action_verb_tuples(doc, main_tok, kw, verb, kw_span)
        return verbs

    elif main_tok.dep_ == 'dobj' and main_tok.head not in kw_span:
        cj = ConjunctsHandler(doc)
        if cj.get_synsets(main_tok.head):
            verb = main_tok.head
            verbs = get_action_verb_tuples(doc, main_tok, kw, verb, kw_span)
            return verbs

    elif main_tok.head.pos_ == 'ADP' and verb_parent_condition(doc, main_tok.head, kw_span):
        prep = main_tok.head
        verb = prep.head
        if verb.text in verbs_stoplist:
            nearest_verb = get_nearest_verb(doc, main_tok, kw_span)
            if nearest_verb:
                verb = nearest_verb
            else:
                return []
        verbs = get_action_verb_tuples(doc, main_tok, kw, verb, kw_span, prep=prep)
        return verbs

    nearest_verb = get_nearest_verb(doc, main_tok, kw_span)
    if nearest_verb:
        verbs = get_action_verb_tuples(doc, main_tok, kw, nearest_verb, kw_span)
        return verbs

    else:
        verbs = []
    return [v for v in verbs if v[0] < main_tok.i]


