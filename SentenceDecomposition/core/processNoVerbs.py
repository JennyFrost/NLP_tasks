from auxiliary_functions import *

suffixes = {'age', 'ance', 'ence', 'ion', 'ment', 'ness', 'ery'}
regex_suff = re.compile(r"(age|ance|ence|ion|ment|ness|ery)\b")
PART = ['part', 'member']
TEAM = ['team', 'department', 'agency']
ACTIVITY = ['campaigns', 'strategy']


def processNoVerbs(doc, indices):
    verbs = []
    prep = ''
    token_ = doc[indices[-1]]
    kw_span = doc[indices[0]:indices[1] + 1]
    main_tok = token_
    # if all(map(lambda x: x in token_.subtree, kw_span)):
    #     main_tok = token_
    if kw_span[0] == token_.head:
        main_tok = kw_span[0]
    elif kw_span[-1] != token_ and kw_span[-1] == token_.head and kw_span[-1] == kw_span[0].head:
        main_tok = kw_span[-1]
    if doc[0].text == 'if':  # or (kw_root.dep_=='nsubj' and kw_root.head.pos_=='AUX'):
        return 'junk', verbs, prep
    if main_tok.text in ROLES or (len(doc) > main_tok.i + 1 and doc[main_tok.i + 1].text in ROLES) \
            or (main_tok.dep_ == 'attr' and main_tok.head.text == 'AUX'):
        # if convert(kw_root.text, WN_NOUN, WN_VERB):
        #   verb = convert(kw_root.text, WN_NOUN, WN_VERB)[0][0]
        #   verbs.append(verb)
        return 'role', verbs, prep
    if main_tok.dep_ == 'nsubj':
        return 'subject', verbs, prep
    if main_tok.dep_ == 'conj':
        main_tok = get_main_token(doc, main_tok, kw_span)
    if main_tok.dep_ == 'ROOT':
        root_lefts = list(map(lambda x: x.dep_, doc[:main_tok.i]))
        if 'amod' in root_lefts or 'nmod' in root_lefts:
            verb_candidates = [token for token in doc[:main_tok.i] if token.dep_ in ('amod', 'nmod')]
            if any([get_synsets(cand) for cand in verb_candidates]):
                for cand in verb_candidates:
                    if get_synsets(cand) and cand.text not in str(kw_span) + str(main_tok):
                        verbs.append(cand.text.strip('-'))
                return 'extracted object', verbs, prep
    if main_tok.head.pos_ == 'ADP':
        head = main_tok.head.head
        if main_tok.head.text == 'of' and head.text in PART:
            # return 'part of a team', verbs, prep
            return '', [], ''
        if main_tok.head.text != 'of':
            prep = main_tok.head.text
        conjs = [head]
        if head.conjuncts:
            conjs += [conj for conj in head.conjuncts if conj.i < main_tok.i]
        if any([bool(re.search(regex_suff, conj.text)) for conj in conjs]):
            for conj in conjs:
                if convert(conj.text, WN_NOUN, WN_VERB):
                    verb = convert(conj.text, WN_NOUN, WN_VERB)[0][0]
                    verbs.append(verb)
        if verbs:
            return 'extracted object', verbs, prep
    # if bool(re.search(regex_suff, kw_root.text)) and convert(kw_root.text, WN_NOUN, WN_VERB):
    #     verb = convert(kw_root.text, WN_NOUN, WN_VERB)[0][0]
    #     if kw_splitted[-2] not in str(kw):
    #       verb += ' ' + kw_splitted[-2]
    #     verbs.append(verb)
    # return 'undefined', verbs, prep
    return '', [], ''
