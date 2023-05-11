import re


class EnumerationHandler:
    """
    Class for processing enumerations
    """

    def __init__(self):

        # dictionaries with generalizing words
        self.BeforeColonDict = {'1 word': ['specialties', 'include', 'skills', 'responsibilities', 'expertise'
            , 'including', 'experience', 'responsibilities', 'achievements', 'competencies'
            , 'includes', 'areas', 'accomplishments', 'specialities', 'tools'
            , 'results', 'strengths', 'included', 'role'],
                                '2 words': ['of expertise', 'core competencies', 'specialties include',
                                            'key achievements'
                                    , 'responsibilities include', 'such as', 'responsible for', 'strengths include',
                                            'skilled in'
                                    , 'key skills', 'skills include', 'skills in', 'expertise in', 'services include',
                                            'following areas'
                                    , 'key accomplishments', 'strengths in', 'key words', 'experience with',
                                            'experience in', 'experience of'
                                    , 'experience includes', 'specialized in', 'knowledge of', 'skilled with',
                                            'expert in'
                                    , 'core competencies', 'success in', 'experienced with', 'professional interests',
                                            'focused on'
                                    , 'specialist in'],
                                '3 words': ['in charge of', 'of expertise include', 'areas of expertise', 'what i do',
                                            'the following areas'
                                    , 'key responsibilities include', 'core competencies include', 'what drives me'
                                    , 'the areas of', 'of expertise in']
                                }

        self.BrandsDict = {'1 word': ['brands', 'clients'],
                           '2 words': ['brands include', 'brands including', 'brands are', 'client experience',
                                       'clients include', 'accounts responsibility',
                                       'main accounts'],
                           '3 words': ['brands such as', 'brand partner experience']
                           }

        self.IndustriesDict = {'1 word': ['industries'],
                               '2 words': ['industries include'],
                               '3 words': ['industry experience in']
                               }

        self.search_len = 0
        # we will search for keywords from dictionary in sentence[:search_len], not in the whole sentence
        for a in self.BeforeColonDict:
            for b in self.BeforeColonDict[a] + self.BrandsDict[a] + self.IndustriesDict[a]:
                if len(b) > self.search_len:
                    self.search_len = len(b)
        self.search_len *= 2

        # compiling regular expressions from dictionaries
        self.skills_pattern1 = re.compile(r'[\w()]*[ :]+|'.join(self.BeforeColonDict['1 word']) + '[\w()]*[ :]+')
        self.skills_pattern2 = re.compile(r'[\w()]*[ :]+|'.join(self.BeforeColonDict['2 words']) + '[\w()]*[ :]+')
        self.skills_pattern3 = re.compile(r'[\w()]*[ :]+|'.join(self.BeforeColonDict['3 words']) + '[\w()]*[ :]+')
        self.industries_pattern1 = re.compile(r'[\w()]*[ :]+|'.join(self.IndustriesDict['1 word']) + '[\w()]*[ :]+')
        self.industries_pattern2 = re.compile(r'[\w()]*[ :]+|'.join(self.IndustriesDict['2 words']) + '[\w()]*[ :]+')
        self.industries_pattern3 = re.compile(r'[\w()]*[ :]+|'.join(self.IndustriesDict['3 words']) + '[\w()]*[ :]+')
        self.brands_pattern1 = re.compile(r'[\w()]*[ :]+|'.join(self.BrandsDict['1 word']) + '[\w()]*[ :]+')
        self.brands_pattern2 = re.compile(r'[\w()]*[ :]+|'.join(self.BrandsDict['2 words']) + '[\w()]*[ :]+')
        self.brands_pattern3 = re.compile(r'[\w()]*[ :]+|'.join(self.BrandsDict['3 words']) + '[\w()]*[ :]+')

        self.inside_brackets_pattern = re.compile(r'\((.*?)\)')  # regular expression for extracting text from brackets

        self.amp_pattern = re.compile(r' & ')  # regexp for &, will be used for splitting
        self.and_pattern = re.compile(r' and ')  # regexp for 'and', will be used for splitting
        self.url_pattern = re.compile(
            r'.*www\.|https://.*|http://.*|.*@.*')  # regexp for website link, will be used for filtering out some items

    # =================================================================================================================

    def extract_items_inside_brackets(self, sent):
        """Extracts text from brackets within sentence
        Usage
        -----
        clear_sent, inside_brackets_items = self.extract_items_inside_brackets(sent)
        Input
        -----
        sent: str, required
            A sentence from which we try to extract text inside brackets
        Output
        ------
        clear_sent, inside_brackets_items: str, list
            clear_sent - sentence without text in brackets
            inside_brackets_items - list of texts inside brackets
        """
        inside_brackets_items = self.inside_brackets_pattern.findall(sent)
        try:
            remove_pattern = re.compile(r'\(' + '\)|\('.join(inside_brackets_items) + '\)')
            clear_sent = remove_pattern.sub('', sent)
        except:
            clear_sent = sent
            for to_remove in ['(' + item + ')' for item in inside_brackets_items]:
                clear_sent = clear_sent.replace(to_remove, '')
            inside_brackets_items = inside_brackets_items

        return clear_sent, inside_brackets_items

    # =================================================================================================================

    def process_enumeration_items(self, items):
        """Processes a list of enumeration items splitting them by '&', '/' and 'and'
        Usage
        -----
        result = self.process_enumeration_items(items)
        Input
        -----
        items: list, required
            A list of primary items which can be splitted into more granular keywords
        Output
        ------
        result: list
            A list of more granular items (if they can be extracted)
        -----
        """
        new_items = []
        for item in items:
            new_items.extend(re.split(',|: ', item))
            # make a new list of more granular items by splitting some of existing ones by , and ': '

        result = []
        for item in new_items:  # process new items one by one
            if self.url_pattern.search(item):  # if item contains website link - filter it out
                continue

            amp_result = []  # a list with items after working with &
            if (len(self.amp_pattern.findall(item)) == 1) and (
                    len(item.split()) <= 5):  # if & occurs 1 time and item is not very long
                pieces = item.split()
                # consider several cases of turning item into keywords
                if (len(pieces) == 3) and (pieces.index('&') == 1):
                    amp_result.extend([pieces[0], pieces[2]])
                elif (len(pieces) == 5) and (pieces.index('&') == 2):
                    amp_result.extend([pieces[0] + ' ' + pieces[1], pieces[3] + ' ' + pieces[4]])
                elif (len(pieces) == 5) and (pieces.index('&') == 1):
                    amp_result.extend(
                        [pieces[0] + ' ' + pieces[3] + ' ' + pieces[4], pieces[2] + ' ' + pieces[3] + ' ' + pieces[4]])
                elif (len(pieces) == 5) and (pieces.index('&') == 3):
                    amp_result.extend(
                        [pieces[0] + ' ' + pieces[1] + ' ' + pieces[2], pieces[0] + ' ' + pieces[1] + ' ' + pieces[4]])
                elif (len(pieces) == 4) and (pieces.index('&') == 1):
                    amp_result.extend([pieces[0] + ' ' + pieces[3], pieces[2] + ' ' + pieces[3]])
                elif (len(pieces) == 4) and (pieces.index('&') == 2):
                    amp_result.extend([pieces[0] + ' ' + pieces[1], pieces[0] + ' ' + pieces[3]])
                else:
                    amp_result.append(item)

            else:
                amp_result.append(item)

            slash_result = []  # a list of items after working with /
            for item1 in amp_result:  # process items from amp_result, not original ones
                if '/' in item1:
                    pieces = re.split(r'/| ', item1)
                    if sum([len(i) == 1 for i in pieces]) >= 2:  # for not splitting words like 'p/l mamagement'
                        slash_result.append(item1)
                    else:
                        slash_result.extend(item1.split('/'))
                else:
                    slash_result.append(item1)

            and_result = []  # a list of items after working with 'and'
            # we apply the logic similar to '&' case
            for item1 in slash_result:
                if (len(self.and_pattern.findall(' ' + item1 + ' ')) == 1) and (len(item1.split()) <= 5):
                    pieces = item1.split()
                    if (len(pieces) == 3) and (pieces.index('and') == 1):
                        and_result.extend([pieces[0], pieces[2]])
                    elif (len(pieces) == 5) and (pieces.index('and') == 2):
                        and_result.extend([pieces[0] + ' ' + pieces[1], pieces[3] + ' ' + pieces[4]])
                    elif (len(pieces) == 4) and (pieces.index('and') == 1):
                        and_result.extend([pieces[0] + ' ' + pieces[3], pieces[2] + ' ' + pieces[3]])
                    elif (len(pieces) == 4) and (pieces.index('and') == 2):
                        and_result.extend([pieces[0] + ' ' + pieces[1], pieces[0] + ' ' + pieces[3]])
                    elif (pieces.index('and') == 0):
                        and_result.append(' '.join(pieces[1:]))
                    else:
                        and_result.extend((' ' + item1 + ' ').split(' and '))
                else:
                    and_result.extend((' ' + item1 + ' ').split(' and '))

            result.extend(and_result)

        return result

    # =================================================================================================================
    def clean_items_list(self, items_list):
        """Cleans enumeration items and splits them into keywords and long phrases
        Usage
        -----
        keywords, long_phrases = self.clean_items_list
        Input
        -----
        items_list: list, required
            A list of enumeration items to clean
        Output
        ------
        keywords, long_phrases - lists
        """
        temp = [' '.join(item.split()) for item in items_list if
                len(' '.join(item.split())) > 1]  # replace multiple spaces with single space and remove single letters
        keywords = []
        long_phrases = []
        for item in temp:  # divide all items between keywords and long phrases based on their length
            if len(item.split()) <= 4:
                keywords.append(item)
            else:
                long_phrases.append(item)

        return keywords, long_phrases

    # =================================================================================================================

    def process_enumeration(self, list_of_sentences):
        """Extracts enumeration items from a list of enumeration sentences
        Usage
        -----
        result = self.process_enumeration(list_of_sentences)
        Input
        -----
        list_of_sentences: list, required
            A list of sentences to process

        Output
        ------
        result: list
            A list of tuples (sentence, type, keywords, long_phrases), where
                - sentence - sentence from input list
                - type - type of sentences, can be 'skills', 'brands', 'industries', 'undefined'
                - keywords - a list of keywords extracted from sentence
                - long_phrases - a list of phrases which failed to turned out into keywords
        """
        result = []
        for sent_ in list_of_sentences:
            sent, items = self.extract_items_inside_brackets(sent_.lower())  # extract text from brackets
            search_part = sent[
                          :self.search_len]  # define the part of sentence where we wiil search for generalizing words

            # consider several cases of searching for generalizing words
            if self.skills_pattern3.search(search_part):
                enumeration_part = sent[self.skills_pattern3.search(sent).end():]
                sentence_type = 'skills'
            elif self.industries_pattern3.search(search_part):
                enumeration_part = sent[self.industries_pattern3.search(sent).end():]
                sentence_type = 'industries'
            elif self.brands_pattern3.search(search_part):
                enumeration_part = sent[self.brands_pattern3.search(sent).end():]
                sentence_type = 'brands'

            elif self.skills_pattern2.search(search_part):
                enumeration_part = sent[self.skills_pattern2.search(sent).end():]
                sentence_type = 'skills'
            elif self.industries_pattern2.search(search_part):
                enumeration_part = sent[self.industries_pattern2.search(sent).end():]
                sentence_type = 'industries'
            elif self.brands_pattern2.search(search_part):
                enumeration_part = sent[self.brands_pattern2.search(sent).end():]
                sentence_type = 'brands'

            elif self.skills_pattern1.search(search_part):
                enumeration_part = sent[self.skills_pattern1.search(sent).end():]
                sentence_type = 'skills'
            elif self.industries_pattern1.search(search_part):
                enumeration_part = sent[self.industries_pattern1.search(sent).end():]
                sentence_type = 'industries'
            elif self.brands_pattern1.search(search_part):
                enumeration_part = sent[self.brands_pattern1.search(sent).end():]
                sentence_type = 'brands'
            else:
                enumeration_part = sent
                sentence_type = 'undefined'

            items.extend(
                re.split(r',|\*', enumeration_part))  # split the sentence by , and * and extend the list of items
            if sentence_type != 'brands':  # process all items if they are not brands
                items = self.process_enumeration_items(items)

            keywords, long_phrases = self.clean_items_list(
                items)  # clean processed items and divide them between keywords and long phrases
            result.append(
                (sent_, sentence_type, keywords, long_phrases))  # append a tuple for the current sentence to result

        return result
