import re


class ExpertiseChecker:
    def __init__(self):

        expertiseWords1 = ['specialties', 'skills', 'responsibilities', 'expertise', 'experience', 'competencies',
                           'specialities', 'expert'
            , 'familiarity', 'specialist', 'professional', 'focus', 'skilled']
        expertiseWords2 = ['of expertise', 'core competencies', 'specialties include', 'key achievements'
            , 'responsibilities include', 'strengths include', 'skilled in', 'skilled at'
            , 'key skills', 'skills include', 'skills in', 'expertise in'
            , 'key accomplishments', 'strengths in', 'experience with', 'experience in', 'experience of'
            , 'experience includes', 'specialized in', 'knowledge of', 'skilled with', 'expert in'
            , 'core competencies', 'success in', 'experienced with', 'experienced in', 'professional interests',
                           'focused on'
            , 'focus on', 'accomplished in', 'years of', 'years in', 'years at', 'decades of', 'decades in',
                           'decades at'
            , 'leader in', 'record in', 'recored of', 'backgroud in', 'experience across', 'experiences across'
            , 'charge of', 'understanding of', 'experience from', 'aspects of', 'my strengths', 'strong background'
            , 'key competencies', 'key experience', 'main competencies', 'core qualifications', 'responsible for',
                           'knowledgeable in', 'proficient in', 'knowledgeable amongst', 'affinity for',
                           'specialization in', 'specialist in',
                           'emphasis in', 'diploma in', 'focus in', 'area of', 'responsibility for', 'focus on',
                           'background in', 'of knowledge',
                           'accomplishments in', 'experts in', 'years within', 'passion for']
        expertiseWords3 = ['in charge of', 'of expertise include', 'areas of expertise', 'what i do',
                           'the following areas'
            , 'key responsibilities include', 'core competencies include', 'what drives me', 'the areas of'
            , 'in field of', 'focus is on']

        self.expertiseWords = set(expertiseWords1 + expertiseWords2 + expertiseWords3)
        self.expertiseWords = self.expertiseWords.union({x.replace(' ', '_') for x in self.expertiseWords})
        self.expertise_pattern = re.compile(
            "|".join([r"\b" + item + r"\b" for item in sorted(self.expertiseWords, key=len, reverse=True)]))

    def checkExpertise(self, sentenceDoc, keywordSpan, originalKeyword):
        """
        Function for checking if a particular keyword have 'expertise in' role in sentence

        Input
        -----
        sentenceDoc: spacy.doc, required
            Spacy doc of the sentence
        keywordSpan: spacy.Span, required
            Spacy span of improved keyword
        originalKeyword: str, required
            Original keyword (str!)

        Output
        ------
        expertise: bool
            True if expertiseKeyword represents the keyword used in 'expertise in' role, otherwise False
        expertiseKeyword: str
            The keyword that plays 'expertise in' role. Note: it can be either keywordToken.text or originalKeyword
        """
        expertise = False
        expertiseKeyword = ''
        keywordToken = keywordSpan[-1]

        if keywordSpan.text.replace(originalKeyword, '') in self.expertiseWords:
            # improved keyword in the form of 'originalKeyword_<smth from expertiseWords>'
            return True

        if (keywordToken.i + 1 < len(sentenceDoc)) and (sentenceDoc[keywordToken.i + 1].text in self.expertiseWords):
            # check for 'improvedKeyword <smth from expertiseWords>' in sentence
            return True

        if any([x + ' ' + keywordSpan.text in sentenceDoc.text for x in self.expertiseWords]):
            return True

        token = keywordToken
        while token.dep_ == 'conj':
            token = token.head

        if (token.text in self.expertiseWords) and (keywordSpan.text not in self.expertiseWords):
            return True

        len_ = len(sentenceDoc)
        if ((len_ > 1) and (sentenceDoc[0].text in self.expertiseWords)) or ((len_ > 2) and (
                sentenceDoc[1].text in self.expertiseWords or sentenceDoc[0].text + ' ' + sentenceDoc[
            1].text in self.expertiseWords)) \
                or ((len_ > 3) and (
                sentenceDoc[2].text in self.expertiseWords or sentenceDoc[1].text + ' ' + sentenceDoc[
            2].text in self.expertiseWords or sentenceDoc[0].text + ' ' + sentenceDoc[1].text in self.expertiseWords)):
            return True

        real_head1 = token.head
        real_head2 = real_head1.head
        real_head3 = real_head2.head
        if (real_head1.text in self.expertiseWords) \
                or (real_head2.text + ' ' + real_head1.text in self.expertiseWords) \
                or (real_head3.text + ' ' + real_head2.text + ' ' + real_head1.text in self.expertiseWords) \
                or (real_head2.text in self.expertiseWords):
            return True

        # foundExpertiseWords = expertise_pattern.findall(sentenceDoc.text)
        # if len(foundExpertiseWords) > 0:
        #   min_distances = []
        #   for word in foundExpertiseWords:
        #     parts = word.split()
        #     dist_to_this_word = [keywordToken.i - (token.i+len(parts)-1) for token in sentenceDoc if token.text == parts[0]]
        #     positive_dist_to_this_word = [x for x in dist_to_this_word if x>0]
        #     min_dist = min(positive_dist_to_this_word) if len(positive_dist_to_this_word) > 0 else 10*len(sentenceDoc)
        #     min_distances.append(min_dist)
        #   print(min_distances)
        #   if min(min_distances) <= 2:
        #     return True, keywordToken.text

        return expertise
