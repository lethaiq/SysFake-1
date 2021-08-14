"""
feature_extraction.py
---------------------

Contains the `ArticleVector` object, the all-in-one interface
for scraping and vectorizing news articles as both URLs and given text.

Dependencies:
-------------
* language_tool_python
* tldextract
* nltk
* sklearn
* goose3
* pyapa
* waybackmachine
"""

import warnings
#import pdb
#import os

from collections import Counter, deque

import goose3
import language_tool_python
import tldextract
#import requests
import nltk

from lxml.etree import ParseError, ParserError #pylint:disable=no-name-in-module
from goose3 import Goose
from pyapa import pyapa

from waybackmachine import WaybackMachine
from waybackmachine.fetch import WaybackMachineError
#from ap_style_checker import StyleChecker
#from newspaper import Article

def process_source_list(filename=""):
    """
    ===
    Utility function to preprocess and read in items from a text file to a list.
    This function is not performed in-place.
    ===
    Parameters
    ===
        source: str
            File name of text to be processed and read. Defaults to an empty string.
        delimiter: str
            Character which separates discrete items in the text file.
    """
    with open(filename, mode='r') as filein:
        list_before_processing = filein.readlines()
        return [j.replace(" ", "").lower().strip()
                for i, j in enumerate(list_before_processing)]

class ArticleVector:
    """
    Purpose: Extract a vector of articles/urls to be stored in feature matrix
    """
    ##### CLASS ATTRIBUTES #####
    ##todo: assign this based on length of features in a global config
    NUM_DIMENSIONS = 18

    reputable_news_sources = process_source_list("./data/reputable_news_sources.txt")
    satire_news_sources = process_source_list("./data/satire_news_sources.txt")
    unreputable_news_sources = process_source_list("./data/unreputable_news_sources.txt")

    ##### INSTANCE ATTRIBUTES #####

    def __init__(self, url="", text="", title="", latest_snapshot=True):
        self.vector = [0] * ArticleVector.NUM_DIMENSIONS
        self.url = url
        self.cleaned_url = self._clean_url()

        if text and url: # usr enters both
            #print('test')
            #article = self.extract_article()
            self.title = title
            self.text = text
        elif not text and url: # user enters url
            article = self._extract_article(latest_snapshot)
            self.title = article.title
            self.text = article.cleaned_text
        elif text and not url: # user enters article text
            self.title = title
            self.text = text

        self.char_counts = Counter(self.text)

        self.paired_tokens = self.tokenize() #list of pairs ex. [('helped', 'VBD')]

        # not entirely sure about this implementation yet -h
        # `tokenize` returns a list of pairs, which the dict constructor uses natively
        self.token_count = nltk.probability.FreqDist(word.lower()
                                                     for word in
                                                     dict(self.paired_tokens).keys())

        if self.text == "":
            warnings.warn('The text for this article is empty.')
            self.num_words = 1 # no division by zero
        else:
            self.num_words = sum(self.token_count.values())

        self.part_of_speech_count = Counter(dict(self.paired_tokens).values())
        self.vector = [self.url_ending_index(),
                       self.from_reputable_source_index(),
                       self.today_index(),
                       self.grammar_index(),
                       self.quotation_index(),
                       self.past_tense_index(),
                       self.present_tense_index(),
                       self.should_index(),
                       self.opinion_index(),
                       self.all_caps_index(),
                       self.from_satire_source_index(),
                       self.exclamation_index(),
                       self.apa_index(),
                       self.name_source_index(),
                       self.interjection_index(),
                       self.you_index(),
                       self.dot_gov_ending_index(),
                       self.from_unreputable_source_index()]

    def _clean_url(self):
        '''
        Extract an ArticleVector's URL domain name, if it possesses a URL.
        ex: clean_url('https://www.nytimes.com/<article identifiers>') -> 'nytimes'
        '''
        return tldextract.extract(self.url).domain if self.url else ''

    def grammar_index(self):
        '''
        returns the number of grammar mistakes of the article divided by the length of the article
        '''

        checker = language_tool_python.LanguageTool('en-US')
        matches = checker.check(self.text) # of grammar errors
        return len(matches) / self.num_words

    def _extract_article(self, use_latest_snapshot):
        '''
        returns a goose article object
        '''
        params = {'browser_user_agent':"Mozilla/5.0"
                                       "(X11; Ubuntu; Linux x86_64; rv:52.0)"
                                       "Gecko/20100101 Firefox/52.0",
                  'strict': False,
                  'enable_image_fetching':False}
        gooser = Goose(params)
        try:
            article = gooser.extract(url=self.url)
        except Exception as scrape_error: #pylint:disable=broad-except
            warnings.warn(f"{type(scrape_error).__name__} occurred, falling back to WaybackMachine")
            if use_latest_snapshot:
                try:
                    snap_url = next(iter(WaybackMachine(self.url)))[0].url
                    article = gooser.extract(url=snap_url)
                except StopIteration:
                    warnings.warn("No snapshots available, returning empty...")
                    article = goose3.Article()
                except (ParseError, ParserError):
                    warnings.warn("Empty document, LXML cannot parse. Returning empty...")
                    article = goose3.Article()
                except WaybackMachineError:
                    warnings.warn("Bad response from archive.org, returning empty...")
                    article = goose3.Article()
                except ValueError:
                    warnings.warn("Various LXML/Goose error, returning empty...")
                    article = goose3.Article()
            else:
                snap_url = deque(WaybackMachine(self.url))[0][0].url
                article = gooser.extract(url=snap_url)
        return article

    def quotation_index(self):
        '''
        Placeholder
        '''
        return self.char_counts['\"'] / self.num_words

    def tokenize(self):
        '''
        returns tokenized and classified versions of text using nltk
        '''
        tokens = nltk.word_tokenize(self.text)
        classified_tokens = nltk.pos_tag(tokens)
        return classified_tokens

    def past_tense_index(self):
        '''
        returns the number of past tense verbs in the text
        '''
        past_index = sum([self.part_of_speech_count['VBD'],
                          self.part_of_speech_count['VBN']])
        return past_index / self.num_words

    def present_tense_index(self):
        '''
        returns the number of present tense verbs in the text over the
        '''
        present_index = sum([self.part_of_speech_count['VBP'],
                             self.part_of_speech_count['VBZ'],
                             self.part_of_speech_count['VBG']])
        return present_index / self.num_words

    def url_ending_index(self):
        '''
        returns 1 if url has reputable ending, 0 otherwise and None if the url is empty.
        '''
        reputable_endings = ['com', 'gov', 'org']
        ending = tldextract.extract(self.url).suffix
        return int(ending in reputable_endings) if self.url else None

    def dot_gov_ending_index(self):
        '''
        returns 1 if url ends in .gov, 0 otherwise and None if the url is empty.
        '''
        ending = tldextract.extract(self.url).suffix
        return int(ending == 'gov') if self.url else None

    def apa_index(self):
        '''
        returns number of apa errors
        '''
        checker = pyapa.ApaCheck()
        matches = checker.match(self.text)
        return len(matches)

    def today_index(self):
        '''
        returns the number of times "today" appears in the article text
        '''
        today_count = self.token_count['today']
        return today_count / self.num_words

    def should_index(self):
        '''
        returns the number of times "should" appears over the total number of words
        '''
        should_count = self.token_count['should']
        return should_count / self.num_words

    def opinion_index(self):
        '''
        returns 1 if 'opinion', 'editorial' or 'commentary' shows up in the url of an article
        '''
        return 1 if any(j in self.url
                        for j in ('opinion', 'commentary', 'editorial')) else 0
        #if 'opinion' in self.url or 'commentary' in self.url or 'editorial' in self.url:
        #    return 1
        #else:
        #    return 0

    def from_reputable_source_index(self):
        '''
        returns 1 if urls has reputable source in it
        '''
        return 1 if any(j in self.cleaned_url
                        for j in ArticleVector.reputable_news_sources) else 0
        #print(ArticleVector.reputable_news_sources)
        #for source in ArticleVector.reputable_news_sources:
        #    #print(source)
        #    if source in self.cleaned_url:
        #        return 1
        #return 0

    def all_caps_index(self):
        '''
        return the number of words in all caps
        in both the title and body divided by the total number of words
        '''
        caps_index = 0
        caps_index += sum([1 if word.isupper() else 0
                           for word in self.title.split(' ')])
        caps_index += sum([1 if word.isupper() else 0
                           for word in self.text.split(' ')])
        return caps_index / self.num_words

    def from_satire_source_index(self):
        '''
        returns 1 if link is from satire news source
        '''
        return 1 if any(j in self.cleaned_url
                        for j in ArticleVector.satire_news_sources) else 0
        #for source in ArticleVector.satire_news_sources:
        #    # only different because satire is full link
        #    if self.cleaned_url in source:
        #        return 1
        #return 0

    def from_unreputable_source_index(self):
        '''
        Placeholder
        '''
        for source in ArticleVector.unreputable_news_sources:
            #print(source)
            if source in self.url:
                return 1
        return 0

    def exclamation_index(self):
        '''
        returns number of exclamation points over total num of words.
        '''

        exclamation_index = self.char_counts['!']
        return exclamation_index / self.num_words

    def name_source_index(self):
        '''
        return the number of proper nouns in the text / total words
        '''
        num_prop_nouns = self.part_of_speech_count['NNP']
        return num_prop_nouns / self.num_words

    def interjection_index(self):
        '''
        return the number of interjections in the text / total words
        '''
        num_interjections = self.part_of_speech_count['UH']
        return num_interjections / self.num_words

    def you_index(self):
        '''
        return the number of times "you" shows up in the text / total words
        '''
        num_yous = self.token_count['you']
        #for word in self.text.split(' '):
        #    if word == 'you':
        #        num_yous += 1
        return num_yous / self.num_words

    #def ap_style_index(self):
    #    '''
    #    Placeholder
    #    '''
    #    checker = StyleChecker(self.text, self.title)
    #    return checker.total_errors

if __name__ == "__main__":
    TEST_URL = "http://www.wsj.com/news/us/"
    v = ArticleVector(url=TEST_URL)
