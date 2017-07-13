import re
from bs4 import BeautifulSoup
import calendar
import requests
from fuzzywuzzy import fuzz
import multiprocessing
from multiprocessing import Pool
from functools import partial
from time import time
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from gensim import corpora, models, utils, similarities
import Pyro4
import logging
import os
from os import listdir
from os.path import isfile, join
import urllib.request
import hashlib
import json


class MyCorpus(object):
    def __iter__(self):
        for line in open('mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())


class HighlightedParser:
    """

    """

    def __init__(self):
        # Define money pattern
        self.money_pattern = "\$([\d\.]+(\s+[mb]illion)?)"

        # Define date pattern
        months = "|".join(calendar.month_name[1:13])
        self.date_pattern = "((" + months + ")\s+\d{,2},\s+\d{,4})"

        # Define the ignore symbols
        self.ignoresymbol_pattern = "[\s-]+"

        self.newline = '<NEWLINE>'

    def model_exist(name, mypath):
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        if name in onlyfiles:
            return True
        else:
            return False

    @staticmethod
    def create_contents(url, path, name):

        """
        Determine by what tag the document is written. Extrarting content.
        """

        name += '.json'
        if HighlightedParser.model_exist(name, path):
            with open(path + '/' + name) as json_data:
                sentences = json.load(json_data)
            print('content exist')
        else:
            html = urllib.request.urlopen(url).read()
            soup = BeautifulSoup(html, 'html.parser')
            text1 = soup.findAll(['div'])
            text2 = soup.findAll(['p'])
            if len(text1) < len(text2):
                text = text2
            else:
                text = text1
            need_del = []
            for i in range(0, len(text)):
                if len(text[i].text.strip()) == 0:
                    need_del.append(i)
            for index in sorted(need_del, reverse=True):
                del text[index]
            sentences = []
            for i in range(0, len(text)):
                contents = text[i].text
                contents = re.sub('[^\x00-\x7F]+', ' ', contents)
                contents = re.sub(' +', ' ', contents)
                contents = re.sub('\n{2,}', ' ', contents)
                if text[i].table:
                    sentences.append(contents)
                else:
                    sent = nltk.sent_tokenize(contents)
                    for j in range(0, len(sent)):
                        if len(nltk.word_tokenize(sent[j])) < 3:
                            continue
                        sentences.append(sent[j])
            with open(path + '/' + name, 'w') as fp:
                json.dump(sentences, fp)
            print('content created')
        return sentences

    def answer(contents, sims, min_ratio, num_of_answ):
        answer = {}
        try:
            for j in range(0, num_of_answ):
                if sims[j][1] < min_ratio / 100:
                    break
                answer.update({contents[sims[j][0]]: str(sims[j][1] * 100) + ' %'})
        except:
            print("Unexpected error:", sys.exc_info()[0])
        return answer

    def find_similar(lsi, doc, corpus, dictionary, path, name):
        name += "simIndex.index"
        if HighlightedParser.model_exist(name, path):
            index = similarities.MatrixSimilarity.load(path + '/' + name)
            print('matrix similarity exist')
        else:
            index = similarities.MatrixSimilarity(lsi[corpus])
            index.save(path + '/' + name)
            print('matrix similarity created')
        vec_bow = dictionary.doc2bow(doc.lower().split())
        vec_lsi = lsi[vec_bow]
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        return sims

    def create_model(documents, path, name):
        if HighlightedParser.model_exist(name + 'deerwester.mm', path):
            print('model exist')
            dictionary = corpora.Dictionary.load(path + '/' + name + 'dictionary.dict')
            corpus = corpora.MmCorpus(path + '/' + name + 'deerwester.mm')
        else:
            stoplist = set(stopwords.words('english'))
            texts = []
            for document in documents:
                texts.append([word for word in document.lower().split() if word not in stoplist])
            # remove words that appear only once
            frequency = defaultdict(int)
            for text in texts:
                for token in text:
                    frequency[token] += 1
            texts = [[token for token in text if frequency[token] > 1]
                     for text in texts]
            dictionary = corpora.Dictionary(texts)
            dictionary.save(path + '/' + name + 'dictionary.dict')  # store the dictionary, for future reference.
            corpus = [dictionary.doc2bow(text) for text in texts]
            corpora.MmCorpus.serialize(path + '/' + name + 'deerwester.mm', corpus)  # store to disk, for later use.
            corpus = corpora.MmCorpus(path + '/' + name + 'deerwester.mm')
            print('model created')
        return corpus, dictionary

    def create_answer(sims, min_ratio, num_of_answ):
        answer = []
        for i in range(0, num_of_answ):
            if sims[i][1] < min_ratio / 100:
                break
            else:
                answer.append(sims[i])
        return answer

    def get_similar_documents(dictionary, lsi, index, request):
        vec_bow = dictionary.doc2bow(request.lower().split())
        vec_lsi = lsi[vec_bow]
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        return sims[0][1]

    def create_request_model(request, name_request, path):
        if HighlightedParser.model_exist(name_request + 'request.index', path):
            dictionary = corpora.Dictionary.load(path + '/' + name_request + 'request.dict')
            lsi = models.LsiModel.load(path + '/' + name_request + 'request.lsi')
            index = similarities.MatrixSimilarity.load(path + '/' + name_request + 'request.index')
            print('request already exist')
        else:
            documents = nltk.sent_tokenize(request)
            stoplist = set(stopwords.words('english'))
            texts = [[word for word in document.lower().split() if word not in stoplist]
                     for document in documents]
            # remove words that appear only once
            frequency = defaultdict(int)
            for text in texts:
                for token in text:
                    frequency[token] += 1
            texts = [[token for token in text if frequency[token] > 1]
                     for text in texts]
            dictionary = corpora.Dictionary(texts)
            dictionary.save(path + '/' + name_request + 'request.dict')
            if len(dictionary) == 0:
                return 0
            corpus = [dictionary.doc2bow(text) for text in texts]
            lsi = models.LsiModel(corpus=corpus, id2word=dictionary, num_topics=100)
            lsi.save(path + '/' + name_request + 'request.lsi')
            index = similarities.MatrixSimilarity(lsi[corpus])
            index.save(path + '/' + name_request + 'request.index')
            print('request created')
        return dictionary, lsi, index

    def group_answer(dictionary, lsi, index, answer, sentences, request, request_len, min_ratio):
        new_answ = sorted(answer)
        hightlighted_texts = [sentences[new_answ[0][0]]]
        current = 0
        for i in range(1, len(new_answ)):
            if new_answ[i][0] == new_answ[i - 1][0]:
                continue
            if new_answ[i][0] == new_answ[i - 1][0] + 1:
                hightlighted_texts[current] = hightlighted_texts[current] + ' ' + sentences[new_answ[i][0]]
                continue
            hightlighted_texts.append(sentences[new_answ[i][0]])
            current += 1
        answ = {}
        for i in range(0, len(hightlighted_texts)):
            ratio = 0
            hightlighted_texts_tosent = nltk.sent_tokenize(hightlighted_texts[i])
            if len(hightlighted_texts_tosent) > request_len:
                ratio = HighlightedParser.get_similar_documents(dictionary, lsi, index, hightlighted_texts[i])
            else:
                for j in range(0, len(hightlighted_texts_tosent)):
                    ratio = ratio + HighlightedParser.get_similar_documents(dictionary, lsi, index,
                                                                            hightlighted_texts_tosent[j])
                ratio = ratio / request_len
            ratio *= 100
            if ratio < min_ratio:
                continue
            else:
                answ.update({hightlighted_texts[i]: str(ratio) + '%'})
        return answ

    @staticmethod
    def lsi_for_finding_request(contents, name, path, request, min_ratio, num_of_answ):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = HighlightedParser.create_model(contents, path, name)
        corpus = model[0]
        id2word = model[1]
        dictionary = model[1]
        if HighlightedParser.model_exist(name + 'model.lsi', path):
            lsi = models.LsiModel.load(path + '/' + name + 'model.lsi')
            print('lsi loaded')
        else:
            lsi = models.LsiModel(corpus, id2word=id2word)
            lsi.save(path + '/' + name + 'model.lsi')
        doc = request
        request_sent = nltk.sent_tokenize(doc)
        name_request = hashlib.md5(request.encode())
        name_request = str(name_request.hexdigest())
        try:
            os.mkdir(path + '/history')
            print('history created')
        except:
            print('history exist')
        path_request = path + '/history'
        if len(request_sent) == 1:
            sims = HighlightedParser.find_similar(lsi, request, corpus, dictionary, path, name)
            answ = HighlightedParser.answer(contents, sims, min_ratio, num_of_answ)
        else:
            request_model = HighlightedParser.create_request_model(request, name_request, path_request)
            request_dictionary = request_model[0]
            request_lsi = request_model[1]
            request_index = request_model[2]
            answ = []
            hightlighted_texts = []
            for i in range(0, len(request_sent)):
                sims = HighlightedParser.find_similar(lsi, request_sent[i], corpus, dictionary, path, name)
                best_sims = HighlightedParser.create_answer(sims, min_ratio, num_of_answ)
                for top_sim in best_sims:
                    hightlighted_texts.append(top_sim)
                answ = HighlightedParser.group_answer(request_dictionary, request_lsi, request_index,
                                                      hightlighted_texts, contents, doc, len(request_sent), min_ratio)
        return answ

    def prepare_text(self, text):
        """
        Delete the noise symbols from the text.
        :param text: [str] Source text.
        :return: [str] Clear text.
        """
        s = text
        s = re.sub("[\u000A\u0085\u2028\u2029]", self.newline, s)
        s = re.sub("{}\s*{}".format(self.newline, self.newline), ".\n", s)
        s = re.sub(self.newline, " ", s)
        s = re.sub("\n{2,}", ".\n", s)
        s = re.sub("\.+", ".", s)
        s = re.sub("\s+", " ", s)
        s = re.sub("\n", " ", s)

        return s.strip()

    def find(self, searching_text, text, ignore_date=False, ignore_money=False):
        """

        :param searching_text:
        :param text:
        :param ignore_date:
        :param ignore_money:
        :return:
        """

        searching_text = self.prepare_text(searching_text)
        searching_text_pattern = searching_text.replace("(", "\(")
        searching_text_pattern = searching_text_pattern.replace(")", "\)")
        text = ' '.join(text)
        if ignore_money:
            searching_text_pattern = re.sub(self.money_pattern, self.money_pattern, searching_text_pattern)
        if ignore_date:
            searching_text_pattern = re.sub(self.date_pattern, self.date_pattern, searching_text_pattern)
        searching_text_pattern = re.sub(' ', self.ignoresymbol_pattern, searching_text_pattern)
        searching_text_pattern = "(?P<target>{})".format(searching_text_pattern)

        match = re.search(searching_text_pattern, text, flags=re.IGNORECASE)

        return match.group('target') if match is not None else ""

    @staticmethod
    def split_sentences(text, min_sentence_length=3):
        min_sentence_length = max(1, min_sentence_length)
        sentences = re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
        return [s.strip() for s in sentences if len(s) >= min_sentence_length]

    @staticmethod
    def partial_ratio(pattern, sentence):
        return fuzz.partial_ratio(pattern, sentence)

    @staticmethod
    def partial_fuzzy_searching(index, pattern, sentences, depth):
        s = ' '.join(sentences[index:index + depth])
        ratio = fuzz.partial_ratio(s, pattern)
        return index, ratio, s

    def fuzzy_find_all(self, pattern, sentences, min_ratio=75, depth=2, min_sentence_length=2):
        """

        :param sentences:
        :param pattern:
        :param text:
        :param min_ratio:
        :param depth:
        :param min_sentence_length:
        :return:
        """

        print("fuzzy searching...")

        t0 = time()

        pattern = self.prepare_text(pattern)
        pattern = pattern.replace("(", "\(")
        pattern = pattern.replace(")", "\)")

        if depth == 0:
            pattern_sentences = self.split_sentences(pattern)
            depth = len(pattern_sentences)

        'run multiprocessing calculation'
        cpu_count = multiprocessing.cpu_count()
        pool = Pool(cpu_count)
        indices = range(0, len(sentences) - depth + 1)
        ratios = pool.map(partial(self.partial_fuzzy_searching,
                                  pattern=pattern,
                                  sentences=sentences,
                                  depth=depth),
                          indices)
        pool.terminate()

        print("Elapsed time: {}(s)".format(time() - t0))

        best_ratios = [(i, r, s) for i, r, s in ratios if r > min_ratio]

        if len(best_ratios) > 0:
            best_ratios.sort(key=lambda r: r[1], reverse=True)
            return best_ratios
        else:
            return ""

    def fuzzy_find_one(self, pattern, text, min_ratio=75, depth=2, min_sentence_length=3):
        best_ratios = self.fuzzy_find_all(pattern, text, min_ratio, depth, min_sentence_length)
        if len(best_ratios) > 0:
            return best_ratios[0]
        else:
            return ""


if __name__ == "__main__":
    # # Add console arguments
    # argparser = argparse.ArgumentParser(description="Highlighted Tool")
    # argparser.add_argument("--url", type=str, default="", help="URL...")
    # argparser.add_argument("--highlighted_url", type=str, default="", help="URL for highlighting")
    # argparser.add_argument("--highlighted_text", type=str, default="", help="Text for highlighting")
    # argparser.add_argument("--ignore_money", type=bool, default=False, help="Ignoring money?")
    # argparser.add_argument("--ignore_date", type=bool, default=False, help="Ignoring dates?")
    # args = argparser.parse_args()
    #
    # old_filing_html = requests.get(args.url)
    # new_filing_html = requests.get(args.highlighted_url)
    # hp = HighlightedParser()
    #
    # old_filing_text = hp.get_processed_html(old_filing_html.text)
    # new_filing_text = hp.get_processed_html(new_filing_html.text)
    #
    # matches = hp.find_text(args.highlighted_text, new_filing_text, ignore_money=args.ignore_money, ignore_date=args.ignore_date)
    # if len(matches) > 0:
    #     print(matches[0])

    # E. g.
    # python3 hlparser.py --url="https://www.sec.gov/Archives/edgar/data/789019/000119312516742796/d245252d10q.htm" --searching_url="https://www.sec.gov/Archives/edgar/data/789019/000156459017000654/msft-10q_20161231.htm" --highlighted_text="Basic earnings per share (“EPS”) is computed based on the weighted average number of shares of common"

    old_filing_url = "https://www.sec.gov/Archives/edgar/data/789019/000119312516742796/d245252d10q.htm"
    new_filing_url = "https://www.sec.gov/Archives/edgar/data/789019/000156459017000654/msft-10q_20161231.htm"

    print("Loading url...")
    old_filing_html = requests.get(old_filing_url)
    new_filing_html = requests.get(new_filing_url)

    perfectly_searching_text = """
    Basic earnings per share (“EPS”) is computed based on the weighted average number of shares of common
    stock outstanding during the period. Diluted EPS is computed based on the weighted average number of
    shares of common stock plus the effect of dilutive potential common shares outstanding during the period
    using the treasury stock method. Dilutive potential common shares include outstanding stock options and
    stock awards.
    """

    mismatch_data_money_searching_text_1 = """
    Foreign currency risks related to certain non-U.S. dollar denominated securities are hedged using
    foreign exchange forward contracts that are designated as fair value hedging instruments.
    As of September 30, 2016 and June 30, 2016, the total notional amounts of these foreign
    exchange contracts sold were $5.2 billion and $5.3 billion, respectively.
    """

    mismatch_data_money_searching_text_1 = """
    Foreign currency risks related to certain non-U.S. dollar denominated securities are hedged using
    foreign exchange forward contracts that are designated as fair value hedging instruments.
    As of December 12, 2015 and June 30, 2017, the total notional amounts of these foreign
    exchange contracts sold were $5.2 billion and $5.3 billion, respectively.
    """

    'foreign currency risks related to certain non-u.s. dollar denominated securities are hedged using ' \
    'foreign exchange forward contracts that are designated as fair value hedging instruments. as of december 31, ' \
    '2016 and june 30, 2016, the total notional amounts of these foreign exchange contracts sold were $4.5 billion and'

    mismatch_data_money_searching_text_2 = """
    Securities held in our equity and other investments portfolio are subject to market price risk.
    Market price risk is managed relative to broad-based global and domestic equity indices using certain
    convertible preferred investments, options, futures, and swap contracts not designated as hedging instruments.
    From time to time, to hedge our price risk, we may use and designate equity derivatives as hedging instruments,
    including puts, calls, swaps, and forwards. As of December 31, 2016, the total notional amounts of equity contracts
    purchased and sold for managing market price risk were $1.2 billion and $1.8 billion, respectively, of which $567 million
    and $764 million, respectively, were designated as hedging instruments. As of June 30, 2016, the total notional amounts of
    equity contracts purchased and sold for managing market price risk were $1.3 billion and $2.2 billion, respectively,
    of which $737 million and $986 million, respectively, were designated as hedging instruments.
    """

    hp = HighlightedParser()

    old_filing_text = hp.extract_html_text(old_filing_html.text)
    new_filing_text = hp.extract_html_text(new_filing_html.text)

    matches_1 = hp.find(perfectly_searching_text, new_filing_text)
    matches_2 = hp.find(mismatch_data_money_searching_text_1, new_filing_text, ignore_date=True, ignore_money=True)
    matches_3 = hp.find(mismatch_data_money_searching_text_2, new_filing_text, ignore_date=True, ignore_money=True)
    fuzzy_match = hp.fuzzy_find_one(mismatch_data_money_searching_text_1, new_filing_text)

    print("Mathces_1: {}\n".format(matches_1))
    print("Mathces_2: {}\n".format(matches_2))
    print("Mathces_3: {}\n".format(matches_3))
    print("fuzzy matching: {}".format(fuzzy_match))

