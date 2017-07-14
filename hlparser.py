import re
from bs4 import BeautifulSoup
import calendar
from fuzzywuzzy import fuzz
import multiprocessing
from multiprocessing import Pool
from functools import partial
from time import time
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from gensim import corpora, models, utils, similarities
import logging
import os
from os import listdir
from os.path import isfile, join
import urllib.request
import hashlib
import json
import numpy as np


class HighlightedParser:
    """
    Highlights text by pattern.
    """
    def __init__(self, lsi_models_path, verbose=True):
        # Define money pattern
        self.money_pattern = "\$([\d\.]+(\s+[mb]illion)?)"
        # Define date pattern
        months = "|".join(calendar.month_name[1:13])
        self.date_pattern = "((" + months + ")\s+\d{,2},\s+\d{,4})"
        # Define the ignore symbols
        self.ignoresymbol_pattern = "[\s-]+"
        self.newline = '<NEWLINE>'
        self.lsi_models_path = lsi_models_path if lsi_models_path else 'data'
        self.verbose = verbose
        self.sentences = 'sentences.json'
        self.model_files = ['corpus.mm', 'dictionary.dict', 'model.lsi', 'simIndex.index']

    def prepare_text(self, text):
        """
        Delete the noise symbols from the text.
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

    def find(self, pattern, text, ignore_date=False, ignore_money=False):
        """
        Search text by pattern.
        """
        pattern = self.prepare_text(pattern)
        searching_text_pattern = pattern.replace("(", "\(")
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

    def fuzzy_find_all(self, pattern, sentences, min_ratio=75, depth=2):
        """
        Find all sentences using fuzzy search.
        """

        if self.verbose:
            print("fuzzy searching...")

        t0 = time()

        pattern = self.prepare_text(pattern)
        pattern = pattern.replace("(", "\(")
        pattern = pattern.replace(")", "\)")

        if depth == 0:
            pattern_sentences = self.split_sentences(pattern)
            depth = len(pattern_sentences)

        # run multiprocessing calculation
        cpu_count = multiprocessing.cpu_count()
        pool = Pool(cpu_count)
        indices = range(0, len(sentences) - depth + 1)
        ratios = pool.map(partial(self.partial_fuzzy_searching,
                                  pattern=pattern,
                                  sentences=sentences,
                                  depth=depth),
                          indices)
        pool.terminate()

        if self.verbose:
            print("Elapsed time: {}(s)".format(time() - t0))

        best_ratios = [(i, r, s) for i, r, s in ratios if r > min_ratio]

        if len(best_ratios) > 0:
            best_ratios.sort(key=lambda r: r[1], reverse=True)
            return best_ratios
        else:
            return ""

    def fuzzy_find_one(self, pattern, text, min_ratio=75, depth=2, min_sentence_length=3):
        """
        Find one sentence using fuzzy search.
        :return:
        """
        best_ratios = self.fuzzy_find_all(pattern, text, min_ratio, depth, min_sentence_length)
        return best_ratios[0] if len(best_ratios) > 0 else ""

    # Latent Semantic Indexing
    def lsi_model_exist(self):
        onlyfiles = [f for f in listdir(self.lsi_models_path) if isfile(join(self.lsi_models_path, f))]
        return np.any(np.in1d(self.model_files, onlyfiles))

    def sentences_exist(self):
        return os.path.isfile(os.path.join(self.lsi_models_path, self.sentences))

    def create_sentences(self, url):
        """
        Determine by what tag the document is written. Extrarting content.
        :param url
        :return sentences: clear sentences of html doc
        """
        model_path = os.path.join(self.lsi_models_path, self.sentences)
        if self.sentences_exist():
            with open(model_path) as json_data:
                sentences = json.load(json_data)
            if self.verbose:
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
            with open(model_path, 'w') as fp:
                json.dump(sentences, fp)
            if self.verbose:
                print('content created')
        return sentences

    def create_lsi_model(self, content):
        """
        Build LSA Model
        :param content: Colection of documents for LS Indexing.
        :return: Corpus, Dictionary, lsi model and matrix of similarity
        """
        if self.lsi_model_exist():
            if self.verbose:
                print('model exist')
            dictionary = corpora.Dictionary.load(os.path.join(self.lsi_models_path, 'dictionary.dict'))
            corpus = corpora.MmCorpus(os.path.join(self.lsi_models_path, 'corpus.mm'))
            lsi = models.LsiModel.load(os.path.join(self.lsi_models_path, 'model.lsi'))
            index = similarities.MatrixSimilarity.load(os.path.join(self.lsi_models_path, 'simIndex.index'))
        else:
            stoplist = set(stopwords.words('english'))
            texts = []
            for document in content:
                texts.append([word for word in document.lower().split() if word not in stoplist])
            # remove words that appear only once
            frequency = defaultdict(int)
            for text in texts:
                for token in text:
                    frequency[token] += 1
            texts = [[token for token in text if frequency[token] > 1] for text in texts]
            dictionary = corpora.Dictionary(texts)
            dictionary.save(os.path.join(self.lsi_models_path, 'dictionary.dict'))  # store the dictionary, for future reference.
            corpus = [dictionary.doc2bow(text) for text in texts]
            corpora.MmCorpus.serialize((os.path.join(self.lsi_models_path, 'corpus.mm')), corpus)  # store to disk, for later use.
            corpus = corpora.MmCorpus(os.path.join(self.lsi_models_path, 'corpus.mm'))
            lsi = models.LsiModel(corpus, id2word=dictionary)
            lsi.save(os.path.join(self.lsi_models_path, 'model.lsi'))
            index = similarities.MatrixSimilarity(lsi[corpus])
            index.save(os.path.join(self.lsi_models_path, 'simIndex.index'))
            if self.verbose:
                print('model created')
        return dictionary, corpus, lsi, index

    @staticmethod
    def answer(sims, min_ratio, num_of_answ):
        answer = [counter for counter in sims if counter[1] > min_ratio]
        return answer[0:num_of_answ]

    def find_similar(dictionary, lsi, index, request):
        vec_bow = dictionary.doc2bow(request.lower().split())
        vec_lsi = lsi[vec_bow]
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        return sims

    def create_request_model(self, request):
        """
        :param request: 
        :return: dictionary, lsi model and matrix of similarity for request
        """
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
        if len(dictionary) == 0:
            return 0
        corpus = [dictionary.doc2bow(text) for text in texts]
        lsi = models.LsiModel(corpus=corpus, id2word=dictionary, num_topics=100)
        index = similarities.MatrixSimilarity(lsi[corpus])
        return dictionary, lsi, index

    def group_answer(dictionary, lsi, index, answer, sentences, request_len, min_ratio):
        """
        :param lsi: lsi model
        :param index: matrix of similarity
        :param answer:
        :param sentences:
        :param request_len:
        :param min_ratio:
        :return: dictionary that consist sentences grouped in response and their level of similarity
        """
        new_answ = sorted(answer)
        hightlighted_texts = [sentences[new_answ[0][0]]]
        current = 0
        for i in range(1, len(new_answ)):
            if new_answ[i][0] == new_answ[i - 1][0]:
                continue
            if new_answ[i][0] == new_answ[i - 1][0] + 1:
                hightlighted_texts[current] += ' ' + sentences[new_answ[i][0]]
                continue
            hightlighted_texts.append(sentences[new_answ[i][0]])
            current += 1
        answ = {}
        for i in range(0, len(hightlighted_texts)):
            ratio = 0
            hightlighted_texts_tosent = nltk.sent_tokenize(hightlighted_texts[i])
            if len(hightlighted_texts_tosent) > request_len:
                ratio = HighlightedParser.find_similar(dictionary, lsi, index, hightlighted_texts[i])[0][1]
            else:
                for j in range(0, len(hightlighted_texts_tosent)):
                    ratio += HighlightedParser.find_similar(dictionary, lsi, index,
                                                            hightlighted_texts_tosent[j])[0][1]
                ratio /= request_len
            ratio *= 100
            if ratio < min_ratio:
                continue
            else:
                answ.update({hightlighted_texts[i]: str(ratio) + '%'})
        return answ

    def lsi_search(self, sentences, request, min_ratio, num_of_answ):
        """
        :param sentences:
        :param request:
        :param min_ratio:
        :param num_of_answ:
        :return: a dictionary containing the highlighted text and its level of similarity
        """
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        request_sent = nltk.sent_tokenize(request)
        model = self.create_lsi_model(sentences)
        answ = {}
        if len(request_sent) == 1:
            sims = HighlightedParser.find_similar(model[0], model[2], model[3], request)
            sims = self.answer(sims, min_ratio/100, num_of_answ)
            for sim in sims:
                answ.update({sentences[sim[0]] : str(sim[1]*100) + '%'})
        else:
            request_model = self.create_request_model(request)
            request_dictionary = request_model[0]
            request_lsi = request_model[1]
            request_index = request_model[2]
            hightlighted_texts = []
            for i in range(0, len(request_sent)):
                sims = HighlightedParser.find_similar(model[0], model[2], model[3], request_sent[i])
                best_sims = HighlightedParser.answer(sims, min_ratio/100, num_of_answ)
                for top_sim in best_sims:
                    hightlighted_texts.append(top_sim)
                answ = HighlightedParser.group_answer(request_dictionary, request_lsi, request_index,
                                                      hightlighted_texts, sentences, len(request_sent),
                                                      min_ratio)
        return answ
