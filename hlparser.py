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
    def lsi_model_exist(self, model_name):
        onlyfiles = [f for f in listdir(self.lsi_models_path) if isfile(join(self.lsi_models_path, f))]
        return bool(model_name in onlyfiles)

    def create_contents(self, url, model_name):
        """
        Determine by what tag the document is written. Extrarting content.
        """
        model_name += '.json'
        if HighlightedParser.lsi_model_exist(model_name, self.lsi_models_path):
            with open(self.lsi_models_path + '/' + model_name) as json_data:
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
            with open(self.lsi_models_path + '/' + model_name, 'w') as fp:
                json.dump(sentences, fp)
            if self.verbose:
                print('content created')
        return sentences

    def create_lsi_model(self, content, model_name):
        """
        Build LSA Model
        :param content: Colection of documents for LS Indexing.
        :param model_name: Model name.
        :return: Corpus and Dictionary
        """
        model_path = os.path.join(self.lsi_models_path, model_name)
        if self.lsi_model_exist(model_name + 'deerwester.mm'):
            if self.verbose:
                print('model exist')
            dictionary = corpora.Dictionary.load(model_path + 'dictionary.dict')
            corpus = corpora.MmCorpus(model_path + 'deerwester.mm')
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
            dictionary.save(model_path + 'dictionary.dict')  # store the dictionary, for future reference.
            corpus = [dictionary.doc2bow(text) for text in texts]
            corpora.MmCorpus.serialize(model_path + 'deerwester.mm', corpus)  # store to disk, for later use.
            corpus = corpora.MmCorpus(model_path + 'deerwester.mm')
            if self.verbose:
                print('model created')
        return corpus, dictionary

    def find_similar(self, lsi, doc, corpus, dictionary, model_name):
        model_name += "simIndex.index"
        model_path = os.path.join(self.lsi_models_path, model_name)
        if self.lsi_model_exist(model_name):
            index = similarities.MatrixSimilarity.load(model_path)
            if self.verbose:
                print('matrix similarity was loaded successfully')
        else:
            index = similarities.MatrixSimilarity(lsi[corpus])
            index.save(model_path)
            if self.verbose:
                print('matrix similarity was created successfully')
        vec_bow = dictionary.doc2bow(doc.lower().split())
        vec_lsi = lsi[vec_bow]
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        return sims

    @staticmethod
    def answer(contents, sims, min_ratio, num_of_answ):
        """???"""
        answer = {}
        try:
            for j in range(0, num_of_answ):
                if sims[j][1] < min_ratio:
                    break
                answer.update({contents[sims[j][0]]: str(sims[j][1])})
        except:
            print("Unexpected error:", sys.exc_info()[0])
        return answer

    def create_answer(sims, min_ratio, num_of_answ):
        answer = []
        for i in range(0, num_of_answ):
            if sims[i][1] < min_ratio:
                break
            else:
                answer.append(sims[i])
        return answer

    @staticmethod
    def get_similar_documents(dictionary, lsi, index, request):
        vec_bow = dictionary.doc2bow(request.lower().split())
        vec_lsi = lsi[vec_bow]
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        return sims[0][1]

    # def create_request_model(request, name_request, path):
    #     if HighlightedParser.lsi_model_exist(name_request + 'request.index', path):
    #         dictionary = corpora.Dictionary.load(path + '/' + name_request + 'request.dict')
    #         lsi = models.LsiModel.load(path + '/' + name_request + 'request.lsi')
    #         index = similarities.MatrixSimilarity.load(path + '/' + name_request + 'request.index')
    #         print('request already exist')
    #     else:
    #         documents = nltk.sent_tokenize(request)
    #         stoplist = set(stopwords.words('english'))
    #         texts = [[word for word in document.lower().split() if word not in stoplist]
    #                  for document in documents]
    #         # remove words that appear only once
    #         frequency = defaultdict(int)
    #         for text in texts:
    #             for token in text:
    #                 frequency[token] += 1
    #         texts = [[token for token in text if frequency[token] > 1]
    #                  for text in texts]
    #         dictionary = corpora.Dictionary(texts)
    #         dictionary.save(path + '/' + name_request + 'request.dict')
    #         if len(dictionary) == 0:
    #             return 0
    #         corpus = [dictionary.doc2bow(text) for text in texts]
    #         lsi = models.LsiModel(corpus=corpus, id2word=dictionary, num_topics=100)
    #         lsi.save(path + '/' + name_request + 'request.lsi')
    #         index = similarities.MatrixSimilarity(lsi[corpus])
    #         index.save(path + '/' + name_request + 'request.index')
    #         print('request created')
    #     return dictionary, lsi, index

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

    def lsi_search(self, content, model_name, request, min_ratio, num_of_answ):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        model_path = os.path.join(self.lsi_models_path, model_name)
        model = self.create_lsi_model(content, model_name)
        corpus = model[0]
        id2word = model[1]
        dictionary = model[1]
        if self.lsi_model_exist(model_name + 'model.lsi'):
            lsi = models.LsiModel.load(model_path + 'model.lsi')
            print('lsi loaded')
        else:
            lsi = models.LsiModel(corpus, id2word=id2word)
            lsi.save(model_path + 'model.lsi')
        doc = request
        request_sent = nltk.sent_tokenize(doc)
        name_request = hashlib.md5(request.encode())
        name_request = str(name_request.hexdigest())
        try:
            os.mkdir(self.lsi_models_path + '/history')
            if self.verbose:
                print('history created')
        except:
            print('history exist')

        path_request = self.lsi_models_path + '/history'
        if len(request_sent) == 1:
            sims = self.find_similar(lsi, request, corpus, dictionary, model_name)
            answ = self.answer(content, sims, min_ratio, num_of_answ)
        else:
            request_model = self.create_request_model(request, name_request, path_request)
            request_dictionary = request_model[0]
            request_lsi = request_model[1]
            request_index = request_model[2]
            answ = []
            hightlighted_texts = []
            for i in range(0, len(request_sent)):
                sims = HighlightedParser.find_similar(lsi, request_sent[i], corpus, dictionary, self.lsi_models_path, model_name)
                best_sims = HighlightedParser.create_answer(sims, min_ratio, num_of_answ)
                for top_sim in best_sims:
                    hightlighted_texts.append(top_sim)
                answ = HighlightedParser.group_answer(request_dictionary, request_lsi, request_index,
                                                      hightlighted_texts, content, doc, len(request_sent), min_ratio)
        return answ
