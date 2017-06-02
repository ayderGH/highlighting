import re
from bs4 import BeautifulSoup
import calendar
import argparse
import requests
from fuzzywuzzy import fuzz
import multiprocessing
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from time import time


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

    @staticmethod
    def extract_html_text(html):
        """
        Represents the HTML text like as the sequence of words and punctuations.
        :return: [str]
        """
        soup = BeautifulSoup(html, 'html5lib')
        pretty_html = soup.prettify(formatter=lambda s: s.replace('\xa0', ' '))
        soup = BeautifulSoup(pretty_html, 'html5lib')
        return soup.get_text()

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
        text = self.prepare_text(text)

        searching_text_pattern = searching_text.replace("(", "\(")
        searching_text_pattern = searching_text_pattern.replace(")", "\)")
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

    def fuzzy_find_all(self, pattern, text, min_ratio=75, depth=2, min_sentence_length=2):
        """

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
        text = self.prepare_text(text)

        pattern = pattern.replace("(", "\(")
        pattern = pattern.replace(")", "\)")

        if depth == 0:
            pattern_sentences = self.split_sentences(pattern)
            depth = len(pattern_sentences)

        sentences = self.split_sentences(text)
        sentences = [s for s in sentences if len(s.split(' ')) > min_sentence_length]

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
