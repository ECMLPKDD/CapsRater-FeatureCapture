import os
import re
import pickle

import itertools
import numpy as np
import pandas as pd
import grammar_check
from rapidfuzz import fuzz, process
from spellchecker import SpellChecker
from nltk.tokenize import RegexpTokenizer
import nltk.word_tokenize

from main import args


def num_words(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    count = len(tokens)
    return count


def sub_chars(string):
    """
    Strips illegal characters from a string.  Used to sanitize input essays.
    Removes all non-punctuation, digit, or letter characters.
    Returns sanitized string.
    string - string
    """
    sub_pat = r"[^A-Za-z\.\?!,';:]"
    char_pat = r"\."
    com_pat = r","
    ques_pat = r"\?"
    excl_pat = r"!"
    sem_pat = r";"
    col_pat = r":"
    whitespace_pat = r"\s{1,}"

    #Replace text.  Ordering is very important!
    nstring = re.sub(sub_pat, " ", string)
    nstring = re.sub(char_pat," .", nstring)
    nstring = re.sub(com_pat, " ,", nstring)
    nstring = re.sub(ques_pat, " ?", nstring)
    nstring = re.sub(excl_pat, " !", nstring)
    nstring = re.sub(sem_pat, " ;", nstring)
    nstring = re.sub(col_pat, " :", nstring)
    nstring = re.sub(whitespace_pat, " ", nstring)

    return nstring


def ngrams(tokens, min_n, max_n):
    """
    Generates ngrams(word sequences of fixed length) from an input token sequence.
    tokens is a list of words.
    min_n is the minimum length of an ngram to return.
    max_n is the maximum length of an ngram to return.
    returns a list of ngrams (words separated by a space)
    """
    all_ngrams = list()
    n_tokens = len(tokens)
    for i in range(n_tokens):
        for j in range(i + min_n, min(n_tokens, i + max_n) + 1):
            all_ngrams.append(" ".join(tokens[i:j]))
    return all_ngrams


def regenerate_good_tokens(string):
    """
    Given an input string, part of speech tags the string, then generates a list of
    ngrams that appear in the string.
    Used to define grammatically correct part of speech tag sequences.
    Returns a list of part of speech tag sequences.
    """
    toks = nltk.word_tokenize(string)
    pos_string = nltk.pos_tag(toks)
    pos_seq = [tag[1] for tag in pos_string]
    pos_ngrams = ngrams(pos_seq, 2, 4)
    # sel_pos_ngrams = f7()
    seen = set()
    seen_add = seen.add
    return [x for x in pos_ngrams if x not in seen and not seen_add(x)]


def get_good_pos_ngrams(train):
    """
    Gets a set of gramatically correct part of speech sequences from an input file called essaycorpus.txt
    Returns the set and caches the file
    """
    essay_corpus = sub_chars(' '.join(train))
    good_pos_ngrams = regenerate_good_tokens(essay_corpus)
    pickle.dump(good_pos_ngrams, open(os.path.join(DATA_PATH, "good_ngram.pkl"), 'wb'))
    return set(good_pos_ngrams)


def get_num_spelling_error(text):
    spell = SpellChecker()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    misspelled = spell.unknown(tokens)
    return len(misspelled)


def get_num_grammar_error(text):
    tool = grammar_check.LanguageTool('en-GB')
    matches = tool.check(text)
    return len(matches)


def get_sim_bet_prompt_n_resp(text, essay_set):
    df = pd.read_excel("./data/essay_set_prompts.xlsx")
    prompt = df[df["essay_set"] == essay_set]["prompt"].tolist()
    match_sent, score, index = zip(*process.extract(' '.join(prompt), text, scorer=fuzz.partial_ratio, limit=1))
    return score


def get_sim_bet_responses(text):
    df = pd.DataFrame()
    for p1, p2 in itertools.combinations(text, 2):
        df = df.append(pd.DataFrame({"1": [p1], "2": [p2]}))
    scores = []
    for i in range(df.shape[0]):
        match_sent, score, index = zip(*process.extract(text[i], df["2"].tolist(), scorer=fuzz.partial_ratio, limit=1))
        scores.append(score)
    return max(scores)


def get_all_features(data):
    feature_1 = num_words(data)
    feature_2a = ' '.join(data).count('!')
    feature_2b = " ".join(data).count('?')
    feature_3 = get_good_pos_ngrams(data)
    feature_4 = get_num_spelling_error(data)
    feature_5 = get_num_grammar_error(data)
    feature_6 = get_sim_bet_prompt_n_resp(data, args.prompt_nbr)
    feature_7 = get_sim_bet_responses(data)
    return np.array((feature_1, feature_2a, feature_2b, feature_3, feature_4, feature_5, feature_6, feature_7))
