#!/usr/bin/env python

'''
    File name: characterize.py
    Author: François Klein
    Date created: 2021-09-29
    Python Version: 3.8
    
    Find the characteristic terms in a given collection (text, ...), compared to a reference collection.
'''


import re
import pandas as pd

from unidecode import unidecode
from collections import Counter
from math import log
from matplotlib import pyplot as plt



### PREPROCESSING ###

def clean_text(string):
    s = re.sub('[\W\d]', ' ', string)       # text characters only
    s = s.lower()                           # lowercase
    s = unidecode(s)                        # ascii characters only
    return s



def read_rule(filename):
    matching = {}
    with open(filename, 'r') as file:
        for line in file:
            terms = line.strip().split(',')
            replacement_term = terms.pop(0)
            new_matchings = {term: replacement_term for term in terms}
            matching = {**matching, **new_matchings}
    return matching



class Levenshtein():
    """
    Compute the Levenshtein distance between 2 strings.
    Implemented with "hidden" dynamic programming:
    it seems simply recursive but there is a cache.
    """
    def __init__(self):
        self.sub_distances = {}
    
    def result(self, s1, s2, res):
        self.sub_distances[(s1, s2)] = res
        return res
    
    def distance(self, s1, s2):
        if (s1, s2) in self.sub_distances:
            return self.sub_distances[(s1, s2)]
        
        if not s1 or not s2:
            return self.result(s1, s2, len(s1) + len(s2))
        
        if s1[0] == s2[0]:
            return self.result(s1, s2, self.distance(s1[1:], s2[1:]))
        
        a = self.distance(s1[1:], s2)
        b = self.distance(s1, s2[1:])
        c = self.distance(s1[1:], s2[1:])
        return self.result(s1, s2, 1 + min([a, b, c]))
    


class DataSource():
    
    def __init__(self):
        """
        Read, organize and clean the original data.
        """
        file_name = '244400404_menus-cantines-nantes-2011-2019.csv'

        self.meals_df = pd.read_csv(file_name, sep=';')
        
        self.meals_df['Date'] = pd.to_datetime(self.meals_df.Date)
        
        _clean_text = lambda row: clean_text(row.Plat)
        self.meals_df['meals'] = self.meals_df.apply(_clean_text, axis=1)
    
    
    def replace(self, rule_names=[]):
        """
        Apply replacement rules for the text in the dataframe.
        
        The rule names are the names of csv files containing the matching for:
        * frequent orthographic errors
        * orthographic variations (lemmatization / stemming)
        * synonyms and abbreviations
        * named entities (an underscore is used to link distinct terms)
        * stop words (replaced by an empty string)
        * categories (list of sub-elements)
        
        The rules order is relevant!
        """
        rules = self._read_rules(rule_names)
        replacement_function = self._build_replacement_function(rules)
        _replacement_function = lambda row: replacement_function(row.meals)
        self.meals_df['meals'] = self.meals_df.apply(_replacement_function, axis=1)
        
    
    def _read_rules(self, rule_names):
        rules = {}
        for rule_name in rule_names:
            matching = read_rule(rule_name + '.csv')
            rules = {**rules, **matching}
        return rules
        
    
    def _build_replacement_function(self, rules):
        def replacement_function(string):
            # A simple string.replace(*rule) (for rule in rules) may replace parts of words
            original_terms = string.split()
            replaced_terms = []
            for term in original_terms:
                for origin, target in rules.items():
                    # ugly: should have been "while term in rules", but avoids infinite loops
                    if term == origin:
                        term = target
                replaced_terms.append(term)
            return ' '.join([term for term in replaced_terms if term])
        return replacement_function
    
    
    def select_data(self, first_date=None, last_date=None, filters={}):
        """
        Create a copy of the original data
        and reduce it by selecting lines depending on the parameters.
        """
        self.filtered_df = pd.DataFrame(self.meals_df)
        
        if first_date:
            self.filtered_df = self.filtered_df[first_date <= self.filtered_df.Date]
        if last_date:
            self.filtered_df = self.filtered_df[self.filtered_df.Date <= last_date]
            
        if 'y' in filters:
            years = filters['y']
            if isinstance(years, int):
                years = [years]
            self.filtered_df = self.filtered_df[self.filtered_df.Date.dt.year.isin(years)]
        if 'm' in filters:
            months = filters['m']
            if isinstance(months, int):
                months = [months]
            self.filtered_df = self.filtered_df[self.filtered_df.Date.dt.month.isin(months)]
        if 'd' in filters:
            days = filters['d']
            if isinstance(days, int):
                days = [days]
            self.filtered_df = self.filtered_df[self.filtered_df.Date.dt.day.isin(days)]
        if 'wd' in filters:
            weekdays = filters['wd']
            if isinstance(weekdays, int):
                weekdays = [weekdays]
            self.filtered_df = self.filtered_df[self.filtered_df.Date.dt.weekday.isin(weekdays)]


    def terms_count(self, len_thresh=2):
        """
        Extract the terms from the selected lines and filter out the shortest ones.
        Aggregate the terms to get their frequency.
        """
        try:
            meals = self.filtered_df.meals
        except AttributeError:
            meals = self.meals_df.meals
        terms = [term for meal in meals for term in meal.split()] # tokenize
        terms = [term for term in terms if len(term) >= len_thresh]
        return Counter(terms)
    
    
    def _suggest(self, suggestions, filename):
        """
        Display suggestions in a format directly registrable as a rule.
        """
        lines = [term + ',' + ','.join(variations) + '\n'
                 for term, variations in suggestions.items()]
        with open(filename, 'w') as file:
            file.writelines(lines)
    

    def suggest_synonyms(self, dist_thresh=0.3):
        """
        Automatically find similar terms in the original data,
        suggest to save them as synonyms in a replacement rule.
        
        For now, just use the Levenshtein distance.
        In the future, could add stemming/lemmatization,
        or the detection of similar first letters (in both a short and a long word) for the abbreviations.
        """
        terms = self.terms_count()
        similar = {}
        for term in terms:
            for previous_term in similar:
                dist = Levenshtein().distance(term, previous_term)
                l = max(len(term), len(previous_term))
                if dist < dist_thresh * l:
                    similar[previous_term].append(term)
                    break
            else:
                similar[term] = []
        
        suggestions = {term: variations for term, variations in similar.items() if variations}
        self._suggest(suggestions, 'ortho_suggestions.csv')
        return
    
    
    def save_changes(self, filename='menus_cantine.csv'):
        self.meals_df.to_csv(filename, sep=',')




### COMPARING ###

def global_freq(kA, kB, NA, NB):
    return (kA + kB) / (NA + NB)



def delta_f(kA, kB, NA, NB):
    """
    Difference of frequencies
    """
    return kA / NA - kB / NB



def rule_of_succession(kA, kB): # a.k.a. "Laplace-Bayes estimator"
    """
    Posterior probability that a term belongs to A
    given observations of this term on A and B.
    """
    return (kA + 1) / (kA + kB + 2)



def simple_score(kA, kB, NA, NB):
    """
    A term characterizes the multiset A (resp. B) if
        - it is a frequent term (in both A and B)
        - the probability that it belongs to A  (resp. B)
          is higher (resp. lower) than the proportion of terms in A (resp. B)
    
    Compute the score    S(t) = f(t) * [r(t) - r0]    where
        f(t) is the frequence of t
        r(t) is the probability given by the rule of succession that t belongs to A
        r0 is the base probability that any term belongs to A
    """
    f = (kA + kB) / (NA + NB)
    r = rule_of_succession(kA, kB)
    r0 = NA / (NA + NB)
    return f * (r - r0)



def relative_entropy(p, q):
    """
    The Kullback–Leibler divergence, a.k.a. relative entropy,
    measures how a probability distribution is different from another one.
    Here we compute the individual contribution of each term to the divergence.
    """
    return p * log(p/q)



def entropy_score(kA, kB, NA, NB):
    """
    Compute the score    S = re(A, B) - re(B, A)    where
        re(A, B) is the relative entropy for the given term between A and B
    
    The relative entropy is based on the probability to find the given term in a multiset (A or B)
    P(next term = t | t is observed k(t) times in the N previous terms)
    which is given by a rule of succession -> (k(t) + 1) / (N + 2)
    """
    pA = (kA + 1) / (NA + 2)
    pB = (kB + 1) / (NB + 2)
    reA = relative_entropy(pA, pB)
    reB = relative_entropy(pB, pA)
    return reA - reB



def max_scores_plot(serie, title):
    background_color = '#293952'
    data_color = '#FDAC53'
    axis_color = 'w'
    
    fig, ax = plt.subplots(1, figsize=(12, 9), facecolor=background_color)
    ax.set_facecolor(background_color)
    
    serie.plot.barh(ax=ax, color=data_color, legend=False)
    ax.tick_params(axis='both', colors=axis_color)
    plt.title(title, loc='left', color=axis_color, fontsize=16)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(axis_color)
    
    ax.invert_yaxis()
    ax.set_yticks([])
    for i, (label, value) in enumerate(zip(serie.index, serie)):
        if value > 0:
            plt.text(0, i, label + '  ', va='center', ha='right', color=data_color)
            plt.text(value, i, f' {value:.4f}', va='center', ha='left', color=axis_color)
        else:
            plt.text(0, i, '  ' + label, va='center', ha='left', color=data_color)
            plt.text(value, i, f'{value:.4f} ', va='center', ha='right', color=axis_color)
    plt.show()



def compare(terms_count_study, terms_count_ref):
    """
    Given 2 multisets ("bags"), find which elements (terms) best characterize each bag.
    
    Notations:
        A: study multiset
        B: reference multiset
        k: multiplicity of a spcecific term -> kA("foo") = count of "foo" in bag A
        N: number of terms in a multiset (with repetitions)
    """
    index=['terms_count_study', 'terms_count_ref']
    terms_df = pd.DataFrame([terms_count_study, terms_count_ref], index=index).T.fillna(0)
    
    NA = terms_df.terms_count_study.sum()
    NB = terms_df.terms_count_ref.sum()
    
    for column in ['global_freq', 'delta_f', 'simple_score', 'entropy_score']:
        func = eval(column)
        parallel_func = lambda row: func(row.terms_count_study, row.terms_count_ref, NA, NB)
        terms_df[column] = terms_df.apply(parallel_func, axis=1)
    
    terms_df.sort_values(by='entropy_score', ascending=False, inplace=True)
    return terms_df



def select_head_and_tail(terms_df, column, items=50):
    terms_df['abs_score'] = terms_df[column].abs()
    max_values = terms_df.nlargest(items, 'abs_score')
    return max_values[column].sort_values(ascending=False)



def compare_scores_plot(df, x='entropy_score', y='delta_f', title=None, x_label=None, y_label=None):
    background_color = '#293952'
    axis_color = 'w'
    
    _, ax = plt.subplots(1, figsize=(12, 9), facecolor=background_color)
    ax.set_facecolor(background_color)
    
    size = df['global_freq'] * 3000
    df.plot.scatter(x=x, y=y, s=size, ax=ax, c='global_freq', cmap='Wistia', colorbar=False)
    ax.tick_params(axis='both', colors=axis_color)
    title = title or f'{x} vs {y}\n'
    ax.set_title(title, color=axis_color, fontsize=16)
    ax.set_xlabel(x_label or x, color=axis_color, fontsize=12, loc='right')
    ax.set_ylabel(y_label or y, color=axis_color, fontsize=12, loc='top')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color(axis_color)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_color(axis_color)
    ax.spines['bottom'].set_position('zero')
    
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.ax.tick_params(colors=axis_color)
    cbar.ax.set_ylabel('Fréquence du terme', color=axis_color, fontsize=12)
    cbar.outline.set_edgecolor('white')

    
    plt.show()



def characterize(explore=False):
    """
    Main function
    Select the relevant data, clean it and compare the terms frequencies.
    """
    data_source = DataSource()
    data_source.replace(['ortho', 'stemming', 'syn'])
    
    data_source.select_data(last_date='2019-06-30')
    data_source.select_data(filters={'wd': [0, 1, 2, 3]})
    # data_source.select_data(last_date='2019-06-30', filters={'wd': 4})
    # data_source.select_data(last_date='2019-06-30')
    terms_count_ref = data_source.terms_count()
    
    data_source.select_data(first_date='2019-07-01')
    data_source.select_data(filters={'wd': 4})
    # data_source.select_data(first_date='2019-07-01', filters={'wd': 4})
    # data_source.select_data(first_date='2019-07-01')
    terms_count_study = data_source.terms_count()
    
    terms_df = compare(terms_count_study, terms_count_ref)
    terms_df.to_csv('charac.csv')
    
    serie = select_head_and_tail(terms_df, 'entropy_score')
    max_scores_plot(serie, "Scores construits avec l'entropie")
    
    serie = select_head_and_tail(terms_df, 'delta_f')
    max_scores_plot(serie, 'Différence des fréquences')
    
    title = "Comparaison entre le score construit sur l'entropie\net la simple différence des fréquences\n"
    x_label = "Différence des fréquences"
    y_label = "Score construit avec l'entropie"
    compare_scores_plot(terms_df, 'delta_f', 'entropy_score', title, x_label, y_label)



if __name__ == "__main__":
    characterize()
    
    # data_source = DataSource()
    # data_source.suggest_synonyms()
