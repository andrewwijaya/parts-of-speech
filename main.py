#!/usr/bin/env python

from io import open

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from conllu import parse_incr
from nltk import FreqDist, WittenBellProbDist
from sklearn.metrics import confusion_matrix

treebank = {'en': 'UD_English-EWT/en_ewt', 'es': 'UD_Spanish-GSD/es_gsd', 'nl': 'UD_Dutch-Alpino/nl_alpino',
            'ja': 'UD_Japanese-GSD/ja_gsd', 'ar': 'UD_Arabic-PADT/ar_padt'}

np.set_printoptions(linewidth=np.inf)


def train_corpus(lang):
    return treebank[lang] + '-ud-train.conllu'


def test_corpus(lang):
    return treebank[lang] + '-ud-test.conllu'


# Remove contractions such as "isn't".
def prune_sentence(sent):
    return [token for token in sent if type(token['id']) is int]


def conllu_corpus(path):
    data_file = open(path, 'r', encoding='utf-8')
    sents = list(parse_incr(data_file))
    return [prune_sentence(sent) for sent in sents]


def convert_sentences_to_tag_lists(sentences):
    """
    Takes a collection of sentences and returns its array of tags.
    """
    tag_array = [["<s>"] + [token['upos'] for token in sentence] + ["</s>"] for sentence in sentences]
    return tag_array


def get_tag_dict(tag_set):
    """
    Takes an array of tags and returns a dictionary of key
    value pairs assigning each tag with a unique integer
    """
    i = 1
    tag_dict = {}
    for tag in tag_set:
        tag_dict[tag] = i
        i += 1
    return tag_dict


def generate_count_table(tags_set, tag_dict, tag_arrays):
    """
    Given a set of tags, tag dictionary, and an array of tag sentences, produce a table representing the count of
    transitions from one tag to the next.
    """
    count_table = np.zeros((len(tags_set), len(tags_set)))
    for tag_sent in tag_arrays:
        for i in range(0, len(tag_sent) - 1):
            tag_index = tag_dict[tag_sent[i]]
            next_tag_index = tag_dict[tag_sent[i + 1]]
            count_table[tag_index - 1][next_tag_index - 1] += 1
    return count_table


def get_tag_list(train_sents):
    """
    Given a collection of sentences, return a unique list of tags occurring in those sentences.
    """
    tag_list = list(set([token["upos"] for sent in train_sents for token in sent]))
    tag_list.append("<s>")
    tag_list.sort()
    tag_list.append("</s>")
    return tag_list


def generate_tag_words_dict(tag_list, train_sents):
    """
    Generates a mapping from tags to integers starting from 1.
    """
    tag_words_dict = {}
    for tag in tag_list:
        tag_words_dict[tag] = []
    for sent in train_sents:
        for token in sent:
            tag_words_dict[token['upos']].append(token['form'])
    return tag_words_dict


def generate_freq_dist_emission_table(tag_words_dict):
    """
    Generates a dictionary of tag to FreqDist. Input parameter is a dictionary mapping each tag to all the words in
    that tag from the corpus.
    """
    freq_dist_dict = {}
    for tag in tag_words_dict:
        freq_dist_dict[tag] = FreqDist(tag_words_dict[tag])
    return freq_dist_dict


def generate_witten_bell_array(tag_list, freq_dist_dict):
    """
    Converts FreqDists to WittenBellProbDists.
    """
    smoothed = {}
    for tag in tag_list:
        smoothed[tag] = WittenBellProbDist(freq_dist_dict[tag], bins=1e10)
    return smoothed


def greedy_algorithm(tag_list, sentence, A, B):
    """
    Implementation of the greedy algorithm. This function takes a sentence and returns the pos tagging for that sentence
    based on transition table A and emission table B.
    """
    pos_tagging = []
    previous_predicted = "<s>"
    tag_dict = get_tag_dict(tag_list)
    for token in sentence:
        likelihood_dict = {}
        for tag in tag_list[1:-1]:
            transition_prob = A[tag_dict[previous_predicted] - 1][tag_dict[tag] - 2]
            emission_prob = B[tag].prob(token['form'])
            likelihood = transition_prob * emission_prob
            likelihood_dict[tag] = likelihood
        predicted_tag = max(likelihood_dict, key=likelihood_dict.get)
        previous_predicted = predicted_tag
        pos_tagging.append(predicted_tag)
    return pos_tagging


def lidstone_smoothing(count_table):
    """
    Performs lidstone smoothing on the transition table and then normalises the table.
    """
    count_table += 0.5
    summ_array_transpose = np.array([np.sum(count_table, axis=1)]).T
    normalized_table = count_table / summ_array_transpose
    return normalized_table


def reverse_tag_dict(tag_dict):
    """
    Reverses tag dict in order to get tag name from integer.
    """
    inv_tag_dict = {v: k for k, v in tag_dict.items()}
    return inv_tag_dict


def viterbi_backtrack(viterbi_index, end_index, sentence):
    """
    This function is used by the viterbi algorithm to recover the POS tagging result. It starts with the end_index and
    traverses the viterbi_index table to return the resulting array of indices representing the POS tagging.
    """
    current_index = end_index
    final_indices = []
    for i in range(len(sentence) - 1, -1, -1):
        prev_index = viterbi_index[int(current_index)][i]
        final_indices.append(prev_index)
        current_index = prev_index
    return final_indices[::-1]


def viterbi_algorithm(tag_list, sentence, A, B):
    """
    This is the forward pass of the Viterbi algorithm. It populates the viterbi_table and also the viterbi_index which
    is used for backtracking purposes later to recover the POS tagging.
    """
    viterbi_table = np.zeros(((len(tag_list) - 2), (len(sentence)) + 1))
    viterbi_index = np.zeros(((len(tag_list) - 2), (len(sentence)) + 1))
    tag_dict = get_tag_dict(tag_list)
    start_tag = "<s>"
    # Initialise
    for tag in tag_list[1:-1]:
        transition_prob = A[tag_dict[start_tag] - 1][tag_dict[tag] - 2]
        emission_prob = B[tag].prob(sentence[0]['form'])
        viterbi_table[tag_dict[tag] - 2][0] = (transition_prob * emission_prob)

    # Main
    for i in range(1, len(sentence)):
        for tag in tag_list[1:-1]:
            max_prob = []
            for prev_tag in tag_list[1:-1]:
                transition_prob = A[tag_dict[prev_tag] - 1][tag_dict[tag] - 2]
                emission_prob = B[tag].prob(sentence[i]['form'])
                max_prob.append(viterbi_table[tag_dict[prev_tag] - 2][i - 1] * transition_prob * emission_prob)
            viterbi_table[tag_dict[tag] - 2][i] = max(max_prob)
            viterbi_index[tag_dict[tag] - 2][i] = max_prob.index(max(max_prob))

    # End tag
    max_prob = []
    for tag in tag_list[1:-1]:
        transition_prob = A[tag_dict[tag] - 1][tag_dict["</s>"] - 2]
        max_prob.append(transition_prob * viterbi_table[tag_dict[tag] - 2][len(sentence) - 1])
    viterbi_table[tag_dict[tag] - 2][len(sentence)] = max(max_prob)
    end_index = max_prob.index(max(max_prob))
    viterbi_index[tag_dict[tag] - 2][len(sentence)] = end_index
    return viterbi_table, viterbi_index, end_index


def get_accuracy(pos_taggings, test_sents):
    """
    This method comparse the predicted tags of a single sentence vs the actual tags from a single training sentence.
    Parameters: pos_tagging is an array of tuples. Each tuple is the word and predicted tag.
                train_sent is a tagged training sentence.
    Returns: value between 0 and 1 signifying accuracy of comparing predicted vs training tags in each sentence.
    """
    matches = 0
    total = 0
    for i in range(0, len(pos_taggings)):
        pos_tagging = pos_taggings[i]
        test_sent = test_sents[i]
        for j in range(0, len(pos_tagging)):
            if pos_tagging[j] == test_sent[j]['upos']:
                matches += 1
        total += len(pos_tagging)
    return matches / total


def forward_algorithm(tag_list, A, B, sentence):
    """
    Forward algorithm implementation. This function populates the alpha table.
    """
    tag_dict = get_tag_dict(tag_list)
    alpha_table = np.zeros(((len(tag_list) - 2), (len(sentence)) + 1))
    start_tag = "<s>"
    # Initialise
    for tag in tag_list[1:-1]:
        transition_prob = A[tag_dict[start_tag] - 1][tag_dict[tag] - 2]
        emission_prob = B[tag].prob(sentence[0]['form'])
        alpha_table[tag_dict[tag] - 2][0] = (transition_prob * emission_prob)

    # Main
    for i in range(1, len(sentence)):
        for tag in tag_list[1:-1]:
            sum_prob = []
            for prev_tag in tag_list[1:-1]:
                transition_prob = A[tag_dict[prev_tag] - 1][tag_dict[tag] - 2]
                emission_prob = B[tag].prob(sentence[i]['form'])
                sum_prob.append(alpha_table[tag_dict[prev_tag] - 2][i - 1] * transition_prob * emission_prob)
            alpha_table[tag_dict[tag] - 2][i] = sum(sum_prob)

    # End tag
    sum_prob = []
    for tag in tag_list[1:-1]:
        transition_prob = A[tag_dict[tag] - 1][tag_dict["</s>"] - 2]
        sum_prob.append(transition_prob * alpha_table[tag_dict[tag] - 2][len(sentence) - 1])
    alpha_table[tag_dict[tag] - 2][len(sentence)] = sum(sum_prob)
    return alpha_table


def backward_algorithm(tag_list, A, B, sentence):
    """
    Backward algorithm implementation. This function populates the beta table.
    """
    tag_dict = get_tag_dict(tag_list)
    beta_table = np.zeros(((len(tag_list) - 2), (len(sentence)) + 1))
    end_tag = "</s>"
    # Initialise
    for tag in tag_list[1:-1]:
        transition_prob = A[tag_dict[tag] - 1][tag_dict[end_tag] - 2]
        beta_table[tag_dict[tag] - 2][len(sentence)] = transition_prob

        # Main
    for i in range(len(sentence) - 2, -2, -1):
        for tag in tag_list[1:-1]:
            sum_prob = []
            for prev_tag in tag_list[1:-1]:
                transition_prob = A[tag_dict[prev_tag] - 1][tag_dict[tag] - 2]
                emission_prob = B[prev_tag].prob(sentence[i + 1]['form'])
                sum_prob.append(beta_table[tag_dict[prev_tag] - 2][i + 2] * transition_prob * emission_prob)
            beta_table[tag_dict[tag] - 2][i + 1] = sum(sum_prob)

    # Start tag
    sum_prob = []
    for tag in tag_list[1:-1]:
        transition_prob = A[tag_dict[tag] - 1][tag_dict["<s>"] - 2]
        sum_prob.append(transition_prob * beta_table[tag_dict[tag] - 2][len(sentence) - 1])
    beta_table[tag_dict[tag] - 2][0] = sum(sum_prob)
    return beta_table


def local_decode(alpha_table, beta_table, tag_list):
    """
    Performs local decoding based on given alpha and beta tables.
    """
    tag_dict = get_tag_dict(tag_list)
    alpha_table = alpha_table[:-1]
    beta_table = beta_table[:-1]
    mult_table = np.multiply(alpha_table, beta_table)
    inv_tag_dict = reverse_tag_dict(tag_dict)
    pos_tagging = []
    for col in mult_table.T:
        colList = col.tolist()
        tag_index = colList.index(max(colList))
        pos_tagging.append(inv_tag_dict[tag_index + 2])
    return pos_tagging


def run_viterbi(test_sents, tag_list, A, B):
    """
    Executes the Viterbi algorithm for all sentences.
    """
    pos_taggings = []
    tag_dict = get_tag_dict(tag_list)
    for test_sent in test_sents:
        viterbi_table, viterbi_index, end_index = viterbi_algorithm(tag_list, test_sent, A, B)
        pos_indices = viterbi_backtrack(viterbi_index, end_index, test_sent)
        inv_tag_dict = reverse_tag_dict(tag_dict)
        pos_indices = pos_indices[1:]
        pos_indices.append(end_index)
        pos_tagging = []
        for pos_index in pos_indices:
            pos_tagging.append(inv_tag_dict[int(pos_index) + 2])
        pos_taggings.append(pos_tagging)
    return pos_taggings


# run greedy on each test sentence and collect results
def run_greedy(test_sents, tag_list, A, B):
    """
    Executes the greedy algorithm for all sentences.
    """
    pos_taggings = []
    for test_sent in test_sents:
        pos_tagging = greedy_algorithm(tag_list, test_sent, A, B)
        pos_taggings.append(pos_tagging)
    return pos_taggings


def run_forward_backward(test_sents, tag_list, A, B):
    """
    Executes the forward backward algorithm for all sentences.
    """
    pos_taggings = []
    for test_sent in test_sents:
        alpha_table = forward_algorithm(tag_list, A, B, test_sent)
        beta_table = backward_algorithm(tag_list, A, B, test_sent)
        pos_tagging = local_decode(alpha_table[:, :-1], beta_table[:, 1:], tag_list)
        pos_taggings.append(pos_tagging)
    return pos_taggings


def flatten(list_of_lists):
    """
    Flattens a list of lists into a single list.
    """
    flattened_list = [item for list in list_of_lists for item in list]
    return flattened_list


def get_actual_tags(sentences):
    """
    Returns a list of all tags flattened into one list.
    """
    tag_arrays = [[token['upos'] for token in sentence] for sentence in sentences]
    flattened_tag_array = flatten(tag_arrays)
    return flattened_tag_array


def create_confusion_matrix_plot(file_name, y, y_hat):
    """
    Create confusion matrix plot given output y and predicted output y_hat.
    Saves the figure to file_name.
    """
    tag_list = list(set(y))
    tag_list.sort()
    plt.rcParams.update({'font.size': 5})
    cf_matrix = confusion_matrix(y, y_hat)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
    ax.set_title(file_name)
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(tag_list)
    ax.yaxis.set_ticklabels(tag_list)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    figure = ax.get_figure()
    figure.savefig('figs/' + file_name + '.png', dpi=400, bbox_inches='tight')
    plt.clf()


def run_all():
    """
    This method executes all required functions necessary for the experiments.
    This method will train a model and run tests for English, Spanish, Dutch, Japanese, and Arabic.
    Accuracies will be printed onto console, confusion matrices and frequency distribution plots will be saved as png
    figures on disk.
    """
    # Language arrays
    langs = ["en", "es", "nl", "ja", "ar"]
    # Prepare corpus
    for lang in langs:
        print("Language: ", lang)

        train_sents = conllu_corpus(train_corpus(lang))
        test_sents = conllu_corpus(test_corpus(lang))

        # Prepare tag lists
        tag_list = get_tag_list(train_sents)
        tag_arrays = convert_sentences_to_tag_lists(train_sents)
        tag_dict = get_tag_dict(tag_list)

        count_table = generate_count_table(tag_list, tag_dict, tag_arrays)
        # delete last row containing transitions from </s>
        count_table = np.delete(count_table, (-1), axis=0)
        # delete first column for <s> as all are zeros
        count_table = np.delete(count_table, 0, 1)
        A = lidstone_smoothing(count_table)
        tag_words_dict = generate_tag_words_dict(tag_list, train_sents)
        freq_dist_dict = generate_freq_dist_emission_table(tag_words_dict)
        B = generate_witten_bell_array(tag_list, freq_dist_dict)

        print(len(train_sents), 'training sentences')
        print(len(test_sents), 'test sentences')

        pos_taggings_dict = {}
        greedy_pos_taggings = run_greedy(test_sents, tag_list, A, B)
        viterbi_pos_taggings = run_viterbi(test_sents, tag_list, A, B)
        forward_backward_pos_taggings = run_forward_backward(test_sents, tag_list, A, B)

        pos_taggings_dict["greedy"] = greedy_pos_taggings
        pos_taggings_dict["viterbi"] = viterbi_pos_taggings
        pos_taggings_dict["forward_backward"] = forward_backward_pos_taggings

        greedy_accuracy = get_accuracy(greedy_pos_taggings, test_sents)
        viterbi_accuracy = get_accuracy(viterbi_pos_taggings, test_sents)
        forward_backward_accuracy = get_accuracy(forward_backward_pos_taggings, test_sents)

        print("greedy accuracy: ", greedy_accuracy)
        print("Viterbi accuracy: ", viterbi_accuracy)
        print("Forward Backward accuracy: ", forward_backward_accuracy)
        test_tag_list = get_tag_list(test_sents)
        print("Train tag set size: (without start and end tags) ", len(tag_list) - 2)
        print("Test tag set size: (without start and end tags) ", len(test_tag_list) - 2)

        # Frequency distribution plots of train and test tag sets
        all_train_actual_tags = get_actual_tags(train_sents)
        FreqDist(all_train_actual_tags)

        # Test freq dist
        plt.rcParams.update({'font.size': 12})
        all_test_actual_tags = get_actual_tags(test_sents)
        fdist_test = FreqDist(all_test_actual_tags)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        fdist_test.plot(120, cumulative=False, title="Test tags freq dist for " + lang)
        plt.savefig("figs/test_tags_freq_dist_" + lang + ".png")
        plt.clf()

        # Create confusion matrices
        for key in pos_taggings_dict:
            flattened_pos_tagging = flatten(pos_taggings_dict[key])
            create_confusion_matrix_plot("confusion_matrix_" + key + "_" + lang, all_test_actual_tags,
                                         flattened_pos_tagging)


run_all()
