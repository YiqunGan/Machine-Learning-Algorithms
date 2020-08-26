import numpy as np

from util import accuracy
from hmm import HMM


# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags

    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    ###################################################
    # Edit here
    S = len(tags)
    state_dict = {tags[i]: i for i in range(S)}
    word_dict = dict()
    tagword_dict = dict()
    pi_list = np.zeros(S)
    A = np.zeros([S, S])
    for sentence in train_data:
        pi_list[state_dict[sentence.tags[0]]] += 1
        for tag, word in zip(sentence.tags, sentence.words):
            if tag not in tagword_dict:
                tagword_dict[tag] = dict()
                if word not in tagword_dict[tag]:
                    tagword_dict[tag][word] = 1
                else:
                    tagword_dict[tag][word] += 1
            else:
                if word not in tagword_dict[tag]:
                    tagword_dict[tag][word] = 1
                else:
                    tagword_dict[tag][word] += 1
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
    pi = pi_list / np.sum(pi_list)
    #for sentence in train_data:
        #for i in range(len(sentence.words)-1):
            #A[state_dict[sentence.tags[i]], state_dict[sentence.tags[i+1]]]+=1

    for state in tags:
        for sentence in train_data:
            for tag1, tag2 in zip(sentence.tags[0::2], sentence.tags[1::2]):
                if state == tag1:
                    A[tags.index(state), tags.index(tag2)] += 1
    for k in range(len(A)):
        if np.sum(A[k, :]) > 0:
            A[k, :] = A[k, :] / np.sum(A[k, :])
    B = np.zeros([S, len(word_dict.keys())])
    wordlist = [*word_dict]
    for key, value in tagword_dict.items():
        for word in value:
            B[tags.index(key)][wordlist.index(word)] = tagword_dict[key][word]
    for k in range(len(B)):
        if np.sum(B[k, :]) > 0:
            B[k, :] = B[k, :] / np.sum(B[k, :])
    obs_dict = {word: i for i, word in enumerate(wordlist)}
    model = HMM(pi, A, B, obs_dict, state_dict)
    #print("test:", train_data[0].index)

    ###################################################
    return model


# TODO:
def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """

    tagging = []
    L = len(model.obs_dict)
    S = len(model.pi)
    for sentence in test_data:
        sentence.words = [sentence.words[i].lower() for i in range(len(sentence.words))]
    for sentence in test_data:
        for word in sentence.words:
            if word not in model.obs_dict:
                model.obs_dict[word] = L
                model.B = np.hstack((model.B, np.ones((S, 1)) * (10 ** (-6))))
        tag = model.viterbi(sentence.words)
    tagging.append(tag)

    return tagging



