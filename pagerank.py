"""
ML for Text Mining
Audrey Zhang
HW1
"""

import numpy as np
from scipy.sparse import dok_matrix, lil_matrix, csc_matrix
from collections import defaultdict
import os
from os import listdir
import datetime
import time


class PageRank:
    """when no outlinks: random walk. with outlinks: alpha """

    def __init__(self, alpha=0.8):

        self.alpha = alpha
        self.transition_mat, self.non_transition_mat = self._get_transition_matrix()
        self.topic_doc_map = self._get_topic_map()
        self.query_topic_distro = self._get_topic_distro(distro_type='query')
        self.user_topic_distro = self._get_topic_distro(distro_type='user')
        self.topic_specific_ranking = None
        self.topic_sensitive_time = None
        self.GPR_ranking = None

    def _get_transition_matrix(self):
        # create an inverted index of all outgoing links for a given page
        transitions = {}
        num_pages = 0
        with open('./data/transition.txt', 'r') as f:
            for line in f:
                temp_line = [int(i) for i in line.strip().split(' ')]

                # substract 1 from page IDs to enable matrix indexing starting at 0
                temp_line[0] -= 1
                temp_line[1] -= 1
                if temp_line[2] == 1:
                    if temp_line[0] not in transitions.keys():
                        transitions[temp_line[0]] = [temp_line[1]]
                    else:
                        transitions[temp_line[0]].append(temp_line[1])
                num_pages = max(num_pages, temp_line[0], temp_line[1])
        self.transitions = transitions

        # add one to num_pages because indexing starts at 0
        self.num_pages = num_pages + 1

        teleport_probability = 1 / self.num_pages
        self.teleportation = np.full(self.num_pages, teleport_probability)

        mat = defaultdict(float)

        # for pages without outgoing links, use lil_matrix to allow for quick update operations over entire rows
        non_transition_mat = lil_matrix((self.num_pages, self.num_pages))

        for page in range(self.num_pages):
            # if page has outgoing links, update the transition probabilities
            if page in transitions.keys():
                transition_prob = 1 / len(transitions[page])
                links = zip([page] * len(transitions[page]), transitions[page])
                for link in links:
                    mat[link] = transition_prob
            else:
                # otherwise, set transition probability = 1/N for all pages
                non_transition_mat[page] = teleport_probability

        # create sparse transition probability matrix
        transition_mat = dok_matrix((self.num_pages, self.num_pages), dtype=np.float64)
        dict.update(transition_mat, mat)

        return transition_mat.tocsr().transpose(), non_transition_mat.tocsr().transpose()

    @staticmethod
    def _get_topic_map():
        topic_doc_map = {}

        with open('./data/doc_topics.txt', 'r') as f:
            for line in f:
                line = [int(i) for i in line.strip().split(' ')]
                if line[1] not in topic_doc_map:
                    topic_doc_map[line[1]] = [line[0] - 1]
                else:
                    topic_doc_map[line[1]].append(line[0] - 1)

        return topic_doc_map

    @staticmethod
    def _get_topic_distro(distro_type):
        if distro_type == 'user':
            path = './data/user-topic-distro.txt'
        elif distro_type == 'query':
            path = './data/query-topic-distro.txt'

        topic_distro = {}

        with open(path, 'r') as f:
            for line in f:
                line = line.split()
                query = tuple((int(line[0]), int(line[1])))
                interest = [float(l[1]) for l in [i.split(':') for i in line[2:]]]
                topic_distro[query] = interest

        return topic_distro

    def _power_iteration(self, beta=None, transition_distro=None, epsilon=1e-8):

        # initialize r_0 as a random probabilistic distribution
        r = np.random.rand(self.num_pages)
        r = r / np.sum(r)

        if transition_distro is None:
            # if no other transition probability distribution is provided, use traditional GPR algorithm
            while True:

                r_new = self.alpha * (
                        self.transition_mat.dot(r) + self.non_transition_mat.dot(r)
                ) + (1 - self.alpha) * self.teleportation

                # check stopping criterion
                if np.linalg.norm(r_new - r) < epsilon:
                    break
                else:
                    r = r_new

        else:
            # otherwise, use modified GPR algorithm to adjust for topic- or user-topic sensitivity
            gamma = 1 - self.alpha - beta

            while True:
                r_new = self.alpha * (self.non_transition_mat.dot(r) + self.transition_mat.dot(r)
                                      ) + beta * transition_distro + gamma * self.teleportation

                if np.linalg.norm(r_new - r) < epsilon:

                    break
                else:
                    r = r_new

        r = r / np.sum(r)
        return r

    def global_page_rank(self):

        if self.GPR_ranking is None:
            start_time = time.time()
            self.GPR_ranking = self._power_iteration()
            end_time = time.time()
            run_time = end_time - start_time

            return self.GPR_ranking, run_time

        else:
            return self.GPR_ranking, None

    def topic_sensitive_page_rank(self):

        start_time = time.time()
        topic_based_ranking = []

        for topic, pages in self.topic_doc_map.items():
            transition_prob = 1 / len(pages)
            transition_distro = np.zeros(self.num_pages)
            transition_distro[pages] = transition_prob

            r = self._power_iteration(beta=0.15, transition_distro=transition_distro)
            topic_based_ranking.append(r)

        end_time = time.time()

        return np.array(topic_based_ranking), end_time - start_time

    def specific_topic_sensitive_pr(self, query, distro_type):

        if self.topic_specific_ranking is None:
            self.topic_specific_ranking, self.topic_sensitive_time = self.topic_sensitive_page_rank()

        start_time = time.time()

        if distro_type == 'user':
            topic_probability = np.array(self.user_topic_distro[query])
        if distro_type == 'query':
            topic_probability = np.array(self.query_topic_distro[query])

        combined_ranking = np.sum(np.multiply(self.topic_specific_ranking.transpose(), topic_probability), axis=1)

        run_time = time.time() - start_time

        return combined_ranking, run_time


def combine_PR_and_relevance(ranking,
                             relevant_docs,
                             relevance_scores,
                             method,
                             weight=0.3):
    """

    :param ranking: PR ranking results
    :param relevant_docs: relevant document IDs based on page_relevance data
    :param relevance_scores: relevance scores based on page_relevance data
    :param method: combination method: 'NS', 'WS', or 'CM'
    :param weight: weight for PR rankings to be used in weighted sum combination method
    :return: ranked results based on the final combined PR and search-relevance scores
    """
    # minus 1 from relevant doc id's to get indices
    relevant_doc_indices = relevant_docs - 1

    # get PR scores for the relevant docs
    results = ranking[relevant_doc_indices]

    if method == 'NS':
        pass

    if method == 'WS':
        results = weight * results + (1 - weight) * relevance_scores

    # custom combination: element-wise multiplication
    if method == 'CM':
        results = (results + relevance_scores) * relevance_scores

    # sort in descending value
    ranking = np.argsort(results)[::-1]
    ranked_docs = relevant_docs[ranking]
    ranked_scores = results[ranking]
    order = np.arange(len(ranked_docs)) + 1
    ranked_results = list(map(list, zip(ranked_docs, order, ranked_scores)))
    return ranked_results


def output_ranking_scores(filename, rankings):
    """writes ranking scores to an output file"""
    with open('./' + filename + '.txt', 'w') as f:
        for i in range(len(rankings)):
            f.write('{} {}\n'.format(i+1, rankings[i]))


def main(path='./data/indri-lists', output_queries=None):
    """
    main function call
    :param path: path to the search-relevance score docs
    :param output_queries: None, or list of query-ids that need the converged PR ranking scores written to output
    """
    run_id = 'run_' + str(datetime.datetime.now())
    pr_algs = ['GPR', 'QTSPR', 'PTSPR']
    combination_methods = ['NS', 'WS', 'CM']

    # initiate time dict to track runtime
    time_tracker = {}

    for p in pr_algs:
        time_tracker[p + '_ranking'] = 0.0

        for c in combination_methods:
            time_tracker[p + '_' + c + '_retrieval'] = 0.0

            # remove previous files if exists
            if os.path.exists('./' + p + '_' + c + '.txt'):
                os.remove('./' + p + '_' + c + '.txt')

    pr = PageRank()

    # get list of queries from sub-folder
    queries = listdir(path)
    n_queries = len(queries)

    for q in queries:
        string_id = q.split('.')[0]
        q_id = tuple([int(i) for i in string_id.split('-')])
        relevant_docs = []
        relevance_scores = []
        with open(path + '/' + q, 'r') as f:
            for line in f:
                line = line.split()
                relevant_docs.append(int(line[2]))
                relevance_scores.append(float(line[4]))
        assert len(relevant_docs) == len(relevance_scores)
        relevant_docs = np.array(relevant_docs)
        relevance_scores = np.array(relevance_scores)

        # iterate through PageRank algorithms:
        for rank_alg in pr_algs:

            if rank_alg == 'GPR':
                ranking, run_time = pr.global_page_rank()
                if run_time is not None:
                    time_tracker['GPR_ranking'] = run_time

                if output_queries is not None:
                    if string_id in output_queries:
                        output_ranking_scores(rank_alg, ranking)

            elif rank_alg == 'QTSPR':
                ranking, run_time = pr.specific_topic_sensitive_pr(q_id, distro_type='query')
                time_tracker[p + '_ranking'] += run_time

                if output_queries is not None:
                    if string_id in output_queries:
                        output_ranking_scores('{}-U{}Q{}'.format(rank_alg, q_id[0], q_id[1]), ranking)

            elif rank_alg == 'PTSPR':
                ranking, run_time = pr.specific_topic_sensitive_pr(q_id, distro_type='user')
                time_tracker[p + '_ranking'] += run_time

                if output_queries is not None:
                    if string_id in output_queries:
                        output_ranking_scores('{}-U{}Q{}'.format(rank_alg, q_id[0], q_id[1]), ranking)

            # iterate through methods for combining PR ranking with relevance scores:
            for method in combination_methods:
                out_name = rank_alg + '_' + method + '.txt'

                start_time = time.time()
                ranked_results = combine_PR_and_relevance(ranking, relevant_docs,
                                                          relevance_scores, method=method)
                end_time = time.time()

                time_tracker[p + '_' + c + '_retrieval'] += (end_time - start_time)

                with open('./' + out_name, 'a') as f:
                    for i in range(len(ranked_results)):
                        f.write(' '.join([string_id, 'Q0'] + [str(m) for m in ranked_results[i]] + [run_id, '\n']))

    # since only 1 iteration of topic sensitive ranking was done for both QTSPR and PTSPR ranking algs
    # add the run_time the ranking time for each algorithm
    time_tracker['QTSPR_ranking'] += pr.topic_sensitive_time
    time_tracker['PTSPR_ranking'] += pr.topic_sensitive_time

    print('ranking and retrieval times, averaged across queries: ')
    for k, v in time_tracker.items():
        print('{}: {}s'.format(k, v/n_queries))


if __name__ == '__main__':
    main(output_queries = '2-1')
