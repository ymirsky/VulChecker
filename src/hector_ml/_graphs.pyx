# cython: language_level=3, wraparound=False

import numpy as np
import scipy.sparse as sp
from structure2vec.graph_collate import Graph

from hector_ml._features cimport FeatureSet


def mean_field_from_node_link_data(
    dict nldata, FeatureSet node_features, FeatureSet edge_features
):
    cdef list nodes = nldata["nodes"]
    cdef Py_ssize_t n_nodes = len(nodes)
    cdef list links = nldata["links"]
    cdef Py_ssize_t n_edges = len(links)

    node_feature_mat = np.zeros((n_nodes, node_features.total_width), dtype=np.float32)
    edge_feature_mat = np.zeros((n_edges, edge_features.total_width), dtype=np.float32)
    adj_row_indexes = np.empty(n_edges * 2, dtype=np.int32)
    adj_col_indexes = np.empty(n_edges * 2, dtype=np.int32)
    adj_data = np.ones(n_edges * 2, dtype=np.float32)
    inc_row_indexes = np.empty(n_edges * 2, dtype=np.int32)
    inc_col_indexes = np.empty(n_edges * 2, dtype=np.int32)
    inc_data = np.ones(n_edges * 2, dtype=np.float32)

    cdef float[:, ::1] node_feature_array = node_feature_mat
    cdef float[:, ::1] edge_feature_array = edge_feature_mat
    cdef int[::1] adj_row_index_array = adj_row_indexes
    cdef int[::1] adj_col_index_array = adj_col_indexes
    cdef int[::1] inc_row_index_array = inc_row_indexes
    cdef int[::1] inc_col_index_array = inc_col_indexes

    cdef dict node_indexes = {}

    cdef Py_ssize_t i
    cdef dict data
    for i, data in enumerate(nodes):
        node_indexes[data["id"]] = i
        node_features.feature_row(node_feature_array, data, i)

    cdef Py_ssize_t source, target
    for i, data in enumerate(links):
        source = node_indexes[data["source"]]
        target = node_indexes[data["target"]]
        adj_row_index_array[2 * i] = source
        adj_col_index_array[2 * i] = target
        adj_row_index_array[2 * i + 1] = target
        adj_col_index_array[2 * i + 1] = source
        inc_row_index_array[2 * i] = source
        inc_col_index_array[2 * i] = i
        inc_row_index_array[2 * i + 1] = target
        inc_col_index_array[2 * i + 1] = i
        edge_features.feature_row(edge_feature_array, data, i)

    return Graph(
        structure=(
            sp.coo_matrix(
                (adj_data, (adj_row_indexes, adj_col_indexes)), (n_nodes, n_nodes)
            ),
            sp.coo_matrix(
                (inc_data, (inc_row_indexes, inc_col_indexes)), (n_nodes, n_edges)
            ),
        ),
        features=(node_feature_mat, edge_feature_mat),
    )
