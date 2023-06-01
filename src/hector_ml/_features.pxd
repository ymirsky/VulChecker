# cython: language_level=3


cdef class FeatureSet:
    cdef readonly tuple features
    cdef readonly Py_ssize_t total_width

    cpdef int feature_row(
        self,
        float[:, :] feature_mat,
        dict data,
        Py_ssize_t row,
    ) except -1
