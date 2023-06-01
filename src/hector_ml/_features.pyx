# cython: language_level=3, wraparound=False


cpdef enum FeatureKind:
    numeric = 1
    optional_numeric = 2
    categorical = 3
    categorical_set = 4


cdef class Feature:
    cdef readonly object name
    cdef readonly Py_ssize_t width

    def __cinit__(self, str name, Py_ssize_t width):
        self.name = name
        self.width = width

    cdef int feature_row(
        self,
        float[:, :] feature_mat,
        dict data,
        Py_ssize_t row,
        Py_ssize_t col,
    ) except -1:
        raise NotImplementedError("Don't call feature_row on the base class.")


cdef class NumericFeature(Feature):
    feature_kind = FeatureKind.numeric

    cdef int feature_row(
        self,
        float[:, :] feature_mat,
        dict data,
        Py_ssize_t row,
        Py_ssize_t col,
    ) except -1:
        feature_mat[row, col] = data[self.name]


cdef class OptionalNumericFeature(Feature):
    feature_kind = FeatureKind.optional_numeric

    cdef int feature_row(
        self,
        float[:, :] feature_mat,
        dict data,
        Py_ssize_t row,
        Py_ssize_t col,
    ) except -1:
        try:
            feature_mat[row, col] = float(data[self.name])
        except (ValueError, TypeError, KeyError):
            feature_mat[row, col + 1] = 1.0


cdef class CategoricalFeature(Feature):
    feature_kind = FeatureKind.categorical

    cdef int feature_row(
        self,
        float[:, :] feature_mat,
        dict data,
        Py_ssize_t row,
        Py_ssize_t col,
    ) except -1:
        cdef Py_ssize_t value
        try:
            value = data[self.name]
        except (TypeError, KeyError):
            pass
        else:
            if 0 <= value < self.width:
                feature_mat[row, col + value] = 1.0


cdef class CategoricalSetFeature(Feature):
    feature_kind = FeatureKind.categorical_set

    cdef int feature_row(
        self,
        float[:, :] feature_mat,
        dict data,
        Py_ssize_t row,
        Py_ssize_t col,
    ) except -1:
        cdef Py_ssize_t value
        try:
            values = list(data[self.name])
        except (TypeError, KeyError):
            pass
        else:
            for value in values:
                if 0 <= value < self.width:
                    feature_mat[row, col + value] = 1.0


cdef class FeatureSet:
    # Repeated from _features.pxd for referece:
    #
    # cdef readonly tuple features
    # cdef readonly Py_ssize_t total_width

    def __cinit__(self, tuple features, Py_ssize_t total_width):
        self.features = features
        self.total_width = total_width

    cpdef int feature_row(
        self,
        float[:, :] feature_mat,
        dict data,
        Py_ssize_t row,
    ) except -1:
        cdef Py_ssize_t col = 0
        cdef Feature feature
        for feature in self.features:
            feature.feature_row(feature_mat, data, row, col)
            col += feature.width

    @staticmethod
    def from_features_and_indexes(raw_features, indexes):
        total_width = 0
        features = []
        for raw_feature in raw_features:
            if raw_feature.kind == FeatureKind.numeric:
                feature = NumericFeature(raw_feature.name, 1)
            elif raw_feature.kind == FeatureKind.optional_numeric:
                feature = OptionalNumericFeature(raw_feature.name, 2)
            elif raw_feature.kind == FeatureKind.categorical:
                feature = CategoricalFeature(
                    raw_feature.name,
                    len(indexes.setdefault(raw_feature.dictionary, {})),
                )
            elif raw_feature.kind == FeatureKind.categorical_set:
                feature = CategoricalSetFeature(
                    raw_feature.name,
                    len(indexes.setdefault(raw_feature.dictionary, {})),
                )
            else:
                raise RuntimeError("Unreachable!")  # pragma: no cover
            features.append(feature)
            total_width += feature.width
        return FeatureSet.__new__(FeatureSet, tuple(features), total_width)
