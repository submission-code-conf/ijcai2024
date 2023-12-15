import tensorflow as tf
from tensorflow import feature_column


class FeatureProcess:
    def __init__(self, feature_map_path, default_embedding_dim=8) -> None:
        assert feature_map_path is not None
        super().__init__()
        lines = open(feature_map_path).readlines()
        print("lines", lines)
        self._name_2_idx_mapping = {}
        self._name_2_spec_mapping = {}
        self._name_2_feature_layer_mapping = {}
        self._idx_range = -1
        self._default_embedding_dim = default_embedding_dim
        for line in lines:
            d = {token.split('=')[0].strip(): token.split('=')[1].strip() for token in line.split(';') if token is not None and len(token) > 0}
            print("line, d: ", line, d)
            idx = int(d['group'])
            print("idx:", idx)
            self._idx_range = max(self._idx_range, idx)
            print("1SSSSSSSS")
            self._name_2_idx_mapping[d['name']] = idx
            print('2SSSSSSSSSSS')
            self._name_2_spec_mapping[d['name']] = d
            print('3SSSSSSSSSS')
        print("5SSSSSSSSSSS")
        self._idx_range += 1
        print("4SSSSSSSSSSS")

        for name, sepc in self._name_2_spec_mapping.items():
            print("name:", name)
            print("sepc:", sepc)
            embedding_dim = int(sepc.get('embedding_dim', str(self._default_embedding_dim)))
            hash_bucket_size = int(sepc.get('hash_bucket_size', '10000'))
            self._name_2_feature_layer_mapping[name] = self._to_sparse_feature_layer(name, embedding_dim, hash_bucket_size)

    def _to_sparse_feature_layer(self, name, embedding_dim=8, hash_bucket_size=10000):
        fc = feature_column.categorical_column_with_hash_bucket(
            name, hash_bucket_size=hash_bucket_size)
        fc = feature_column.embedding_column(fc, dimension=embedding_dim)
        feature_layer = tf.keras.layers.DenseFeatures(fc)
        return feature_layer

    def _to_sparse_feature_layers(self, names, embedding_dim=8, hash_bucket_size=10000):
        return [self._to_sparse_feature_layer(name, embedding_dim, hash_bucket_size) for name in names]

    def to_sparse_feature_embeddings(self, features):
        embeddings = [None for i in range(self._idx_range)]
        for name, feature_layer in self._name_2_feature_layer_mapping.items():
            embedding = feature_layer(features)
            embeddings[self._name_2_idx_mapping[name]] = embedding
        return embeddings
