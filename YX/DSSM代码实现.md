# DSSM代码实现

DSSM双塔模型的网络结构如下所示：

```python
from tensorflow.python.keras.models import Model

from deepctr.layers.utils import combined_dnn_input
from deepctr.feature_column import input_from_feature_columns, build_input_features
from deepctr.layers.core import DNN, PredictionLayer
from utils import Cosine_Similarity


def DSSM(user_dnn_feature_columns, item_dnn_feature_columns, gamma=1, dnn_use_bn=True, dnn_hidden_units=(300, 300, 128), dnn_activation='tanh',l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary'):
    """Instantiates the Deep Structured Semantic Model architecture.
    :param user_dnn_feature_columns:An iterable containing user's features used by deep part of the model.
    :param item_dnn_feature_columns:An iterable containing item's the features used by deep part of the model.
    :param gamma: smoothing factor in the softmax function for DSSM
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    # 输入层
    user_features = build_input_features(user_dnn_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features, user_dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    item_features = build_input_features(item_dnn_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features, item_dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
	
    # 表示层
    user_dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  dnn_use_bn, seed=seed, name="user_embedding")(user_dnn_input)

    item_dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  dnn_use_bn, seed=seed, name="item_embedding")(item_dnn_input)
	
    # 匹配层
    score = Cosine_Similarity(user_dnn_out, item_dnn_out, gamma=gamma)

    output = PredictionLayer(task, False)(score)

    model = Model(inputs=user_inputs_list+item_inputs_list, outputs=output)

    return model
```

根据双塔模型的网络结构，逐层分析DSSM的代码实现过程，下面以movieLens为例说明：

### 1、输入层

movieLens数据如下：




输入特征按照数据类型可划分为sparse features和dense features，对这两类特征执行不同的处理。对于movieLens，数据划分如下：

```python
sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation']
dense_features = ['user_mean_rating', 'item_mean_rating']
target = ['rating']
```

由于使用双塔结构，对user和item划分如下：

```python
user_sparse_features, user_dense_features = ['user_id', 'gender', 'age', 'occupation'], ['user_mean_rating']
item_sparse_features, item_dense_features = ['movie_id'], ['item_mean_rating']
```

除此之外，user和item分别包括一个sequence feature，如下：

```python
user_varlen_sparse_feature = ['user_hist']
item_varlen_sparse_feature = ['genres']
```

针对sparse features需要进行Embedding处理，dense features做标准化处理，输入特征的处理如下：

```python

def input_from_feature_columns(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False):
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) 
    												if feature_columns else []    # Sparse features
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) 
    												if feature_columns else []    # Sequence features
	
    # Sparse features embedding
    embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
                                                    seq_mask_zero=seq_mask_zero)
    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features, feature_columns)
    
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")
	
    # Sequence features embedding
    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
                                                                 varlen_sparse_feature_columns)
    
    # Embedding features
    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
    
    if not support_group:
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
        
    return group_embedding_dict, dense_value_list
```

其中，函数create_embedding_matrix用于构建embedding层，构建过程如下：

```python
def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True):
    from . import feature_column as fc_lib

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict

def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                        embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix + '_emb_' + feat.embedding_name)
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(
                                l2_reg),
                            name=prefix + '_seq_emb_' + feat.name,
                            mask_zero=seq_mask_zero)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb
    return sparse_embedding
```

函数embedding_lookup用于获取sparse features的embedding表示，构建过程如下：

```python
def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(return_feat_list) == 0 or feature_name in return_feat_list):
            if fc.use_hash:
                lookup_idx = Hash(fc.vocabulary_size, mask_zero=(feature_name in mask_feat_list))(
                    sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]

            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict
```

函数varlen_embedding_lookup用于获取sequence features的embedding表示，由于sequence features是不定长的序列数据，所以获取的embedding需要进行pooling操作，获取指定维度的向量表示：

```python
def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=True)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict


def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns, to_list=False):
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = fc.length_name
        if feature_length_name is not None:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm)(
                    [embedding_dict[feature_name], features[feature_length_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
                [seq_input, features[feature_length_name]])
        else:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm, supports_masking=True)(
                    [embedding_dict[feature_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=True)(
                seq_input)
        pooling_vec_list[fc.group_name].append(vec)
    if to_list:
        return chain.from_iterable(pooling_vec_list.values())
    return pooling_vec_list
```

函数get_dense_input用于获取dense features，一般不做处理（输入前作标准化处理），如需处理可使用transform_fn做特征转换，构建过程如下：

```python
def get_dense_input(features, feature_columns):
    from . import feature_column as fc_lib
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        if fc.transform_fn is None:
            dense_input_list.append(features[fc.name])
        else:
            transform_result = Lambda(fc.transform_fn)(features[fc.name])
            dense_input_list.append(transform_result)
    return dense_input_list
```

输入层处理后的数据如下：

```python
>>> user_sparse_embedding_list
# [<tf.Tensor 'sparse_emb_user_id_6/Identity:0' shape=(None, 1, 4) dtype=float32>,
#  <tf.Tensor 'sparse_emb_gender_6/Identity:0' shape=(None, 1, 4) dtype=float32>,
#  <tf.Tensor 'sparse_emb_age_6/Identity:0' shape=(None, 1, 4) dtype=float32>,
#  <tf.Tensor 'sparse_emb_occupation_6/Identity:0' shape=(None, 1, 4) dtype=float32>,
#  <tf.Tensor 'sequence_pooling_layer_14/Identity:0' shape=(None, 1, 4) dtype=float32>]
```

```python
>>> user_dense_value_list
# [<tf.Tensor 'user_mean_rating_6:0' shape=(None, 1) dtype=float32>]
```

经函数combined_dnn_input合并后，输出数据如下：

```python
>>> user_dnn_input
# <tf.Tensor 'concatenate_27/Identity:0' shape=(None, 21) dtype=float32>
```

经过以上处理后的数据进入表示层。

### 2、表示层

表示层的网络定义如下：

```python
# 表示层
user_dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,dnn_use_bn, seed=seed, name="user_embedding")(user_dnn_input)

item_dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,dnn_use_bn, seed=seed, name="item_embedding")(item_dnn_input)
```

其中，DNN的网络结构定义为dnn_hidden_units=(300, 300, 128)。

DNN的实现过程如下：

```python
class DNN(Layer):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **output_activation**: Activation function to use in the last layer.If ``None``,it will be same as ``activation``.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
```

表示层输出user和item的embedding，通过匹配层计算user embedding和item embedding的相似度。

### 3、匹配层

计算user embedding和item embedding的cos相似度：

```python
def Cosine_Similarity(query, candidate, gamma=1, axis=-1):
    query_norm = tf.norm(query, axis=axis)
    candidate_norm = tf.norm(candidate, axis=axis)
    cosine_score = reduce_sum(tf.multiply(query, candidate), -1)
    cosine_score = div(cosine_score, query_norm*candidate_norm+1e-8)
    cosine_score = tf.clip_by_value(cosine_score, -1, 1.0)*gamma
    return cosine_score
```

cos函数得到的结果进入 sigmoid 函数输出预测结果：

```python
class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
```

## 3、模型训练

### 3.1 数据预处理

```python
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from dssm import DSSM
from deepctr.feature_column import SparseFeat, get_feature_names, VarLenSparseFeat, DenseFeat
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils import Negative_Sample

data_path = './data/movielens.txt'
train, test, data = data_process(data_path)
train = get_user_feature(train)
train = get_item_feature(train)

sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation']
dense_features = ['user_mean_rating', 'item_mean_rating']
target = ['rating']

user_sparse_features, user_dense_features = ['user_id', 'gender', 'age', 'occupation'], ['user_mean_rating']
item_sparse_features, item_dense_features = ['movie_id', ], ['item_mean_rating']

# 1.Label Encoding for sparse features,and process sequence features
for feat in sparse_features:
    lbe = LabelEncoder()
    lbe.fit(data[feat])
    train[feat] = lbe.transform(train[feat])
    test[feat] = lbe.transform(test[feat])
mms = MinMaxScaler(feature_range=(0, 1))
mms.fit(train[dense_features])
train[dense_features] = mms.transform(train[dense_features])

# 2.preprocess the sequence feature
genres_key2index, train_genres_list, genres_maxlen = get_var_feature(train, 'genres')
user_key2index, train_user_hist, user_maxlen = get_var_feature(train, 'user_hist')

user_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4) 
                        for i, feat in enumerate(user_sparse_features)] + [DenseFeat(feat, 1,) for feat in user_dense_features]
item_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4, use_hash=True) 
                        for i, feat in enumerate(item_sparse_features)] + [DenseFeat(feat, 1,) for feat in item_dense_features]

item_varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=1000, embedding_dim=4),
                                                maxlen=genres_maxlen, combiner='mean', weight_name=None)]

user_varlen_feature_columns = [VarLenSparseFeat(SparseFeat('user_hist', vocabulary_size=3470,embedding_dim=4), 
                                                maxlen=user_maxlen, combiner='mean', weight_name=None)]

# 3.generate input data for model
user_feature_columns += user_varlen_feature_columns
item_feature_columns += item_varlen_feature_columns

#add user history as user_varlen_feature_columns
train_model_input = {name:train[name] for name in sparse_features + dense_features}
train_model_input["genres"] = train_genres_list
train_model_input["user_hist"] = train_user_hist

from deepctr.feature_column import input_from_feature_columns, build_input_features
user_features = build_input_features(user_feature_columns)
user_inputs_list = list(user_features.values())
user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features, user_feature_columns,
                                                                     1e-6, 1024)
```

### 3.2 训练模型

```python
# 4.Define Model,train
model = DSSM(user_feature_columns, item_feature_columns, task='binary')
model.summary()

optimizer = Adam(lr=0.005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'], )

history = model.fit(train_model_input, train[target].values,
                    batch_size=512, epochs=10, verbose=2, validation_split=0.1,)
model.save_weights('../saved_model/movielens_dssm.ckpt')
```

### 3.3 预测

```python
# 5.test data preprocessing
model.load_weights('../saved_model/movielens_dssm.ckpt')
test = pd.merge(test, train[['movie_id', 'item_mean_rating']].drop_duplicates(), on='movie_id', how='left').fillna(0.5)
test = pd.merge(test, train[['user_id', 'user_mean_rating']].drop_duplicates(), on='user_id', how='left').fillna(0.5)
test = pd.merge(test, train[['user_id', 'user_hist']].drop_duplicates(), on='user_id', how='left').fillna('1')
test[dense_features] = mms.transform(test[dense_features])

test_genres_list = get_test_var_feature(test, 'genres', genres_key2index, genres_maxlen)
test_user_hist = get_test_var_feature(test, 'user_hist', user_key2index, user_maxlen)

test_model_input = {name : test[name] for name in sparse_features + dense_features}
test_model_input["genres"] = test_genres_list
test_model_input["user_hist"] = test_user_hist

# 6.predict and evaluate
pred_ans = model.predict(test_model_input, batch_size=256)
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

# 7.user embedding & item embedding
user_embedding_model = Model(inputs=model.input, outputs=model.get_layer("user_embedding").output)
item_embedding_model = Model(inputs=model.input, outputs=model.get_layer("item_embedding").output)
user_embedding = user_embedding_model.predict(test_model_input)
item_embedding = item_embedding_model.predict(test_model_input)

print("user embedding shape: ", user_embedding.shape)
print("item embedding shape: ", item_embedding.shape)

np.save('../saved_model/user_embedding.npy', user_embedding)
np.save('../saved_model/item_embedding.npy', item_embedding)
```









