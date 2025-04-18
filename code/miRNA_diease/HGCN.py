import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

class HGCN():
    def __init__(self, hidden_dims, lambda_):
        self.lambda_ = lambda_
        self.n_layers = len(hidden_dims) - 1
        self.W, self.v = self.define_weights(hidden_dims)
        self.C = {}

    # A: 邻接矩阵，X: 节点特征矩阵，R 和 S: 节点的超边信息
    def __call__(self, A, X, R, S):
        # Encoder
        H = X
        for layer in range(self.n_layers):
            H = self.__encoder(A, H, layer)

        # Final node representations
        self.H = H

        # Decoder (这里的解码器结构可以根据需要调整)
        for layer in range(self.n_layers - 1, -1, -1):
            H = self.__decoder(H, layer)

        X_ = H

        # 计算节点特征的重建损失
        features_loss = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(X - X_, 2))))

        # 计算超图结构损失
        self.S_emb = tf.nn.embedding_lookup(self.H, S)
        self.R_emb = tf.nn.embedding_lookup(self.H, R)
        structure_loss = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb * self.R_emb, axis=-1)))
        structure_loss = tf.reduce_sum(structure_loss)

        # 总损失
        self.loss = features_loss + self.lambda_ * structure_loss

        return self.loss, self.H, self.C

    def __encoder(self, A, H, layer):
        H = tf.matmul(H, self.W[layer])
        self.C[layer] = self.hypergraph_attention_layer(A, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable(f"W{i}", shape=(hidden_dims[i], hidden_dims[i+1]))

        Ws_att = {}
        for i in range(self.n_layers):
            v = {}
            v[0] = tf.get_variable(f"v{i}_0", shape=(hidden_dims[i+1], 1))
            v[1] = tf.get_variable(f"v{i}_1", shape=(hidden_dims[i+1], 1))
            Ws_att[i] = v

        return W, Ws_att

    def hypergraph_attention_layer(self, A, M, v, layer):
        with tf.variable_scope(f"layer_{layer}"):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)

            unnormalized_attentions = tf.SparseTensor(
                indices=logits.indices,
                values=tf.nn.sigmoid(logits.values),
                dense_shape=logits.dense_shape
            )
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(
                indices=attentions.indices,
                values=attentions.values,
                dense_shape=attentions.dense_shape
            )

            return attentions
