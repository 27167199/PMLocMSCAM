import tensorflow.compat.v1 as tf
from HGCN import HGCN
import scipy.sparse as sp
tf.disable_eager_execution()

# 将 TensorFlow 稀疏矩阵转换为 Numpy 稀疏矩阵
def convert_sparse_tf2np(input):
    return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])),
                          shape=(input[layer][2][0], input[layer][2][1])) for layer in input]

class HGCNTrainer():
    def __init__(self, args):
        self.args = args
        self.build_placeholders()
        hgcn = HGCN(args.hidden_dims, args.lambda_)
        self.loss, self.H, self.C = hgcn(self.A, self.X, self.R, self.S)
        self.optimize(self.loss)
        self.build_session()

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.S = tf.placeholder(tf.int64)
        self.R = tf.placeholder(tf.int64)

    def build_session(self, gpu=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if gpu == False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def train_hypergraph_model(self, A, X, S, R):
        for epoch in range(self.args.n_epochs):
            self.run_epoch(epoch, A, X, S, R)

    def run_epoch(self, epoch, A, X, S, R):
        loss, _ = self.session.run([self.loss, self.train_op],
                                   feed_dict={self.A: A,
                                              self.X: X,
                                              self.S: S,
                                              self.R: R})
        return loss

    def infer_hypergraph_embeddings(self, A, X, S, R):
        H, C = self.session.run([self.H, self.C],
                                feed_dict={self.A: A,
                                           self.X: X,
                                           self.S: S,
                                           self.R: R})
        return H, convert_sparse_tf2np(C)
