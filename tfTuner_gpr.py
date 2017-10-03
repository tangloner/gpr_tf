'''
Created on Sep 19, 2017

@author: Bo Tang @ SUSTech, CUHK
'''

import sys
import numpy as np
import tensorflow as tf
import lhsmdu
from time import time

class GPRResult(object):

    def __init__(self, ypreds=None, sigmas=None):
        self.ypreds = ypreds
        self.sigmas = sigmas

class GPR_GDResult(GPRResult):
    
    def __init__(self, ypreds=None, sigmas=None, minL=None, minL_conf=None):
        super(GPR_GDResult, self).__init__(ypreds, sigmas)
        self.minL = minL
        self.minL_conf = minL_conf

class GPR(object):
    
    MAX_TRAIN_SIZE = 7000
    BATCH_SIZE = 3000
    NUM_THREADS = 4
    
    def __init__(self, length_scale=1.0, magnitude=1.0):
        assert np.isscalar(length_scale)
        assert np.isscalar(magnitude)
        assert length_scale > 0 and magnitude > 0
        self.length_scale = length_scale
        self.magnitude = magnitude
        self.X_train = None
        self.y_train = None
        self.xy_ = None
        self.K = None
        self.graph = None
        self.vars = None
        self.ops = None

    def build_graph(self):
        self.vars = {}
        self.ops = {}
        self.graph = tf.Graph()
        with self.graph.as_default():
            mag_const = tf.constant(self.magnitude,dtype=np.float32,name='magnitude')
            ls_const = tf.constant(self.length_scale,dtype=np.float32,name='length_scale')

            v1 = tf.placeholder(tf.float32, name="v1")
            v2 = tf.placeholder(tf.float32, name="v2")
            dist_op = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1,v2),2),1), name='dist_op')
            
            self.vars['v1_h'] = v1
            self.vars['v2_h'] = v2
            self.ops['dist_op'] = dist_op

            # Nodes for kernel computation
            X_dists = tf.placeholder(tf.float32, name='X_dists')
            ridge_ph = tf.placeholder(tf.float32, name='ridge')
            K_op = mag_const*tf.exp(-X_dists / ls_const)
            K_ridge_op = K_op + tf.diag(ridge_ph)

            self.vars['X_dists_h'] = X_dists
            self.vars['ridge_h'] = ridge_ph
            self.ops['K_op'] = K_op
            self.ops['K_ridge_op'] = K_ridge_op

            # Nodes for xy computation

            K = tf.placeholder(tf.float32, name='K')
            K_inv = tf.placeholder(tf.float32, name='K_inv')
            xy_ = tf.placeholder(tf.float32, name='xy_')
            yt_ = tf.placeholder(tf.float32, name='yt_')
            K_inv_op = tf.matrix_inverse(K)
            #K_inv_op = tf.check_numerics(K_inv_op, "K_inv: ")
            xy_op = tf.matmul(K_inv, yt_)
            #xy_op = tf.check_numerics(xy_op, "xy_: ")
            
            self.vars['K_h'] = K
            self.vars['K_inv_h'] = K_inv
            self.vars['xy_h'] = xy_
            self.vars['yt_h'] = yt_
            self.ops['K_inv_op'] = K_inv_op
            self.ops['xy_op'] = xy_op
 
            # Nodes for yhat / sigma computation
            K2 = tf.placeholder(tf.float32, name="K2")
            K3 = tf.placeholder(tf.float32, name="K3")
            yhat_ = tf.cast(tf.matmul(tf.transpose(K2), xy_), tf.float32);
            sv1 = tf.matmul(tf.transpose(K2), tf.matmul(K_inv, K2))
            sig_val = tf.cast((tf.sqrt(tf.diag_part(K3-sv1))), tf.float32)

            self.vars['K2_h'] = K2
            self.vars['K3_h'] = K3
            self.ops['yhat_op'] = yhat_
            self.ops['sig_op'] = sig_val

            # Compute y_best (min y)
            y_best_op = tf.cast(tf.reduce_min(yt_,0,True),tf.float32)
            self.ops['y_best_op'] = y_best_op

            sigma = tf.placeholder(tf.float32, name='sigma')
            yhat = tf.placeholder(tf.float32, name='yhat')

    def check_X_y(self, X, y):
        from sklearn.utils.validation import check_X_y

        if X.shape[0] > GPR.MAX_TRAIN_SIZE:
            raise Exception("X_train size cannot exceed {} ({})".format(GPR.MAX_TRAIN_SIZE, X.shape[0]))
        return check_X_y(X,y, multi_output=True, allow_nd=True, y_numeric=True, estimator="GPR")

    def check_fitted(self):
        if self.X_train is None or self.y_train is None or self.xy_ is None or self.K is None:
            raise Exception("The model must be trained before making predictions!")

    def check_array(self, X):
        from sklearn.utils.validation import check_array
        return check_array(X, allow_nd=True, estimator="GPR")

    def check_output(self, X):
        finite_els = np.isfinite(X)
        if not np.all(finite_els):
            raise Exception("Input contains non-finite values: {}".format(X[~finite_els]))

    def fit(self, X_train, y_train, ridge=1.0):
        self._reset()
        X_train, y_tain = self.check_X_y(X_train, y_train)
        self.X_train = np.float32(X_train)
        self.y_train = np.float32(y_train)
        sample_size = self.X_train.shape[0]

        if np.isscalar(ridge):
            ridge = np.ones(sample_size)*ridge
        assert ridge.ndim == 1

        X_dists = np.zeros((sample_size, sample_size), dtype=np.float32)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(intra_op_parallelism_threads=self.NUM_THREADS)) as sess:
            dist_op = self.ops['dist_op']
            v1, v2 = self.vars['v1_h'], self.vars['v2_h']
            for i in range(sample_size):
                X_dists[i] = sess.run(dist_op, feed_dict={v1:self.X_train[i], v2:self.X_train})

            K_ridge_op =self.ops['K_ridge_op']
            X_dists_ph = self.vars['X_dists_h']
            ridge_ph = self.vars['ridge_h']

            self.K = sess.run(K_ridge_op, feed_dict={X_dists_ph:X_dists, ridge_ph:ridge})

            K_ph = self.vars['K_h']
            K_inv_op = self.ops['K_inv_op']
            self.K_inv = sess.run(K_inv_op, feed_dict={K_ph:self.K})

            xy_op = self.ops['xy_op']
            K_inv_ph = self.vars['K_inv_h']
            yt_ph = self.vars['yt_h']
            self.xy_ = sess.run(xy_op, feed_dict={K_inv_ph:self.K_inv, yt_ph:self.y_train})

        return self

    def predict(self, X_test):
        self.check_fitted();
        X_test = np.float32(self.check_array(X_test))
        test_size = X_test.shape[0]
        sample_size = self.X_train.shape[0]

        arr_offset = 0;
        yhats = np.zeros([test_size, 1])
        sigmas = np.zeros([test_size, 1])

        with tf.Session(graph=self.graph, config=tf.ConfigProto(intra_op_parallelism_threads=self.NUM_THREADS)) as sess:
            dist_op = self.ops['dist_op']
            v1, v2 = self.vars['v1_h'], self.vars['v2_h']

            K_op = self.ops['K_op']
            X_dists = self.vars['X_dists_h']

            yhat_ = self.ops['yhat_op']
            K_inv_ph = self.vars['X_dists_h']
            K2 = self.vars['K2_h']
            K3 = self.vars['K3_h']
            xy_ph = self.vars['xy_h']

            while arr_offset < test_size:
                if arr_offset + GPR.BATCH_SIZE > test_size:
                    end_offset = test_size
                else:
                    end_offset = arr_offset + GPR.BATCH_SIZE

                X_test_batch = X_test[arr_offset:end_offset]
                batch_len = end_offset - arr_offset

                dists1 = np.zeros([sample_size, batch_len])
                for i in range(sample_size):
                    dists1[i] = sess.run(dist_op, feed_dict={v1:self.X_train[i], v2:X_test_batch})
                sig_val = self.ops['sig_op']
                K2_ = sess.run(K_op, feed_dict={X_dists, dists1})
                yhat = sess.run(yhat_, feed_dict={K2:K2_, xy_ph:self.xy_})
                dists2 = np.zeros([batch_len,batch_len])
                for i in range(batch_len):
                    dists2[i] = sess.run(dist_op, feed_dict={v1:X_test_batch[i], v2:X_test_batch})
                K3_ = sess.run(K_op, feed_dict={X_dists:dists2})

                sigma = np.zeros([1,batch_len], np.float32)
                sigma[0] = sess.run(sig_val, feed_dict={K_inv_ph:self.K_inv, K2:K2_, K3:K3_})
                sigma = np.transpose(sigma)
                yhats[arr_offset:end_offset] = yhat
                sigmas[arr_offset:end_offset] = sigma
                arr_offset = end_offset

            self.check_output(yhats)
            self.check_output(sigmas)
            return GPRResult(yhats, sigmas)


    def get_params(self, deep=True):
        return {"length_scale": self.length_scale, "magnitude": self.magnitude, "X_train": self.X_train, "y_train": self.y_train, "xy_": self.xy_, "K": self.K, "K_inv": self.K_inv}

    def set_params(self, **parameters):
        for param, val in parameters.iteritems():
            setattr(self, param, val)
        return self

    def _reset(self):
        import gc

        self.X_train = None
        self.y_train = None
        self.xy_ = None
        self.K = None
        self.K_inv = None
        self.graph = None
        self.build_graph()
        gc.collect()


class GPR_GD(GPR):
    DEFAULT_LENGTH_SCALE = 1.0
    DEFAULT_MAGNITUDE = 1.0
    DEFAULT_RIDGE = 1.0
    DEFAULT_LEARNING_RATE = 0.01
    DEFAULT_EPSILON = 1e-6
    DEFAULT_MAX_ITER = 100
    DEFAULT_RIDGE = 1.0
    DEFAULT_SIGMA_MULTIPLIER = 3.0
    DEFAULT_MU_MULTIPLIER = 1.0

    GP_BETA_UCB = "UCB"
    GP_BETA_CONST = "CONST"

    def __init__(self, length_scale=DEFAULT_LENGTH_SCALE, 
                 magnitude=DEFAULT_MAGNITUDE, 
                 learning_rate=DEFAULT_LEARNING_RATE, 
                 epsilon=DEFAULT_EPSILON, 
                 max_iter=DEFAULT_MAX_ITER, 
                 sigma_multiplier=DEFAULT_SIGMA_MULTIPLIER, 
                 mu_multiplier = DEFAULT_MU_MULTIPLIER):
        super(GPR_GD, self).__init__(length_scale, magnitude)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.sigma_multiplier = sigma_multiplier
        self.mu_multiplier = mu_multiplier

    def fit(self, X_train, y_train, ridge = DEFAULT_RIDGE):
        super(GPR_GD, self).fit(X_train,y_train,ridge)

        with tf.Session(graph=self.graph, 
                config=tf.ConfigProto(intra_op_parallelism_threads=self.NUM_THREADS)) as sess:
            xt_ = tf.Variable(self.X_train[0], tf.float32)
            xt_ph = tf.placeholder(tf.float32)
            xt_assign_op = xt_.assign(xt_ph)

            init = tf.global_variables_initializer()
            sess.run(init)
            K2_mat = tf.transpose(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(xt_, self.X_train),2),1)),0))
            K2__ = tf.cast(self.magnitude * tf.exp(-K2_mat/self.length_scale),tf.float32)
            yhat_gd = tf.cast(tf.matmul(tf.transpose(K2__), self.xy_),tf.float32)
            sig_val = tf.cast((tf.sqrt(self.magnitude-tf.matmul(tf.transpose(K2__), tf.matmul(self.K_inv, K2__)) )), tf.float32)

            Loss = tf.squeeze(tf.subtract(self.mu_multiplier*yhat_gd,self.sigma_multiplier*sig_val))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)
            train = optimizer.minimize(Loss)

            self.vars['xt_'] = xt_
            self.vars['xt_ph'] = xt_ph
            self.ops['xt_assign_op'] = xt_assign_op
            self.ops['yhat_gd'] = yhat_gd
            self.ops['sig_val2'] = sig_val
            self.ops['loss_op'] = Loss
            self.ops['train_op'] = train

        return self


    def predict(self, X_test, contraint_helper=None, categorical_feature_method='hillclimbing',categorical_feature_steps=3):
        self.check_fitted();
        X_test = np.float32(self.check_array(X_test))
        test_size = X_test.shape[0]
        nfeats = self.X_train.shape[1]

        arr_offset = 0
        yhats = np.zeros([test_size, 1])
        sigmas = np.zeros([test_size, 1])
        minLs = np.zeros([test_size, 1])
        minL_confs = np.zeros([test_size,nfeats])

        with tf.Session(graph=self.graph, config=tf.ConfigProto(intra_op_parallelism_threads=self.NUM_THREADS)) as sess:
            while arr_offset < test_size:
                if arr_offset+GPR.BATCH_SIZE > test_size:
                    end_offset = test_size
                else:
                    end_offset = arr_offset + GPR.BATCH_SIZE;
                
                X_test_batch = X_test[arr_offset:end_offset]
                batch_len = end_offset - arr_offset

                xt_ = self.vars['xt_']
                init = tf.global_variables_initializer()
                sess.run(init)

                sig_val = self.ops['sig_val2']
                yhat_gd = self.ops['yhat_gd']
                Loss = self.ops['loss_op']
                train = self.ops['train_op']

                xt_ph = self.vars['xt_ph']
                assign_op = self.ops['xt_assign_op']

                yhat = np.empty((batch_len, 1))
                sigma = np.empty((batch_len, 1))
                minL = np.empty((batch_len, 1))
                minL_conf = np.empty((batch_len, nfeats))

                for i in range(batch_len):
                    print "---------------------------------------------------"
                    yhats_it = np.empty((self.max_iter+1,))*np.nan
                    sigmas_it = np.empty((self.max_iter+1,))*np.nan
                    losses_it = np.empty((self.max_iter+1,))*np.nan
                    confs_it = np.empty((self.max_iter+1,nfeats))*np.nan

                    sess.run(assign_op, feed_dict={xt_ph:X_test_batch[i]})
                    for step in range(self.max_iter):
                        print "Sample {}, iter {}: ".format(i, step)
                        yhats_it[step] = sess.run(yhat_gd)[0][0]
                        sigmas_it[step] = sess.run(sig_val)[0][0]
                        losses_it[step] = sess.run(Loss)
                        confs_it[step] = sess.run(xt_)

                        print "   yhat:  {}".format(yhats_it[step])
                        print "   sigma:  {}".format(sigmas_it[step])
                        print "   loss:  {}".format(losses_it[step])
                        print "   conf:  {}".format(confs_it[step])
                        sess.run(train)
                    if step == self.max_iter - 1:
                        yhats_it[-1] = sess.run(yhat_gd)[0][0]
                        sigmas_it[-1] = sess.run(sig_val)[0][0]
                        losses_it[-1] = sess.run(Loss)
                        confs_it[-1] = sess.run(xt_)
                        assert np.all(np.isfinite(yhats_it))
                        assert np.all(np.isfinite(sigmas_it))
                        assert np.all(np.isfinite(losses_it))
                        assert np.all(np.isfinite(confs_it))

                    if np.all(~np.isfinite(losses_it)):
                        min_loss_idx = 0
                    else:
                        min_loss_idx = np.nanargmin(losses_it)

                    yhat[i] = yhats_it[min_loss_idx]
                    sigma[i] = sigmas_it[min_loss_idx]
                    minL[i] = losses_it[min_loss_idx]
                    minL_conf[i] = confs_it[min_loss_idx]

                minLs[arr_offset:end_offset] = minL
                minL_confs[arr_offset:end_offset] = minL_conf
                yhats[arr_offset:end_offset] = yhat
                sigmas[arr_offset:end_offset] = sigma
                arr_offset = end_offset

        self.check_output(yhats)
        self.check_output(sigmas)
        self.check_output(minLs)
        self.check_output(minL_confs)

        return GPR_GDResult(yhats, sigmas, minLs, minL_confs)

def gd_tf(xs, ys, xt, ridge, length_scale=1.0, magnitude=1.0, max_iter=50):
    print "xs = {}".format(xs.shape)
    print "ys = {}".format(ys.shape)
    print "xt = {}".format(xt.shape)

    with tf.Graph().as_default():
        sample_size = xs.shape[0]
        nfeats = xs.shape[1]
        test_size = xt.shape[0]
        ini_size = xt.shape[0]

        yhats = np.zeros([test_size,1])
        sigmas = np.zeros([test_size,1])
        minL = np.zeros([test_size,1])
        new_conf = np.zeros([test_size,nfeats])

        xs = np.float32(xs)
        ys = np.float32(ys)

        xt_ = tf.Variable(xt[0], tf.float32)

        sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
        init = tf.global_variables_initializer()
        sess.run(init)

        ridge = np.float32(ridge)

        v1 = tf.placeholder(tf.float32, name="v1")
        v2 = tf.placeholder(tf.float32, name="v2")
        dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1,v2),2),1))

        tmp = np.zeros([sample_size,sample_size])

        for i in range(sample_size):
            tmp[i] = sess.run(dist, feed_dict={v1:xs[i], v2:xs})

        tmp = tf.cast(tmp,tf.float32)
        K = magnitude*tf.exp(-tmp/length_scale) + tf.diag(ridge)

        K2_mat = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(xt_,xs),2),1))
        k2_mat = tf.transpose(tf.expand_dims(K2_mat,0))
        K2 = tf.cast(tf.exp(-K2_mat/length_scale),tf.float32)

        x = tf.matmul(tf.matrix_inverse(K), ys)
        x = sess.run(x)

        yhat_ = tf.cast(tf.matmul(tf.transpose(K2), x), tf.float32)
        sig_val = tf.cast((tf.sqrt(magnitude-tf.matmul(tf.transpose(K2), tf.matmul(tf.matrix_inverse(K), K2)) )), tf.float32)

        print sess.run(yhat_).shape
        print sess.run(sig_val).shape


        yhat_ = tf.check_numerics(yhat_, message='yhat: ')
        sig_val = tf.check_numerics(sig_val, message='sig_val: ')
        Loss = tf.squeeze(tf.subtract(yhat_,sig_val))
        Loss = tf.check_numerics(Loss, message='Loss: ')

        print sess.run(Loss)

        optimizer = tf.train.AdamOptimizer(0.1)
        train = optimizer.minimize(Loss)
        inti. tf.global_variables_initializer()
        sess.run(init)

        for i in range(ini_size):
            assign_op = xt_.assign(xt[i])
            sess.run(assign_op)
            for step in range(max_iter):
                print i, step, sess.run(Loss)
                sess.run(train)
            yhats[i] = sess.run(yhat_)[0][0]
            sigmas[i] = sess.run(sig_val)[0][0]
            minL[i] = sess.run(Loss)
            new_conf[i] = sess.run(xt_)
        
        return yhats, sigmas, minL, new_conf


def main():
    check_gd_equivalence()


def create_random_matrices(n_samples=3000, n_feats=12, n_test=444):
    X_train = np.random.rand(n_samples, n_feats)
    y_train = np.random.rand(n_samples, 1)
    X_test = np.random.rand(n_test, n_feats)

    length_scale = np.random.rand()
    magnitude = np.random.rand()
    ridge = np.ones(n_samples)*np.random.rand()

    return X_train, y_train, X_test, length_scale, magnitude, ridge

def check_equivalence():
    X_train, y_train, X_test, length_scale, magnitude, ridge = create_random_matrices()

    print "Running GPR method ..."
    start = time()
    yhats1, sigmas1, eips1 = gp_tf(X_train, y_train, X_test, ridge, length_scale, magnitude)
    print "GPR method: {0:.3f} seconds".format(time()-start)

    print "Running GPR class ..."
    start = time()
    gpr = GPR(length_scale, magnitude)
    gpr.fit(X_train, y_train, ridge)
    yhats2, sigmas2, eips2 = gpr.predict(X_test)
    print "GPR class: {0:.3f} seconds".format(time()-start)

    assert np.allclose(yhats1,yhats2)
    assert np.allclose(sigmas1,sigmas2)
    assert np.allclose(eips1,eips2)

def check_gd_equivalence():
    X_train, y_train, X_test, length_scale, magnitude, ridge = create_random_matrices(n_test=2)
 
    print "Running GPR_GD class ..."
    start = time()
    gpr_gd = GPR_GD(length_scale, magnitude, max_iter=5)
    gpr_gd.fit(X_train, y_train, ridge)
    ypres2 = gpr_gd.predict(X_test)
    print "GPR_GD class: {0:.3f} seconds\n".format(time()-start)

def read_matrices(n_feats, filepath):
    train_file = open(filepath)
    train_data = train_file.readlines() 
    n_samples = len(train_data)  
    X_train = np.array([s.strip().split(",")[0: (n_feats-1)] for s in train_data])
    Y_train = np.array([s.strip().split(",")[(n_feats-1):] for s in train_data])
    
    length_scale = np.random.rand()
    magnitude = np.random.rand()
    ridge = np.ones(n_samples)*np.random.rand()
    
    return X_train, Y_train, length_scale, magnitude, ridge

def explore_x_test(n_feats, test_size, X_train):
    return X_test

def LHS_samples(n_feats, test_size):
    lhs = lhsmu.sample(n_feats, test_size)
    X_test = lhsmdu.createRandomStandardUniformMatrix(n_feats,test_size)
    return X_test

def select_x_test(X_train):
    X_test = X_train
    return X_test


if __name__ == "__main__":
    main()
