import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import datetime


class Node:
    def __init__(self, value=0, child=[], network=None):
        self.value = value
        self.child = child
        self.father = None
        self.item_id = None
        self.network = network

    def get_child(self):
        return self.child

    def get_child_id(self):
        # 找出是第几个孩子，如果是其儿子，则返回子节点,否则返回-1
        c = self.father.get_child()
        for i, t in enumerate(c):
            if t == self:
                return i
        else:
            return -1


class HuffmanTree:
    def __init__(self, value, id, state_dim, user_state_dim, branch=16, hidden=64, learning_rate=1e-3, seed=1,
                 max_seq_length=32):
        self.value = value
        self.id = id
        self.state_dim = state_dim
        self.user_state_dim = user_state_dim
        self.branch = branch
        self.hidden_size = hidden
        self.lr = learning_rate
        self.seed, self.max_seq_length = seed, max_seq_length
        assert len(self.value) == len(self.id)
        self.tree = [Node(value=_) for _ in value]  # initialize the tree nodes
        self.root_child_count = 0
        for i, t in enumerate(self.tree):
            t.item_id = id[i]
        self.tree_copy = self.tree.copy()
        self.buildTree()
        self.codebook = self.getCode()
        self.input_state = tf.placeholder(dtype=tf.float32, shape=[None, self.max_seq_length, state_dim])
        self.input_state_length = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.input_reward = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.input_action = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.input_user_feature = tf.placeholder(dtype=tf.float32, shape=[None, self.user_state_dim])
        self.buildNetwork_v4()  # 构建网络
        self.action_prob_list = self.build_tree_embedding()
        action_prob = tf.nn.embedding_lookup(self.action_prob_list, self.input_action)
        self.loss = -tf.reduce_mean(self.input_reward * tf.log(action_prob + 1e-13))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def buildTree(self):
        # Organize the tree nodes to huffman tree.
        def getValue(n):
            return n.value

        while len(self.tree) >= self.branch:
            self.tree.sort(key=lambda n: n.value)
            child = self.tree[:self.branch]
            v = sum([getValue(_) for _ in child])
            new_node = Node(value=v, child=child)
            for c in child:
                c.father = new_node
            self.tree = self.tree[self.branch:]
            self.tree.append(new_node)
        self.root_child_count = len(self.tree)
        if len(self.tree) > 1:  # 如果最后根节点大于1，则需要新建个根节点
            v = sum([getValue(_) for _ in self.tree])
            new_node = Node(value=v, child=self.tree)
            for c in self.tree:
                c.father = new_node
            self.tree = new_node
        else:
            self.tree = self.tree[0]

    def getCode(self):
        # 得到所有的code
        codes = {}
        for node in self.tree_copy:
            code = ''
            node_id = node.item_id
            while node.father is not None:
                code = str(node.get_child_id()) + ' ' + code
                node = node.father
            codes[node_id] = code
        return codes

    def feature_extract(self, input_state, input_state_length):
        '''
        Create RNN feature extractor for recommendation systems.
        :return:
        '''
        with tf.variable_scope('feature_extract', reuse=tf.AUTO_REUSE):
            basic_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
            _, states = tf.nn.dynamic_rnn(basic_cell, input_state, dtype=tf.float32, sequence_length=input_state_length)
            # basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size)
            # _, states = tf.nn.dynamic_rnn(basic_cell, input_state, dtype=tf.float32, sequence_length=input_state_length)
            # states = states[0]
        return states

    def feature_extract_atem(self, input_state, input_state_length):
        '''
        build the ATEM model
        :param input_state: input state.
        :param input_length: input state length
        :return: the ATEM model
        '''
        with tf.variable_scope('feature_extractor_atem', reuse=False):
            item_embedding = slim.fully_connected(input_state, self.hidden_size)  # [N, max_seq_length,hidden_size]
            attention_layer = slim.fully_connected(item_embedding, 1)
            attention_weight = tf.nn.softmax(attention_layer, 1)  # [N, max_seq_length, 1]
            output_layer = attention_weight * item_embedding  # [N, 1, hidden_size]
            output_layer = tf.reduce_sum(output_layer, axis=1)
        return output_layer

    def feature_extract_caser(self, input_state, input_state_length):
        with tf.variable_scope('feature_extract_caser', reuse=False):
            input_state = tf.expand_dims(input_state, axis=3)
            output_v = tf.layers.conv2d(input_state, self.hidden_size, [self.max_seq_length, 1],
                                        activation=tf.nn.relu)
            out_v = tf.layers.flatten(output_v)  # [N, self.state_dim*self.hidden_size]
            output_hs = list()
            for h in np.arange(self.max_seq_length) + 1:
                conv_out = tf.layers.conv2d(input_state, self.hidden_size, [h, self.state_dim],
                                            activation=tf.nn.relu)
                conv_out = tf.reshape(conv_out, [-1, self.max_seq_length - h + 1, self.hidden_size])
                pool_out = tf.layers.max_pooling1d(conv_out, [self.max_seq_length + 1 - h], 1)
                pool_out = tf.squeeze(pool_out, 1)
                output_hs.append(pool_out)
            out_h = tf.concat(output_hs, axis=1)
            out = tf.concat([out_v, out_h], axis=1)
            z = tf.layers.dense(out, self.hidden_size, activation=tf.nn.relu)
        return z

    def feature_extract_drr_att_our(self, input_state):
        with tf.variable_scope("feature_extract_drr_att"):
            item_embedding = slim.fully_connected(input_state, self.hidden_size)  # [N, max_seq_length, hidden]
            user_embedding = slim.fully_connected(self.input_user_feature, self.hidden_size)  # [N, hidden]
            user_embedding = tf.reshape(user_embedding, [-1, 1, self.hidden_size])
            all_input = tf.concat([item_embedding, user_embedding], axis=1)  # [N, max_seq_length+1, hidden]
            attention_layer = slim.fully_connected(all_input, self.hidden_size)
            attention_layer = slim.fully_connected(attention_layer, 1)
            attention_weight = tf.nn.softmax(attention_layer, 1)
            output = attention_weight * all_input
            att_output = tf.reduce_sum(output, axis=1)
            # all_output = tf.concat([att_output, average_pool], axis=1)
            # return all_output
            return att_output

    def mlp(self, id=None, softmax_activation=False):
        '''
        Create a multi-layer neural network as tree node.
        :param id: tree node id
        :param reuse: reuse for the networks
        :return: a multi-layer neural network with output dim equals to branch size.
        '''
        with tf.variable_scope('node_' + str(id), reuse=tf.AUTO_REUSE):
            state = self.feature_extract_drr_att_our(self.input_state)
            all_state = tf.concat([self.input_user_feature, state], axis=1)
            l1 = slim.fully_connected(all_state, self.branch)
            # l2 = slim.fully_connected(l1, self.hidden_size)
            # l3 = slim.fully_connected(l2, self.branch)
            if softmax_activation:
                outputs = tf.nn.softmax(l1)
            else:
                outputs = l1
        return outputs

    def buildNetwork_v1(self):
        # Build tree-structured neural networks for each node, without parameter sharing
        queue = []
        current_line = 0
        queue.append([current_line, self.tree])
        count = 0
        while len(queue) > 0:
            line, node = queue.pop(0)
            if line != current_line:  # for parameter sharing
                current_line = line
            node.network = self.mlp(id=str(count), softmax_activation=True)
            if len(node.child) != 0:
                for n in node.child:
                    queue.append([line + 1, n])
                    count += 1

    def buildNetwork_v2(self):
        # Build tree-structured neural networks for each node, layer parameter sharing
        queue = []
        current_line = 0
        queue.append([current_line, self.tree])
        count = 0
        networks = [self.mlp(id=str(count), softmax_activation=True)]
        while len(queue) > 0:
            line, node = queue.pop(0)
            if line != current_line:  # for parameter sharing
                current_line = line
                networks.append(self.mlp(id=str(count), softmax_activation=True))
            node.network = networks[-1]
            if len(node.child) != 0:
                for n in node.child:
                    queue.append([line + 1, n])
                    count += 1

    def buildNetwork_v3(self):
        # Build tree-structured neural networks for each node, layer parameter sharing with extra decision unit
        queue = []
        current_line = 0
        queue.append([current_line, self.tree])
        count = 0
        t_node = self.mlp(id=str(count), softmax_activation=False)
        # t_node = slim.fully_connected(t_node, num_outputs=self.branch, activation_fn=tf.nn.relu)
        # t_node = slim.fully_connected(t_node, num_outputs=self.branch, activation_fn=tf.nn.relu)
        t_node = slim.fully_connected(t_node, num_outputs=self.branch, activation_fn=tf.nn.softmax)
        networks = [t_node]
        while len(queue) > 0:
            line, node = queue.pop(0)
            if line != current_line:  # for parameter sharing
                current_line = line
                t_node = self.mlp(id=str(count), softmax_activation=False)
                # t_node = slim.fully_connected(t_node, num_outputs=self.branch, activation_fn=tf.nn.relu)
                # t_node = slim.fully_connected(t_node, num_outputs=self.branch, activation_fn=tf.nn.relu)
                t_node = slim.fully_connected(t_node, num_outputs=self.branch, activation_fn=tf.nn.softmax)
                networks.append(t_node)
            node.network = networks[-1]
            if len(node.child) != 0:
                for n in node.child:
                    queue.append([line + 1, n])
                    count += 1

    def buildNetwork_v4(self):
        # Build tree-structured neural networks for each node, all parameter sharing with extra decision unit
        queue = []
        current_line = 0
        queue.append([current_line, self.tree])
        count = 0
        root_node = self.mlp(id=str('node'), softmax_activation=False)
        # t_node = slim.fully_connected(t_node, num_outputs=self.branch, activation_fn=tf.nn.relu)
        # t_node = slim.fully_connected(t_node, num_outputs=self.branch, activation_fn=tf.nn.relu)
        t_node = slim.fully_connected(root_node, num_outputs=self.branch, activation_fn=tf.nn.softmax)
        networks = [t_node]
        while len(queue) > 0:
            line, node = queue.pop(0)
            if line != current_line:  # for parameter sharing
                current_line = line
                # t_node = slim.fully_connected(t_node, num_outputs=self.branch, activation_fn=tf.nn.relu)
                # t_node = slim.fully_connected(t_node, num_outputs=self.branch, activation_fn=tf.nn.relu)
                t_node = slim.fully_connected(root_node, num_outputs=self.branch, activation_fn=tf.nn.softmax)
                networks.append(t_node)
            node.network = networks[-1]
            if len(node.child) != 0:
                for n in node.child:
                    queue.append([line + 1, n])
                    count += 1

    def learn(self, user_feature, state, state_length, action, reward):
        state_length = state_length.flatten()
        action = action.flatten()
        reward = reward.flatten()
        user_feature = np.reshape(user_feature, [-1, self.user_state_dim])
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.input_state: state,
            self.input_state_length: state_length,
            self.input_reward: reward, self.input_action: action,
            self.input_user_feature: user_feature})
        return np.mean(loss)

    def getAction(self, input_user_feature, state, state_length):
        # 根据输入的state和state_length得到一个动作的采样
        def softmax(x):
            """Compute the softmax in a numerically stable way."""
            x = x - np.max(x)
            exp_x = np.exp(x)
            softmax_x = exp_x / np.sum(exp_x)
            return softmax_x

        t_node = self.tree
        action_prob = self.sess.run(t_node.network,
                                    feed_dict={self.input_state: state, self.input_state_length: state_length,
                                               self.input_user_feature: input_user_feature})
        count = 0
        while len(t_node.child) != 0:
            if count == 0:  # root node
                np.random.seed(self.seed)
                c = np.random.choice(self.root_child_count, p=softmax(action_prob.flatten()[:self.root_child_count]))
            else:
                np.random.seed(self.seed)
                c = np.random.choice(self.branch, p=action_prob.flatten())
            t_node = t_node.child[c]
            action_prob = self.sess.run(t_node.network,
                                        feed_dict={self.input_state: state, self.input_state_length: state_length,
                                                   self.input_user_feature: input_user_feature})
            count += 1
        return t_node.item_id

    def getActionProb(self, action):
        # 根据输入的state和state_length找出某个特定action的概率,是个tensor
        path = list(map(int, self.codebook[action].split()))
        node = self.tree
        action_prob = 1
        for i in range(len(path)):
            t = node.network[:, path[i]]
            action_prob *= node.network[:, path[i]]
            node = node.child[path[i]]
        return action_prob

    def state_padding(self, input_state, input_state_length):
        if input_state_length > self.max_seq_length:
            input_state = input_state[-self.max_seq_length:]
            input_state_length = self.max_seq_length
        input_state = np.array(input_state).reshape([input_state_length, self.state_dim])
        if input_state_length < self.max_seq_length:
            # padding the zero matrix.
            padding_mat = np.zeros([self.max_seq_length - input_state_length, self.state_dim])
            input_state = np.vstack((input_state, padding_mat))
        return input_state

    def build_tree_embedding(self):
        # 构建从id到variable的embedding
        embedding = []
        for i in range(len(self.id)):
            embedding.append(self.getActionProb(i))
        return tf.stack(embedding)

    def get_all_action_prob(self, input_user_feature, input_state, input_state_length):
        input_state = np.reshape(input_state, [-1, self.max_seq_length, self.state_dim])
        input_user_feature = np.reshape(input_user_feature, [-1, self.user_state_dim])
        input_state_length = np.reshape(input_state_length, [-1, ])
        return self.sess.run(self.action_prob_list, feed_dict={self.input_state: input_state,
                                                               self.input_state_length: input_state_length,
                                                               self.input_user_feature: input_user_feature})


# for debugging
if __name__ == '__main__':
    id = np.arange(100)
    np.random.seed(1)
    frequency = np.arange(100)
    h_tree = HuffmanTree(value=frequency, id=id, branch=3, state_dim=10)

    state = np.random.rand(1, 32, 10)
    state_length = 10
    action = h_tree.getAction(state, [state_length])
    action_prob = h_tree.getActionProb(1)

    for i in range(100):
        start = datetime.datetime.now()
        loss = h_tree.learn(state, [state_length], [1], [10])
        end = datetime.datetime.now()
        print('Step {}\n loss:{} time:{}'.format(i, loss, (end - start).seconds))
    print('Training time:{}'.format((end - start).seconds))
