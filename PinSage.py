#!/usr/bin/python3

import numpy as np;
import networkx as nx;
import tensorflow as tf;

class Convolve(tf.keras.Model):

  def __init__(self, hidden_channels):

    super(Convolve, self).__init__();
    self.Q = tf.keras.layers.Dense(units = hidden_channels, activation = tf.keras.layers.LeakyReLU());
    self.W = tf.keras.layers.Dense(units = hidden_channels, activation = tf.keras.layers.LeakyReLU());

  def call(self, inputs):

    # embeddings.shape = (batch, node number, in_channels)
    embeddings = inputs[0];
    tf.debugging.Assert(tf.equal(tf.shape(tf.shape(embeddings))[0], 3), [embeddings.shape]);
    # weights.shape = (node number, node number)
    weights = inputs[1];
    tf.debugging.Assert(tf.equal(tf.shape(tf.shape(weights))[0], 2), [weights.shape]);
    tf.debugging.Assert(tf.equal(tf.shape(weights)[0], tf.shape(weights)[1]), [weights.shape]);
    tf.debugging.Assert(tf.equal(tf.shape(embeddings)[1], tf.shape(weights)[0]), [weights.shape]);
    # neighbor_set.shape = (node number, neighbor number);
    neighbor_set = inputs[2];
    tf.debugging.Assert(tf.equal(tf.shape(tf.shape(neighbor_set))[0], 2), [neighbor_set.shape]);
    # neighbor_embeddings.shape = (batch, node number, neighbor number, in channels)
    # gather from tensor of dim (node number, batch, in_channel)
    # with indices of dim (node number, neighbor num, 1)
    # to get tensor of dim (node number, neighor num, batch, in_channel)
    # transpose to get tensor of dim (batch, node_number, neighbor num, in channel)
    neighbor_embeddings = tf.keras.layers.Lambda(lambda x, neighbor_set:
      tf.transpose(
        tf.gather_nd(
          tf.transpose(x, (1, 0, 2)), 
          tf.expand_dims(neighbor_set, axis = -1)
        ),
        (2, 0, 1, 3)
      ), 
      arguments = {'neighbor_set': neighbor_set}
    )(embeddings);
    # neighbor_hiddens.shape = (batch, node_number, neighbor number, hidden channels)
    neighbor_hiddens = self.Q(neighbor_embeddings);
    # indices.shape = (node_number, neighbor number, 2)
    node_nums = tf.keras.layers.Lambda(lambda x: tf.tile(tf.expand_dims(tf.range(tf.shape(x)[0]), axis = 1),(1, tf.shape(x)[1])))(neighbor_set);
    indices = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis = -1))([node_nums, neighbor_set]);
    # neighbor weights.shape = (node number, neighbor number)
    neighbor_weights = tf.keras.layers.Lambda(lambda x, indices: tf.gather_nd(x, indices), arguments = {'indices': indices})(weights);
    # neighbor_weights.shape = (1, node number, neighbor number, 1)
    neighbor_weights = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, 0), -1))(neighbor_weights);
    # weighted_sum_hidden.shape = (batch, node_number, hidden channels)
    weighted_sum_hidden = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = 2) / (tf.math.reduce_sum(x[1], axis = 2) + 1e-6))([neighbor_hiddens, neighbor_weights]);
    # concated_hidden.shape = (batch, node number, in_channels + hidden channels)
    concated_hidden = tf.keras.layers.Concatenate(axis = -1)([embeddings, weighted_sum_hidden]);
    # hidden_new.shape = (batch, node number, hidden_channels)
    hidden_new = self.W(concated_hidden);
    # normalized.shape = (batch, node number, hidden_channels)
    normalized = tf.keras.layers.Lambda(lambda x: x / (tf.norm(x, axis = 2, keepdims = True) + 1e-6))(hidden_new);
    return normalized;

class PinSage(tf.keras.Model):

  def __init__(self, hidden_channels, graph = None, edge_weights = None):

    # hidden_channels is list containing output channels of every convolve.
    assert type(hidden_channels) is list;
    if graph is not None: assert type(graph) is nx.classes.graph.Graph;
    if edge_weights is not None: assert type(edge_weights) is list;
    super(PinSage, self).__init__();
    # create convolves for every layer.
    self.convs = list();
    for i in range(len(hidden_channels)):
      self.convs.append(Convolve(hidden_channels[i]));
    # get edge weights from pagerank. (from, to)
    self.edge_weights = self.pagerank(graph) if graph is not None else edge_weights;
    
  def call(self, inputs):

    # embeddings.shape = (batch, node number, in channels)
    embeddings = inputs[0];
    tf.debugging.Assert(tf.equal(tf.shape(embeddings)[1], tf.shape(self.edge_weights)[0]), [embeddings.shape]);
    # sample_neighbor_num.shape = ()
    sample_neighbor_num = inputs[1];
    tf.debugging.Assert(tf.equal(tf.shape(tf.shape(sample_neighbor_num))[0], 0), [sample_neighbor_num]);
    # sample a fix number of neighbors according to edge weights.
    # neighbor_set.shape = (node num, neighbor num)
    neighbor_set = tf.random.categorical(self.edge_weights, sample_neighbor_num);
    for conv in self.convs:
      embeddings = conv([embeddings, self.edge_weights, neighbor_set]);
    return embeddings;

  def pagerank(self, graph, damp_rate = 0.2):

    # node id must from 0 to any nature number.
    node_ids = sorted([id for id in graph.node]);
    assert node_ids == list(range(len(node_ids)));
    # adjacent matrix.
    weights = np.zeros((len(graph.node), len(graph.node),), dtype = np.float32);
    for f in graph.nodes:
      for t in list(graph.adj[f]):
        weights[f,t] = 1.;
    weights = tf.constant(weights);
    # normalize adjacent matrix line by line.
    line_sum = tf.math.reduce_sum(weights, axis = 1, keepdims = True) + 1e-6;
    normalized = weights / line_sum;
    # dampping vector.
    dampping = tf.ones((len(graph.nodes),), dtype = tf.float32);
    dampping = dampping / tf.constant(len(graph.nodes), dtype = tf.float32);
    dampping = tf.expand_dims(dampping, 0); # line vector.
    # learning pagerank.
    v = dampping;
    while True:
      v_updated = (1 - damp_rate) * tf.linalg.matmul(v, normalized) + damp_rate * dampping;
      d = tf.norm(v_updated - v);
      if tf.equal(tf.less(d,1e-4),True): break;
      v = v_updated;
    # edge weight is pagerank.
    weights = weights * tf.tile(v,(tf.shape(weights)[0],1));
    line_sum = tf.math.reduce_sum(weights, axis = 1, keepdims = True) + 1e-6;
    normalized = weights / line_sum;
    return normalized;

if __name__ == "__main__":

  assert tf.executing_eagerly();
  g = nx.Graph();
  g.add_node(0);
  g.add_node(1);
  g.add_node(2);
  g.add_node(3);
  g.add_edge(0,1);
  g.add_edge(0,2);
  g.add_edge(2,3);
  pinsage = PinSage([10,10,10], g);
  print(pinsage.edge_weights)

