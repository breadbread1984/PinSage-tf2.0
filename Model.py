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
    # neighbor_set.shape = (neighbor number);
    neighbor_set = inputs[2];
    tf.debugging.Assert(tf.equal(tf.shape(tf.shape(neighbor_set))[0], 1), [neighbor_set.shape]);
    # node_id.shape = ();
    node_id = inputs[3];
    tf.debugging.Assert(tf.equal(tf.shape(tf.shape(node_id))[0], 0), [node_id.shape]);
    # neighbor_embeddings.shape = (batch, neighbor number, in channels)
    neighbor_embeddings = tf.keras.layers.Lambda(lambda x, neighbor_set: tf.gather(x, neighbor_set, axis = 1), arguments = {'neighbor_set': neighbor_set})(embeddings);
    # neighbor_hiddens.shape = (batch, neighbor number, hidden channels)
    neighbor_hiddens = self.Q(neighbor_embeddings);
    # incoming weights.shape = (node number, 1)
    incoming_weights = tf.keras.layers.Lambda(lambda x, node_id: tf.gather(x, [node_id], axis = 1), arguments = {'node_id': node_id})(weights);
    # neighbor weights.shape = (1, neighbor number, 1)
    neighbor_weights = tf.keras.layers.Lambda(lambda x, neighbor_set: tf.gather(x, neighbor_set, axis = 0), arguments = {'neighbor_set': neighbor_set})(incoming_weights);
    neighbor_weights = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 0))(neighbor_weights);
    # weighted_sum_hidden.shape = (batch, hidden channels)
    weighted_sum_hidden = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = 1) / (tf.math.reduce_sum(x[1], axis = 1) + 1e-6))([neighbor_hiddens, neighbor_weights]);
    # node_embedding.shape = (batch, in_channels)
    node_embedding = tf.keras.layers.Lambda(lambda x, node_id: tf.squeeze(tf.gather(x, [node_id], axis = 1), axis = 1), arguments = {'node_id': node_id})(embeddings);
    # concated_hidden.shape = (batch, in_channels + hidden channels)
    concated_hidden = tf.keras.layers.Concatenate(axis = -1)([node_embedding, weighted_sum_hidden]);
    # hidden_new.shape = (batch, hidden_channels)
    hidden_new = self.W(concated_hidden);
    # normalized.shape = (batch, hidden_channels)
    normalized = tf.keras.layers.Lambda(lambda x: x / (tf.norm(x, axis = 1, keepdims = True) + 1e-6))(hidden_new);
    return normalized;

class PinSage(tf.keras.Model):

  def __init__(self, hidden_channels, graph):

    # hidden_channels is list containing output channels of every convolve.
    assert type(hidden_channels) is list;
    assert type(graph) is nx.classes.graph.Graph;
    super(PinSage, self).__init__();
    # create convolves for every layer.
    self.convs = list();
    for i in range(len(hidden_channels)):
      self.convs.append(Convolve(hidden_channels[i]));
    # get edge weights from pagerank. (from, to)
    self.edge_weights = self.pagerank(graph);
    
  def call(self, inputs):

    # embeddings.shape = (batch, node number, in channels)
    embeddings = inputs[0];
    tf.debugging.Assert(tf.equal(tf.shape(embeddings)[1] == tf.shape(self.weights)[0]), [embeddings.shape]);

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
    # edge weight is calculated by multiply of two related nodes.
    weights = tf.math.minimum(weights, tf.linalg.matmul(tf.transpose(v),v));
    # normalized edge weights line by line.
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

