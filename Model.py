#!/usr/bin/python3

import tensorflow as tf;
import networkx as nx;

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
      self.convs.append(Convolve(channels[i]));
    # node id must from 0 to any nature number.
    node_ids = sorted([id for id in graph.node]);
    assert node_ids == list(range(len(node_ids)));
    # get weight from pagerank.
    
  def call(self, inputs):

    embeddings = inputs[0];

if __name__ == "__main__":

  assert tf.executing_eagerly();
  convolve = Convolve(10);
