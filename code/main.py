import tensorflow as tf
import random
import numpy as np
import os

from GCN_SeAN import *

flags = tf.app.flags
flags.DEFINE_string("config_id",'6',"Hyperparam config id")
flags.DEFINE_string("data_dir", "../data/en-dstc2", "Data directory ")
flags.DEFINE_string("data_name", "/preprocessed-dialog-dstc2-final_data.json", "Preprocessed Data Name")
flags.DEFINE_string("logs_dir", "logs", "Logs directory ")
flags.DEFINE_string("checkpoint_dir", 'checkpoints', "checkpoint directory")

flags.DEFINE_string("rnn_unit", 'gru', "GRU or LSTM")
flags.DEFINE_boolean("edge_gate",True,"Use Gating mechanism for GCN edges")
flags.DEFINE_boolean("use_rnn",True,"Use RNN for encoding the Context and Query before using GCN")
flags.DEFINE_boolean("use_dep_labels",False,"Use Dependency Labels in the GCN params")
flags.DEFINE_float("learning_rate", 0.0008, "Learning rate for Adam Optimizer")
flags.DEFINE_float("l2", 0.001, "L2 regularization for the parameters")
flags.DEFINE_integer("batch_size",32, "Batch size for training")
flags.DEFINE_integer("epochs",30, "Number of epochs to train for")
flags.DEFINE_integer("num_hops",1, "Number of hops for GCN")
flags.DEFINE_integer("max_gradient_norm",5, "Max grad norm. 0 for no clipping")
flags.DEFINE_float("dropout", 0.9, "keep probability for keeping unit in GCN")
flags.DEFINE_integer("word_emb_dim",300, "hidden dimensions of the word embeddings.")
flags.DEFINE_integer("rnn_hidden_units",300, "hidden dimensions of the RNN units.")
flags.DEFINE_integer("gcn_hidden_units",300, "hidden dimensions of the GCN nodes.")
flags.DEFINE_integer("seed",1234, "Random Seed")
flags.DEFINE_string("init", 'trunc', "[none,xavier,unit_variance,trunc]")
flags.DEFINE_boolean("train",True,"Train or Infer")
flags.DEFINE_boolean("debug",False,"Debug mode or not")
FLAGS = flags.FLAGS



def create_model(sess,FLAGS):
    """
    Create a new model and if model already exists then restores the saved model
    """
    model =GCN_SeAN(FLAGS)
    
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir+FLAGS.config_id)
    if ckpt:
        model.logger.info("Restoring model parameters from %s" %
              ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)   
    else:
        model.logger.info("Created new model.")
        sess.run(tf.global_variables_initializer())

    return model


def exact_match(p,l):
    """
    Computes per-response accuracy between predictions (p) and ground-truth responses (l)
    """
    c=0
    for i1,i in enumerate(l):
        if p[i1]==l[i1]:
            c+=1
    
    return c/len(l)
 

def save_model(sess,model):
    """
    Save the model after each epoch
    """
    if not os.path.exists(FLAGS.checkpoint_dir+FLAGS.config_id):
        os.makedirs(FLAGS.checkpoint_dir+FLAGS.config_id)
    save_path = model.saver.save(sess, os.path.join(FLAGS.checkpoint_dir+FLAGS.config_id, "model.ckpt"))
        
    model.logger.info("Model saved in file: %s" % save_path)
    

def train(sess,model):
    for epoch in range(model.params.epochs):
        loss,bleu,preds,labels = model.run_epoch(sess,epoch)
        val_loss,val_bleu,val_preds,val_labels = model.predict(sess,'dev')
        
        val_acc = exact_match(val_preds,val_labels)
        model.logger.info('Val Loss:{}, Val BLEU: {}, Val Acc: {}'.format(val_loss, val_bleu,val_acc))
        save_model(sess,model)
    
    
def infer(sess,model,data_name):
    """
    Function to run inference on the computation graph. Dumps the predicted responses and ground truth responses
    into txt files for evaluation:
        sess: Tensorflow Session
        model: The saved model used to perform inference
        data_name: can be 'train', 'test' or 'valid' to perform inference on the train, test and valid datasets
    """    
    loss, bl_test, preds_test, labels_test = model.predict(sess,data_name=data_name)
    model.logger.info('Test Loss: {}, Test BLEU: {}'.format(loss,bl_test))
    
    fp1 =open('predictions'+str(FLAGS.config_id)+'.txt','w+')
    for item in preds_test:
        fp1.write("%s\n"%item)
    fp1.close()
    
    fp2 =open('labels'+str(FLAGS.config_id)+'.txt','w+')
    for item in labels_test:
        fp2.write("%s\n"%item)
    fp2.close()


if __name__=='__main__':
    
    tf.reset_default_graph()
    with tf.Graph().as_default():

        tf.set_random_seed(FLAGS.seed)
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
    
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        with tf.Session(config=config) as sess:

            model = create_model(sess,FLAGS)
            
            if FLAGS.train:
                train(sess,model)
                infer(sess,model,'test')
            else:
                infer(sess,model,'test')
