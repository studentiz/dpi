####################################################################
#### Single-cell multi-omics modeling with deep parametric inference
####################################################################

import sys, os
import pickle
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from scipy.stats import ranksums
from joblib import Parallel, delayed
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from matplotlib.axes import Axes
from anndata import AnnData
from typing import Optional, Union
from scipy.sparse import issparse
from tqdm import tqdm
from tqdm import trange
import textwrap
import time
import datetime
from tensorflow.keras import optimizers
import pandas as pd
import statsmodels.api as sm
import numba as nb
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scipy.io as sp_io
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
import datetime
import os
from sklearn import decomposition
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import scanpy as sc
import random
import csv
from tensorflow.keras.layers import Lambda, Input, Dense, Activation, AlphaDropout, GaussianNoise, GaussianDropout, Layer, Dropout, Multiply, Reshape, RepeatVector, Permute, ReLU, Concatenate, LayerNormalization
from tensorflow.keras.layers import concatenate, multiply, average
from tensorflow.keras import backend as K
from tensorflow.keras.losses import kullback_leibler_divergence, mean_squared_logarithmic_error, binary_crossentropy, poisson, sparse_categorical_crossentropy, squared_hinge, categorical_crossentropy, cosine_similarity, mean_squared_error, logcosh
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.colors as colors
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skbio.stats.composition import clr
import seaborn as sns
import umap

mm_rna = MinMaxScaler()
mm_pro = MinMaxScaler()

mix_model = None
rna_model = None
pro_model = None

def saveobj2file(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    return("save obj to " + filepath)

def loadobj(filepath):
    tempobj = None
    with open(filepath, 'rb') as f:
        tempobj = pickle.load(f)
    
    return tempobj

def Reverse(lst):
    return [ele for ele in reversed(lst)]

def add_genes(sc_data, genes):
    for gene in genes:
        index = sc_data.var.index==gene
        sc_data.var["highly_variable"][index] = True

def preprocessing(sc_data):
    sc_data.var['mt'] = sc_data.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(sc_data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

def normalize(sc_data, protein_expression_obsm_key, **kwargs):
    sc_data.var_names_make_unique()
    sc.pp.log1p(sc_data)
    sc_data.obsm["pro_nor"] = clr(1+sc_data.obsm[protein_expression_obsm_key]).astype("float16")
    
def scale(sc_data):
    sc_data.obsm["rna_observed"] = sc_data.X.astype("float16")
    sc_data.obsm["pro_observed"] = sc_data.obsm["pro_nor"].astype("float16")
    sc_data.mm_rna = mm_rna
    sc_data.mm_pro = mm_pro
    sc_data.obsm["rna_nor"] = sc_data.mm_rna.fit_transform(sc_data.obsm["rna_observed"]).astype("float16")
    sc_data.obsm["pro_nor"] = sc_data.mm_pro.fit_transform(sc_data.obsm["pro_observed"]).astype("float16")

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.0)
    return z_mean + K.exp(0.5*z_log_var) * epsilon

def build_Gamma(x):
    return tf.math.exp(tf.math.lgamma(x+1))

def build_NB(args):
    theta, miu, x = args
    fenmu = theta+miu
    x_log = x
    return build_Gamma(x+theta)/build_Gamma(theta)*tf.pow((theta/fenmu), theta)*tf.pow((miu/fenmu), x_log)

def build_Poisson(args):
    lmd, x = args
    return tf.math.pow(np.e, -1*lmd)*tf.math.pow(lmd, x)/build_Gamma(x)

def loss_plot(sc_data):
    plt.plot(sc_data.estimator.history["loss"],'-o',label='Training_loss')
    plt.plot(sc_data.estimator.history["val_loss"],'-o',label='Validation_loss')
    plt.legend(loc='upper right')
    plt.title("Loss of DPI Model")

def rna_loss_func(rna_mean, rna_sigma, rna_input, rna_denoised, rna_input_pd, rna_denoised_pd, allfeaturedims):    
    reconstruction_loss_rna = K.mean(cosine_similarity(rna_input, rna_denoised)+cosine_similarity(rna_input_pd, rna_denoised_pd)) * allfeaturedims
    kl_loss_rna = -0.5 * K.sum(1 + rna_sigma - K.square(rna_mean) - K.exp(rna_sigma), axis=-1)
    rna_loss = reconstruction_loss_rna + kl_loss_rna
    
    return rna_loss

def pro_loss_func(pro_mean, pro_sigma, pro_input, pro_denoised, pro_input_pd, pro_denoised_pd, allfeaturedims):    
    reconstruction_loss_pro = K.mean(cosine_similarity(pro_input, pro_denoised)+cosine_similarity(pro_input_pd, pro_denoised_pd)) * allfeaturedims
    kl_loss_pro = -0.5 * K.sum(1 + pro_sigma - K.square(pro_mean) - K.exp(pro_sigma), axis=-1)
    pro_loss = reconstruction_loss_pro + kl_loss_pro
    
    return pro_loss

def mix_loss_func(rna_mean, rna_sigma, rna_input, rna_denoised, pro_mean, pro_sigma, pro_input, pro_denoised, mix_mean, mix_sigma, rec_rna_input, rec_rna_denoised, rec_pro_input, rec_pro_denoised, rna_input_pd, rna_denoised_pd, pro_input_pd, pro_denoised_pd, allfeaturedims, recfeaturedims):
    
    rna_loss = rna_loss_func(rna_mean, rna_sigma, rna_input, rna_denoised, rna_input_pd, rna_denoised_pd, allfeaturedims)
    pro_loss = pro_loss_func(pro_mean, pro_sigma, pro_input, pro_denoised, pro_input_pd, pro_denoised_pd, allfeaturedims)

    reconstruction_loss_mix1 = K.mean(mean_squared_error(rec_rna_input, rec_rna_denoised)) * allfeaturedims
    reconstruction_loss_mix2 = K.mean(mean_squared_error(rec_pro_input, rec_pro_denoised)) * allfeaturedims
    kl_loss_mix = -0.5 * K.sum(1 + mix_sigma - K.square(mix_mean) - K.exp(mix_sigma), axis=-1)
    mix_loss = reconstruction_loss_mix1 + reconstruction_loss_mix2 + kl_loss_mix
    
    return rna_loss + pro_loss + mix_loss

NorAct = lambda x: tf.clip_by_value(tf.nn.relu(x), 1e-6, 100)
ThetaAct = lambda x: tf.clip_by_value(tf.nn.relu(x), 1e-6, 100)
Relu01 = lambda x: tf.clip_by_value(tf.nn.relu(x), 0, 1)

def build_rna_model(sc_data, net_dim_rna_list, net_dim_rna_mean, seed=100, alphadropout=0, lr=0.001):
    inikernel = initializers.glorot_uniform(seed=seed)
    
    input_dim_rna = sc_data.obsm["rna_nor"].shape[1]
    
    input_rna = Input(shape=(input_dim_rna,), name='input_rna')
    h_rna_encoder = input_rna
    
    for net_dim_rna in net_dim_rna_list:
        h_rna_encoder = Dense(net_dim_rna, kernel_initializer=inikernel, activation="softplus")(h_rna_encoder)

        if alphadropout>0:
            h_rna_encoder = AlphaDropout(alphadropout)(h_rna_encoder)
    
    if 0==len(net_dim_rna_list):
        h_rna_encoder = Dense(net_dim_rna_mean, kernel_initializer=inikernel, activation="softplus")(h_rna_encoder)
        if alphadropout>0:
            h_rna_encoder = AlphaDropout(alphadropout)(h_rna_encoder)
    
    h_rna_encoder = LayerNormalization()(h_rna_encoder)
   
    h_rna_mean = Dense(net_dim_rna_mean, name='encoder_rna_mean', kernel_initializer=inikernel)(h_rna_encoder)
    h_rna_sigma = Dense(net_dim_rna_mean, name='encoder_rna_sigma', kernel_initializer=inikernel)(h_rna_encoder)
    z_rna = Lambda(sampling, output_shape=(net_dim_rna_mean,), name='z_rna')([h_rna_mean, h_rna_sigma])
    h_rna_decoder_z = Dense(net_dim_rna_mean, name='decoder_rna_z', kernel_initializer=inikernel, activation="linear")(z_rna)
    
    h_rna_decoder = h_rna_decoder_z
    
    h_rna_decoder = h_rna_decoder_z
    rna_theta = h_rna_decoder_z
    rna_miu = h_rna_decoder_z
    
    output_rna = Dense(input_dim_rna, kernel_initializer=inikernel, activation=Relu01, name="rna_denoised")(h_rna_decoder)
    
    rna_theta = Dense(input_dim_rna, kernel_initializer=inikernel, activation=ThetaAct, name="rna_theta")(rna_theta)
    rna_miu = Dense(input_dim_rna, kernel_initializer=inikernel, activation=NorAct, name="rna_miu")(rna_miu)

    rna_input_pd = Lambda(build_NB, output_shape=(input_dim_rna,))([rna_theta, rna_miu, input_rna])
    rna_input_pd = Activation("relu", name="rna_input_pd")(rna_input_pd)
    
    rna_denoised_pd = Lambda(build_NB, output_shape=(input_dim_rna,))([rna_theta, rna_miu, output_rna])
    rna_denoised_pd = Activation("relu", name="rna_denoised_pd")(rna_denoised_pd)
    
    
    self_model = Model(inputs=[input_rna], outputs=[output_rna])
    allfeaturedims = input_dim_rna
    self_model.add_loss(rna_loss_func(h_rna_mean, h_rna_sigma, input_rna, output_rna, rna_input_pd, rna_denoised_pd, allfeaturedims))
    self_model.compile(optimizer=tf.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-05, clipnorm=1))
    
    rna_model = self_model
    
    sc_data.rna_model = rna_model
    
    return self_model

def build_pro_model(sc_data, net_dim_pro_list, net_dim_pro_mean, seed=100, alphadropout=0, lr=0.001):
    inikernel = initializers.glorot_uniform(seed=seed)
    
    input_dim_pro = sc_data.obsm["pro_nor"].shape[1]
    
    input_pro = Input(shape=(input_dim_pro,), name='input_pro')
    h_pro_encoder = input_pro
    
    for net_dim_pro in net_dim_pro_list:
        h_pro_encoder = Dense(net_dim_pro, kernel_initializer=inikernel, activation="softplus")(h_pro_encoder)
        if(alphadropout>0):
            h_pro_encoder = AlphaDropout(alphadropout)(h_pro_encoder)
    
    if 0==len(net_dim_pro_list):
        h_pro_encoder = Dense(net_dim_pro_mean, kernel_initializer=inikernel, activation="softplus")(h_pro_encoder)
        if(alphadropout>0):
            h_pro_encoder = AlphaDropout(alphadropout)(h_pro_encoder)
    
    h_pro_encoder = LayerNormalization()(h_pro_encoder)
    
    h_pro_mean = Dense(net_dim_pro_mean, name='encoder_pro_mean', kernel_initializer=inikernel)(h_pro_encoder)
    h_pro_sigma = Dense(net_dim_pro_mean, name='encoder_pro_sigma', kernel_initializer=inikernel)(h_pro_encoder)
    z_pro = Lambda(sampling, output_shape=(net_dim_pro_mean,), name='z_pro')([h_pro_mean, h_pro_sigma])
    h_pro_decoder_z = Dense(net_dim_pro_mean, name='decoder_pro_z', kernel_initializer=inikernel, activation="linear")(z_pro)
        
    h_pro_decoder = h_pro_decoder_z
    
    h_pro_decoder = h_pro_decoder_z
    pro_lmd = h_pro_decoder_z
    
    output_pro = Dense(input_dim_pro, kernel_initializer=inikernel, activation=Relu01, name="pro_denoised")(h_pro_decoder)

    pro_lmd = Dense(input_dim_pro, kernel_initializer=inikernel, activation=NorAct, name="pro_lmd")(pro_lmd)
    
    pro_input_pd = Lambda(build_Poisson, output_shape=(input_dim_pro,))([pro_lmd, input_pro])
    pro_input_pd = Activation("relu", name="pro_input_pd")(pro_input_pd)        

    pro_output_pd = Lambda(build_Poisson, output_shape=(input_dim_pro,))([pro_lmd, output_pro])
    pro_output_pd = Activation("relu", name="pro_denoised_pd")(pro_output_pd)

    self_model = Model(inputs=[input_pro], outputs=[output_pro])
    allfeaturedims = input_dim_pro
    self_model.add_loss(pro_loss_func(h_pro_mean, h_pro_sigma, input_pro, output_pro, pro_input_pd, pro_output_pd, allfeaturedims))
    self_model.compile(optimizer=tf.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-05, clipnorm=1))
    
    pro_model = self_model
    
    sc_data.pro_model = pro_model
    
    return self_model    

def build_mix_model(sc_data, net_dim_rna_list=[1024, 256, 128], net_dim_pro_list=[128], net_dim_rna_mean=128, net_dim_pro_mean=128, net_dim_mix=128, seed=100, alphadropout=0, lr=0.001):
    
    input_dim_rna = sc_data.obsm["rna_nor"].shape[1]
    input_dim_pro = sc_data.obsm["pro_nor"].shape[1]
    
    inikernel = initializers.glorot_uniform(seed=seed)
    
    rna_model = build_rna_model(sc_data, net_dim_rna_list, net_dim_rna_mean, seed, alphadropout, lr)
    pro_model = build_pro_model(sc_data, net_dim_pro_list, net_dim_pro_mean, seed, alphadropout, lr)

    input_mix_rna_mean = rna_model.get_layer("encoder_rna_mean").output
    input_mix_pro_mean = pro_model.get_layer("encoder_pro_mean").output
    mix_mean = Concatenate()([input_mix_rna_mean, input_mix_pro_mean])
    #mix_mean = Dense(net_dim_mix, kernel_initializer=inikernel)(mix_mean)
    #mix_mean = LayerNormalization()(mix_mean)
    mix_mean = Dense(net_dim_mix, name="mix_mean", kernel_initializer=inikernel)(mix_mean)

    input_mix_rna_sigma = rna_model.get_layer("encoder_rna_sigma").output
    input_mix_pro_sigma = pro_model.get_layer("encoder_pro_sigma").output
    mix_sigma = Concatenate()([input_mix_rna_sigma, input_mix_pro_sigma])
    mix_sigma = Dense(net_dim_mix, name="mix_sigma", kernel_initializer=inikernel)(mix_sigma)

    z_mix = Lambda(sampling, output_shape=(net_dim_mix,), name='z_mix')([mix_mean, mix_sigma])
    mix_docoder = Dense(net_dim_rna_mean+net_dim_pro_mean, name="mix_docoder", kernel_initializer=inikernel)(z_mix)
    mix_docoder_rna = Dense(net_dim_rna_mean, name='mix_docoder_rna', kernel_initializer=inikernel)(mix_docoder)
    mix_docoder_pro = Dense(net_dim_pro_mean, name='mix_docoder_pro', kernel_initializer=inikernel)(mix_docoder)
    
    rna_mean = input_mix_rna_mean
    rna_sigma = input_mix_rna_sigma
    rna_input = rna_model.get_layer("input_rna").output
    rna_denoised = rna_model.get_layer("rna_denoised").output
    
    pro_mean = input_mix_pro_mean
    pro_sigma = input_mix_pro_sigma
    pro_input = pro_model.get_layer("input_pro").output
    pro_denoised = pro_model.get_layer("pro_denoised").output
    
    rec_rna_input = rna_model.get_layer("decoder_rna_z").output
    rec_rna_denoised = mix_docoder_rna
    rec_pro_input = pro_model.get_layer("decoder_pro_z").output
    rec_pro_denoised = mix_docoder_pro
    
    rna_input_pd = rna_model.get_layer("rna_input_pd").output
    rna_denoised_pd = rna_model.get_layer("rna_denoised_pd").output
    pro_input_pd = pro_model.get_layer("pro_input_pd").output
    pro_denoised_pd = pro_model.get_layer("pro_denoised_pd").output
    
    allfeaturedims = input_dim_rna+input_dim_pro
    recfeaturedims = net_dim_rna_mean+net_dim_pro_mean

    self_model = Model(inputs=[rna_input, pro_input], outputs=[rna_denoised, pro_denoised])
    self_model.add_loss(mix_loss_func(rna_mean, rna_sigma, rna_input, rna_denoised, pro_mean, pro_sigma, pro_input, pro_denoised, mix_mean, mix_sigma, rec_rna_input, rec_rna_denoised, rec_pro_input, rec_pro_denoised, rna_input_pd, rna_denoised_pd, pro_input_pd, pro_denoised_pd, allfeaturedims, recfeaturedims))
    self_model.compile(optimizer=tf.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-05, clipnorm=1))
    
    mix_model = self_model
    
    sc_data.mix_model = mix_model
    
    return self_model

def fit(sc_data, mode="mm", epochs=500, batch_size=64, validation_split=0.1, patience=3, shuffle=True, verbose=1):
    es = EarlyStopping(monitor="loss", patience=3, verbose=2)
    
    mix_model = sc_data.mix_model
    rna_model = sc_data.rna_model
    pro_model = sc_data.pro_model
    
    if "mm"==mode and not(mix_model is None):
        estimator = mix_model.fit(x=[sc_data.obsm["rna_nor"], sc_data.obsm["pro_nor"]], epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=es, shuffle=shuffle, verbose=verbose)
    elif "rna"==mode and not(rna_model is None):
        estimator = rna_model.fit(x=[sc_data.obsm["rna_nor"]], epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=es, shuffle=shuffle, verbose=verbose)
    elif "pro"==mode and not(pro_model is None):
        estimator = pro_model.fit(x=[sc_data.obsm["pro_nor"]], epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=es, shuffle=shuffle, verbose=verbose)
    else:
        raise Exception(print("unkonw mode."))
    
    sc_data.estimator = estimator

def loss_plot(sc_data):
    plt.plot(sc_data.estimator.history["loss"],'-o',label='Training_loss')
    plt.plot(sc_data.estimator.history["val_loss"],'-o',label='Validation_loss')
    plt.legend(loc='upper right')
    plt.title("Loss of DPI Model")

def get_features(sc_data, mode="mm", return_value=False):
    
    mix_model = sc_data.mix_model
    rna_model = sc_data.rna_model
    pro_model = sc_data.pro_model
    
    if "mm"==mode and not(mix_model is None):        
        mode_features = Model(inputs=mix_model.inputs, outputs=mix_model.get_layer("encoder_rna_mean").output).predict([sc_data.obsm["rna_nor"], sc_data.obsm["pro_nor"]])
        sc_data.obsm["rna_features"] = mode_features
        
        mode_features = Model(inputs=mix_model.inputs, outputs=mix_model.get_layer("encoder_pro_mean").output).predict([sc_data.obsm["rna_nor"], sc_data.obsm["pro_nor"]])
        sc_data.obsm["pro_features"] = mode_features
        
        mode_features = Model(inputs=mix_model.inputs, outputs=mix_model.get_layer("mix_mean").output).predict([sc_data.obsm["rna_nor"], sc_data.obsm["pro_nor"]])
        sc_data.obsm["mix_features"] = mode_features
        
    elif "rna"==mode and not(rna_model is None):
        mode_features = Model(inputs=rna_model.inputs, outputs=rna_model.get_layer("encoder_rna_mean").output).predict([sc_data.obsm["rna_nor"], sc_data.obsm["pro_nor"]])
        sc_data.obsm["rna_features"] = mode_features
    elif "pro"==mode and not(pro_model is None):
        mode_features = Model(inputs=pro_model.inputs, outputs=pro_model.get_layer("encoder_pro_mean").output).predict([sc_data.obsm["rna_nor"], sc_data.obsm["pro_nor"]])
        sc_data.obsm["pro_features"] = mode_features
    else:
        raise Exception(print("unkonw mode."))
    
    if return_value:
        return mode_features

def get_spaces(sc_data, mode="mm", return_value=False):
    
    mix_model = sc_data.mix_model
    rna_model = sc_data.rna_model
    pro_model = sc_data.pro_model
    
    if "mm"==mode and not(mix_model is None):        
        mode_features = Model(inputs=mix_model.inputs, outputs=mix_model.get_layer("z_rna").output).predict([sc_data.obsm["rna_nor"], sc_data.obsm["pro_nor"]])
        sc_data.obsm["rna_latent_space"] = mode_features
        
        mode_features = Model(inputs=mix_model.inputs, outputs=mix_model.get_layer("z_pro").output).predict([sc_data.obsm["rna_nor"], sc_data.obsm["pro_nor"]])
        sc_data.obsm["pro_latent_space"] = mode_features
        
        mode_features = Model(inputs=mix_model.inputs, outputs=mix_model.get_layer("z_mix").output).predict([sc_data.obsm["rna_nor"], sc_data.obsm["pro_nor"]])
        sc_data.obsm["mm_parameter_space"] = mode_features
        
    elif "rna"==mode and not(rna_model is None):
        mode_features = Model(inputs=rna_model.inputs, outputs=rna_model.get_layer("z_rna").output).predict([sc_data.obsm["rna_nor"], sc_data.obsm["pro_nor"]])
        sc_data.obsm["rna_latent_space"] = mode_features
        
    elif "pro"==mode and not(pro_model is None):
        mode_features = Model(inputs=pro_model.inputs, outputs=pro_model.get_layer("z_pro").output).predict([sc_data.obsm["rna_nor"], sc_data.obsm["pro_nor"]])
        sc_data.obsm["pro_latent_space"] = mode_features
    else:
        raise Exception(print("unkonw mode."))
    
    if return_value:
        return mode_features

def space_plot(sc_data, spacetype="mm_parameter_space", **kwargs):
    temp = np.mean(sc_data.obsm[spacetype], axis=0)
    temp = (temp-np.min(temp))/(np.max(temp)-np.min(temp))
    temp = temp * (1-(-1))+(-1)
    sns.histplot(temp, **kwargs)
    plt.show()
    plt.close()

def fix_rna_denoised(sc_data):
    tempindex = sc_data.obsm["rna_observed"]>sc_data.obsm["rna_denoised"]
    sc_data.obsm["rna_denoised"][tempindex] = sc_data.obsm["rna_observed"][tempindex]

def fix_pro_denoised(sc_data):
    tempindex = sc_data.obsm["pro_denoised"]>sc_data.obsm["pro_observed"]
    sc_data.obsm["pro_denoised"][tempindex] = sc_data.obsm["pro_observed"][tempindex]

def get_denoised_rna(sc_data, return_value=False):
    
    mix_model = sc_data.mix_model
    rna_model = sc_data.rna_model
    pro_model = sc_data.pro_model
    
    if not(mix_model is None):
        temp_denoised_rna = Model(inputs=mix_model.inputs, outputs=mix_model.get_layer("rna_denoised").output).predict([sc_data.obsm["rna_nor"], sc_data.obsm["pro_nor"]])
    elif not(rna_model is None):
        temp_denoised_rna = Model(inputs=rna_model.inputs, outputs=rna_model.get_layer("rna_denoised").output).predict([sc_data.obsm["rna_nor"], sc_data.obsm["pro_nor"]])
    else:
        raise Exception(print("unkonw mode."))
    
    temp_denoised_rna = MinMaxScaler().fit_transform(temp_denoised_rna)
    
    sc_data.obsm["rna_denoised"] = sc_data.mm_rna.inverse_transform(temp_denoised_rna)
    
    fix_rna_denoised(sc_data)
    
    if return_value:
        return sc_data.obsm["rna_denoised"]

def get_denoised_pro(sc_data, return_value=False):
    
    mix_model = sc_data.mix_model
    rna_model = sc_data.rna_model
    pro_model = sc_data.pro_model    
    
    if not(mix_model is None):
        temp_denoised_pro = Model(inputs=mix_model.inputs, outputs=mix_model.get_layer("pro_denoised").output).predict([sc_data.obsm["rna_nor"], sc_data.obsm["pro_nor"]])
    elif not(rna_model is None):
        temp_denoised_pro = Model(inputs=pro_model.inputs, outputs=pro_model.get_layer("pro_denoised").output).predict([sc_data.obsm["rna_nor"], sc_data.obsm["pro_nor"]])
    else:
        raise Exception(print("unkonw mode."))
    
    temp_denoised_pro = MinMaxScaler().fit_transform(temp_denoised_pro)
    
    sc_data.obsm["pro_denoised"] = sc_data.mm_pro.inverse_transform(temp_denoised_pro)
    
    fix_pro_denoised(sc_data)
    
    if return_value:
        return sc_data.obsm["pro_denoised"]

def umap_run(sc_data, n_neighbors="auto", **kwargs):
    umap_data = sc_data.obsm["mix_features"]
    if "auto"==n_neighbors:
        n_neighbors = sc_data.uns["neighbors"]["params"]["n_neighbors"]
        
    sc_data.umap_mapper = umap.UMAP(n_neighbors=n_neighbors, **kwargs).fit(umap_data)
    sc_data.obsm["X_umap"] = sc_data.umap_mapper.transform(umap_data)
    
def umap_plot(sc_data, featuretype="rna", **kwargs):
    if "rna"==featuretype:
        rna_umap(sc_data, **kwargs)
    elif "protein"==featuretype:
        pro_umap(sc_data, **kwargs)
    else:
        raise Exception(print("unkonw faturetype."))
        
def rna_umap(sc_data, **kwargs):
    rna_adata = sc.AnnData(sc_data.obsm["rna_nor"].copy(), obs=sc_data.obs)
    rna_adata.var = sc_data.var
    rna_adata.layers["rna_denoised"] = sc_data.obsm["rna_denoised"]
    rna_adata.layers["rna_observed"] = sc_data.obsm["rna_observed"]
    rna_adata.obsm["X_umap"] = sc_data.obsm["X_umap"].copy() 
    rna_adata.X = sc_data.mm_rna.inverse_transform(rna_adata.X)
    sc.pl.umap(rna_adata, **kwargs)

def pro_umap(sc_data, **kwargs):
    temp_df = pd.DataFrame(sc_data.obsm["pro_nor"].copy())
    temp_df.columns = sc_data.uns["adt"]
    temp_df.index = sc_data.obs.index
    pro_adata = sc.AnnData(temp_df, obs=sc_data.obs)
    pro_adata.layers["pro_denoised"] = sc_data.obsm["pro_denoised"].copy()
    pro_adata.layers["pro_observed"] = sc_data.obsm["pro_observed"].copy()
    pro_adata.obsm["X_umap"] = sc_data.obsm["X_umap"].copy()
    pro_adata.X = sc_data.mm_pro.inverse_transform(pro_adata.X)
    sc.pl.umap(pro_adata, **kwargs)

def create_grid_space(umap_xy, spec=0.8):
    umap_x = umap_xy[:,0]
    umap_y = umap_xy[:,1]

    grid_x_space = np.arange(np.min(umap_x), np.max(umap_x), spec)
    grid_y_space = np.arange(np.min(umap_y), np.max(umap_y), spec)
    grid_x_space, grid_y_space = np.meshgrid(grid_x_space, grid_y_space)
    
    return grid_x_space, grid_y_space

@nb.jit()
def umap_xy_trans2_grid_space(umap_x, umap_y, grid_x_space, grid_y_space, spec=0.8):
    cell_grid_x = np.zeros(umap_x.shape)
    cell_grid_y = np.zeros(umap_y.shape)
    cell_grid_index = np.zeros(umap_x.shape[0])
    
    grid_index = 0
    for x_i in range(grid_x_space.shape[0]):
        for y_i in range(grid_y_space.shape[1]):
            min_x_dis_p = spec/2
            min_y_dis_p = spec/2
            
            for cell_i in range(cell_grid_x.shape[0]):
                min_x_dis = abs(umap_x[cell_i]-grid_x_space[x_i, y_i])
                min_y_dis = abs(umap_y[cell_i]-grid_y_space[x_i, y_i])                    
                    
                if(min_x_dis<min_x_dis_p and min_y_dis<min_y_dis_p):
                    cell_grid_x[cell_i] = grid_x_space[x_i, y_i]
                    cell_grid_y[cell_i] = grid_y_space[x_i, y_i]
                    cell_grid_index[cell_i] = grid_index
            
            grid_index += 1
            
    return cell_grid_x, cell_grid_y, cell_grid_index

import numba as nb

def create_grid_space(umap_xy, spec=0.8):
    umap_x = umap_xy[:,0]
    umap_y = umap_xy[:,1]

    grid_x_space = np.arange(np.min(umap_x), np.max(umap_x), spec)
    grid_y_space = np.arange(np.min(umap_y), np.max(umap_y), spec)
    grid_x_space, grid_y_space = np.meshgrid(grid_x_space, grid_y_space)
    
    return grid_x_space, grid_y_space

@nb.jit()
def umap_xy_trans2_grid_space(umap_x, umap_y, grid_x_space, grid_y_space, spec=0.8):
    cell_grid_x = np.zeros(umap_x.shape)
    cell_grid_y = np.zeros(umap_y.shape)
    cell_grid_index = np.zeros(umap_x.shape[0])
    
    grid_index = 0
    for x_i in range(grid_x_space.shape[0]):
        for y_i in range(grid_y_space.shape[1]):
            min_x_dis_p = spec/2
            min_y_dis_p = spec/2
            
            for cell_i in range(cell_grid_x.shape[0]):
                min_x_dis = abs(umap_x[cell_i]-grid_x_space[x_i, y_i])
                min_y_dis = abs(umap_y[cell_i]-grid_y_space[x_i, y_i])                    
                
                if(min_x_dis<min_x_dis_p and min_y_dis<min_y_dis_p):
                    cell_grid_x[cell_i] = grid_x_space[x_i, y_i]
                    cell_grid_y[cell_i] = grid_y_space[x_i, y_i]
                    cell_grid_index[cell_i] = grid_index
            
            grid_index += 1
            
    return cell_grid_x, cell_grid_y, cell_grid_index

def perturbation_run(sc_data, feature, amplitude=1, obs="", perturbed_clusters="all", featuretype="rna", mode="multiple"):
    
    perturbed_cellindex = []
    
    if "all"==perturbed_clusters:
        perturbed_cellindex = np.array(range(sc_data.shape[0]))
    else:
        if ""!=obs:
            for perturbed_cluster in perturbed_clusters:
                tempindex = np.where(perturbed_cluster==sc_data.obs[obs])[0].tolist()
                perturbed_cellindex += tempindex
        else:
            raise Exception("obs parameter is not set.")
    
    sc_data.uns["perturbed_cellindex"] = perturbed_cellindex
    
    if "rna"==featuretype:
        sc_data.obsm["rna_perturbation"] = sc_data.obsm["rna_nor"].copy()
        temp = sc_data.var["highly_variable"]==True
        temp_index = np.where(temp.index==feature)[0]
        if len(temp_index)<=0:
            raise Exception(print("can not find ",feature," in highly_variable features"))
        
        rna_exp = sc_data.obsm["rna_perturbation"][perturbed_cellindex, temp_index]
        
        if "multiple"==mode:
            rna_exp = amplitude*rna_exp
        elif "normality"==mode:
            perturbation_nor = np.random.normal(0, 1, rna_exp.shape[0])
            rna_exp = rna_exp+amplitude*perturbation_nor
        else:
            raise Exception("Unknown perturbation mode.")
                
        sc_data.obsm["rna_perturbation"][perturbed_cellindex, temp_index] = rna_exp
                
        sc_data.obsm["pro_perturbation"] = sc_data.obsm["pro_nor"].copy()
        
    elif "protein"==featuretype:
        sc_data.obsm["pro_perturbation"] = sc_data.obsm["pro_nor"].copy()
        temp_index = np.where(sc_data.obsm["protein_expression"].columns==feature)[0]
        if len(temp_index)<=0:
            raise Exception(print("can not find ",feature," in protein features"))
        
        pro_exp = sc_data.obsm["pro_perturbation"][perturbed_cellindex, temp_index]
        
        if "multiple"==mode:
            pro_exp = amplitude*pro_exp
        elif "normality"==mode:
            perturbation_nor = np.random.normal(0, 1, pro_exp.shape[0])
            pro_exp = pro_exp+amplitude*perturbation_nor
        else:
            raise Exception("Unknown perturbation mode.")
        
        sc_data.obsm["pro_perturbation"][perturbed_cellindex, temp_index] = pro_exp
                
        sc_data.obsm["rna_perturbation"] = sc_data.obsm["rna_nor"].copy()
    
    else:
        raise Exception(print("unkonw perturbation mode. DPI only support rna or protein perturbation"))
        
    perturbation_mix_mean = Model(inputs=sc_data.mix_model.inputs, outputs=sc_data.mix_model.get_layer("mix_mean").output).predict([sc_data.obsm["rna_perturbation"], sc_data.obsm["pro_perturbation"]])
    sc_data.obsm["perturbation_mix_mean"] = perturbation_mix_mean
    sc_data.obsm["X_umap_perturbation"] = sc_data.umap_mapper.transform(perturbation_mix_mean)
    
def perturbation_vis(sc_data, spec=0.5, obs="", intercept=(-1,1), arrow_color="black", masked=False, showlegend=False, **kwargs):
        
    perturbed_cellindex = sc_data.uns["perturbed_cellindex"]
        
    mapper_transform = sc_data.obsm["X_umap"]
    mapper_transform_perturbation = sc_data.obsm["X_umap_perturbation"]
    
    grid_x_space, grid_y_space = create_grid_space(mapper_transform, spec)
    
    umap_x = mapper_transform[perturbed_cellindex, 0]
    umap_y = mapper_transform[perturbed_cellindex, 1]
    cell_grid_x, cell_grid_y, cell_grid_index = umap_xy_trans2_grid_space(umap_x, umap_y, grid_x_space, grid_y_space, spec)    

    umap_x = mapper_transform_perturbation[perturbed_cellindex, 0]
    umap_y = mapper_transform_perturbation[perturbed_cellindex, 1]
    cell_perturbation_grid_x, cell_perturbation_grid_y, cell_perturbation_grid_index = umap_xy_trans2_grid_space(umap_x, umap_y, grid_x_space, grid_y_space, spec)

    U = np.zeros(cell_grid_x.shape[0])
    V = np.zeros(cell_grid_y.shape[0])
    for cell_grid_index_item in np.unique(cell_grid_index):
        temp_index = np.where(cell_grid_index==cell_grid_index_item)
        U[temp_index] = np.mean(cell_perturbation_grid_x[temp_index] - cell_grid_x[temp_index])
        V[temp_index] = np.mean(cell_perturbation_grid_y[temp_index] - cell_grid_y[temp_index])
    
    X = cell_grid_x
    Y = cell_grid_y
    
    U[np.where(U>=1)]=intercept[1]
    U[np.where(U<=-1)]=intercept[0]
    V[np.where(V>=1)]=intercept[1]
    V[np.where(V<=-1)]=intercept[0]
    
    df = pd.DataFrame({'X':X, 'Y':Y, 'U':U, 'V':V})
    df = df.drop_duplicates()
    
    X = df["X"]
    Y = df["Y"]
    U = df["U"]
    V = df["V"]
    
    df = pd.DataFrame({"x":mapper_transform[perturbed_cellindex, 0], "y":mapper_transform[perturbed_cellindex, 1], "cluster":sc_data[perturbed_cellindex].obs[obs]})
    
    if showlegend:
        sns.scatterplot(data=df, x="x", y="y", hue="cluster", **kwargs)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)
    else:
        sns.scatterplot(data=df, x="x", y="y", hue="cluster", legend=False, **kwargs)
        
    plt.scatter(X, Y, s=0.01, c=arrow_color, alpha=1, marker=".", linewidth=0)
    plt.quiver(X, Y, U, V, width=0.0015, scale=50, headwidth=5, color=arrow_color)
    if masked:
        plt.scatter(mapper_transform_perturbation[perturbed_cellindex, 0], mapper_transform_perturbation[perturbed_cellindex, 1], s=10, color="#aafdbf", alpha=0.02)
        
def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T) 
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

def cell_state_degree_plot(sc_data, title, colorbar=True, **kwargs):
    temp = sc_data.obsm["mix_features"]-sc_data.obsm["perturbation_mix_mean"]
    temp = np.power(temp, 2)
    temp = np.sum(temp, axis=1)
    temp = np.sqrt(temp)
    x = sc_data.obsm["X_umap"][:,0]
    y = sc_data.obsm["X_umap"][:,1]
    df = pd.DataFrame({"x":x, "y":y, "degree":temp})
    plt.axis('off')
    plt.scatter(x=df["x"], y=df["y"], c=df["degree"], **kwargs)
    plt.title(title)
    if colorbar:
        plt.colorbar()

def cell_state_umap_degree_plot(sc_data, title, colorbar=True, **kwargs):
    temp = sc_data.obsm["X_umap"]-sc_data.obsm["X_umap_perturbation"]
    temp = np.power(temp, 2)
    temp = np.sum(temp, axis=1)
    temp = np.sqrt(temp)
    x = sc_data.obsm["X_umap"][:,0]
    y = sc_data.obsm["X_umap"][:,1]
    df = pd.DataFrame({"x":x, "y":y, "degree":temp})
    plt.axis('off')
    plt.scatter(x=df["x"], y=df["y"], c=df["degree"], **kwargs)
    plt.title(title)
    if colorbar:
        plt.colorbar()

def fluctuation_vis(sc_data, **kwargs):
    cos_similar_matrix = get_cos_similar_matrix(sc_data.obsm["perturbation_mix_mean"], sc_data.obsm["mix_features"])
    fluctuation = cos_similar_matrix[range(0,perturbation_mix_mean.shape[0]), range(0,perturbation_mix_mean.shape[0])]

    df = pd.DataFrame({"x":sc_data.obsm["X_umap"][:,0], "y":sc_data.obsm["X_umap"][:,1], "fluctuation":fluctuation})
    ax = sns.scatterplot(data=df, x="x", y="y", hue="fluctuation", legend=False, linewidth=0, vmin=-1, vmax=1, **kwargs)
    norm = plt.Normalize(-1, 1)
    sm = plt.cm.ScalarMappable(cmap="icefire", norm=norm)
    sm.set_array([])
    ax.figure.colorbar(sm)

def cell_vector_field_run(sc_data):
    sc_data.obsm["X_VF"] = sc_data.obsm["perturbation_mix_mean"]-sc_data.obsm["mix_features"]
    sc_data.obsm["X_mix_features"] = sc_data.obsm["mix_features"]

def cell_vector_field_vis(sc_data, E_key="umap", **kwargs):
    plot_vector_field(sc_data, E_key=E_key, **kwargs)

def referencedata(sc_data_ref, sc_data, celltypes):
    temp_index = np.argmax(sc_data_ref.obsm["ref_mix_mean_cosmtx"], axis=1)
    sc_data_ref.obs["labels"] = sc_data.obs[celltypes][temp_index].values
    
    return sc_data_ref

def annotate(ref_data, ref_labelname, query_data):
    query_data.obsm["mix_mean"] = Model(inputs=ref_data.mix_model.inputs, outputs=ref_data.mix_model.get_layer("mix_mean").output).predict([query_data.obsm["rna_nor"], query_data.obsm["pro_nor"]])
    query_data.obsm["X_umap"] = ref_data.umap_mapper.transform(query_data.obsm["mix_mean"])
    
    query_data.obsm["ref_mix_mean_cosmtx"] = dpi.get_cos_similar_matrix(query_data.obsm["mix_mean"], ref_data.obsm["mix_features"]).astype("float16")
    
    referencedata(query_data, ref_data, celltypes=ref_labelname)
    
    return query_data

def l2_norm(x, axis=-1):
    if issparse(x):
        return np.sqrt(x.multiply(x).sum(axis=axis).A1)
    else:
        return np.sqrt(np.sum(x * x, axis = axis))

def cosine_similarity_velo(sc_data: AnnData, zs_key: str, reverse: bool = False, use_rep_neigh: Optional[str] = None, vf_key: str = 'VF', run_neigh: bool = True, n_neigh: int = 20, t_key: Optional[str] = None, var_stabilize_transform: bool = False) -> csr_matrix:
    
    Z = sc_data.obsm[f'X_{zs_key}']
    V = sc_data.obsm[f'X_{vf_key}']
    if reverse:
        V = -V
    if var_stabilize_transform:
        V = np.sqrt(np.abs(V)) * np.sign(V)

    ncells = sc_data.n_obs

    if run_neigh:
        sc.pp.neighbors(sc_data, use_rep = f'X_{use_rep_neigh}', n_neighbors = n_neigh)
    n_neigh = sc_data.uns['neighbors']['params']['n_neighbors'] - 1
    indices_matrix = sc_data.obsp['distances'].indices.reshape(-1, n_neigh)

    if t_key is not None:
        ts = sc_data.obs[t_key].values
        indices_matrix2 = np.zeros(indices_matrix.shape, dtype = int)
        for i in range(ncells):
            idx = np.abs(ts - ts[i]).argsort()[:(n_neigh + 1)]
            idx = np.setdiff1d(idx, i) if i in idx else idx[:-1]
            indices_matrix2[i] = idx
        indices_matrix = np.hstack([indices_matrix, indices_matrix2])

    vals, rows, cols = [], [], []
    for i in range(ncells):
        idx = np.unique(indices_matrix[i])
        idx2 = indices_matrix[idx].flatten()
        idx2 = np.setdiff1d(idx2, i)
        idx = np.unique(np.concatenate([idx, idx2]))
        dZ = Z[idx] - Z[i, None]
        if var_stabilize_transform:
            dZ = np.sqrt(np.abs(dZ)) * np.sign(dZ)
        cos_sim = np.einsum("ij, j", dZ, V[i]) / (l2_norm(dZ, axis = 1) * l2_norm(V[i]))
        vals.extend(cos_sim)
        rows.extend(np.repeat(i, len(idx)))
        cols.extend(idx)

    res = coo_matrix((vals, (rows, cols)), shape = (ncells, ncells))
    res.data = np.clip(res.data, -1, 1)
    return res.tocsr()


def quiver_autoscale(E: np.ndarray, V: np.ndarray):
    fig, ax = plt.subplots()
    scale_factor = np.abs(E).max()

    Q = ax.quiver(
        E[:, 0] / scale_factor,
        E[:, 1] / scale_factor,
        V[:, 0],
        V[:, 1],
        angles = 'xy',
        scale = None,
        scale_units = 'xy',
    )
    Q._init()
    fig.clf()
    plt.close(fig)
    return Q.scale / scale_factor


def vector_field_embedding(sc_data: AnnData, T_key: str, E_key: str, scale: int = 10, self_transition: bool = False):

    T = sc_data.obsp[T_key]

    if self_transition:
        max_t = T.max(1).A.flatten()
        ub = np.percentile(max_t, 98)
        self_t = np.clip(ub - max_t, 0, 1)
        T.setdiag(self_t)

    T = T.sign().multiply(np.expm1(abs(T * scale)))
    T = T.multiply(csr_matrix(1.0 / abs(T).sum(1)))
    if self_transition:
        T.setdiag(0)
        T.eliminate_zeros()

    E = sc_data.obsm[f'X_{E_key}']
    V = np.zeros(E.shape)

    for i in range(sc_data.n_obs):
        idx = T[i].indices
        dE = E[idx] - E[i, None]
        dE /= l2_norm(dE)[:, None]
        dE[np.isnan(dE)] = 0
        prob = T[i].data
        V[i] = prob.dot(dE) - prob.mean() * dE.sum(0)

    V /= 3 * quiver_autoscale(E, V)
    return V


def vector_field_embedding_grid(E: np.ndarray, V: np.ndarray, smooth: float = 0.5, stream: bool = False) -> tuple:

    grs = []
    for i in range(E.shape[1]):
        m, M = np.min(E[:, i]), np.max(E[:, i])
        diff = M - m
        m = m - 0.01 * diff
        M = M + 0.01 * diff
        gr = np.linspace(m, M, 50)
        grs.append(gr)

    meshes = np.meshgrid(*grs)
    E_grid = np.vstack([i.flat for i in meshes]).T

    n_neigh = int(E.shape[0] / 50)
    nn = NearestNeighbors(n_neighbors = n_neigh, n_jobs = -1)
    nn.fit(E)
    dists, neighs = nn.kneighbors(E_grid)

    scale = np.mean([g[1] - g[0] for g in grs]) * smooth
    weight = norm.pdf(x = dists, scale = scale)
    weight_sum = weight.sum(1)

    V_grid = (V[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, weight_sum)[:, None]

    if stream:
        E_grid = np.stack(grs)
        V_grid = V_grid.T.reshape(2, 50, 50)

        mass = np.sqrt((V_grid * V_grid).sum(0))
        min_mass = 1e-5
        min_mass = np.clip(min_mass, None, np.percentile(mass, 99) * 0.01)
        cutoff1 = (mass < min_mass)

        length = np.sum(np.mean(np.abs(V[neighs]), axis = 1), axis = 1).reshape(50, 50)
        cutoff2 = (length < np.percentile(length, 5))

        cutoff = (cutoff1 | cutoff2)
        V_grid[0][cutoff] = np.nan
    else:
        min_weight = np.percentile(weight_sum, 99) * 0.01
        E_grid, V_grid = E_grid[weight_sum > min_weight], V_grid[weight_sum > min_weight]
        V_grid /= 3 * quiver_autoscale(E_grid, V_grid)

    return E_grid, V_grid


def plot_vector_field(sc_data: AnnData, reverse: bool = False, zs_key: str = 'mix_features', vf_key: str = 'VF', run_neigh: bool = True, use_rep_neigh: str = 'mix_features', t_key: Optional[str] = None, n_neigh: int = 20, var_stabilize_transform: bool = False, E_key: str = 'umap', scale: int = 10, self_transition: bool = False, smooth: float = 0.5, grid: bool = False, stream: bool = True, stream_density: int = 2, stream_color: str = 'k', linewidth: int = 1, arrowsize: int = 1, density: float = 1., arrow_size_grid: int = 5, color: Optional[str] = None, ax: Optional[Axes] = None, **kwargs):
    
    sc_data.obsp['cosine_similarity'] = cosine_similarity_velo(sc_data, reverse = reverse, zs_key = zs_key, vf_key = vf_key, run_neigh = run_neigh, use_rep_neigh = use_rep_neigh, t_key = t_key, n_neigh = n_neigh, var_stabilize_transform = var_stabilize_transform)
    
    sc_data.obsm['X_DV'] = vector_field_embedding(sc_data, T_key = 'cosine_similarity', E_key = E_key, scale = scale, self_transition = self_transition)
    
    nan_index = np.isnan(sc_data.obsm[f'X_DV'])
    sc_data.obsm[f'X_DV'][nan_index] = 0

    E = sc_data.obsm[f'X_{E_key}']
    V = sc_data.obsm[f'X_DV']

    if grid:
        stream = False

    if grid or stream:
        E, V = vector_field_embedding_grid(E = E, V = V, smooth = smooth, stream = stream)

    if stream:
        lengths = np.sqrt((V * V).sum(0))
        linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
        stream_kwargs = dict(linewidth = linewidth, density = stream_density, zorder = 3, color = stream_color, arrowsize = arrowsize, arrowstyle = '-|>', maxlength = 4, integration_direction = 'both')
        ax.streamplot(E[0], E[1], V[0], V[1], **stream_kwargs)
    else:
        if density < 1:
            idx = np.random.choice(len(E), int(len(E) * density), replace = False)
            E = E[idx]
            V = V[idx]
        
        quiver_kwargs = dict(angles = 'xy', scale_units = 'xy', edgecolors = 'k', scale = 1 / arrow_size_grid, width = 0.001, headlength = 12, headwidth = 10, headaxislength = 8, color = 'grey', linewidth = 1, zorder = 3)
        ax.quiver(E[:, 0], E[:, 1], V[:, 0], V[:, 1], **quiver_kwargs)

    ax = sc.pl.embedding(sc_data, basis = E_key, color = color, ax = ax, show = False, **kwargs)

    return ax

def cell_state_vector_field(sc_data, feature, amplitude=2, obs="", perturbed_clusters="all", featuretype="rna", mode="multiple", zs_key="mix_features", vf_key='VF', use_rep_neigh="mix_features", E_key="umap", legend_loc='none', frameon=False, size=100, alpha=0.2, density=0.5, grid=False, **kwargs):
    perturbation_run(sc_data, feature, amplitude, obs, perturbed_clusters, featuretype, mode)
    cell_vector_field_run(sc_data)
    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
    return plot_vector_field(sc_data, E_key=E_key, zs_key=zs_key, vf_key=vf_key, use_rep_neigh=use_rep_neigh, color=obs, ax=axs, legend_loc=legend_loc, frameon=frameon, size=size, alpha=alpha, density=density, grid=grid, **kwargs)