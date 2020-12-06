# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print('Importing Libraries')


# python
import joblib

# sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import  OneHotEncoder

# data processing and visualisation
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import SeqIO
from scipy import stats


print('Done Importing Libraries')
print('Loading and Parsing fasta file')

def generate_kmers(sequence, window, slide=1):
    """Make kmers from a sequence
        Args:
            - sequence (str); sequence to compute kmer
            - window (int); size of kmers
            - slide (int); no of bases to move 'window' along sequence
            default = 1
            
        Return:
            - 'list' object of kmers
        
        Example:
            - >>> DNA_kmers('ATGCGTACC', window=4, slide=4)
                ['ATGC', 'GTAC']
    """
    
    all_possible_seq = []
    kmers = []
    
    for base in range(0, len(sequence), slide): # indices
        # extend by window
        all_possible_seq.append(sequence[base:base + window]) 
        
    # remove all kmers != window    
    for seq in all_possible_seq:
        if len(seq) == window:
            kmers.append(seq)

    return kmers


# read and parse the fasta file as SeqIO object
file = SeqIO.parse(
    'biomart_transcriptome_all.fasta',
    'fasta'
)

# select ids and corresponding sequences
sequence_ids = []
sequences = []
for gene in file:
    sequence_ids.append(gene.id)
    sequences.append(str(gene.seq))
    
# create a table of gene ids; select only gene short name and type
id_tab = pd.Series(sequence_ids).str.split('|', expand=True).iloc[:, 2]


# join gene_id_tab with corresponding seqs
transcripts = pd.concat([id_tab, pd.Series(sequences)], axis=1)
# set column names
transcripts.columns = ['gene', 'seq']

print('Done Parsing fasta file')
print('Loading and Parsing CRAC data')

# read N_protein CRAC data
N_protein = pd.read_excel('SB20201008_hittable_unique.xlsx', sheet_name='rpm > 10')

print('Done Parsing CRAC data, Now generating Kmers')

# select common genes between N_protein and transcripts
N_genes = set(N_protein['Unnamed: 0'])
t_genes = set(transcripts.gene)
common_genes = N_genes.intersection(t_genes)

# filter transcripts data with common genes and remove duplicates
transcripts_N = transcripts.drop_duplicates(
    subset='gene').set_index('gene').loc[common_genes]


# create kmers from seq
transcripts_N['kmers'] = transcripts_N.seq.apply(generate_kmers, window=4, slide=4)

# view of kmers data
transcripts_N.kmers


# seperate kmers into columns. pad short seqs with '_'
kmer_matrix = transcripts_N.kmers.apply(pd.Series).fillna('_')

print('Done generating Kmers, Creating feature matrix X and response vector y')

# convert kmers to ints
ohe = OneHotEncoder(sparse=True)
ohe_kmers = ohe.fit_transform(kmer_matrix)


# response vector
y = pd.concat([kmer_matrix[0],
           N_protein.drop_duplicates(subset='Unnamed: 0').set_index('Unnamed: 0')], axis=1)['133_FH-N_229E']

print('Done crewating feature matrix and response vector y, Now splitting data')


# split data into train and test sets
XTrain, XTest, yTrain, yTest = train_test_split(ohe_kmers, y, test_size=0.2, random_state=1)

print('Done splitting data, Now training the model')

# instantiate the regressor
linreg = LinearRegression()

# train on data
linreg.fit(XTrain, yTrain)

# check performance on test set
yPred = linreg.predict(XTest)
r2Score = (metrics.r2_score(y_true=yTest, y_pred=yPred))

print('Done training the model, Now saving the model')

# save model
_ = joblib.dump(linreg, 'BaseModel.sav')

print('Done saving the model, Now computing model metrics')

# plot of yTest vs yPred
g = sns.regplot(x = yTest, y = yPred, scatter_kws={'alpha':0.2})

# set axes labels
_ = plt.xlabel('yTest')
_ = plt.ylabel('yPred')

# pearson correlation test
r, p = stats.pearsonr(yTest, yPred)
_ = g.annotate('r={}, p={}'.format(r, p), (-8, 2))
_ = plt.savefig('modelPearsonCorr.png')


print('r2 Score = {}, pearsonr = {}'.format(r2Score, r))
print('Dine')