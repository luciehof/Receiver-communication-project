#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math

# GLOBAL VARIABLES
# text of 80 chars -> 640 bits
# we encode each bit into 51200/640 = 80 bits
# each byte is encoded with 80*8 bits = 640 bits -> 2^9

N = 40960
CODEWORD_SIZE = 128 #if m = 2⁸, m = 2n => n = 2⁷
REPETITION = 4

# with CODEWORD_SIZE = 2^i
i = 7


# auxiliar function to compute the matrix M

def compute_M(j):
    if j== 0:
        return np.array([1])
    M_1 = np.array([[1,1],[1,-1]])
    temp = M_1
    for l in range (j-1):
        up = np.concatenate((temp,temp), axis=1)
        bottom = np.concatenate((temp,-temp), axis=1)
        temp = np.concatenate((up,bottom), axis=0)
    return temp


# auxiliar function to take the k-th column of M

def columnk_M(M,k):
    if M.shape == (1,):
        return M[0]
    if k>=0 and k<np.size(M, 1):
        return M[:,k-1]
    else:
        print("bad index for column: "+str(k))


# auxiliar function to encode a byte
# encode it to its sign mutiply by the column k of matrix M

def encode_mk(M,k,sign):
    return (sign*columnk_M(M,k))


# function which encodes all the text
# first open the file
# get the "sign" of each byte : if <= CODEWORD_SIZE it is +, otherwise it is -
# get the value of the column of M for each byte, equal to its value in binary modulo the CODEWORD_SIZE
# proceed by doing a repetition code with 5 repetition : encode each byte for each repetition


def transmitter(filename):

    f=open(filename, 'rb')
    data = f.read()
    f.close()
    X = np.array([])
    sign = 1

    for j in range (len(data)):
        print (data[j])

        # get the byte
        if (data[j]<=CODEWORD_SIZE):
            sign = 1
        else :
            sign = -1

        l = np.mod(data[j], CODEWORD_SIZE)

        # create intermediary X_I_SIZE*CODEWORD_SIZE bits X_i
        X_i = np.array([])
        for r in range (REPETITION):
            V = encode_mk(compute_M(i),l,sign)
            X_i = np.concatenate((X_i,V), axis=0)
        print (X_i)
        # append intermediary X_i to N-bits vector X
        X = np.concatenate((X,X_i), axis=0)
    print (X)

    return np.savetxt("encoded.txt", X.astype(int), fmt="%s")

transmitter("input.txt")
