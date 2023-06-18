#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math

# GLOBAL VARIABLES
# text of 80 chars -> 640 bits
# we encode each bit into 51200/640 = 80 bits
# each byte is encoded with 80*8 bits = 640 bits -> 2^9

N = 40960
CODEWORD_SIZE = 128 #if m = 2⁸, m = 2n => n = 2⁷
X_I_SIZE = 512 #CODEWORD_SIZE * REPETITION
REPETITION = 4

# with CODEWORD_SIZE = 2^i
i = 7


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


# auxiliar function to compute the matrix U = M_i * Y
# Y_i is a X_I_SIZE-bits subvector of N-bits Y.
# j should correspond to the global var i

def compute_U(M,Y_i,j):
    if j==0:
        return Y_i
    if j==1:
        return np.dot(M,Y_i)
    U_bis = compute_M(j-1)

    U_t = np.dot(U_bis,Y_i[0:int(CODEWORD_SIZE/2)])
    U_b = np.dot(U_bis,Y_i[int(CODEWORD_SIZE/2):])
    return np.concatenate((U_t+U_b,U_t-U_b),axis=0)


# auxiliar function which returns the value of the decoded text byte
# take the maximum (absolute) value in the matrix U

def decode_U(U):
    U_bis=abs(U)
    k=np.argmax(U_bis)
    if U[int(k)]<0:
        return k+1+128
    else :
        return k+1



# function which decodes the received file
# separate the received vector in the 5 repetition (for each byte)
# and then decode the sum of the 5 matrix U obtained

def receiver(filename):

    # open filename to get corresponding binary vector (N-bits) Y
    Y = np.genfromtxt(filename)
    Y=list(Y)
    output = open("output.txt", 'w')
    Y_0 = np.array([])
    Y_1 = np.array([])
    Y_2 = np.array([])
    Y_3 = np.array([])


    # for all subvector (X_I_SIZE) Y_i in Y, we decode the corresponding byte
    for j in range(0,512*80, X_I_SIZE):
        Y_i = Y[j:j+X_I_SIZE]
        Y_0 = Y_i[0:128]
        Y_1 = Y_i[128:256]
        Y_2 = Y_i[256:384]
        Y_3 = Y_i[384:512]

        curr_byte = decode_U(compute_U(compute_M(i), Y_0, i)+compute_U(compute_M(i), Y_1, i)+compute_U(compute_M(i), Y_2, i)+compute_U(compute_M(i), Y_3, i))
        print(chr(curr_byte))

        # each time we get a byte, we write it in a new output file
        output.write(chr(curr_byte))

    output.close()



receiver("received.txt")
