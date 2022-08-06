Datset reference: Christianson Nicolas H., Sizemore Blevins Ann and Bassett Danielle S. 2020. Architecture and evolution of semantic networks in mathematics textsProc. R. Soc. A.4762019074120190741
http://doi.org/10.1098/rspa.2019.0741


"all_index.npy" is a list of 10 arrays; the elements of each of those 10 arrays are the index terms (i.e., concept phrases) corresponding to the nodes of each graph extracted from the texts, given in the order they appear in the corresponding adjacency matrix.

 

"cooc_mats.npy" gives a list of the adjacency matrices for each of the graphs as they exist at the end of each text. That is, matrix k gives, in its (i, j)th position, the number of sentences in which the ith and jth concepts of text k co-occur throughout the whole text (where i and j correspond to the concepts denoted by all_index[k][i] and all_index[k][j]). Diagonal elements are the total number of sentences in which each concept occurred.

 

"filt_mats.npy" gives a list of the filtration matrices for each of the graphs. That is, every element (i, j) of each matrix k gives the sentence number of text k in which the concepts i and j first co-occurred in the text. If two concepts never co-occurred, then the value (i, j) will be infinity. The diagonal gives the sentence number in which each concept first occurred. The sentence number is 1-indexed, i.e., if concepts i and j first co-occurred in the 1st sentence, filt_mat[k][i, j] = 1.

 

Within each list, the data (arrays, matrices, etc.) are ordered by author last name as follows: Treil, Axler, Edwards, Lang, Petersen, Robbiano, Bretscher, Greub, Hefferson, Strang. 

Each file with name <i>_opt_networks.npy contains a numpy array of all optimized networks for the textbook at index i. Each such file contains networks optimized at beta values in np.linspace(1e-3, .2, 15).
