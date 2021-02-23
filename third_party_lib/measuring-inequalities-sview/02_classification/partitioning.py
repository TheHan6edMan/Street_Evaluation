import numpy as np

def partition_by_class(labels, label_test, seed=None):
    unique_labels = np.unique(labels)
    unique_test_labels = np.unique(label_test)
    train_partition = []
    test_partition = []
    np.random.seed(seed)
    for lab in unique_labels:
        if lab in unique_test_labels:
            rows = np.where(labels == lab)[0]
            rows = np.random.permutation(rows)
            test_partition += list(rows)
        else:
            rows = np.where(labels == lab)[0]
            rows = np.random.permutation(rows)
            train_partition += list(rows)
    return np.random.permutation(train_partition), \
           np.random.permutation(test_partition)

def _partition_stratified(labels, train_size, seed=None):
    unique_labels = np.unique(labels)
    train_partition = []
    test_partition = []
    np.random.seed(seed)
    for lab in unique_labels:
        rows = np.where(labels == lab)[0]
        rows = np.random.permutation(rows)
        label_size = np.sum(labels == lab)
        split_point = np.int(label_size * train_size)
        train_partition += list(rows[:split_point])
        test_partition += list(rows[split_point:])
    return np.random.permutation(train_partition), np.random.permutation(test_partition)

def _partition_stratified_validation(labels, train_size, valid_size, seed=None):
    unique_labels = np.unique(labels)
    train_partition = []
    validation_partition = []
    test_partition = []
    for lab in unique_labels:
        rows = np.where(labels == lab)[0]
        lab_size = np.sum(labels == lab)
        tr_split_point = np.int(lab_size * train_size)
        train_partition += list(rows[:tr_split_point])
        tr_sp_va_split_point = np.int(lab_size*(train_size + valid_size))
        validation_partition += list(rows[tr_split_point: tr_sp_va_split_point])
        test_partition += list(rows[tr_sp_va_split_point:])
    np.random.seed(seed)
    return np.random.permutation(train_partition), \
           np.random.permutation(validation_partition), \
           np.random.permutation(test_partition)

def divide_in_kparts(v, K):
    # divide array/list v into K parts, if len(v) is not divisible by K, the returned `parts` will be a ragged list
    lenv = len(v)  # len(v) == 28
    step_k = np.int(lenv / K)  # 5, 5, 5, 5, 8
    parts = []  # append
    for k in range(K-1):  # 0, 1, 2, 3
        parts += [v[k * step_k : (k + 1) * step_k]]  # v[5:10] == [5, 6, 7, 8, 9]  [[5, 6, 7, 8, 9]]
    parts += [v[(K-1)*step_k:]]  # [[0, 1,2 ,3, 4], [5, 6, ..], [20, 21, ..28]]
    return parts

def cpart(partition):
    """ As this function has been called, it always receives an input value which is a 2-D regular or irregular list. 
        It returns a vector with the length of the number of sublists, 
        whose element value is the relative distribution of the length of the sublist in the input list
        i.e., the number of elements of each row in the input list."""
    counts = np.zeros(len(partition))
    for n in range(len(partition)):
        counts[n] = len(partition[n])
        if counts[n] == 0:
            counts[n] = 1e-7
    return counts / np.sum(counts)

def entropy(portion):
    """ return the entropy of input vector `portion`"""
    ent = 0
    for n in range(len(portion)):
        ent = ent - portion[n] * np.log(portion[n])
    return ent

def cross_entropy(portion, target):
    ent = 0
    for n in range(len(portion)):
        ent = ent - target[n] * np.log(portion[n])
    return ent

def portions(part, labels):
    """ part: passed as partition[n],
        which means the rows where a same unique clabel `x` appears on in `clabels`,
        and suppose `clabel_name == "lsoa" `, it means the operationa done here are all at the same lsoa level"""
    un_labels = np.unique(labels)
    portion = np.zeros(len(un_labels))
    for n, uniq_label in enumerate(un_labels):
        """ `np.sum(labels[part] == uniq_label)` returns the times of some unique label appears on some certain rows in labels
            and suppose `clabel_name == "lsoa" `, the unique label is 7 and the rows are [1, 2, 3, 4] representing a certain lsoa level
            then `np.sum(labels[part] == uniq_label)` is the times that lable "7" appears in this level"""
        portion[n] = np.sum(labels[part] == uniq_label)  # 1~10
        if portion[n] == 0:
            portion[n] = 1e-7
    portion = portion / (np.sum(portion))  # 
    return portion

def epart(partition, labels):
    entropies = np.zeros(len(partition))
    for n in range(len(partition)):
        """ Each element of `ports` is the frequency of a unique label appearing in all the data on some specified rows of labels
            so it canbe viewed as a relative distribution of times of occurrences of the unique label in `labels`
                suppose the clabel_name is lsoa, `crows` is [1, 2, 3, 4], unique labels are 1, 2, ..., 10
                then ports[1] may be the frequency of label 1 appearing in all the data on rows [1, 2, 3, 4] of labels
                so ports can be viewed as a distribution"""
        ports = portions(partition[n], labels)
        entropies[n] = entropy(ports)
        """ entropies, shape == (K,)
            and according to desciption about `ports` above,
            entropies[n] can be viewed as some 'information' the distribution contains in part `partition[n]`
            (since you need to make sure each part of the partition contains as much information of the data as possible) """
    return entropies

def pargmax(partition, crows, labels, coeff=1.0):
    partition_copy = partition.copy()
    """ container to store the entropies of the trials of new partition tested in the `for` loop below"""
    vals = np.zeros(len(partition))
    for n in range(len(partition)):
        partition_copy[n] = np.append(partition_copy[n], crows)  # suppose the new `crows` will be appended on the (n+1)-th part
        """ the information of new partition. See more details about the infomation at the comments of `epart`"""
        
        entropies = epart(partition_copy, labels)
        """ the relative distribution of length of each row in `partition_copy`,
            i.e. the relative distribution of the number of rows of each part according to the new partition"""
        counts = cpart(partition_copy)
        """ coeff = 5 * entropy1 / entropy2;
            `entropy(counts)` can be viewed as the information the distribution of the number of rows of each part contains,
            since you need to make sure each part contains roughly the same number of rows;
            `entropies.mean()` is a way to take all the information contained in each part of the new partition into consideration;
            so the opperation below can make sence"""
        vals[n] = entropies.mean() + entropy(counts) * coeff
        # reset the copy as the original one
        partition_copy = partition.copy()
        
    return np.argmax(vals)

#????
def pargmax_cross(partition, crows, labels, target_dist, coeff=1.0):
    partition_ = partition.copy()
    vals = np.zeros(len(partition))
    for n in range(len(partition)):
        partition_[n] = np.append(partition_[n], crows)
        entropies_ = epart(partition_, labels)
        counts_ = cpart(partition_)
        vals[n] = entropies_.mean() - cross_entropy(counts_, target_dist) * coeff
        partition_ = partition.copy()
    return np.argmax(vals)

### THE NEXT TWO FUNCTIONS NEED TO BE TESTED.
def partition_stratified_validation(labels, train_size, valid_size, seed=None, clabels=None):
    if clabels is None:
        return _partition_stratified_validation(labels, train_size, valid_size, seed=None)

    print('This might take a while - get a coffeeee.')
    np.random.seed(seed)
    target_dist = np.array([train_size, valid_size, 1.0-train_size-valid_size])
    unique_clabels = np.unique(clabels)
    num_unique_labels = len(np.unique(labels))
    coeff = entropy(np.ones(num_unique_labels) * 1. / num_unique_labels) / entropy(target_dist)
    partition = list(np.zeros((3, 1)))
    for k in range(3):
        partition[k] = np.array([],dtype=np.uint16)

    for clab, m in zip(unique_clabels, range(len(unique_clabels))):
        crows = np.where(clabels == clab)[0]
        if m == 0:
            partition[0] = np.append(partition[0], crows)
        else:
            mpart = pargmax_cross(partition, crows, labels, target_dist, coeff=coeff*5)
            partition[mpart] = np.append(partition[mpart], crows)
        if m % 50 == 0:
            print('{} / {} done'.format(m, len(unique_clabels)))
            print('cparts: {}'.format(cpart(partition)))
    for k in range(3):
        partition[k] = np.random.permutation(partition[k])
    return partition

def partition_stratified(labels, train_size, seed=None, clabels=None, label_test=None):
    if clabels is None:
        return _partition_stratified(labels, train_size, seed=None)

    if label_test is not None:
        return partition_by_class(K, labels, seed=None, label_test=label_test)

    print('This might take a while - get a coffeeee.')
    unique_clabels = np.unique(clabels)
    np.random.seed(seed)
    target_dist = np.array([train_size, 1.0 - train_size])
    num_unique_labels = len(np.unique(labels))
    coeff = entropy(np.ones(num_unique_labels) * 1. / num_unique_labels) / entropy(target_dist)
    partition = list(np.zeros((2,1)))
    for k in range(2):
        partition[k] = np.array([],dtype=np.uint16)

    for clab, m in zip(unique_clabels, range(len(unique_clabels))):
        crows = np.where(clabels == clab)[0]
        if m == 0:
            partition[0] = np.append(partition[0], crows)
        else:
            mpart = pargmax_cross(partition, crows, labels, target_dist, coeff=coeff*5)
            partition[mpart] = np.append(partition[mpart], crows)
        if m % 50 == 0:
            print('{} / {} done'.format(m, len(unique_clabels)))
            print('cparts: {}'.format(cpart(partition)))
    for k in range(2):
        partition[k] = np.random.permutation(partition[k])
    return partition

def partition_stratified_kfold(K, labels, seed=None, clabels=None):
    if clabels is None:
        return _partition_stratified_kfold(K, labels, seed=None)

    print('This might take a while - get a coffeeee.')

    unique_clabels = np.unique(clabels)  # the unique constrain labels, type:
    np.random.seed(seed)
    num_unique_labels = len(np.unique(labels))  # 1~10
    # let num_unique_labels = u, then coeff = ln(u)/ln(K)
    coeff = entropy( np.ones(num_unique_labels) * 1. / num_unique_labels ) / entropy( np.ones(K) * 1. / K)
    # init a container to store the assignment of K parts, shape = (K, 0)
    partition = list(np.zeros((K, 1)))
    for k in range(K):
        partition[k] = np.array([], dtype=np.uint16)

    for uniq_clab, m in zip(unique_clabels, range(len(unique_clabels))):  # enumerate(unique_clabels)
        """ while dividing the data into sections K,
            the allocation of data on the `crows` is done between those rows """
        crows = np.where(clabels == uniq_clab)[0]  # lsoa == A, B
        if m == 0:                                 # m       0, 1
            partition[0] = np.append(partition[0], crows)

        else:
            mpart = pargmax(partition, crows, labels, coeff=coeff*5)
            partition[mpart] = np.append(partition[mpart], crows)
        
        if m % 50 == 0:
            print('{} / {} done'.format(m, len(unique_clabels)))
            print('cparts: {}'.format(cpart(partition)))
    
    for k in range(K):
        partition[k] = np.random.permutation(partition[k])
    return partition

def _partition_stratified_kfold(K, labels, seed=None):
    unique_labels = np.unique(labels)  # 1, 2, ..., 10
    np.random.seed(seed)
    for lab, n in zip(unique_labels, range(len(unique_labels))):
        rows_of_lab = np.where(labels == lab)[0]
        rows_of_lab = np.random.permutation(rows_of_lab)
        """ let int(rows_of_lab / K) = s, v_{i} = rows_of_lab[i],
            then kfold_parts = [[v0...v_{s-1}], [v_{s}...v{2s-1}], ...]
            i.e. divide the rows where the unique label appears in labels into K parts
            while trying to make those rows be divided as evenly as possible into those K parts
            NOTE: there's only one division for one unique label in every cycle, so you got what the `else` part meant to do"""
        kfold_parts = divide_in_kparts(rows_of_lab, K)    # label = 1 [part1_1, part1_2, part1_3...]
        if n == 0:                                        # label = 2 [part2_1, part2_2, part2_3...]
            partitions = kfold_parts  # [part1_1, part1_2, part1_3...]
        else:
            for k in range(K):
                partitions[k] = np.append(partitions[k], kfold_parts[k])
    for k in range(K):
        partitions[k] = np.random.permutation(partitions[k])
    return partitions

def get_partition_stratified_kfold(kp, partitions):
    test_partition = partitions[kp]
    train_partition = []
    for k in range(len(partitions)):
        if k != kp:
            train_partition += list(partitions[k])
    return train_partition, test_partition

def decimate_partition_stratified(current_partition, labels, psize=1.0):
    train_part, _ = partition_stratified(labels[current_partition], train_size=psize)
    return list(np.asarray(current_partition)[train_part])

def partition(labels, train_size, seed=None):
    train_partition = []
    validation_partition = []
    test_partition = []
    np.random.seed(seed)
    rows = np.random.permutation(labels.size)
    lab_size = labels.size
    row_end = np.int(lab_size * train_size)
    train_partition += list(rows[:row_end])
    row_beg = row_end
    test_partition += list(rows[row_beg:])
    return train_partition, test_partition

def combine_partitions(init_part, groups):
    ngroups = len(groups)
    fin_part = []
    for n in range(ngroups):
        part_ = np.array([], dtype=np.uint16)
        for g in groups[n]:
            part_ = np.append(part_, init_part[g])
        fin_part = fin_part + [part_]
    return fin_part
