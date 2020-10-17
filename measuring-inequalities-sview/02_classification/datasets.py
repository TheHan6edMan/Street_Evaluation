import numpy as np
import partitioning
import pickle
import h5py
import pandas as pd

def map_labels(labels):
    return labels - 1

class Dataset:
    def __init__(self, image_hdf5_file, label_pickle_file, label_name, clabel_name=None):
        """ image_hdf5_file: path of the file storing the extracted 4096-D vectors
            label_pickle_file: path of the file storing the lables
            label_name: names of the labels, i.e. columns in the pd.DataFrame
            clabel_name: names of the labels used to constrain partitioning."""
        self.imf = h5py.File(image_hdf5_file, 'r')  # according to 03_make_hdf5.py, it may be data/learning-ready-data/<output_name>_images_imgnet.hdf5
        self.codes = self.imf['features']  # an ndarry, containing the 4096-D codes, shape == (len(imgs), 4, 4096)
        self.raw_label_data = pd.read_pickle(label_pickle_file)  # according to 03_make_hdf5.py, it may be data/learning-ready-data/<output_name>_variables.p
        self.labels = map_labels(self.raw_label_data[label_name].as_matrix())  # labels, i.e. deciles
        self.vmap = self.raw_label_data[['img_id', 'pcd', 'oa11', 'lsoa11']].as_matrix()  # ??????
        self.label_name = label_name  # label names, type: str

        """ clabel_name is the name of the label that will be used to constrain partitioning.
                e.g. Images coming from the same lsoa will not be separated during partitioning,
                and all of them will remain in the same partition. 
            if it is None, then no constrain is given."""
        if clabel_name is not None:
            self.clabels = self.raw_label_data[clabel_name].as_matrix()  # constrain labels, dim = 2, type: #? str ?
        else:
            self.clabels = None
            
        # population label distribution
        self.label_types, self.label_counts = np.unique(self.labels, return_counts=True)  # label_types: every unique lable, label_counts: the relative counts
        self.label_dist = self.label_counts / np.float(self.label_counts.sum())
        self.num_labels = self.label_types.size
        self.batch_inds = np.zeros(self.num_labels, dtype=np.int)  # batch indices
        print('Label types: {}'.format(self.label_types))
        print('Label counts: {}'.format(self.label_counts))
        print('Label distributions: {}'.format(self.label_dist))

        # indices to keep track of which samples are used so far within training. important to count epochs, rather than iterations.
        self.batch_ind = 0
        self.batch_ind_test = 0
        self.batch_ind_valid = 0

        # place holders that will be specified if some of the following functions are called
        self.train_part = []  # indices of rows allocated to the training set
        self.test_part = []
        self.validation_part = []

    # this function gets features and labels of samples
    def get_data_part(self, rows, noise_std=1):
        srows = sorted(rows)
        d1 = np.squeeze(self.codes[srows, 0, :])  # d1.shape == (len(rows), 4096)
        d2 = np.squeeze(self.codes[srows, 1, :])
        d3 = np.squeeze(self.codes[srows, 2, :])
        d4 = np.squeeze(self.codes[srows, 3, :])
        if noise_std is not None:
            d1 += np.random.normal(loc=0, scale=0.05, size=d1.shape)
            d2 += np.random.normal(loc=0, scale=0.05, size=d2.shape)
            d3 += np.random.normal(loc=0, scale=0.05, size=d3.shape)
            d4 += np.random.normal(loc=0, scale=0.05, size=d4.shape)
        l = np.zeros([d1.shape[0], self.num_labels], dtype=np.float)  # (len(rows), num_labels)
        lpart = self.labels[srows]  # labels on the spicified `rows`
        for lab_, k in zip(self.label_types, range(self.num_labels)):
            l[lpart == lab_, k] = 1  # ????
        return d1, d2, d3, d4, l

    def get_train_batch(self, batch_size):
        rows = self.train_part[self.batch_ind: self.batch_ind + batch_size]
        d1, d2, d3, d4, l = self.get_data_part(rows)
        self.batch_ind += batch_size
        if self.batch_ind >= len(self.train_part):
            self.batch_ind = 0
        return d1, d2, d3, d4, l

    def get_balanced_train_batch(self, batch_size):
        rows = []
        _num_lables = self.label_types.size
        num_per_label = np.int(batch_size / np.float(_num_lables))
        for l in self.label_types:
            lrows = np.random.permutation(np.where(self.labels[self.train_part] == l)[0])
            lrows = lrows[:num_per_label].astype(np.int)
            rows += list(np.asarray(self.train_part)[lrows])
        d1, d2, d3, d4, l = self.get_data_part(rows)
        return d1, d2, d3, d4, l

    def get_train_data(self):
        rows = self.train_part
        d1, d2, d3, d4, l = self.get_data_part(rows)
        return d1, d2, d3, d4, l

    def get_validation_batch(self, batch_size):
        rows = self.validation_part[self.batch_ind_valid: self.batch_ind_valid + batch_size]
        d1, d2, d3, d4, l = self.get_data_part(rows)
        self.batch_ind_valid += batch_size
        if self.batch_ind_valid >= len(self.validation_part):
            self.batch_ind_valid = 0
        return d1, d2, d3, d4, l

    def get_balanced_validation_batch(self, batch_size):
        rows = []
        _num_labels = self.label_types.size
        num_per_label = np.int(batch_size / np.float(_num_labels))
        for l in self.label_types:
            lrows = np.random.permutation(np.where(self.labels[self.validation_part] == l)[0])
            lrows = lrows[:num_per_label].astype(np.int)
            rows += list(np.asarray(self.validation_part)[lrows])
        d1, d2, d3, d4, l = self.get_data_part(rows)
        return d1, d2, d3, d4, l

    def get_validation_data(self):
        rows = self.validation_part
        d1, d2, d3, d4, l = self.get_data_part(rows)
        return d1, d2, d3, d4, l

    def get_test_batch(self, batch_size):
        rows = self.test_part[self.batch_ind_test: self.batch_ind_test + batch_size]
        d1, d2, d3, d4, l = self.get_data_part(rows)
        self.batch_ind_test += batch_size
        if self.batch_ind_test >= len(self.test_part):
            self.batch_ind_test = 0
        return d1, d2, d3, d4, l

    def get_test_data(self):
        rows = self.test_part
        d1, d2, d3, d4, l = self.get_data_part(rows)
        return d1, d2, d3, d4, l

    def write_preds(self, preds, fname):
        srows = sorted(self.test_part)
        data_matrix = np.append(self.vmap[srows,:], self.labels[srows, np.newaxis], axis=1)
        data_matrix = np.append(data_matrix, preds[:, np.newaxis], axis=1)
        pred_matrix = pd.DataFrame(data=data_matrix, columns=['img_id', 'pcd', 'oa11', 'lsoa11', self.label_name, 'predicted'])
        pred_matrix.to_csv(fname,index=False)

class Dataset_CrossValidation(Dataset):
    def __init__(self, image_hdf5_file, label_pickle_file, label_name, clabel_name=None):
        Dataset.__init__(self,
                         image_hdf5_file,
                         label_pickle_file,
                         label_name,
                         clabel_name=clabel_name)

    def pick_label(self, part_gen, part_file, part_kn=5, part_kp=0, psize=1.0, seed=None):
        ''' This runs at every creation instance
            label_type: 'cat' (categorial), 'cont' (continuous)
            `part_gen`: #
            `part_file: name of the file of the k partitions (pickle)
            `part_kn`: total number of partitions in cross-validation
            `part_kp`: the index of the part used as test set in cross-validation
            `psize`: percentage of the training used'''
        if part_gen == 1:
            """ this part creates partitions from the data and saves them in a specified file.
                the partitioning is k-folds and stratified (arranged in a sequence of grades or ranks)
                it also allows constraints, see above comment, as clabels."""
            print('==================================== generating partitions from selected classes =====================================')
            self.kpartitions = partitioning.partition_stratified_kfold(part_kn,
                                                                       self.labels,
                                                                       seed=seed,
                                                                       clabels=self.clabels)
            print('==================================== generating partitions =====================================')
            # ??? what's the difference
            self.kpartitions = partitioning.partition_stratified_kfold(part_kn,
                                                                      self.labels,
                                                                      seed=seed,
                                                                      clabels=self.clabels)
            pickle.dump(self.kpartitions, open(part_file, 'wb'))
        else:
            """
                reads a partitioning that was written before,
                e.g. if 5 fold cross-validation is used, then this file simply will have 5 partitions written in it.
                self.kpartitions is a list with 5 members and each member has a list of data sample ids."""
            self.kpartitions = pickle.load(open(part_file, 'rb'))
        """
            creates training and test part from the self.kpartitions.
            e.g.    in 5 fold cross validation, it uses the fold part_kp (1 or 2 or ... or 5) as test and combines the remaining 4 as training data."""
        _train_part, self.test_part = partitioning.get_partition_stratified_kfold(part_kp, self.kpartitions)
        """
            psize indicates the percentage of training data to be used during training.
            If it is less then 1.0 then we take a subset of the training data with the same proportions of classes, i.e. stratified.
            Note that inside `decimate_partition_stratified`, we randomly permute over the samples,
            so every time you run this code, training will happen with another subset of size psize."""
        if psize < 1.0:
            self.train_part = partitioning.decimate_partition_stratified(_train_part, self.labels, psize=psize)
        else:
            self.train_part = _train_part

    def get_validation_batch(self, batch_size):
        return self.get_train_batch(batch_size)

    def get_balanced_validation_batch(self, batch_size):
        return self.get_balanced_train_batch(batch_size)

class Dataset_TVT(Dataset):
    def __init__(self, image_hdf5_file, label_pickle_file, label_name, clabel_name=None):
        Dataset.__init__(self,
                         image_hdf5_file,
                         label_pickle_file,
                         label_name,
                         clabel_name=clabel_name)

    def pick_label(self, part_gen, part_file, train_size, valid_size, psize=1.0, seed=None):
        ''' This runs at every creation instance
            part_gen: indicator to specifify whether the data has been allocated
            part_file: path to write or read the partition result
            train_size: percentage of tt data used as training set
            valid_size: percentage of tt data used for validation
            psize: the percentage of training data to be used during training
            label_type: 'cat' (categorial), 'cont' (continuous)'''
        if part_gen == 1:
            """ this part creates partitions from the data and saves them in a specified file.
                the partitioning is stratified and only 3 parts: train, test and validation.
                it also allows constrains, see above comment, as clabels."""
            print('==================================== generating partitions =====================================')
            _train_part, self.validation_part, self.test_part = partitioning.partition_stratified_validation(self.labels,
                                                                                                                 train_size,
                                                                                                                 valid_size,
                                                                                                                 seed=seed,
                                                                                                                 clabels=self.clabels)
            pickle.dump(_train_part, open(part_file + '_train', 'wb'))
            pickle.dump(self.validation_part, open(part_file + '_validation', 'wb'))
            pickle.dump(self.test_part, open(part_file + '_test', 'wb'))
        else:

            """ reads a partitioning that was written before.
                there are three files: validation, test and train"""
            _train_part = pickle.load(open(part_file + '_train', 'rb'))
            self.validation_part = pickle.load(open(part_file + '_validation', 'rb'))
            self.test_part = pickle.load(open(part_file + '_test', 'rb'))


        """ psize indicates the percentage of training data to be used during training.
            If it is 1.0, then we use all the training data. So `self.train_part = _train_part`
            If it is less then 1.0 then we take a subset of the training data with the same proportions of classes, i.e. stratified.
            Note that inside decimate_partition_stratified code, we randomly permute over the samples.
            So every time you run this code, training will happen with another subset of size psize."""
        if psize < 1.0:
            self.train_part = partitioning.decimate_partition_stratified(_train_part, self.labels, psize=psize)
        else:
            self.train_part = _train_part

    def write_preds_validation(self, preds, fname):
        srows = sorted(self.validation_part)
        data_matrix=np.append(self.vmap[srows,:],self.labels[srows,np.newaxis],axis=1)
        data_matrix=np.append(data_matrix,preds[:,np.newaxis],axis=1)
        pred_matrix=pd.DataFrame(data=data_matrix,columns=['img_id', 'pcd', 'oa11', 'lsoa11',self.label_name,'predicted'])
        pred_matrix.to_csv(fname,index=False)

class Dataset_TT(Dataset):
    def __init__(self, image_hdf5_file, label_pickle_file, label_name, clabel_name=None):
        Dataset.__init__(self,
                         image_hdf5_file,
                         label_pickle_file,
                         label_name,
                         clabel_name=clabel_name)

    def pick_label(self, part_gen, part_file, train_size, psize=1.0, seed=None):
        ''' part_gen: indicator to specifify whether the data has been allocated
            part_file: path to write or read the partition result
            train_size: percentage of tt data used as training set
            psize: the percentage of training data to be used during training
            This runs at every creation instance
            label_type: 'cat' (categorial), 'cont' (continuous)
        '''
           
        if part_gen == 1:
            """ this part creates partitions from the data and saves them in a specified file.
                the partitioning is stratified and only 2 parts: train, and test """
            print('==================================== generating partitions =====================================')
            _train_part, self.test_part = partitioning.partition_stratified(self.labels,
                                                                            train_size,
                                                                            seed=seed,
                                                                            clabels=self.clabels)
            pickle.dump(_train_part, open(part_file + '_train', 'wb'))
            pickle.dump(self.test_part, open(part_file + '_test', 'wb'))
        else:
            """ reads a partitioning that was written before.
                there are three files: validation, test and train"""
            _train_part = pickle.load(open(part_file + '_train', 'rb'))
            self.test_part = pickle.load(open(part_file + '_test', 'rb'))


        """ psize indicates the percentage of training data to be used during training.
            If it is 1.0, then we use all the training data. So `self.train_part = _train_part`
            If it is less then 1.0 then we take a subset of the training data with the same proportions of classes, i.e. stratified.
            Note that inside decimate_partition_stratified code, we randomly permute over the samples.
            So every time you run this code, training will happen with another subset of size psize."""
        if psize < 1.0:
            self.train_part = partitioning.decimate_partition_stratified(_train_part, self.labels, psize=psize)
        else:
            self.train_part = _train_part

    def get_validation_batch(self, batch_size):
        return self.get_train_batch(batch_size)

    def get_balanced_validation_batch(self, batch_size):
        return self.get_balanced_train_batch(batch_size)

    def write_preds_validation(self, preds, fname):
        srows = sorted(self.validation_part)
        data_matrix=np.append(self.vmap[srows,:],self.labels[srows,np.newaxis],axis=1)
        data_matrix=np.append(data_matrix,preds[:,np.newaxis],axis=1)
        pred_matrix=pd.DataFrame(data=data_matrix,columns=['img_id', 'pcd', 'oa11', 'lsoa11',self.label_name,'predicted'])
        pred_matrix.to_csv(fname,index=False)

class Dataset_TT_byclass(Dataset):
    def __init__(self, image_hdf5_file, label_pickle_file, label_name, clabel_name=None, label_test=None):
        """ label_test: labels of the selected class used as test set"""
        Dataset.__init__(self,
                         image_hdf5_file,
                         label_pickle_file,
                         label_name,
                         clabel_name=clabel_name)
        self.label_test = label_test

    def pick_label(self, part_gen, part_file, train_size, psize=1.0, seed=None):
        ''' part_gen: indicator to specifify whether the data has been allocated
            part_file: path to write or read the partition result
            train_size: percentage of tt data used as training set
            psize: the percentage of training data to be used during training
            This runs at every creation instance
            label_type: 'cat' (categorial), 'cont' (continuous)'''
        if part_gen == 1:
            """ this part creates partitions from the data and saves them in a specified file.
                the partitioning is stratified and only 2 parts: train, and test ."""
            print('==================================== generating partitions =====================================')
            _train_part, self.test_part = partitioning.partition_by_class(self.labels, 
                                                                            self.label_test, 
                                                                            seed=seed)
            pickle.dump(_train_part, open(part_file + '_train', 'wb'))
            pickle.dump(self.test_part, open(part_file + '_test', 'wb'))
            
        else:
            """ reads a partitioning that was written before.
                there are three files: validation, test and train"""
            _train_part = pickle.load(open(part_file + '_train', 'rb'))
            self.test_part = pickle.load(open(part_file + '_test', 'rb'))


        """ psize indicates the percentage of training data to be used during training.
            If it is 1.0, then we use all the training data. So `self.train_part = _train_part`
            If it is less then 1.0 then we take a subset of the training data with the same proportions of classes, i.e. stratified.
            Note that inside decimate_partition_stratified code, we randomly permute over the samples.
            So every time you run this code, training will happen with another subset of size psize."""
        if psize < 1.0:
            self.train_part = partitioning.decimate_partition_stratified(_train_part, self.labels, psize=psize)
        else:
            self.train_part = _train_part

    def get_validation_batch(self, batch_size):
        return self.get_train_batch(batch_size)

    def get_balanced_validation_batch(self, batch_size):
        return self.get_balanced_train_batch(batch_size)
        
    def write_preds_validation(self, preds, fname):
        srows = sorted(self.validation_part)
        data_matrix=np.append(self.vmap[srows,:],self.labels[srows,np.newaxis],axis=1)
        data_matrix=np.append(data_matrix,preds[:,np.newaxis],axis=1)
        pred_matrix=pd.DataFrame(data=data_matrix,columns=['img_id', 'pcd', 'oa11', 'lsoa11',self.label_name,'predicted'])
        pred_matrix.to_csv(fname,index=False)
