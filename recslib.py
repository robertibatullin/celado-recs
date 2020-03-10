#!/usr/bin/env python
# coding: utf-8

# # Библиотека общих функций для рекомендательной системы

#TODO
#вставлять текущий год и месяц в соотв. поля при предикте
#разбить на трейн-тест заранее, файлы с именами

import os, re
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import coverage_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def read_list(filename):
    rows = open(filename, 'r', encoding='utf8').readlines()
    items = map(lambda s:s.strip(), rows )
    items = filter(lambda s:len(s) > 0 and s[0] != '#', items)
    return list(items)

class RecommenderSystem():
    
    def __read_config(self):
        cfg_path = os.path.join(self.folder, 'config.txt')
        cfg_lines = read_list(cfg_path)
        vars_to_train_index = cfg_lines.index('VARS TO TRAIN =')
        self.vars_to_train = cfg_lines[vars_to_train_index+1:]
        cfg_lines = cfg_lines[:vars_to_train_index]
        splits = [s.split('=') for s in cfg_lines]
        dct = {s[0].strip():s[1].strip() for s in splits}
        self.customer_id = dct['CUSTOMER ID']
        self.offer_id = dct['OFFER ID']
        self.contract_id = dct['CONTRACT ID']
        self.id_vars = [self.customer_id, self.offer_id, self.contract_id]
        self.target_var = dct['TARGET']
        self.contract_year = dct['CONTRACT YEAR']
        self.contract_year_epoch = int(dct['CONTRACT YEAR EPOCH'])
        self.contract_month = dct['CONTRACT MONTH']
        self.clf_name = dct['CLASSIFIER']
    
    def __init__(self,
                 model_path = 'model',
                 version = 'last',
                 mode = 'read',#explore|train|predict|precompute|read (from precomputed predictions) 
                 current_year = None,
                 current_month = None,
                 min_samples_in_class = 5,
                 test_size = 0.25,
                 **kwargs): 
        self.model_path = model_path
        version_list = list(filter(lambda s:re.match('\d+\.\d+', s) is not None,
                              os.listdir(self.model_path) ))
        version_list = list(filter(lambda s:os.path.isdir(os.path.join(self.model_path,s)),
                              version_list))
        if version == 'last':
            self.version = sorted(version_list)[-1]
        else:
            self.version = version
        print(f'Initializing recommender system (version {self.version}) for {mode} mode')
        folder = os.path.join(self.model_path,
                              self.version)
        self.folder = folder
        self.__read_config()
        now = datetime.now()
        self.current_year = now.year if current_year == None else current_year
        self.current_month = now.month if current_month == None else current_month
        self.current_year = self.current_year - self.contract_year_epoch
        
        files_required = ['customers.csv',
                        'offers.csv',
                        'contracts.csv',
                        ]
        for file in files_required:
            if not os.path.exists(os.path.join(folder, file)):
                raise FileNotFoundError(f'Required file {file} not found. Preprocess data first.')

        self.clf_filename = os.path.join(self.folder, 
                                         self.clf_name+'.pkl')
        self.mode = mode
        self.threshold = min_samples_in_class
        self.test_size = test_size

        if not self.mode in ('train','precompute','create'):
            files_required = ['probs.csv', 'model_cols.txt', self.clf_name+'.pkl']
            for file in files_required:
                if not os.path.exists(os.path.join(folder, file)):
                    raise FileNotFoundError(f'Required file {file} not found. Train model first.')
            self.Xcols = read_list(os.path.join(folder, 'model_cols.txt'))
                
        print('RecommenderSystem initialized')
                
        if mode in ('predict','precompute')\
        and os.path.exists(self.clf_filename):
            print(f'Loading trained model "{self.clf_filename}"')
            self.clf = pickle.load(open(self.clf_filename,'rb'))
            print('Model loaded')
                                    
        if mode == 'train':                
            self.train(**kwargs)
            
        if mode in ('train', 'precompute'):
            self.precompute()
            
        if mode == 'explore':
            mean_recalls = self.recall_at_k(10, 
                                            500, 
                                            verbose=False)
                        
    def set_mode(self, new_mode):
        self.mode = new_mode
                        
    def unite(self, other, version):
        print('Uniting RecommenderSystem versions {0} and {1}'.format(self.version, other.version))
        folder = os.path.join(self.model_path,
                      version)
        if not os.path.exists(folder):
            os.mkdir(folder)
        for wd in ['customers.csv',
                   'offers.csv',
                   'contracts.csv']:#['opportunity.csv', 'fleet.csv']:
            f1 = pd.read_csv(os.path.join(self.folder, wd))
            f2 = pd.read_csv(os.path.join(other.folder, wd))
            f = pd.concat([f1,f2], join='outer', ignore_index=True)
            f.drop_duplicates(inplace=True)
            f.to_csv(os.path.join(folder, wd), index=False)
        return RecommenderSystem(model_path = self.model_path,
                                 version = version, 
                                 mode='create',
                                 test_size = self.test_size)
        
    def __normalize_encode(self, data):
        #переводим текстовые переменные в категориальные
        #кодируем one-hot
        #float -> int
        #сливаем и сортируем столбцы по алфавиту
        data_id_vars = list(set(data.columns).intersection(set(self.id_vars)))
        if data_id_vars != []:
            id_cols = data[data_id_vars]
            cat_cols = data.drop(data_id_vars,axis=1).select_dtypes(include=['object'])
        else:
            id_cols = None
            cat_cols = data.select_dtypes(include=['object'])
        num_cols = data.select_dtypes(exclude=['object'])
        num_cols.drop(self.id_vars, axis=1, inplace=True, errors='ignore')
#        try:
#            num_cols = num_cols.drop('Offer: Month',axis=1)
#            cat_cols = cat_cols.join(data['Offer: Month'])
#        except KeyError:
#            pass
        cat_cols = cat_cols.astype('category')    
        one_hot = pd.get_dummies(cat_cols)
        to_drop = list(filter(lambda s:s[-3:]=='_no' \
                              or s[-4:]=='_0.0' \
                              or s[-2:]=='_0',
                              one_hot.columns))
        one_hot.drop(to_drop, axis=1, inplace=True, errors='ignore') 
        data1 = num_cols.join(one_hot)
        if id_cols is not None:
            data1 = data1.join(id_cols)
    
        data1.sort_index(axis=1,inplace=True)
        return data1

    def __group_by_id(self, data):
        #группируем все данные одного ids в 1 строку
        #Existing переменные суммируем (bool), Customer + Offer усредняем (num)
        data_ = data.fillna(0)
        ids = self.id_vars + [self.target_var]
        ids_ = [id_ for id_ in ids if id_ in data.columns]
        to_sum = [col for col in data.columns if 'Existing:' in col]
        to_mean = [col for col in data.columns if 'Customer:' in col or 'Offer:' in col]
        result_mean = data_.groupby(ids_)[to_mean].mean()
        result_sum = data_.groupby(ids_)[to_sum].sum()
        del data_
        result = result_mean.join(result_sum)
        result.reset_index(inplace=True)
        result.sort_index(axis=1, inplace=True)
        return result
    
    def __filter_small_classes(self, data):
        target_vals = pd.value_counts(data[self.target_var])
        rare_target_vals = target_vals[target_vals < self.threshold].index
        filtered = data[~data[self.target_var].isin(rare_target_vals)]
        return filtered
        
    def __get_Xy(self,
               use = 'for train', #or for predict
               customer_df = None
               ):
        if customer_df is None:
            customers = pd.read_csv(os.path.join(self.folder, 'customers.csv'))
        else:
            customers = customer_df
        offers = pd.read_csv(os.path.join(self.folder, 'offers.csv'))
        contracts = pd.read_csv(os.path.join(self.folder, 'contracts.csv'))
        how = 'left' if use == 'for train' else 'right'            
        off_contr = contracts.merge(offers, on=self.offer_id, how=how)
        data = off_contr.merge(customers, on=self.customer_id, how=how)
        del off_contr, contracts, offers, customers
        if use == 'for train':
            data.dropna(inplace=True)
        data.drop_duplicates().reset_index(drop=True, inplace=True)
        target = data[self.target_var]
        ids = data[self.id_vars]
        data = data.reindex(self.vars_to_train, axis=1)
        data = self.__normalize_encode(data.drop(self.target_var, axis=1))
        data = data.join(target).join(ids)
        data = self.__group_by_id(data)
        if use == 'for train' and self.threshold > 1:
            data = self.__filter_small_classes(data)            
        target = data[self.target_var]
        ids = data[self.id_vars]
        if use == 'for train':
            target_encoded, target_unique = pd.factorize(target)
            with open(os.path.join(self.folder, 'target_names.csv'),'w') as f:
                f.write('\n'.join(target_unique))
                target = target_encoded
        X_index = data[self.customer_id]
        X = data.drop(self.id_vars+[self.target_var], axis=1, errors='ignore')
        X_cols = X.columns
        if use == 'for train' and self.threshold > 1:
            print('Filtered off classes with less than {} members'.format(self.threshold))
            print (info('Full', X, target, X_index))
        if use == 'for train':
            return X, target, X_cols, X_index
        elif use == 'for predict':
            return X, X_cols, X_index
    
    def train(self, **kwargs):
        
        def cov_error(clf_, X_, y_, class_names = None): #for scoring
            pred_proba = clf_.predict_proba(X_)
            y_bin = pd.get_dummies(y_)
            if class_names is not None:
                for c in class_names:
                    if not c in y_bin.columns:
                        y_bin[c] = 0
            y_bin.sort_index(axis=1, inplace=True)
            return -coverage_error(y_bin.values, pred_proba)
        
        def output_cov_error(X_, y_, class_names = None):
            return -int(cov_error(clf, X_, y_, class_names))
                
        def print_metrics():
            print('Train set coverage error', output_cov_error(X_train,y_train))
            print('Test set coverage error', 
                  output_cov_error(X_test,
                                   y_test,
                                   class_names = pd.unique(y_train)))
            
        X, y, self.Xcols, Xind = self.__get_Xy()
        
        with open(os.path.join(self.folder, 'model_cols.txt'), 'w') as f:
                f.write('\n'.join(self.Xcols))
        
        if self.test_size > 0:
            print('Train-test splitting')
            if self.threshold == 1:
                #удваиваем датасет, чтобы в каждом классе было не меньше 2 экз
                X = pd.concat([X,X])
                y = pd.concat([y,y])
                Xind = pd.concat([Xind, Xind])
            X_train, X_test, y_train, y_test,\
            self.train_cnums, self.test_cnums = train_test_split(X, y,
                                                                 Xind,
                                                                 test_size = self.test_size,
                                                                 stratify = y)
        else:
            X_train, X_test, y_train, y_test,\
            self.train_cnums, self.test_cnums = X, X, y, y, Xind, Xind
            
        print (info('Train', X_train, y_train, self.train_cnums))
        print (info('Test', X_test, y_test, self.test_cnums))

        X_train = sparse.csr_matrix(X_train)
        X_test = sparse.csr_matrix(X_test)
        del X, y
        
        if self.clf_name == 'rf':
            clf = RandomForestClassifier(class_weight='balanced',
                                         n_estimators=40,
                                         random_state=0,
                                         **kwargs)
            print('Training random forest')
        elif self.clf_name == 'log':
            clf = make_pipeline( StandardScaler(with_mean=False), 
                                LogisticRegression(multi_class='multinomial',
                                                   solver='saga',
                                                   C=0.002) )
            print('Training logistic regression')
        elif self.clf_name == 'kn':
            clf = make_pipeline( StandardScaler(with_mean=False), 
                                KNeighborsClassifier(n_neighbors=50,
                                        **kwargs) )
            print('Training k-neighbors classifier')
        elif self.clf_name == 'svm':
            clf = make_pipeline( StandardScaler(with_mean=False), 
                                SVC(kernel='rbf',
                                    probability=True,
                                    **kwargs) )
            print('Training SVC')
        if self.mode == 'explore':
            if self.clf_name == 'log':
                param_grid = {'logisticregression__penalty':['l2'],
                              'logisticregression__C':[0.002, 0.005, 0.008]}
            elif self.clf_name == 'rf':
                param_grid = {'n_estimators':[10, 20, 40, 100],
                              'criterion':['gini','entropy']
                              }
            grid = GridSearchCV(clf, 
                                scoring = cov_error,
                                verbose = 0,
                                param_grid = param_grid)
#            X_train, X_test, y_train, y_test = train_test_split(X,y, 
#                                                                test_size=0.4,
#                                                                stratify = y)
#            print('Train set num classes',len(pd.unique(y_train)))
#            print('Test set num classes',len(pd.unique(y_test)))
#            clf = clf.fit(X_train,y_train)
            grid.fit(X_train, y_train)
            clf = grid.best_estimator_
            print_metrics()
            print(grid.best_params_)
        elif self.mode == 'train':
            clf = clf.fit(X_train,y_train)
            print_metrics()
            self.clf = clf            
            print('Saving model')
            pickle.dump(clf, open(self.clf_filename, 'wb'))
            print('Model saved to '+self.clf_filename)

    def predict(self, customer_df, num_top = None):
        target_names = read_list(os.path.join(self.folder, 'target_names.csv'))
        X, _, customer_nums = self.__get_Xy(use='for predict', 
                                          customer_df = customer_df)
        X = X.reindex(self.Xcols, axis=1, fill_value=0)
        X[self.contract_month] = self.current_month
        X[self.contract_year] = self.current_year
        data = self.clf.predict_proba(X)
        pp = pd.DataFrame(index = customer_nums,
                          columns = target_names,
                          data = data)
        pp.reset_index(inplace=True)
        pp.drop_duplicates(inplace=True)
        pp = pp.groupby(self.customer_id).mean().reset_index()
        if num_top is None: #одна таблица на всех покупателей 
            return pp
        else: #для каждого покупателя своя таблица, отсортированная по вероятности
            results = []
            for cid in pp.index:
                result = pp.loc[cid].copy()
                header = result[self.customer_id]
                result.drop(self.customer_id, inplace=True)
                result.sort_values(ascending = (num_top<0), inplace=True)
                result = result.head(np.abs(num_top))
                result = pd.DataFrame({header:result.index,
                        'Propensity to buy':result})
                results.append(result)
            return results
        
    def precompute(self):        
        print('Precomputing probabilities')
        pp = self.predict(customer_df = None)
        pp.to_csv(os.path.join(self.folder, 'probs.csv'), 
                  sep=';',
                  index=False)
        print('Precomputed probabilities saved to {}/probs.csv.'.format(self.folder))

    def get_top_from_table(self, input_id, #customer name or model name
                           find, #customers|offers
                           num_top,
                           show_really_sold = False):
        print('Getting top from precomputed table')
        probs = pd.read_csv(os.path.join(self.folder, 'probs.csv'),
                         sep=';', index_col=0)
        try:
            if find == 'offers':
                series = probs.loc[input_id].copy()
            elif find == 'customers':
                series = probs[input_id].copy()
        except KeyError or IndexError:
            return None
        series.sort_values(ascending=(num_top < 0), inplace=True)
        series = series.head(np.abs(num_top))
        df = pd.DataFrame(series).reset_index()
        if find == 'offers':
            df.rename(columns = {'index':'Offer',
                                 df.columns[1]:'Propensity to buy'}, inplace=True)
            if show_really_sold:
                df['Really sold?'] = df['Offer'].apply(lambda offer:self.is_really_sold(input_id, offer))
        elif find == 'customers':
            df.rename(columns = {df.columns[0]:'Customer',
                                 df.columns[1]:'Propensity to buy'}, inplace=True)
            if show_really_sold:
                df['Really sold?'] = df['Customer'].apply(
                        lambda customer:self.is_really_sold(customer, input_id))                
        return df

    def is_really_sold(self, customer, offer):
        df = pd.read_csv(os.path.join(self.folder,'contracts.csv'))
        rs = df[(df[self.customer_id]==customer) & (df[self.target_var]==offer)]
        res = 'yes' if len(rs)>0 else 'no'
        return res

    def retrain(self, source_folder):
        #в source_folder должны быть записаны готовые файлы csv
        time = datetime.now().strftime('%Y-%m-%d %H:%M')
        new = RecommenderSystem(model_path = self.model_path,
                                version = source_folder,
                                mode = 'create')
        version_ = self.version.split('.')
        new_version_ = [version_[0], str(int(version_[1])+1)]
        new_version = '.'.join(new_version_)
        updated = self.unite(new, new_version)

#        for filename in os.listdir(new.folder):
#            os.remove(os.path.join(new.folder, filename))
#        os.removedirs(new.folder)

        updated.set_mode('train')
        updated.train()
        updated.precompute()
        log_str = 'Recommender system updated from version {2} to {0} with new data from "{1}"'.format(
                new_version,
                os.path.split(source_folder)[-1],
                self.version)
        print(time, log_str, file = open(os.path.join(self.model_path,
                                     'update.log'),'a'))
        return updated, log_str

    def get_real_sells(self, customer_id):
        contracts = pd.read_csv(os.path.join(self.folder,'contracts.csv'))
        real_sells = contracts[contracts[self.customer_id] == customer_id][self.target_var]
        real_sells.drop_duplicates(inplace=True)
        return real_sells
                
    def recall_at_k(self, Ks, n_samples=None, verbose=False):
        print('Calculating recall scores')
        probs = pd.read_csv(os.path.join(self.folder, 'probs.csv'),
                            sep=';', index_col = 0) 
        try:
            test_cnums = self.test_cnums
        except AttributeError:
            test_size = len(probs) if self.test_size == 0 else int(self.test_size*len(probs))
            test_cnums = np.random.choice(probs.index, 
                                     size=test_size, 
                                     replace=False)
        if n_samples is None:
            print('Testing all {} samples from test set'.format(len(test_cnums)))
            cnums = test_cnums
        else:
            print('Testing {} random samples from test set'.format(n_samples))
            cnums = np.random.choice(test_cnums, 
                                     size=n_samples, 
                                     replace=False)
        if type(Ks) is int:
            top_sizes = [Ks]
            max_top_size = Ks
        elif type(Ks) is list:
            top_sizes = Ks
            max_top_size = np.max(Ks)
        recalls = {top_size:{'pos':[], 'neg':[]} for top_size in top_sizes}
        for cnum in cnums:
            real_sells = self.get_real_sells(customer_id = cnum)
            if len(real_sells) == 0:
                real_sells = {'nothing'}
                n_neg, n_pos = 1, 0
            else:
                real_sells = set(real_sells)
                n_neg = 1 if 'nothing' in real_sells else 0
                n_pos = len(real_sells) - n_neg
            maxtop = probs.loc[cnum].sort_values(ascending=False).head(max_top_size)
            if verbose:
                print(cnum) 
                print('RELEVANT: ', ', '.join(real_sells))
            for top_size in top_sizes:
                top = maxtop.head(top_size)
                top = list(top.index)
                if verbose:
                    print(f'RECOMMENDED TOP {max_top_size}:', ', '.join(top))
                n_pos_matches = len([s for s in top if s in real_sells and s!='nothing'])
                n_neg_matches = 1 if 'nothing' in top and 'nothing' in real_sells else 0
                if n_neg > 0:
                    neg_recall = n_neg_matches/n_neg
                    recalls[top_size]['neg'].append(neg_recall)
                    if verbose:
                        print(f'NEG RECALL AT TOP {max_top_size}:',neg_recall)
                if n_pos > 0: 
                    pos_recall = n_pos_matches/n_pos
                    recalls[top_size]['pos'].append(pos_recall)
                    if verbose:
                        print(f'POS RECALL AT TOP {max_top_size}:',pos_recall)
                if verbose:
                    print()
        means = {}    
        for top_size in top_sizes:
            pos = recalls[top_size]['pos']
            neg = recalls[top_size]['neg']
            pos = 0 if len(pos) == 0 else np.mean(pos)
            neg = 0 if len(neg) == 0 else np.mean(neg)
            means[top_size] = {'pos':pos, 'neg':neg}
            print('TOP SIZE:',top_size)
            print('Recall for positive predictions:',pos)
            print('Recall for negative predictions:',neg)
        return means
    
    def write_and_show_results(self, results, lang='en'):
        tables, footers = [], []
        for i,result in enumerate(results):
            filename = now.strftime("%Y-%m-%d_%H-%M")+'_'+str(i)+'.xlsx'
            url = 'http://receiptparser.pythonanywhere.com/zeppelin/results/'+filename
            filepath = os.path.join(self.model_path,
                                    'results',
                                    filename)
            result.to_excel(filepath, index=False)
            tables.append(  result.to_html(index=False) )
            if lang == 'ru':
                footers.append( f'Результат сохранён в файл <a href={url}>{filename}</a>' )
            elif lang == 'en':
                footers.append( f'Results saved to the file <a href={url}>{filename}</a>' )
        return tables, footers    
        
def read_table(source):
    if source.split('.')[-1].lower() == 'xlsx':
        return pd.read_excel(source)
    elif source.split('.')[-1].lower() == 'csv':
        return pd.read_csv(source)
    else:
        raise TypeError('Input file must be .xlsx or .csv')

def info(name, X, y, Xind):
    s = '{} dataset\n'.format(name)
    s += '\t{} samples\n'.format(X.shape[0])
    s += '\t{} features\n'.format(X.shape[1])
    s += '\t{} classes\n'.format(len(pd.unique(y)))
    s += '\t{} unique sample IDs'.format(len(pd.unique(Xind)))
    return s
