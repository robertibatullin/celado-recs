#!/usr/bin/env python
# coding: utf-8

# # Библиотека общих функций для рекомендательной системы

#TODO
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

def printto(*args, out=None):
    if out == None:
        print(*args)
    else:
        print(*args, file=open(out,'a'))

class RecommenderSystem():
    
    def __read_config(self):
        cfg_path = os.path.join(self.folder, 'config.txt')
        if not os.path.exists(cfg_path):
            raise FileNotFoundError('config.txt not found.')
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
                 output = None,
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
        self.output = output
        printto(f'Initializing recommender system (version {self.version}) for {mode} mode',
              out=self.output)
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
        
        files_required = []
        if mode == 'precompute':
            files_required = ['model_cols.txt', self.clf_name+'.pkl']
        elif mode == 'predict':
            files_required = ['probs.csv', 'model_cols.txt', self.clf_name+'.pkl']
        elif mode in ('read','explore'):
            files_required = ['probs.csv']

        if not os.path.exists(os.path.join(folder, file)):
            raise FileNotFoundError(f'Required file {file} not found. Train model first.')
        
        if 'model_cols.txt' in files_required:
            self.Xcols = read_list(os.path.join(folder, 'model_cols.txt'))
                
        printto('RecommenderSystem initialized', out=self.output)
                
        if mode in ('predict','precompute')\
        and os.path.exists(self.clf_filename):
            printto(f'Loading trained model "{self.clf_filename}"', out=self.output)
            self.clf = pickle.load(open(self.clf_filename,'rb'))
            printto('Model loaded', out=self.output)
                                    
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
        printto('Uniting RecommenderSystem versions {0} and {1}'.format(self.version, 
                other.version), out=self.output)
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
        to_sum = data_.drop(ids_, axis=1).filter(regex='^Existing: ')
        to_sum = to_sum.join(data_[ids_])
        to_mean = data.drop(ids_, axis=1).filter(regex='(^Customer: )|(^Offer: )')
        to_mean = to_mean.join(data_[ids_])
        result_mean = to_mean.groupby(ids_).mean()
#        print('groupby mean ready',result_mean.shape)
        result_sum = to_sum.groupby(ids_).sum()
#        print('groupby sum ready',result_sum.shape)
        del data_, to_mean, to_sum
        result = result_mean.join(result_sum)
        del result_mean, result_sum
        result.reset_index(inplace=True)
        result.sort_index(axis=1, inplace=True)
        return result
    
    def __filter_small_classes(self, data):
        target_vals = pd.value_counts(data[self.target_var])
        popular_target_vals = target_vals[target_vals >= self.threshold].index
        filtered = data[data[self.target_var].isin(popular_target_vals)]
        return filtered
        
    def __get_Xy(self,
               use = 'for train', #or for predict
               customer_df = None
               ):
        if customer_df is None:
            customers = pd.read_csv(os.path.join(self.folder, 'customers.csv'))
        else:
            customers = customer_df
#        print('customers',customers.shape)
        offers = pd.read_csv(os.path.join(self.folder, 'offers.csv'))
#        print('offers',offers.shape)
        contracts = pd.read_csv(os.path.join(self.folder, 'contracts.csv'))
#        print('contracts',contracts.shape)
        how = 'left' if use == 'for train' else 'right'            
        off_contr = contracts.merge(offers, on=self.offer_id, how=how)
        data = off_contr.merge(customers, on=self.customer_id, how=how)
#        print('data',data.shape)
        del off_contr, contracts, offers, customers
        if use == 'for train':
            data.dropna(inplace=True)
#        print('data dropna',data.shape)
        data.drop_duplicates().reset_index(drop=True, inplace=True)
#        print('data drop dupl',data.shape)
        target = data[self.target_var]
        ids = data[self.id_vars]
        data = data.reindex(self.vars_to_train, axis=1)
        data = self.__normalize_encode(data.drop(self.target_var, axis=1))
        data = data.join(target).join(ids)
#        print('data encoded',data.shape)
        data = self.__group_by_id(data)
#        print('data grouped',data.shape)
        if use == 'for train' and self.threshold > 1:
            data = self.__filter_small_classes(data)  
#            print('data filtered',data.shape)
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
            printto('Filtered off classes with less than {} members'.format(self.threshold),
                    out=self.output)
            printto(info('Full', X, target, X_index), out=self.output)
        if use == 'for train':
            return X, target, X_cols, X_index
        elif use == 'for predict':
            return X.fillna(0), X_cols, X_index
    
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
            printto('Train set coverage error', output_cov_error(X_train,y_train), out=self.output)
            printto('Test set coverage error', 
                  output_cov_error(X_test,
                                   y_test,
                                   class_names = pd.unique(y_train)), out=self.output)
            
        X, y, self.Xcols, Xind = self.__get_Xy()
        
        with open(os.path.join(self.folder, 'model_cols.txt'), 'w') as f:
                f.write('\n'.join(self.Xcols))
        
        if self.test_size > 0:
            printto('Train-test splitting', out=self.output)
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
            
        printto (info('Train', X_train, y_train, self.train_cnums), out=self.output)
        printto (info('Test', X_test, y_test, self.test_cnums), out=self.output)

        X_train = sparse.csr_matrix(X_train)
        X_test = sparse.csr_matrix(X_test)
        
#        X = pd.DataFrame(X, columns=self.Xcols)
#        lx = len(X)
#        print(lx)
#        for c in X.columns:
#            if lx != len(X[c].dropna()):
#                print(c, len(X[c].dropna()))
        del X, y
        
        if self.clf_name == 'rf':
            clf = RandomForestClassifier(class_weight='balanced',
                                         n_estimators=40,
                                         random_state=0,
                                         **kwargs)
            printto('Training random forest', out=self.output)
        elif self.clf_name == 'log':
            clf = make_pipeline( StandardScaler(with_mean=False), 
                                LogisticRegression(multi_class='multinomial',
                                                   solver='saga',
                                                   C=0.002) )
            printto('Training logistic regression', out=self.output)
        elif self.clf_name == 'kn':
            clf = make_pipeline( StandardScaler(with_mean=False), 
                                KNeighborsClassifier(n_neighbors=50,
                                        **kwargs) )
            printto('Training k-neighbors classifier', out=self.output)
        elif self.clf_name == 'svm':
            clf = make_pipeline( StandardScaler(with_mean=False), 
                                SVC(kernel='rbf',
                                    probability=True,
                                    **kwargs) )
            printto('Training SVC', out=self.output)
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
            printto(grid.best_params_, out=self.output)
        elif self.mode == 'train':
            clf = clf.fit(X_train,y_train)
            print_metrics()
            self.clf = clf            
            printto('Saving model', out=self.output)
            pickle.dump(clf, open(self.clf_filename, 'wb'))
            printto('Model saved to '+self.clf_filename, out=self.output)

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
        printto('Precomputing probabilities', out=self.output)
        pp = self.predict(customer_df = None)
        pp.set_index(self.customer_id, inplace=True)
        pp = pp.unstack().reset_index()
        pp.columns = [self.offer_id,
                      self.customer_id,
                      'PropensityToBuy']
        pp[['ProductId','NewUsed']] = pp[self.offer_id].str.split(':',expand=True)
        pp.drop(self.offer_id,axis=1,inplace=True)
        pp = pp[pp['PropensityToBuy']>0]
        pp.sort_index(axis=1, inplace=True)
        pp.to_csv(os.path.join(self.folder, 'probs.csv'), 
                  sep=';',
                  index=False)
        printto('Precomputed probabilities saved to {}/probs.csv.'.format(self.folder),
                out=self.output)

    def get_top_from_table(self, input_id, #customer or model id
                           find, #customers|offers
                           num_top,
                           show_really_sold = False):
        printto('Getting top from precomputed table', out=self.output)
        probs = pd.read_csv(os.path.join(self.folder, 'probs.csv'),
                         sep=';')
        if find == 'offers':
            source_id, target_id = 'CompanyId', 'ProductId'
        elif find == 'customers':
            source_id, target_id = 'ProductId', 'CompanyId'
        frg = probs[probs[source_id]==input_id].copy()
        del probs
        frg.sort_values('PropensityToBuy', ascending=False, inplace=True)
        if input_id == 'nothing':
            cols = [target_id, 'PropensityToBuy']
        else:
            cols = [target_id, 'NewUsed', 'PropensityToBuy']
        frg = frg[cols].head(num_top)
        if show_really_sold:
            if find == 'offers':
                frg['Really sold?'] = frg[target_id].apply(
                        lambda offer:self.is_really_sold(input_id, offer))
            elif find == 'customers':
                frg['Really sold?'] = frg[target_id].apply(
                        lambda customer:self.is_really_sold(customer, input_id))
        return frg

    def is_really_sold(self, customer, offer):
        ctr = pd.read_csv(os.path.join(self.folder,'contracts.csv'))
        customer_sells = ctr[[self.customer_id, 
                              self.offer_id, 
                              self.target_var]].copy()
        del ctr
        customer_sells = customer_sells[
                customer_sells[self.customer_id]==customer]
        if len(customer_sells) == 0:
            if offer == 'nothing':
                return 'yes'
            else:
                return 'no'
        split = customer_sells[self.target_var].str.split(':', expand=True)
        successes = customer_sells[split[0]==offer]
        if len(successes) > 0:
            return 'yes'
        else:
            return 'no'

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
        printto('Calculating recall scores', out=self.output)
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
            printto('Testing all {} samples from test set'.format(len(test_cnums)), out=self.output)
            cnums = test_cnums
        else:
            printto('Testing {} random samples from test set'.format(n_samples), out=self.output)
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
                printto(cnum, out=self.output) 
                printto('RELEVANT: ', ', '.join(real_sells), out=self.output)
            for top_size in top_sizes:
                top = maxtop.head(top_size)
                top = list(top.index)
                if verbose:
                    printto(f'RECOMMENDED TOP {max_top_size}:', ', '.join(top), out=self.output)
                n_pos_matches = len([s for s in top if s in real_sells and s!='nothing'])
                n_neg_matches = 1 if 'nothing' in top and 'nothing' in real_sells else 0
                if n_neg > 0:
                    neg_recall = n_neg_matches/n_neg
                    recalls[top_size]['neg'].append(neg_recall)
                    if verbose:
                        printto(f'NEG RECALL AT TOP {max_top_size}:',neg_recall, out=self.output)
                if n_pos > 0: 
                    pos_recall = n_pos_matches/n_pos
                    recalls[top_size]['pos'].append(pos_recall)
                    if verbose:
                        printto(f'POS RECALL AT TOP {max_top_size}:',pos_recall, out=self.output)
                if verbose:
                    printto('', out=self.output)
        means = {}    
        for top_size in top_sizes:
            pos = recalls[top_size]['pos']
            neg = recalls[top_size]['neg']
            pos = 0 if len(pos) == 0 else np.mean(pos)
            neg = 0 if len(neg) == 0 else np.mean(neg)
            means[top_size] = {'pos':pos, 'neg':neg}
            printto('TOP SIZE:',top_size, out=self.output)
            printto('Recall for positive predictions:',pos, out=self.output)
            printto('Recall for negative predictions:',neg, out=self.output)
        return means
    
    def write_and_show_results(self, results, lang='en'):
        tables, footers = [], []
        for i,result in enumerate(results):
            now = datetime.now()
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
