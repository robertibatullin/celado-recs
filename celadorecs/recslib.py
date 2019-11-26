#!/usr/bin/env python
# coding: utf-8

# # Библиотека общих функций для анализа данных Caterpillar

import os
from shutil import copyfileobj
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

now = datetime.now()
current_year = now.year
current_month = now.month

telemetry_columns = ['Existing: Fleet age',
                     'Existing: Моточасы 2017-2018\n(в работе)',
                  'Existing: Моточасы 2017-2018\n(всего)',
                  'Existing: Моточасы 2017-2018\n(холостой ход)',
                  'Existing: Общая наработка',
                  'Existing: Расход топлива 2017-2018\n(в работе)',
                  'Existing: Расход топлива 2017-2018\n(всего)',
                  'Existing: Расход топлива 2017-2018\n(холостой ход)']
fleet_columns = ['Customer: Physical Region',
                 'Existing: New (0) / Used (1)',
                 'Existing: Product Family',
                 'Existing: Product Group',
                 'Existing: Manufacturer',
                 'Existing: Model',
                 'Existing: Quantity',
                 'Customer Number', 
                 'Customer Name',
                 'Customer: OKVED', 
                 'Customer: Company Age',
                 'Customer: Гарантия',
                 'Customer: Наличие б.у',
                 'Customer: Наличие продления гарантии EPP',
                 'Customer: Продажи техники в регионе (прогноз 2019)'
                 ] #+ telemetry_columns
customers_columns = ['Customer Number',
                     'Customer Name',
                     'INN',
                     'Customer: Physical Region',
                     'Customer: OKVED',
                     'Customer: Company Age',
#                     'Offer: New (0) / Used (1)',
                     'Offer: Product Family',
                     'Offer: Product Group',
                     'Offer: Manufacturer',
                     'Offer: Model',
                     'Offer: Month',
                     'Offer: Quantity', 'Offer: Won/Lost Years Ago',
                     'Lost (0)/Won (1)',
                     'Opportunity number',
                     'Customer: Выручка', 'Customer: Основные средства', 
                     'Customer: Покупки техники', 'Customer: Чистая прибыль (убыток)',
                     'Customer: ЧислСотруд', 'Customer: НедобросовПостав',
                     'Customer: Продажи техники в регионе (прогноз 2019)'                     
                     ]
common_customer_columns = list(set(fleet_columns).intersection(customers_columns))
id_vars = ['Customer Number', 'Customer Name', 
           'Opportunity number', 'INN']
vars_to_train = ['Customer: Physical Region',
                 'Customer: OKVED',
                 'Customer: Company Age',
                 'Customer: Гарантия',
                 'Customer: Наличие б.у',
                 'Customer: Наличие продления гарантии EPP',                
                 'Existing: New (0) / Used (1)',
#                 'Existing: Product Family',
#                 'Existing: Product Group',
#                 'Existing: Manufacturer',
                 'Existing: Model',
                 'Existing: Quantity',
#                 'Offer: New (0) / Used (1)',
#                 'Offer: Product Family',
#                 'Offer: Product Group',
#                 'Offer: Manufacturer',
#                 'Offer: Model',
                 'Offer: Won/Lost Years Ago',
                 'Offer: Month',
#                 'Offer: Quantity', 
#                 'Lost (0)/Won (1)'
                 'Sold model',
                 'Customer: Продажи техники в регионе (прогноз 2019)'
                 'Customer: Выручка', 'Customer: Основные средства', 
                 'Customer: Покупки техники', 'Customer: Чистая прибыль (убыток)',
                 'Customer: ЧислСотруд', 'Customer: НедобросовПостав'
                 ]

class RecommenderSystem():
    
    def __init__(self,
                 source_filename = None,
                 dealer = 'dealer1',
                 version = 'last',
                 target_var = 'Offer: Model',
                 mode = 'read',#explore|train|predict|precompute|read (from precomputed predictions) 
                 min_samples_in_class = 1,
                 test_size = 0,
                 **kwargs): 
        dealer_dir = os.path.join('model',
                                  'dealer_data',
                                  dealer)
        if version == 'last':
            versions = [dirname for dirname in os.listdir(dealer_dir) \
                        if dirname != 'temp' and \
                        os.path.isdir(os.path.join(dealer_dir,dirname))]
            self.version = sorted(versions)[-1]
        else:
            self.version = version
        print('Initializing recommender system for dealer "{0}" (version {2}) for {1} mode'.format(
                dealer, mode, self.version))
        folder = os.path.join(dealer_dir,
                              self.version)
        self.dealer = dealer
        self.folder = folder
        self.target_var = target_var
        self.clf_name = 'rf'
        self.clf_filename = os.path.join(self.folder, 
                                         self.clf_name+'.pkl')
        self.mode = mode
        self.threshold = min_samples_in_class
        self.test_size = test_size

        if not os.path.exists(folder):
            os.mkdir(folder)
        #формируем файл data - комбинации fleet*offer
        #сохраняем этот и вспомогательные файлы
        if os.path.exists(os.path.join(folder, 'data_for_train.csv'))\
        and os.path.exists(os.path.join(folder, 'data_for_predict.csv')):
            data = pd.read_csv(os.path.join(folder, 'data_for_predict.csv'))
        else:
            #preprocess df_fleet
            if os.path.exists(os.path.join(folder, 'fleet.csv')):
                df_fleet = pd.read_csv(os.path.join(folder, 'fleet.csv'))
                df_fleet = df_fleet[[col for col in fleet_columns if col in df_fleet.columns]]
            else:
                df = pd.read_excel(source_filename, sheet_name = 'Fleet Data')
                df_fleet = preprocess_fleet_data(df)
                df_fleet.to_csv(os.path.join(folder, 'fleet.csv'), index=False)
                del df
            #preprocess df_opp
            if os.path.exists(os.path.join(folder, 'opportunity.csv')):
                df_opp = pd.read_csv(os.path.join(folder, 'opportunity.csv'))
                df_opp = df_opp[[col for col in customers_columns if col in df_opp.columns]]
            else:
                df = pd.read_excel(source_filename, sheet_name = 'Customers Data')
                df_opp = preprocess_opportunity_data(df)
                df_opp.to_csv(os.path.join(folder, 'opportunity.csv'), index=False)
                del df
                
            #opportunity: удаляем покупателей, по которым нет данных, дубликаты и лишние столбцы
            df_opp.dropna(inplace=True)
            df_opp.drop_duplicates(inplace=True)
#            df_opp['Sold Quantity'] = df_opp['Offer: Quantity']*df_opp['Lost (0)/Won (1)']
            df_opp.drop('Offer: Quantity',axis=1,inplace=True)
        
            df_opp['Sold model'] = df_opp[target_var].where(df_opp['Lost (0)/Won (1)']==1).copy()
            df_opp['Sold model'].fillna(df_opp['Lost (0)/Won (1)'], inplace=True)
            df_opp['Sold model'].replace({0:'nothing'}, inplace=True)
            df_opp['Sold model'] = df_opp['Sold model'].astype('category')

            #Fleet: убираем дубликаты и лишние столбцы
            df_fleet.drop_duplicates(inplace=True)
            
            #Объединяем таблицы
            
            common_columns = [col for col in common_customer_columns if col in df_fleet.columns and col in df_opp.columns]

            data = df_fleet.merge(df_opp, how='outer', on=common_columns)   
            del df_fleet, df_opp
            
            data.drop_duplicates(inplace=True)
            data.fillna({'Existing: Quantity':0,
                  'Customer: Гарантия':0,
                  'Customer: Сервисный контакт уровень 1':0,
                  'Customer: Наличие б.у':0,
                  'Customer: Наличие продления гарантии EPP':0,
                  'Existing: New (0) / Used (1)':'no',
                  'Existing: Product Family':'no',
                  'Existing: Product Group':'no',
                  'Existing: Manufacturer':'no',
                  'Existing: Model':'no'}, inplace=True)    

            data.sort_index(axis=1,inplace=True)

        #    data['Customer: OKVED'] = data['Customer: OKVED'].apply(str).str[:2]
            false_float_cols = [col for col in data.select_dtypes(include=['float']).columns
                                if not col in telemetry_columns]
            for col in false_float_cols:
                data[col].fillna(-1, inplace=True)
                data[col] = data[col].apply(int)
            data[false_float_cols].replace({-1:None}, inplace=True)
            
            data.to_csv(os.path.join(folder, 'data_for_predict.csv'), index=False)

            data = data.dropna()            
            data.to_csv(os.path.join(folder, 'data_for_train.csv'), index=False)
                  
        #переменные, используемые в обучении модели
        if not mode in ('train', 'create'): #читать Xcols
            if os.path.exists(os.path.join(folder, 'model_cols.txt')):
                with open(os.path.join(folder, 'model_cols.txt'), 'r', encoding='utf-8') as f:
                    model_cols = f.readlines()
                self.Xcols = [mc.strip() for mc in model_cols]
            else:
                raise FileNotFoundError('File "model_cols.txt" not found. Train model first')
                
               
#        формируем вспомогательные файлы
        if not os.path.exists(os.path.join(folder, 'offers_text.csv')):
            offer_data = data.filter(regex='Offer: |Opportunity number').drop_duplicates().dropna()
            offer_data.to_csv(os.path.join(folder, 'offers_text.csv'),index=False)
        
        if not os.path.exists(os.path.join(folder, 'customers_text.csv')):
            customer_data = data.filter(regex='Customer: |Existing: |Customer Number|Customer Name').drop_duplicates()
            customer_data.to_csv(os.path.join(folder, 'customers_text.csv'),index=False)
    #    в кодированном виде
        if not os.path.exists(os.path.join(folder, 'offers.csv')) or not os.path.exists(os.path.join(folder, 'customers.csv')):
            norm_data = normalize_encode(data, target_var)
            if not os.path.exists(os.path.join(folder, 'offers.csv')):
                offer_data = norm_data.filter(regex='Offer: |Opportunity number').drop_duplicates().dropna()
                offer_data.to_csv(os.path.join(folder, 'offers.csv'),index=False)
            
            if not os.path.exists(os.path.join(folder, 'customers.csv')):
                customer_data = norm_data.filter(regex='Customer: |Existing: |Customer Number|Customer Name').drop_duplicates()
                customer_data = group_by_id(customer_data, ['Customer Name','Customer Number'])
                customer_data.to_csv(os.path.join(folder, 'customers.csv'),index=False)

        print('RecommenderSystem initialized')
                
        if mode in ('predict','precompute')\
        and os.path.exists(self.clf_filename):
            print('Loading pkl')
            print(self.clf_filename)
            self.clf = pickle.load(open(self.clf_filename,'rb'))
            print('Loaded')
                                    
        if mode in ('explore', 'train'):                
            self.train(**kwargs)
            
        if mode in ('train', 'precompute'):
            self.precompute()
                        
    def set_mode(self, new_mode):
        self.mode = new_mode
                        
    def unite(self, other, version):
        if self.dealer != other.dealer:
            return None
        print('Uniting RecommenderSystem versions {0} and {1}'.format(self.version, other.version))
        folder = os.path.join('model',
                      'dealer_data',
                      self.dealer,
                      version)
        if not os.path.exists(folder):
            os.mkdir(folder)
        for wd in ['opportunity.csv', 'fleet.csv']:
            f1 = pd.read_csv(os.path.join(self.folder, wd))
            f2 = pd.read_csv(os.path.join(other.folder, wd))
            f = pd.concat([f1,f2], join='outer', ignore_index=True)
            f.to_csv(os.path.join(folder, wd), index=False)
        return RecommenderSystem(dealer = self.dealer,
                                 version = version, 
                                 mode='create',
                                 test_size = self.test_size)
    
    def get_data(self, kind='customer', **ids):
        print('Getting data')
        if kind == 'customer':
            if 'number' in ids:
                feature, id_ = 'Customer Number', 'number'
            elif 'name' in ids:
                feature, id_ = 'Customer Name', 'name'
        elif kind == 'offer':
            if 'number' in ids:
                feature, id_ = 'Opportunity number', 'number'
            elif 'name' in ids:
                feature, id_ = self.target_var, 'name'
        df = pd.read_csv(os.path.join(self.folder,kind+'s_text.csv'))
        data = df[df[feature]==ids[id_]].copy()
        data.drop([col for col in data.columns if not col in vars_to_train+id_vars],
                  axis=1, inplace=True)
        data.drop_duplicates(inplace=True)
        print('Got data')
        return data
    
    def get_mean_normalized_data(self, kind='customer', **ids):
        if kind == 'customer':
            if 'number' in ids:
                feature, id_ = 'Customer Number', 'number'
            elif 'name' in ids:
                feature, id_ = 'Customer Name', 'name'
        elif kind == 'offer':
            if 'number' in ids:
                feature, id_ = 'Opportunity number', 'number'
            elif 'name' in ids:
                feature, id_ = self.target_var, 'name'
        df = pd.read_csv(os.path.join(self.folder,kind+'s.csv'))
        data = df[df[feature]==ids[id_]].copy()
        if len(data) < 2:
            return data
        else:
#            means = data.select_dtypes(exclude=['object']).mean().to_frame().T
#            means.sort_index(axis=1, inplace=True)
            means = group_by_id(data, [feature])
            means[feature] = id_
            return means
        
    def get_real_sells(self, **customer_ids):
        df_customers = pd.read_csv(os.path.join(self.folder,'opportunity.csv'))
        df_customers['Sold Quantity'] = df_customers['Offer: Quantity']*df_customers['Lost (0)/Won (1)']
        df_customers.drop(['Lost (0)/Won (1)','Offer: Quantity'],axis=1,inplace=True)
        if 'customer_number' in customer_ids:
            real_sells = df_customers[df_customers['Customer Number']==customer_ids['customer_number']].copy()
        elif 'customer_name' in customer_ids:
            real_sells = df_customers[df_customers['Customer Name']==customer_ids['customer_name']].copy() 
        real_sells.drop_duplicates(inplace=True)
        real_sells.sort_index(axis=1,inplace=True)
        return real_sells

    def is_really_sold(self, customer_number, offer):
        df_customers = pd.read_csv(os.path.join(self.folder,'opportunity.csv'))
        if offer == 'nothing':
            rs = df_customers[(df_customers['Customer Number']==customer_number) &\
                              (df_customers['Lost (0)/Won (1)'])]
            res = 'yes' if len(rs)==0 else 'no'
        else:
            rs = df_customers[(df_customers['Customer Number']==customer_number) &\
                              (df_customers['Offer: Model']==offer) &\
                              (df_customers['Lost (0)/Won (1)'])]
            res = 'yes' if len(rs)>0 else 'no'
        return res
    
    def compare_real_sells_and_prediction(self, customer_number_list):
        for i,cn in enumerate(customer_number_list):
            df1 = self.get_mean_normalized_data('customer', number=cn)
            rs = self.get_real_sells(cn)
            df2 = pd.DataFrame([df1.join(self.get_mean_normalized_data('offer', number=ofn)).iloc[0] for ofn in rs['Opportunity number']])
            if i==0:
                df = df2
                result = rs
            else:
                df = df.append(df2)
                result = result.append(rs)
        result['Predicted success probability'] = self.predict_proba_for_customer_and_offer(df)
        result.drop('Opportunity number', axis=1, inplace=True)
        result.drop_duplicates(inplace=True)
        return result

    def get_xy(self, source,
               use_y,
               threshold=None,
               verbose = True): #for training, false for precomputing
                    
        data = pd.read_csv(source)
        data.drop([col for col in data.columns if not col in vars_to_train+id_vars],
                  axis=1, inplace=True)

        data = normalize_encode(data.drop('Sold model', axis=1), 
                                self.target_var).join(data['Sold model'])
        data.drop(['Sold Quantity','Lost (0)/Won (1)'], axis=1, inplace=True, errors='ignore')
        
        data.drop('INN',axis=1,inplace=True)

        data = group_by_id(data, ['Customer Number', 'Customer Name', 'Sold model'])
        
        if self.threshold > 1:
            data = filter_small_classes(data, self.threshold)
            
        if use_y:
            soldmodel_encoded, soldmodel_unique = pd.factorize(data['Sold model'])
            with open(os.path.join(self.folder, 'model_names.csv'),'w') as f:
                f.write('\n'.join(soldmodel_unique))
            data['Sold model'] = soldmodel_encoded
            y = data['Sold model']

        Xind = data['Customer Number']
        X = data.drop(id_vars+['Sold model'], axis=1, errors='ignore')
        Xcols = X.columns

        del data
        if verbose:
            if threshold > 1:
                print('Filtered off classes with less than {} members'.format(threshold))
            if use_y:
                print (info('Full', X, y, Xind))
        if use_y:
            return X, y, Xcols, Xind
        else:
            return X, Xcols, Xind   
    
    def precompute(self):        
        print('Precomputing probabilities')
        with open(os.path.join(self.folder, 'model_names.csv'),'r', encoding='utf-8') as f:
            model_names = f.readlines()
        model_names = [mn.strip() for mn in model_names]

        X, _, cnums = self.get_xy(
                os.path.join(self.folder, 'data_for_predict.csv'),
                use_y = False,
                threshold = 1,
                verbose = False)
                
        X = X[[col for col in X.columns if col in self.Xcols]]
        
        data = self.clf.predict_proba(X)
        pp = pd.DataFrame(index = cnums,
                          columns = model_names,
                          data = data)
        pp['Customer Number'] = pp.index
        pp.drop_duplicates(inplace=True)
        pp = pp.groupby('Customer Number').mean().reset_index() #? группировать по индексу
        pp.to_csv(os.path.join(self.folder, 'probs.csv'), 
                  sep=';',
                  index=False)
        print('Precomputed probabilities saved to {}/probs.csv.'.format(self.folder))

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
            
        X, y, self.Xcols, Xind = self.get_xy(
                os.path.join(self.folder, 'data_for_train.csv'),
                use_y = True,
                threshold = self.threshold)
        
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
            
    def retrain(self, new_data_file):
        time = datetime.now().strftime('%Y-%m-%d %H:%M')
        new = RecommenderSystem(new_data_file,
                                dealer = self.dealer,
                                version = 'temp',
                                mode = 'create')
        version_ = self.version.split('.')
        new_version_ = [version_[0], str(int(version_[1])+1)]
        new_version = '.'.join(new_version_)
        updated = self.unite(new, new_version)

        for filename in os.listdir(new.folder):
            os.remove(os.path.join(new.folder, filename))
        os.removedirs(new.folder)

        updated.set_mode('train')
        updated.train()
        updated.precompute()
        log_str = 'Recommender system updated from version {2} to {0} with new data from "{1}"'.format(
                new_version,
                os.path.split(new_data_file)[-1],
                self.version)
        print(time, log_str, file = open(os.path.join('model',
                                     'dealer_data',
                                     self.dealer,
                                     'update.log'),'a'))
        return updated, log_str
                
    def predict_proba_for_customer(self, customer_data):
        #на входе dataframe customer кодированная и нормированная
        #на выходе вероятность успеха
        cdata = customer_data.copy()
        for month in range(1,13):
            cdata['Offer: Month_'+str(month)] = (month == current_month)
        cdata['Offer: Won/Lost Years Ago'] = 0
        extra_cols = list(set(cdata.columns)-set(self.Xcols))
        absent_cols = list(set(self.Xcols)-set(cdata.columns))
        if extra_cols != []:
            cdata.drop(extra_cols,axis=1,inplace=True)
        if absent_cols != []:
            cdata = cdata.join(pd.DataFrame(columns=absent_cols))
        cdata.sort_index(axis=1,inplace=True)
        cdata.drop(id_vars+['Sold Quantity',
                             'Lost (0)/Won (1)',
                             self.target_var,
                             'Sold model'], axis=1, 
                             errors='ignore',inplace=True)
        cdata.fillna(0,inplace=True)
        return self.clf.predict_proba(cdata)
        
    def get_top_offers(self, customer_data, num_offers):
        with open(os.path.join(self.folder, 'model_names.csv'),'r', encoding='utf-8') as f:
            model_names = f.readlines()
        model_names = [mn.strip() for mn in model_names]
        pp = self.predict_proba_for_customer(customer_data)
        assert pp.shape[1] == len(model_names)
        dct = {model_names[i]:pp[:,i] for i in range(pp.shape[1])}
        result = pd.DataFrame(dct)
        result.sort_values(0, axis=1, ascending=(num_offers<0), inplace=True)
        result = result[result.columns[:np.abs(num_offers)]]
        return result
    
    def get_top_from_table(self, input_id, #customer name or model name
                           find, #customers|offers
                           num_top,
                           show_really_sold = False):
        print('Getting top from table')
        df = pd.read_csv(os.path.join(self.folder, 'probs.csv'),
                         sep=';', index_col=0)
        try:
            if find == 'offers':
                series = df.loc[input_id].copy()
            elif find == 'customers':
                series = df[input_id].copy()
        except KeyError or IndexError:
            return None
        series.sort_values(ascending=(num_top < 0), inplace=True)
        series = series.head(np.abs(num_top))
        df = pd.DataFrame(series).reset_index()
        if find == 'offers':
            df.rename(columns = {'index':'Model',
                                 df.columns[1]:'Propensity to buy'}, inplace=True)
            if show_really_sold:
                df['Really sold?'] = df['Model'].apply(lambda offer:self.is_really_sold(input_id, offer))
        elif find == 'customers':
            df.rename(columns = {'index':'Customer',
                                 df.columns[1]:'Propensity to buy'}, inplace=True)
            if show_really_sold:
                df['Really sold?'] = df['Customer Number'].apply(lambda customer:self.is_really_sold(customer, input_id))                
        return df
   
    def predict_preprocessed(self, preproc_df, num_top):
        if len(preproc_df) == 0:
            return None
        df = normalize_encode(preproc_df, 'Sold model')
        df = group_by_id(df, ['Customer Name','Customer Number'])
        result_dfs = []
        for cnum in df['Customer Number']:
            result_df = self.get_top_offers(df[df['Customer Number']==cnum], 
                                            num_top).T.reset_index()
            result_df.rename(columns={'index':'Model',
                                      0:cnum},
                        inplace=True)
            result_dfs.append(result_df)
        return result_dfs
        
    def predict(self, raw_df, num_top):
        if len(raw_df) == 0:
            return None
        df = preprocess_fleet_data(raw_df)
        return self.predict_preprocessed(df, num_top)
    
    def get_customer_name(self, customer_number):
        path = os.path.join(self.folder, 'customers_text.csv')
        df = pd.read_csv(path)
        cnames = df[df['Customer Number']==customer_number]['Customer Name']
        if len(cnames)==0:
            return None
        else:
            return cnames.iloc[0]
        
    def recall_at_k(self, Ks, n_samples=None):
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
            real_sells = self.get_real_sells(customer_number = cnum)
            real_sells.fillna(0, inplace=True)
            real_sells = real_sells[real_sells['Sold Quantity']>0][self.target_var]
            if len(real_sells) == 0:
                real_sells = ['nothing']
            else:
                real_sells = set(real_sells)
            maxtop = probs.loc[cnum].sort_values(ascending=False).head(max_top_size)
            for top_size in top_sizes:
                top = maxtop.head(top_size)
                top_ = top
                top = list(top.index)
                n_matches = len([s for s in top if s in real_sells])
                recall = n_matches*1./len(real_sells)
                if real_sells == ['nothing']:
                    recalls[top_size]['neg'].append(recall)
                else:
                    recalls[top_size]['pos'].append(recall)
#                    if recall == 1 and len(real_sells)>1:
#                        print(cnum) 
#                        print(real_sells)
#                        print(top_)
        means = {}    
        for top_size in top_sizes:
            means[top_size] = {'pos':np.mean(recalls[top_size]['pos']),
                               'neg':np.mean(recalls[top_size]['neg'])}        
        return means
        
def info(name, X, y, Xind):
    s = '{} dataset\n'.format(name)
    s += '\t{} samples\n'.format(X.shape[0])
    s += '\t{} features\n'.format(X.shape[1])
    s += '\t{} classes\n'.format(len(pd.unique(y)))
    s += '\t{} unique sample IDs'.format(len(pd.unique(Xind)))
    return s

def filter_small_classes(data, threshold):
    models = pd.value_counts(data['Sold model'])
    rare_models = models[models < threshold].index
    filtered = data[~data['Sold model'].isin(rare_models)]
    return filtered
    
def read_table(source):
    if source.split('.')[-1].lower() == 'xlsx':
        return pd.read_excel(source)
    elif source.split('.')[-1].lower() == 'csv':
        return pd.read_csv(source)
    else:
        raise TypeError('Input file must be .xlsx or .csv')

def write_and_show_results(results, lang='en'):
    tables, footers = [], []
    for i,result in enumerate(results):
        filename = os.path.join('results',
                                now.strftime("%Y-%m-%d_%H-%M")+'_'+str(i)+'.xlsx')
        result.to_excel(filename, index=False)
        tables.append(  result.to_html(index=False) )
        if lang == 'ru':
            footers.append( 'Результат сохранён в файл {}'.format(filename) )
        elif lang == 'en':
            footers.append( 'Results saved to the file {}'.format(filename) )
    return tables, footers
  
def group_by_id(data, ids):
    #группируем все данные одного ids в 1 строку
    #Existing переменные суммируем (bool), Customer + Offer усредняем (num)
    to_sum = [col for col in data.columns if 'Existing:' in col]
    to_mean = [col for col in data.columns if 'Customer:' in col or 'Offer:' in col]
    aggdict = dict(zip(to_sum, [np.sum]*len(to_sum)))
    aggdict.update(dict(zip(to_mean, [np.mean]*len(to_mean))))
    ids_ = [id_ for id_ in ids if id_ in data.columns]
    result = data.fillna(0).groupby(ids_)[to_mean+to_sum]
    result = result.aggregate(aggdict)
    result.reset_index(inplace=True)
    result.sort_index(axis=1, inplace=True)
    return result

def normalize_encode(data, target_var):
    #переводим текстовые переменные в категориальные
    #кодируем one-hot
    #float -> int
    #сливаем и сортируем столбцы по алфавиту
    #
    data_id_vars = list(set(data.columns).intersection(set(id_vars)))
    if data_id_vars != []:
        id_cols = data[data_id_vars]
        cat_cols = data.drop(data_id_vars,axis=1).select_dtypes(include=['object'])
    else:
        id_cols = None
        cat_cols = data.select_dtypes(include=['object'])
    num_cols = data.select_dtypes(exclude=['object'])
    num_cols.drop(id_vars, axis=1, inplace=True, errors='ignore')
    try:
        num_cols = num_cols.drop('Offer: Month',axis=1)
        cat_cols = cat_cols.join(data['Offer: Month'])
    except KeyError:
        pass
    cat_cols.drop(target_var, axis=1, inplace=True, errors='ignore')
    
    cat_cols = cat_cols.astype('category')    
    one_hot = pd.get_dummies(cat_cols)
    one_hot.drop(['Existing: New (0) / Used (1)_no',
                  'Existing: New (0) / Used (1)_0.0',
                  'Existing: Product Family_no',
                  'Existing: Product Group_no',
                  'Existing: Manufacturer_no',
                  'Existing: Model_no'], axis=1, inplace=True, errors='ignore') 
    data1 = num_cols.join(one_hot)
    if id_cols is not None:
        data1 = data1.join(id_cols)

    data1.sort_index(axis=1,inplace=True)
    return data1

def get_year_month(date_col):
    date_col_ = pd.to_datetime(date_col,
                               errors = 'coerce',
                               infer_datetime_format=True)
    year = date_col_.dt.year
    month = date_col_.dt.month
    return year, month            
        
def preprocess_fleet_data(raw_df):
    df = raw_df.copy()
    try:
        df['Existing: Fleet age'] = current_year - df['Возраст машины']
    except KeyError:
        df['Existing: Fleet age'] = 0
    df.rename(columns = {col:'Existing: '+col for col in ['Machine Business Division',
                                                      'Product Family', 'Product Group',
                                                      'Manufacturer', 'Model',
                                                      'Моточасы 2017-2018\n(в работе)',
                                                      'Моточасы 2017-2018\n(всего)',
                                                      'Моточасы 2017-2018\n(холостой ход)',
                                                      'Общая наработка',
                                                      'Расход топлива 2017-2018\n(в работе)',
                                                      'Расход топлива 2017-2018\n(всего)',
                                                      'Расход топлива 2017-2018\n(холостой ход)']},
         inplace=True)
    df.rename(columns = {'New / Used':'Existing: New (0) / Used (1)'},
             inplace=True)
    df.rename(columns = {col:'Customer: '+col for col in ['% запчастей купленных у оф. Дилера',
                                                          '% работ сделанных у оф. Дилера',
                                                          'Physical Region', 'OKVED','Гарантия',
                                                          'PWC','Местонахождения склада',                                                      
                                                          'Сервисный контакт уровень 1', 
                                                          'Наличие б.у', 'Наличие продления гарантии EPP']},
             inplace=True)
    df.rename(columns = {'Customer Name ':'Customer Name'},
             inplace=True)
    df['Existing: Quantity'] = 1
    cdc_year, _ = get_year_month(df['Company Date Create'])
    df['Customer: Company Age'] = current_year-cdc_year
    for col in ['Customer: Наличие продления гарантии EPP',
           'Customer: Наличие б.у',
           'Customer: Гарантия']:
        if col in df.columns:
            df[col] = df[col].apply({'No':0, 'Yes':1}.get)
    df['Existing: New (0) / Used (1)'] =\
    df['Existing: New (0) / Used (1)'].apply({'New machines':0,
       'Used machines':1, 'New':0, 'Used':1}.get)
    try:
        df['Customer Number']
    except KeyError:
        df['Customer Number'] = 'C'
    df = df[[col for col in fleet_columns if col in df.columns]]
    df.drop_duplicates(inplace=True)
    return df

def preprocess_opportunity_data(raw_df):
    df = raw_df.copy()
    df.rename(columns = {col:'Offer: '+col for col in ['Machine Business Division',
                                                      'Product Family', 'Product Group',
                                                      'Manufacturer', 'Model', 'Quantity']},
         inplace=True)
    df.rename(columns = {col:'Customer: '+col for col in ['Physical Region', 'OKVED']},
             inplace=True)
    df['Offer: Model'] = df['Offer: Model'] + ':'+ df['New / Used']
    df.drop('New / Used', axis=1, inplace=True)
    df.rename(columns = {'Customer Name ':'Customer Name',
                         'Opportunity':'Opportunity number'},
             inplace=True)
    wld_year, wld_month = get_year_month(df['Won/Lost Date'])
    df['Offer: Won/Lost Years Ago'] = current_year-wld_year
    df['Offer: Month'] = wld_month
    if 'Company Date Create' in df.columns:
        cdc_year, _ = get_year_month(df['Company Date Create'])
        df['Customer: Company Age'] = current_year-cdc_year
    df['Lost (0)/Won (1)'] = df['Opportunity Status'].apply({'Lost':0,
                                                         'Won':1,
                                                        'No Deals':None}.get)
    try:
        df['Opportunity number']
    except KeyError:
        df['Opportunity number'] = 'OPP'
    df = df[[col for col in customers_columns if col in df.columns]]
    df.drop_duplicates(inplace=True)
    return df

def preprocess_offer_data(raw_df):
    df = raw_df.copy()
    df.rename(columns = {col:'Offer: '+col for col in ['Machine Business Division',
                                                      'Product Family', 'Product Group',
                                                      'Manufacturer', 'Model', 'Quantity']},
         inplace=True)
    df.rename(columns = {'New / Used':'Offer: New (0) / Used (1)',
                        'Opportunity':'Opportunity number'},
             inplace=True)
    wld_year, wld_month = get_year_month(df['Won/Lost Date'])
    df['Offer: Won/Lost Years Ago'] = current_year-wld_year
    df['Offer: Month'] = wld_month
    df['Offer: New (0) / Used (1)'] = df['Offer: New (0) / Used (1)'].apply({'New':0,b'Used':1}.get)
    df = df[['Offer: Machine Business Division',
             'Offer: Product Family', 'Offer: Product Group',
             'Offer: Manufacturer', 'Offer: Model', 'Offer: Quantity',
             'Offer: Won/Lost Years Ago', 'Offer: Month', 'Offer: New (0) / Used (1)']]
    df['Opportunity number'] = 'OPP'
    df.drop_duplicates(inplace=True)
    return df

if __name__ == "__main__": 
   
    ds = RecommenderSystem(
            source_filename = '/home/robert/projects/celado/caterpillar/raw data/Data for Customers Clustering 19.02.2019.xlsx',
            dealer = 'dealer1',
            version = '1.0',
#            test_size = 0.25,
            mode = 'read')
    
    cn = 'C46785'#'C51514'
    top = ds.get_top_from_table(cn, 'offers', 10, True)
    print(top)

#    new_data_file = '/home/robert/projects/celado/caterpillar/release/sample_inputs/new1.xlsx'
#    ds, log_str = ds.retrain(new_data_file)
#    print(log_str)

#    top_sizes = [10]#,5,1]
#    recalls = ds.recall_at_k(top_sizes, 200)
#    for top_size in top_sizes:
#        print('At top-{}:'.format(top_size))
#        print('Mean recall for negative (sold nothing) samples: {:.4f}'.format(recalls[top_size]['neg']))
#        print('Mean recall for positive (sold something) samples: {:.4f}'.format(recalls[top_size]['pos']))
#        print()
