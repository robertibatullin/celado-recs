#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:40:57 2019

@author: robert
"""
from flask import Flask, render_template, request, url_for, session
from recslib import *
import os, webbrowser
from threading import Timer

app = Flask(__name__)
app.secret_key = 'sk'

def check_password(dealer, password, base):
    with open(base,'r',encoding='utf-8') as f:
        rows = f.readlines()
    passwords = {row.split(',')[0]:row.split(',')[1].strip() \
                       for row in rows[1:]}
    if not dealer in passwords:
        return 'No such dealer'
    if password != passwords[dealer]:
        return 'Incorrect password'
    return 'OK'

@app.route('/')
def index():
    return render_template('login.html',
                           message = '')

@app.route('/login', methods=['POST','GET'])
def login():
    dealer = request.form['dealer']
    password = request.form['password']
    password_check_msg = check_password(dealer, password, 
                                        os.path.join('model', 'users.csv'))
    if password_check_msg == 'OK':
        session['dealer']  = dealer         
        session['lang'] = request.form['lang']
        return render_template('input.html')
    else:
        return render_template('login.html', 
                               message = password_check_msg)
    
@app.route('/input', methods=['POST','GET'])
def input_():
    return render_template('input.html')

@app.route('/input_customer', methods=['POST','GET'])
def input_customer():
    return render_template('input_customer.html')

@app.route('/input_offer', methods=['POST','GET'])
def input_offer():
    return render_template('input_offer.html')

@app.route('/input_retrain', methods=['POST','GET'])
def input_retrain():
    return render_template('input_retrain.html')

@app.route('/show_offers', methods=['POST'])
def show_offers():
    num = int(request.form['num'])
    if request.form['maxmin'] == 'min':
        num = -num
    if request.form['cnum'] != '':
        mode = 'read'
    elif 'input_table' in request.files:
        mode = 'predict'
    else:
        return render_template('notfound.html', 
                           message = {'en':'No data input.',
                                      'ru':'Данные не введены'}[session['lang']])
    ds = RecommenderSystem(dealer = session['dealer'], 
                           mode=mode)
    if mode == 'predict':
        file = request.files['input_table']
        file_type = file.filename.split('.')[-1]
        temp_filename = os.path.join(os.getcwd(), 'temp.'+file_type)
        file.save(temp_filename)
        raw_df = read_table(temp_filename)
        print(raw_df)
        results = ds.predict(raw_df, num)
        os.remove(temp_filename)
    elif mode == 'read':
        if request.form['cnum'] != '':
            cnum = 'C'+request.form['cnum']
            result = ds.get_top_from_table(cnum, 'offers', num, True)
            if result is None:
                return render_template('notfound.html', 
                                   message = {'en':'Customer not found in the database.',
                                              'ru':'Клиент не найден в базе'}[session['lang']])
            else:
                results = [result]
    tables, footers = write_and_show_results(results, session['lang'])
    result = {footers[i]:tables[i] for i in range(len(tables))}
    return render_template('show.html', 
                           header = 'Propensity to buy',
                           result = result)

@app.route('/show_customers', methods=['POST'])
def show_customers():
    num = int(request.form['num'])
    if request.form['maxmin'] == 'min':
        num = -num
    if request.form['mname'] == '':
        return render_template('notfound.html', 
                           message = {'en':'No data input.',
                                      'ru':'Данные не введены'}[session['lang']])
    mname = request.form['mname']
    if mname != 'nothing':
        mname += ':'+request.form['new_used']
    ds = RecommenderSystem(dealer = session['dealer'], 
                           mode='read')
    result = ds.get_top_from_table(mname, 'customers', num, True)
    if result is None:
        return render_template('notfound.html', 
                           message = {'en':'Model not found in the database.',
                                      'ru':'Модель не найдена в базе'}[session['lang']])
    tables, footers = write_and_show_results([result], session['lang'])
    result = {footers[i]:tables[i] for i in range(len(tables))}
    return render_template('show.html', 
                           header = mname,
                           result = result)
    
@app.route('/show_retrain', methods=['POST'])
def show_retrain():
    if not 'retrain_table' in request.files:
        return render_template('notfound.html', 
                           message = {'en':'No data input.',
                                      'ru':'Данные не введены'}[session['lang']])
    ds = RecommenderSystem(dealer = session['dealer'], 
                           mode = 'create')
    file = request.files['retrain_table']
    file_type = file.filename.split('.')[-1]
    temp_filename = os.path.join(os.getcwd(), file.filename)
    file.save(temp_filename)
    ds, result = ds.retrain(temp_filename)
    os.remove(temp_filename)
    return render_template('show_r.html', 
                           header = {'en':'Retrain result',
                                     'ru':'Результат дообучения'}[session['lang']],
                           result = result)
    
@app.route('/sample_data', methods=['GET'])
def sample_data():
    return render_template('sample_data.html')

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

def run():
    Timer(1, open_browser).start()
    app.run()

if __name__ == "__main__":
    run()

