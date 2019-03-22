# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:01:58 2017

@author: hadooop
"""

import pymysql

def update(host,user,password,db,sql):
    conn = pymysql.connect(host,user,password,db)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        conn.commit()
    except:
        conn.rollback()
    finally:
        conn.close()
    return 

def search(host,user,password,db,sql):
    conn = pymysql.connect(host,user,password,db)
    cursor = conn.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    conn.close()
    
    return result

def delete(host,user,password,db,sql):
    conn = pymysql.connect(host,user,password,db)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        conn.commit()
    except:
        conn.rollback()
    finally:
        conn.close()
    return 

def insert(host,user,password,db,sql):
    conn = pymysql.connect(host,user,password,db)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        conn.commit()
    except:
        conn.rollback()
    finally:
        conn.close()
    return 