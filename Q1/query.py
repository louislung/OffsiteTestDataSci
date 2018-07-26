#!/usr/bin/python


##################
# Import modules #
##################
import csv
import psycopg2
import os
from pathlib import Path


####################
# Define parameter #
####################
hostname = 'postgresql-test.crcvhvacob77.us-west-2.rds.amazonaws.com'
username = 'louis'
password = 'louis234'
database = 'postgres'
script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
ddl_dir = script_dir / 'ddl'
data_dir = script_dir / 'data'
table = 'piwik_track'
date_install = '2017-04-01'
date_from = '2017-04-02'
date_to = '2017-04-08'

################
# Main Program #
################
def doQuery(conn, sql, bind_value = ()) :
    cur = conn.cursor()
    if type(sql) == str:
        cur.execute(sql, bind_value)
    else:
        cur.execute(open(sql, "r").read(), bind_value)
    cur.close()


def doQuerySelect(conn, sql) :
    cur = conn.cursor()
    cur.execute(sql)
    result = cur.fetchone()
    cur.close()
    return result[0]

if __name__ == '__main__':
    print('[INFO] Connecting to postgresSQL DB')
    myConnection = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
    print('[INFO] Connect success')
    """
    #Drop table
    sql_droptable ='DROP TABLE IF EXISTS {0}'.format(table)
    print('[INFO] Drop table')
    doQuery(myConnection, sql_droptable)
    print('[INFO] Drop table success')

    #Create table
    print('[INFO] Create table')
    doQuery(myConnection, ddl_dir / (table + '.sql'))
    print('[INFO] Create table success')

    #Insert data
    print('[INFO] Insert data')
    sql_insert = 'INSERT INTO {0} (time, uid, event_name, source_ip) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;'.format(table)
    with open(data_dir / (table + '.csv')) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            doQuery(myConnection, sql_insert, (row))
    print('[INFO] Insert data success')
    """
    #Select results
    sql_select = "SELECT COUNT(DISTINCT uid) USER_CNT FROM piwik_track " \
                 "WHERE EVENT_NAME = 'FIRST_INSTALL' " \
                 "AND time = date '{0}' " \
                 "AND uid IN (SELECT uid FROM piwik_track WHERE time >= date '{1}' and time <= date '{2}');".format(date_install,date_from,date_to)
    user_cnt = doQuerySelect(myConnection, sql_select)
    print('Answers:')
    print('{0} user(s) who install the app (i.e. with FIRST_INSTALL event) on {1} and use our app at least once (i.e. with any event) between {2} and {3}.'.format(user_cnt,date_install,date_from,date_to))

    myConnection.commit()
    myConnection.close()