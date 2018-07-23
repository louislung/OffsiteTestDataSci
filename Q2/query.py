#!/usr/bin/python


##################
# Import modules #
##################
import os
from pathlib import Path


####################
# Define parameter #
####################
script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
data_dir = script_dir / 'data'
log_file = 'cdn.log'
date_from = '2017-08-24'
time_from = '00:00:00'
date_to = '2017-08-25'
time_to = '23:59:59'


################
# Main Program #
################
def get_data(input_filename, delimiter = b'\t'):
    with open(input_filename, 'r+b') as f:
        for record in f:                 # traverse sequentially through the file
            x = record.split(delimiter)  # parsing logic goes here (binary, text, JSON, markup, etc)
            yield x                      # emit a stream of things
            #  (e.g., words in the line of a text file,
            #   or fields in the row of a CSV file)



if __name__ == '__main__':
    print('main')
    log_generator = get_data(data_dir / log_file)
    next(log_generator) #skip header
    for row in log_generator:
        print(row)
        if str(row[0] == '2017­08­24':
            print('2017­08­24')
    print('end')