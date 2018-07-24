#!/usr/bin/python


##################
# Import modules #
##################
import os
from pathlib import Path
import time
import io
import re


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
suffix = '.jpg'
suffix_regex = re.compile(re.escape(suffix) + '$')


################
# Main Program #
################
def get_data(input_filename, delimiter = '\t'):
    with io.open(input_filename, mode = 'r+', encoding = 'utf-8') as f:
        for record in f:
            x = record.rstrip().split(delimiter)
            yield x


if __name__ == '__main__':
    log_generator = get_data(data_dir / log_file)
    header = next(log_generator) #skip header
    size = 0
    for row in log_generator:
        timestamp = time.strptime((row[header.index('# date')] + ' ' + row[header.index('time')]),'%Y-%m-%d %H:%M:%S')
        if timestamp >= time.strptime(date_from + ' ' + time_from,'%Y-%m-%d %H:%M:%S') and timestamp <= time.strptime(date_to + ' ' + time_to,'%Y-%m-%d %H:%M:%S'):
            if suffix_regex.search(row[header.index('url')]):
                size += float(row[header.index('size')])

    print('Answers:')
    print('Total size of file with suffix ({0}) = {1}'.format(suffix,size))
