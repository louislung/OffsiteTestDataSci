Run Result
----------

## How to get result

Run the following command

.. code:: bash
    
    python query.py
    

## What the script does

The script will perform the following in order
1. Connect to postgreSQL database which set up on AWS
2. Drop table piwik_track if exists
3. Create table piwik_track if not exists
4. Insert data from data/piwik_track.csv into table created in step 3
5. Run sql to get the answer