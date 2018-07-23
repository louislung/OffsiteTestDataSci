CREATE TABLE IF NOT EXISTS public.piwik_track (
time timestamp,
uid varchar(256),
event_name varchar(256),
source_ip varchar(50),
CONSTRAINT piwik_track_pkey PRIMARY KEY(time,uid,event_name,source_ip)
)