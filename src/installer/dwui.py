# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 14:16:33 2015

@author: jdh
"""

"""
Datacube ui v2

"""


from flask import Flask, send_file, abort
from flask_cors import CORS
import redis
import psycopg2
import random
from threading import Lock
from datetime import datetime
from dc_tasks import get_tile_listing,obtain_cloudfree_mosaic,assemble_mosaic,translate_files,get_bounds,obtain_file_from_math,apply_color_table_to_files, obtain_median_mosaic,obtain_pca_test, obtain_forest_noforest
from datacube.api.model import DatasetType, Ls57Arg25Bands, Satellite, Ls8Arg25Bands
import gdal
from celery import group,chord,chain
import os
import tarfile, time

app = Flask(__name__)
app.use_x_sendfile = True;
cors = CORS(app)

jobs = redis.StrictRedis(host='localhost',port=6379,db=10) #implement our own custom cache
db = None
try:
    db = psycopg2.connect(host='127.0.0.1',port='5432',dbname='postgres',user='cube_user',password='GAcube0')
except:
    print "Cannot connect to datacube database. Please check settings"
    exit()


processes = {}
no_scale = []
tcounter = 1
mutex = Lock()
def make_rid():
    global tcounter
    mutex.acquire()
    try:
        tcounter+=1
        rid = str(tcounter)
        processes[rid] = None
        return rid
    finally:
        mutex.release()

'''
def make_rid():
    mutex.acquire()
    try:
        rid = str(random.randint(0,99999999))
        while rid in processes:
            rid = str(random.randint(0,99999999))
        processes[rid] = None
        return rid
    finally:
        mutex.release()
'''
@app.route("/status")
def list_job_status():
   res = "{\"request\":\"OK\",\"tasks\":["
   plt = []
   json_arr = []
   for proc in processes:
      print processes[proc]
      p = "{\"id\":\""+str(proc)+"\",\"status\":\""+str(processes[proc].state)+"\"}"
      plt.append(p)
   res+=",".join(plt)+"]}"
   return res
      


@app.route("/cancel/<rid>")
def cancel_task(rid):
    if rid in processes:
        try:
            processes[rid].revoke()
            processes[rid].terminate()
        except:
            pass
    return "{\"request\":\"OK\"}"

@app.route("/get_stats/<rid>")
def get_stats(rid):
    if rid in processes:
        if processes[rid].ready():
            tmpvrt = processes[rid].get()[1:]
            tmpvrt = [i[0:-3]+'csv' for i in tmpvrt]
            rid2 = make_rid()
            processes[rid2] = get_bounds.delay(tmpvrt)
            return "{\"request\":\"OK\",\"files\":\""+','.join(tmpvrt)+"\"}"
    return "{\"request\":\"WAIT\"}"


@app.route("/tileinfo/<xa>/<ya>/<start>/<end>/<sats>/<ma>/submit")
def get_tile_info_by_month(rid):
    
    return "{\"request\":\"WAIT\"}"

@app.route("/list/<xa>/<ya>/<start>/<end>/<sats>/<ba>/submit")
def get_tiles(xa,ya,start,end,sats,ba):
    rid = make_rid()
    processes[rid]=get_tile_listing.delay([int(x) for x in xa.split(',')],[int(y) for y in ya.split(',')],datetime.strptime(start,"%Y-%m-%d"),datetime.strptime(end,"%Y-%m-%d"),[Satellite[s] for s in sats.split(',')],[DatasetType.ARG25],None)
    return "{\"request\":\""+rid+"\"}"

@app.route("/list/<xa>/<ya>/<start>/<end>/<sats>/<ma>/<ba>/submit")
def get_tiles_with_months(xa,ya,start,end,sats,ma,ba):
    rid = make_rid()
    processes[rid]=get_tile_listing.delay([int(x) for x in xa.split(',')],[int(y) for y in ya.split(',')],datetime.strptime(start,"%Y-%m-%d"),datetime.strptime(end,"%Y-%m-%d"),[Satellite[s] for s in sats.split(',')],[DatasetType.ARG25],[int(m) for m in ma.split(',')])
    return "{\"request\":\""+rid+"\"}"

@app.route("/list/<rid>/view")
def get_tile_list(rid):
    if rid in processes:
        if processes[rid].ready():
            return processes[rid].get()
    return "{\"request\":\"WAIT\"}"

@app.route("/mosaic/<xa>/<ya>/<start>/<end>/<sats>/<ba>/preview")
def submit_preview_request(xa,ya,start,end,sats,ba):
    try:    
        rid = make_rid()
        x_arr = [int(x) for x in xa.split(',')]
        y_arr = [int(y) for y in ya.split(',')]
        s_arr = [Satellite[s] for s in sats.split(',')]
        endDate=datetime.strptime(end,"%Y-%m-%d")
        startDate=datetime.strptime(start,"%Y-%m-%d")
        b_arr = []
        be_arr = ba.split(',')
        for s in s_arr:
            for b in be_arr:
                if s ==Satellite.LS5 or s==Satellite.LS7:
                    try:
                        tmp = Ls57Arg25Bands[b]
                        b_arr.append(tmp)
                    except KeyError:
                        pass
                elif s==Satellite.LS8:
                    try:
                        tmp = Ls8Arg25Bands[b]
                        b_arr.append(tmp)
                    except KeyError:
                        pass                
        processes[rid] = chord(group(obtain_cloudfree_mosaic.s(x,y,startDate,endDate, b_arr, s_arr,5,4000,4000,"GTiff",gdal.GDT_CInt16,None) for x in x_arr for y in y_arr ),assemble_mosaic.s()).apply_async()
        return "{\"request\":\""+rid+"\"}"
    except:
        return "{\"request\":\"WAIT\"}"
    

@app.route("/mosaic/<xa>/<ya>/<start>/<end>/<sats>/<ma>/<ba>/preview")
def submit_preview_with_months_request(xa,ya,start,end,sats,ba,ma):
    rid = make_rid()
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    endDate=datetime.strptime(end,"%Y-%m-%d")
    startDate=datetime.strptime(start,"%Y-%m-%d")
    m_arr = [int(m) for m in ma.split(',')]
    b_arr = []
    be_arr = ba.split(',')
    for s in s_arr:
        for b in be_arr:
            if s ==Satellite.LS5 or s==Satellite.LS7:
                try:
                    tmp = Ls57Arg25Bands[b]
                    b_arr.append(tmp)
                except KeyError:
                    pass
            elif s==Satellite.LS8:
                try:
                    tmp = Ls8Arg25Bands[b]
                    b_arr.append(tmp)
                except KeyError:
                    pass
                    
    processes[rid] = chord(group(obtain_cloudfree_mosaic.s(x,y,startDate,endDate, b_arr, s_arr,5,4000,4000,"GTiff",gdal.GDT_CInt16,m_arr) for x in x_arr for y in y_arr ),assemble_mosaic.s()).apply_async()
    return "{\"request\":\""+rid+"\"}"


@app.route("/mosaic/<xa>/<ya>/<start>/<end>/<sats>/<ba>/submit")
def submit_mosaic_request(xa,ya,start,end,sats,ba):
    rid = make_rid()
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    endDate=datetime.strptime(end,"%Y-%m-%d")
    startDate=datetime.strptime(start,"%Y-%m-%d")
    b_arr = []
    be_arr = ba.split(',')
    for s in s_arr:
        for b in be_arr:
            if s ==Satellite.LS5 or s==Satellite.LS7:
                try:
                    tmp = Ls57Arg25Bands[b]
                    b_arr.append(tmp)
                except KeyError:
                    pass
            elif s==Satellite.LS8:
                try:
                    tmp = Ls8Arg25Bands[b]
                    b_arr.append(tmp)
                except KeyError:
                    pass
                    
    processes[rid] = chord(group(obtain_cloudfree_mosaic.s(x,y,startDate,endDate, b_arr, s_arr,0,4000,4000,"GTiff",gdal.GDT_CInt16,None) for x in x_arr for y in y_arr ),assemble_mosaic.s()).apply_async()
    return "{\"request\":\""+rid+"\"}"

@app.route("/mosaic/<xa>/<ya>/<start>/<end>/<sats>/<ma>/<ba>/submit")
def submit_mosaic_with_months_request(xa,ya,start,end,sats,ba,ma):
    rid = make_rid()
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    endDate=datetime.strptime(end,"%Y-%m-%d")
    startDate=datetime.strptime(start,"%Y-%m-%d")
    m_arr = [int(m) for m in ma.split(',')]
    b_arr = []
    be_arr = ba.split(',')
    for s in s_arr:
        for b in be_arr:
            if s ==Satellite.LS5 or s==Satellite.LS7:
                try:
                    tmp = Ls57Arg25Bands[b]
                    b_arr.append(tmp)
                except KeyError:
                    pass
            elif s==Satellite.LS8:
                try:
                    tmp = Ls8Arg25Bands[b]
                    b_arr.append(tmp)
                except KeyError:
                    pass
                    
    processes[rid] = chord(group(obtain_cloudfree_mosaic.s(x,y,startDate,endDate, b_arr, s_arr,0,4000,4000,"GTiff",gdal.GDT_CInt16,m_arr) for x in x_arr for y in y_arr ),assemble_mosaic.s()).apply_async()
    return "{\"request\":\""+rid+"\"}"
    
@app.route("/preview/<xa>/<ya>/<start>/<end>/<sats>/<ma>/tci_b/submit")
def preview_tci_b_with_months_submit(xa,ya,start,end,sats,ma):
    rid = make_rid()
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    endDate=datetime.strptime(end,"%Y-%m-%d")
    startDate=datetime.strptime(start,"%Y-%m-%d")
    m_arr = [int(m) for m in ma.split(',')]
    bands=[Ls57Arg25Bands.BLUE, Ls57Arg25Bands.GREEN, Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1, Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
    expressions_list=["0.3561*band1 + 0.3972*band2 + 0.3904*band3 + 0.6966*band4 + 0.2286*band5 + 0.1596*band6",\
"-0.3344*band1 - 0.3544*band2 - 0.4556*band3 + 0.6966*band4 - 0.0242*band5 - 0.2630*band6",\
"0.2626*band1 + 0.2141*band2 + 0.0926*band3 + 0.0656*band4 - 0.7629*band5 - 0.5388*band6"]
    no_scale.append(rid)
    processes[rid]=chord(chord((obtain_cloudfree_mosaic.s(x,y,startDate,endDate, bands, s_arr,5,4000,4000,"GTiff",gdal.GDT_CInt16,m_arr)|obtain_file_from_math.s(expressions_list=expressions_list,file_format="GTiff",data_type=gdal.GDT_CFloat32,input_ndv=-999,output_ndv=-999) for x in x_arr for y in y_arr),assemble_mosaic.s()),apply_color_table_to_files.s(output_band=1,color_table="/tilestore/colortables/brightness-color.txt")).apply_async()
    return "{\"request\":\""+rid+"\"}"
@app.route("/preview/<xa>/<ya>/<start>/<end>/<sats>/tci_b/submit")
def preview_tci_b_submit(xa,ya,start,end,sats):
    rid = make_rid()
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    endDate=datetime.strptime(end,"%Y-%m-%d")
    startDate=datetime.strptime(start,"%Y-%m-%d")
    
    bands=[Ls57Arg25Bands.BLUE, Ls57Arg25Bands.GREEN, Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1, Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
    expressions_list=["0.3561*band1 + 0.3972*band2 + 0.3904*band3 + 0.6966*band4 + 0.2286*band5 + 0.1596*band6",\
"-0.3344*band1 - 0.3544*band2 - 0.4556*band3 + 0.6966*band4 - 0.0242*band5 - 0.2630*band6",\
"0.2626*band1 + 0.2141*band2 + 0.0926*band3 + 0.0656*band4 - 0.7629*band5 - 0.5388*band6"]
    no_scale.append(rid)
    processes[rid]=chord(chord((obtain_cloudfree_mosaic.s(x,y,startDate,endDate, bands, s_arr,5,4000,4000,"GTiff",gdal.GDT_CInt16,None)|obtain_file_from_math.s(expressions_list=expressions_list,file_format="GTiff",data_type=gdal.GDT_CFloat32,input_ndv=-999,output_ndv=-999) for x in x_arr for y in y_arr),assemble_mosaic.s()),apply_color_table_to_files.s(output_band=1,color_table="/tilestore/colortables/brightness-color.txt")).apply_async()
    return "{\"request\":\""+rid+"\"}"
    
@app.route("/mosaic/<xa>/<ya>/<start>/<end>/<sats>/<ma>/tci_b/submit")
def mosaic_tci_b_with_months_submit(xa,ya,start,end,sats,ma):
    rid = make_rid()
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    endDate=datetime.strptime(end,"%Y-%m-%d")
    startDate=datetime.strptime(start,"%Y-%m-%d")
    m_arr = [int(m) for m in ma.split(',')]
    bands=[Ls57Arg25Bands.BLUE, Ls57Arg25Bands.GREEN, Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1, Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
    expressions_list=["0.3561*band1 + 0.3972*band2 + 0.3904*band3 + 0.6966*band4 + 0.2286*band5 + 0.1596*band6",\
"-0.3344*band1 - 0.3544*band2 - 0.4556*band3 + 0.6966*band4 - 0.0242*band5 - 0.2630*band6",\
"0.2626*band1 + 0.2141*band2 + 0.0926*band3 + 0.0656*band4 - 0.7629*band5 - 0.5388*band6"]
    no_scale.append(rid)
    processes[rid]=chord(chord((obtain_cloudfree_mosaic.s(x,y,startDate,endDate, bands, s_arr,0,4000,4000,"GTiff",gdal.GDT_CInt16,m_arr)|obtain_file_from_math.s(expressions_list=expressions_list,file_format="GTiff",data_type=gdal.GDT_CFloat32,input_ndv=-999,output_ndv=-999) for x in x_arr for y in y_arr),assemble_mosaic.s()),apply_color_table_to_files.s(output_band=1,color_table="/tilestore/colortables/brightness-color.txt")).apply_async()
    return "{\"request\":\""+rid+"\"}"
@app.route("/mosaic/<xa>/<ya>/<start>/<end>/<sats>/tci_b/submit")
def mosaic_tci_b_submit(xa,ya,start,end,sats):
    rid = make_rid()
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    endDate=datetime.strptime(end,"%Y-%m-%d")
    startDate=datetime.strptime(start,"%Y-%m-%d")
    
    bands=[Ls57Arg25Bands.BLUE, Ls57Arg25Bands.GREEN, Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1, Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
    expressions_list=["0.3561*band1 + 0.3972*band2 + 0.3904*band3 + 0.6966*band4 + 0.2286*band5 + 0.1596*band6","-0.3344*band1 - 0.3544*band2 - 0.4556*band3 + 0.6966*band4 - 0.0242*band5 - 0.2630*band6","0.2626*band1 + 0.2141*band2 + 0.0926*band3 + 0.0656*band4 - 0.7629*band5 - 0.5388*band6"]
    no_scale.append(rid)
    processes[rid]=chord(chord((obtain_cloudfree_mosaic.s(x,y,startDate,endDate, bands, s_arr,0,4000,4000,"GTiff",gdal.GDT_CInt16,None)|obtain_file_from_math.s(expressions_list=expressions_list,file_format="GTiff",data_type=gdal.GDT_CFloat32,input_ndv=-999,output_ndv=-999) for x in x_arr for y in y_arr),assemble_mosaic.s()),apply_color_table_to_files.s(output_band=1,color_table="/tilestore/colortables/brightness-color.txt")).apply_async()
    return "{\"request\":\""+rid+"\"}"
    
    
@app.route("/preview/<xa>/<ya>/<start>/<end>/<sats>/<ma>/tci_g/submit")
def preview_tci_g_with_months_submit(xa,ya,start,end,sats,ma):
    rid = make_rid()
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    endDate=datetime.strptime(end,"%Y-%m-%d")
    startDate=datetime.strptime(start,"%Y-%m-%d")
    m_arr = [int(m) for m in ma.split(',')]
    bands=[Ls57Arg25Bands.BLUE, Ls57Arg25Bands.GREEN, Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1, Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
    expressions_list=["0.3561*band1 + 0.3972*band2 + 0.3904*band3 + 0.6966*band4 + 0.2286*band5 + 0.1596*band6",\
"-0.3344*band1 - 0.3544*band2 - 0.4556*band3 + 0.6966*band4 - 0.0242*band5 - 0.2630*band6",\
"0.2626*band1 + 0.2141*band2 + 0.0926*band3 + 0.0656*band4 - 0.7629*band5 - 0.5388*band6"]
    no_scale.append(rid)
    processes[rid]=chord(chord((obtain_cloudfree_mosaic.s(x,y,startDate,endDate, bands, s_arr,5,4000,4000,"GTiff",gdal.GDT_CInt16,m_arr)|obtain_file_from_math.s(expressions_list=expressions_list,file_format="GTiff",data_type=gdal.GDT_CFloat32,input_ndv=-999,output_ndv=-999) for x in x_arr for y in y_arr),assemble_mosaic.s()),apply_color_table_to_files.s(output_band=2,color_table="/tilestore/colortables/greenness-color.txt")).apply_async()
    return "{\"request\":\""+rid+"\"}"
@app.route("/preview/<xa>/<ya>/<start>/<end>/<sats>/tci_g/submit")
def preview_tci_g_submit(xa,ya,start,end,sats):
    rid = make_rid()
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    endDate=datetime.strptime(end,"%Y-%m-%d")
    startDate=datetime.strptime(start,"%Y-%m-%d")
    
    bands=[Ls57Arg25Bands.BLUE, Ls57Arg25Bands.GREEN, Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1, Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
    expressions_list=["0.3561*band1 + 0.3972*band2 + 0.3904*band3 + 0.6966*band4 + 0.2286*band5 + 0.1596*band6",\
"-0.3344*band1 - 0.3544*band2 - 0.4556*band3 + 0.6966*band4 - 0.0242*band5 - 0.2630*band6",\
"0.2626*band1 + 0.2141*band2 + 0.0926*band3 + 0.0656*band4 - 0.7629*band5 - 0.5388*band6"]
    no_scale.append(rid)
    processes[rid]=chord(chord((obtain_cloudfree_mosaic.s(x,y,startDate,endDate, bands, s_arr,5,4000,4000,"GTiff",gdal.GDT_CInt16,None)|obtain_file_from_math.s(expressions_list=expressions_list,file_format="GTiff",data_type=gdal.GDT_CFloat32,input_ndv=-999,output_ndv=-999) for x in x_arr for y in y_arr),assemble_mosaic.s()),apply_color_table_to_files.s(output_band=2,color_table="/tilestore/colortables/greenness-color.txt")).apply_async()
    return "{\"request\":\""+rid+"\"}"
    
@app.route("/mosaic/<xa>/<ya>/<start>/<end>/<sats>/<ma>/tci_g/submit")
def mosaic_tci_g_with_months_submit(xa,ya,start,end,sats,ma):
    rid = make_rid()
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    endDate=datetime.strptime(end,"%Y-%m-%d")
    startDate=datetime.strptime(start,"%Y-%m-%d")
    m_arr = [int(m) for m in ma.split(',')]
    bands=[Ls57Arg25Bands.BLUE, Ls57Arg25Bands.GREEN, Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1, Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
    expressions_list=["0.3561*band1 + 0.3972*band2 + 0.3904*band3 + 0.6966*band4 + 0.2286*band5 + 0.1596*band6",\
"-0.3344*band1 - 0.3544*band2 - 0.4556*band3 + 0.6966*band4 - 0.0242*band5 - 0.2630*band6",\
"0.2626*band1 + 0.2141*band2 + 0.0926*band3 + 0.0656*band4 - 0.7629*band5 - 0.5388*band6"]
    no_scale.append(rid)
    processes[rid]=chord(chord((obtain_cloudfree_mosaic.s(x,y,startDate,endDate, bands, s_arr,0,4000,4000,"GTiff",gdal.GDT_CInt16,m_arr)|obtain_file_from_math.s(expressions_list=expressions_list,file_format="GTiff",data_type=gdal.GDT_CFloat32,input_ndv=-999,output_ndv=-999) for x in x_arr for y in y_arr),assemble_mosaic.s()),apply_color_table_to_files.s(output_band=2,color_table="/tilestore/colortables/greenness-color.txt")).apply_async()
    return "{\"request\":\""+rid+"\"}"
@app.route("/mosaic/<xa>/<ya>/<start>/<end>/<sats>/tci_g/submit")
def mosaic_tci_g_submit(xa,ya,start,end,sats):
    rid = make_rid()
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    endDate=datetime.strptime(end,"%Y-%m-%d")
    startDate=datetime.strptime(start,"%Y-%m-%d")
    
    bands=[Ls57Arg25Bands.BLUE, Ls57Arg25Bands.GREEN, Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1, Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
    expressions_list=["0.3561*band1 + 0.3972*band2 + 0.3904*band3 + 0.6966*band4 + 0.2286*band5 + 0.1596*band6",\
"-0.3344*band1 - 0.3544*band2 - 0.4556*band3 + 0.6966*band4 - 0.0242*band5 - 0.2630*band6",\
"0.2626*band1 + 0.2141*band2 + 0.0926*band3 + 0.0656*band4 - 0.7629*band5 - 0.5388*band6"]
    no_scale.append(rid)
    processes[rid]=chord(chord((obtain_cloudfree_mosaic.s(x,y,startDate,endDate, bands, s_arr,0,4000,4000,"GTiff",gdal.GDT_CInt16,None)|obtain_file_from_math.s(expressions_list=expressions_list,file_format="GTiff",data_type=gdal.GDT_CFloat32,input_ndv=-999,output_ndv=-999) for x in x_arr for y in y_arr),assemble_mosaic.s()),apply_color_table_to_files.s(output_band=2,color_table="/tilestore/colortables/greenness-color.txt")).apply_async()
    return "{\"request\":\""+rid+"\"}"
    
    
@app.route("/preview/<xa>/<ya>/<start>/<end>/<sats>/<ma>/tci_w/submit")
def preview_tci_w_with_months_submit(xa,ya,start,end,sats,ma):
    rid = make_rid()
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    endDate=datetime.strptime(end,"%Y-%m-%d")
    startDate=datetime.strptime(start,"%Y-%m-%d")
    m_arr = [int(m) for m in ma.split(',')]
    bands=[Ls57Arg25Bands.BLUE, Ls57Arg25Bands.GREEN, Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1, Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
    expressions_list=["0.3561*band1 + 0.3972*band2 + 0.3904*band3 + 0.6966*band4 + 0.2286*band5 + 0.1596*band6",\
"-0.3344*band1 - 0.3544*band2 - 0.4556*band3 + 0.6966*band4 - 0.0242*band5 - 0.2630*band6",\
"0.2626*band1 + 0.2141*band2 + 0.0926*band3 + 0.0656*band4 - 0.7629*band5 - 0.5388*band6"]
    no_scale.append(rid)
    processes[rid]=chord(chord((obtain_cloudfree_mosaic.s(x,y,startDate,endDate, bands, s_arr,5,4000,4000,"GTiff",gdal.GDT_CInt16,m_arr)|obtain_file_from_math.s(expressions_list=expressions_list,file_format="GTiff",data_type=gdal.GDT_CFloat32,input_ndv=-999,output_ndv=-999) for x in x_arr for y in y_arr),assemble_mosaic.s()),apply_color_table_to_files.s(output_band=3,color_table="/tilestore/colortables/wetness-color.txt")).apply_async()
    return "{\"request\":\""+rid+"\"}"
@app.route("/preview/<xa>/<ya>/<start>/<end>/<sats>/tci_w/submit")
def preview_tci_w_submit(xa,ya,start,end,sats):
    rid = make_rid()
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    endDate=datetime.strptime(end,"%Y-%m-%d")
    startDate=datetime.strptime(start,"%Y-%m-%d")
    
    bands=[Ls57Arg25Bands.BLUE, Ls57Arg25Bands.GREEN, Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1, Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
    expressions_list=["0.3561*band1 + 0.3972*band2 + 0.3904*band3 + 0.6966*band4 + 0.2286*band5 + 0.1596*band6",\
"-0.3344*band1 - 0.3544*band2 - 0.4556*band3 + 0.6966*band4 - 0.0242*band5 - 0.2630*band6",\
"0.2626*band1 + 0.2141*band2 + 0.0926*band3 + 0.0656*band4 - 0.7629*band5 - 0.5388*band6"]
    no_scale.append(rid)
    processes[rid]=chord(chord((obtain_cloudfree_mosaic.s(x,y,startDate,endDate, bands, s_arr,5,4000,4000,"GTiff",gdal.GDT_CInt16,None)|obtain_file_from_math.s(expressions_list=expressions_list,file_format="GTiff",data_type=gdal.GDT_CFloat32,input_ndv=-999,output_ndv=-999) for x in x_arr for y in y_arr),assemble_mosaic.s()),apply_color_table_to_files.s(output_band=3,color_table="/tilestore/colortables/wetness-color.txt")).apply_async()
    return "{\"request\":\""+rid+"\"}"
    
@app.route("/mosaic/<xa>/<ya>/<start>/<end>/<sats>/<ma>/tci_w/submit")
def mosaic_tci_w_with_months_submit(xa,ya,start,end,sats,ma):
    rid = make_rid()
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    endDate=datetime.strptime(end,"%Y-%m-%d")
    startDate=datetime.strptime(start,"%Y-%m-%d")
    m_arr = [int(m) for m in ma.split(',')]
    bands=[Ls57Arg25Bands.BLUE, Ls57Arg25Bands.GREEN, Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1, Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
    expressions_list=["0.3561*band1 + 0.3972*band2 + 0.3904*band3 + 0.6966*band4 + 0.2286*band5 + 0.1596*band6",\
"-0.3344*band1 - 0.3544*band2 - 0.4556*band3 + 0.6966*band4 - 0.0242*band5 - 0.2630*band6",\
"0.2626*band1 + 0.2141*band2 + 0.0926*band3 + 0.0656*band4 - 0.7629*band5 - 0.5388*band6"]
    no_scale.append(rid)
    processes[rid]=chord(chord((obtain_cloudfree_mosaic.s(x,y,startDate,endDate, bands, s_arr,0,4000,4000,"GTiff",gdal.GDT_CInt16,m_arr)|obtain_file_from_math.s(expressions_list=expressions_list,file_format="GTiff",data_type=gdal.GDT_CFloat32,input_ndv=-999,output_ndv=-999) for x in x_arr for y in y_arr),assemble_mosaic.s()),apply_color_table_to_files.s(output_band=3,color_table="/tilestore/colortables/wetness-color.txt")).apply_async()
    return "{\"request\":\""+rid+"\"}"
@app.route("/mosaic/<xa>/<ya>/<start>/<end>/<sats>/tci_w/submit")
def mosaic_tci_w_submit(xa,ya,start,end,sats):
    rid = make_rid()
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    endDate=datetime.strptime(end,"%Y-%m-%d")
    startDate=datetime.strptime(start,"%Y-%m-%d")
    
    bands=[Ls57Arg25Bands.BLUE, Ls57Arg25Bands.GREEN, Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1, Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
    expressions_list=["0.3561*band1 + 0.3972*band2 + 0.3904*band3 + 0.6966*band4 + 0.2286*band5 + 0.1596*band6",\
"-0.3344*band1 - 0.3544*band2 - 0.4556*band3 + 0.6966*band4 - 0.0242*band5 - 0.2630*band6",\
"0.2626*band1 + 0.2141*band2 + 0.0926*band3 + 0.0656*band4 - 0.7629*band5 - 0.5388*band6"]
    no_scale.append(rid)
    processes[rid]=chord(chord((obtain_cloudfree_mosaic.s(x,y,startDate,endDate, bands, s_arr,0,4000,4000,"GTiff",gdal.GDT_CInt16,None)|obtain_file_from_math.s(expressions_list=expressions_list,file_format="GTiff",data_type=gdal.GDT_CFloat32,input_ndv=-999,output_ndv=-999) for x in x_arr for y in y_arr),assemble_mosaic.s()),apply_color_table_to_files.s(output_band=3,color_table="/tilestore/colortables/wetness-color.txt")).apply_async()
    return "{\"request\":\""+rid+"\"}"


@app.route("/mosaic/<rid>/bounds/submit")
def get_mosaic_bounds(rid):
    if rid in processes:
        if processes[rid].ready():
            tmpvrt = processes[rid].get()[0]
            rid2 = make_rid()
            processes[rid2] = get_bounds.delay(tmpvrt)
            return "{\"request\":\""+rid2+"\"}"
    return "{\"request\":\"WAIT\"}"
            
@app.route("/mosaic/<rid>/bounds/view")
def view_mosaic_bounds(rid):
    if rid in processes:
        if processes[rid].ready():
            b_str = processes[rid].get()
            return "{\"request\":\""+b_str+"\"}"
    return "{\"request\":\"WAIT\"}"

@app.route("/mosaic/<rid>/<file_format>/view/submit")
def view_mosaic_image_submit(rid,file_format):
    if rid in processes:
        print processes[rid]
        if processes[rid].ready():
            tmpvrt = processes[rid].get()[0]
            print tmpvrt
            rid2 = make_rid()
            print "CALLING TRANSLATE FILES"
            out_size = "5% 5%"
            out_scale = "0 4096 0 255"
            data_t = "Byte"
            band_l = [1,2,3]
            if file_format == "JPEG" or file_format == "GTiff":
                out_size = "100% 100%"
            if file_format == "GTiff":
                out_scale = None
                data_t = None
                band_l = None
            if rid in no_scale:
                out_scale = None
            processes[rid2] = chain(assemble_mosaic.s(tmpvrt),translate_files.s(file_format=file_format,output_scale=out_scale,output_size=out_size,output_datatype=data_t,output_bands=band_l,additional_arguments=None)).apply_async()
            return "{\"request\":\""+rid2+"\"}"
    return "{\"request\":\"WAIT\"}"
@app.route("/mosaicfc/<rid>/<file_format>/view/submit")
def view_mosaicfc_image_submit(rid,file_format):
    if rid in processes:
        print processes[rid]
        if processes[rid].ready():
            tmpvrt = processes[rid].get()[0]
            print tmpvrt
            rid2 = make_rid()
            print "CALLING TRANSLATE FILES"
            out_size = "5% 5%"
            out_scale = "0 4096 0 255"
            data_t = "Byte"
            band_l = [2,3,1]
            if file_format == "JPEG" or file_format == "GTiff":
                out_size = "100% 100%"
            if file_format == "GTiff":
                out_scale = None
                data_t = None
                band_l = None
            if rid in no_scale:
                out_scale = None
            processes[rid2] = chain(assemble_mosaic.s(tmpvrt),translate_files.s(file_format=file_format,output_scale=out_scale,output_size=out_size,output_datatype=data_t,output_bands=band_l,additional_arguments=None)).apply_async()
            return "{\"request\":\""+rid2+"\"}"
    return "{\"request\":\"WAIT\"}"
@app.route("/mosaic/<rid>/view")
def view_mosaic_image(rid):
    if rid in processes:
        if processes[rid].ready():
            tmpvrt = processes[rid].get()
            print "Image is :"
            print tmpvrt
            if len(tmpvrt)>=1:
                return "{\"request\":\""+tmpvrt[0]+"\",\"alternative\":\""+tmpvrt[1]+"\"}"
    return "{\"request\":\"WAIT\"}"

@app.route("/stats/<xa>/<ya>/<start>/<end>/<sats>/submit")
def submit_stats_request(xa,ya,start,end,sats):
    return "Not availible yet"

@app.route("/tilestore/tile_cache/<img>")
def send_image(img):
    if '..' in img or img.startswith('/'):
        abort(403) #should be enough for now
    tar_img = os.path.join("/tilestore/tile_cache/",img)
    if os.path.isfile(tar_img):
        return send_file(tar_img)
    else:
        abort(404)
    
@app.route("/tilestore/tile_cache/<img>/download")
def send_image_attachment(img):
    if '..' in img or img.startswith('/'):
        abort(403) #should be enough for now
    tar_img = os.path.join("/tilestore/tile_cache/",img)
    if os.path.isfile(tar_img):
        return send_file(tar_img,as_attachment=True)
    else:
        abort(404)

@app.route("/")
def get_index():
    return """
    <p>Services:</p>
    <ul>
    <li>list/lng1,lngs2,...,lngN/lat1,lat2,...,latN/YYYY-MM-DD/YYYY-MM-DD/LS7,LS8[/1,2,3...,12]/submit <i>Request a listing of data in 1deg x 1deg squares</i></li>
    <li>list/1234567/view <i>View a requested listing</i></li>
    
    </ul>
    """
@app.route("/median/<xa>/<ya>/<start>/<end>/<sats>/<min_val>/submit")
def median_mosaic(xa, ya,start, end, sats, min_val):
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    mval=int(min_val)
    rid = make_rid()
    bands=[Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1, Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
    en=datetime.strptime(end,"%Y-%m-%d")
    st=datetime.strptime(start,"%Y-%m-%d")
    processes[rid]=chord(group(obtain_median_mosaic.s(mval,X,Y,st,en,bands,s_arr,0,4000,4000,"GTiff",gdal.GDT_CFloat32,None) for X in x_arr for Y in y_arr),assemble_mosaic.s()).apply_async()
    return "{\"request\":\""+rid+"\", \"type\":\"median mosaic\"}"
@app.route("/pca/<xa>/<ya>/<p1_start>/<p1_end>/<p2_start>/<p2_end>/<sats>/<min_val>/submit")
def pca(xa,ya,p1_start, p1_end,p2_start, p2_end, sats, min_val):
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    mval=int(min_val)
    rid = make_rid()
    en1=datetime.strptime(p1_end,"%Y-%m-%d")
    st1=datetime.strptime(p1_start,"%Y-%m-%d")
    en2=datetime.strptime(p2_end,"%Y-%m-%d")
    st2=datetime.strptime(p2_start,"%Y-%m-%d")
    bands=[Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1, Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
    processes[rid]=group(obtain_pca_test.s(mval, X,Y,st1,en1,st2,en2,bands,s_arr,0,4000,4000,"GTiff",gdal.GDT_CFloat32,None) for X in x_arr for Y in y_arr).apply_async()
    processes[rid].state="working"
    return "{\"request\":\""+rid+"\", \"type\":\"pca\"}"
@app.route("/forest/<xa>/<ya>/<start>/<end>/<sats>/<min_val>/submit")
def forest(xa, ya,start, end, sats, min_val):
    x_arr = [int(x) for x in xa.split(',')]
    y_arr = [int(y) for y in ya.split(',')]
    s_arr = [Satellite[s] for s in sats.split(',')]
    mval=int(min_val)
    rid = make_rid()
    #bands=[Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1, Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
    en=datetime.strptime(end,"%Y-%m-%d")
    st=datetime.strptime(start,"%Y-%m-%d")
    processes[rid]=chord(group(obtain_forest_noforest.s(X,Y,st,en,min_ok=min_val) for X in x_arr for Y in y_arr),assemble_mosaic.s()).apply_async()
    return "{\"request\":\""+rid+"\", \"type\":\"forest\"}"
@app.route("/tar/<rid>/download")    
def downloadTar(rid):
    """
        Create a tar file from the file list returned by request identified by rid
    """
    if rid in processes:
        if processes[rid].ready():
            if processes[rid].state=="working":
                processes[rid].state="success"
            fs=processes[rid].get()
            files=[f for f in fs if f!="None"]
            strFile="/tilestore/tile_cache/"+rid+"-"+str(time.time())+".tar.gz"
            tar = tarfile.open(strFile, "w:gz")
            for name in files:
                tar.add(name)
            tar.close()
            return "{\"request\":\""+strFile+"\"}"
    return "{\"request\":\"WAIT\"}"
            

def i_main():
    
    app.debug = True
    app.run(host='0.0.0.0', port=8080)
    

if __name__=='__main__':
    i_main()
    
