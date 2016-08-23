# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:01:33 2015

@author: jdh

Tile task system for celery
"""
from datacube.api.query import SortType
from matplotlib.mlab import PCA
from datetime import datetime,timedelta
import logging
import os
from osgeo import gdal
import osr #agregado para exportar el archivo de pca (TODO: Evitarlo)
import numpy
import numpy as np
import numexpr as ne
import Image
import sklearn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cPickle as pickl
from sklearn.preprocessing import normalize
from datacube.api.model import DatasetType, Ls57Arg25Bands, Satellite, Ls8Arg25Bands
from datacube.api.utils import NDV, empty_array, get_dataset_metadata, get_dataset_data_with_pq, raster_create,get_dataset_data, PqaMask
from datacube.api.query import list_tiles
from datacube.api.model import DatasetType
from datacube.api.model import Ls57Arg25Bands, TciBands, NdviBands, EviBands
from datacube.api.query import list_tiles
from datacube.api.utils import get_mask_pqa, get_dataset_data_masked, OutputFormat
import time
from pprint import pprint
import itertools
import random
import string
from gdalconst import *
from datacube_worker import celery, cache, database
import Image
import math
from scipy.cluster.vq import kmeans,vq
#app = Celery('tasks',backend='redis://localhost',broker='amqp://')
satellites = {'ls7':Satellite.LS7,'ls8':Satellite.LS8}
FILE_EXT = {"png":".png","GTiff":".tif","VRT":".vrt","JPEG":".jpeg"}

@celery.task()
def get_tile_info(xa,ya,start,end,satellite,datasets,months=None):
    """
    Get Tile Info
    """
    tiles = list_tiles(x=xa,y=ya,acq_min=start,acq_max=end,satellites = satellite,dataset_types=datasets)
    data = "{\"request\":\"DONE\",\"tiles\":["
    data_arr = []
    for tile in tiles:
        
        if months:
            print tile.start_datetime.month
            if tile.start_datetime.month in months:
                data_arr.append()
        else:
            data_arr.append("{\"x\":"+str(tile.x)+",\"y\":"+str(tile.y)+",\"date\":\""+str(tile.start_datetime)+"\"}")
    data+=','.join(data_arr)+"]}"
    return data
    
@celery.task()
def get_tile_listing(xa,ya,start,end,satellite,datasets,months=None):
    """
    List tiles. Months will only show the requested months
    """
    tiles = list_tiles(x=xa,y=ya,acq_min=start,acq_max=end,satellites = satellite,dataset_types=datasets)
    data = "{\"request\":\"DONE\",\"tiles\":["
    data_arr = []
    for tile in tiles:
        if months:
            print tile.start_datetime.month
            if tile.start_datetime.month in months:
                data_arr.append("{\"x\":"+str(tile.x)+",\"y\":"+str(tile.y)+",\"date\":\""+str(tile.start_datetime)+"\"}")
        else:
            data_arr.append("{\"x\":"+str(tile.x)+",\"y\":"+str(tile.y)+",\"date\":\""+str(tile.start_datetime)+"\"}")
    data+=','.join(data_arr)+"]}"
    return data
    

@celery.task()
def obtain_cloudfree_mosaic(x,y,start,end, bands, satellite,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_CInt16,months=None):
    StartDate = start
    EndDate = end
    print "starting cloudfree mosaic"    
    best_data = {}
    band_str = "+".join([band.name for band in bands])
    sat_str = "+".join([sat.name for sat in satellite])
    cache_id = [str(x),str(y),str(start),str(end),band_str,sat_str,str(xsize),str(ysize),file_format,str(iterations)]
    f_name = "_".join(cache_id)
    f_name = f_name.replace(" ","_")
    c_name = f_name
    cached_res = cache.get(c_name)
    if cached_res:
        return str(cached_res)
    f_name = os.path.join("/tilestore/tile_cache",f_name)
    tiles = list_tiles(x=[x], y=[y],acq_min=StartDate,acq_max=EndDate,satellites=satellite,dataset_types=[DatasetType.ARG25,DatasetType.PQ25], sort=SortType.ASC)
    tile_metadata = None
    tile_count = 0
    tile_filled = False
    stats_file = open(f_name+'.csv','w+')

    for tile in tiles:
        if tile_filled:
           break
        if months:
            print tile.start_datetime.month
            if not tile.start_datetime.month in months:
                continue
        #print "merging on tile "+str(tile.x)+", "+str(tile.y)
        tile_count+=1
        dataset =  DatasetType.ARG25 in tile.datasets and tile.datasets[DatasetType.ARG25] or None
        if dataset is None:
            print "No dataset availible"
            tile_count-=1
            continue
        tile_metadata = get_dataset_metadata(dataset)
        if tile_metadata is None:
            print "NO METADATA"
            tile_count-=1
            continue
        pqa = DatasetType.PQ25 in tile.datasets and tile.datasets[DatasetType.PQ25] or None
        mask = None
        mask = get_mask_pqa(pqa,[PqaMask.PQ_MASK_CLEAR],mask=mask)
	
	if tile.dataset.find('LC8') >= 0:
            nbands = map(lambda x: Ls8Arg25Bands(x.value+1),bands)
	else:
            nbands = bands
        band_data = get_dataset_data_masked(dataset, mask=mask,bands=nbands)

	if tile.dataset.find('LC8') >= 0:
             band_data = dict(map(lambda (k,v): (Ls57Arg25Bands(k.value-1),v), band_data.iteritems()))

        swap_arr = None
        best = None
        for band in bands:
            if not band in best_data:
                #print "Adding "+band.name
                #print band_data[band]
                best_data[band]=band_data[band]
                best = numpy.array(best_data[band])
                swap_arr=numpy.in1d(best.ravel(),-999).reshape(best.shape)
            else:
                best = numpy.array(best_data[band])
                swap_arr=numpy.in1d(best.ravel(),-999).reshape(best.shape)
                b_data = numpy.array(band_data[band])
                # extend array if source data is smaller than best data
                while b_data.shape[1] < swap_arr.shape[1]:
                    col = numpy.zeros((b_data.shape[0],1))
                    col.fill(-999)
                    b_data = numpy.append(b_data,col,axis=1)

                while b_data.shape[0] < swap_arr.shape[0]:
                    row = numpy.zeros((1,b_data.shape[1]))
                    row.fill(-999)
                    b_data = numpy.append(b_data,row,axis=0)

                best[swap_arr]=b_data[swap_arr]
                best_data[band]=numpy.copy(best)
                del b_data
        stats_file.write(str(tile.start_datetime.year)+','+str(tile.start_datetime.month)+','+str(len(best[swap_arr]))+"\n")
        del swap_arr
        del best
        if iterations > 0:
            if tile_count>iterations:
                print "Exiting after "+str(iterations)+" iterations"
                break
    numberOfBands=len(bands)

    if numberOfBands == 0:
       return "None"
    if bands[0] not in best_data:
       print "No data was merged for "+str(x)+", "+str(y)
       return "None"

    print "mosaic created"
    numberOfPixelsInXDirection=len(best_data[bands[0]])
    print numberOfPixelsInXDirection
    numberOfPixelsInYDirection=len(best_data[bands[0]][0])   
    print numberOfPixelsInYDirection
    pixels = numberOfPixelsInXDirection
    if numberOfPixelsInYDirection > numberOfPixelsInXDirection:
        pixels = numberOfPixelsInYDirection
    if tile_count <1:
        print "No tiles found for "+str(x)+", "+str(y)
        return "None"
    driver = gdal.GetDriverByName(file_format)
    if driver is None:
        print "No driver found for "+file_format
        return "None"
    #print f_name+'.tif'
    raster = driver.Create(f_name+'.tif', pixels, pixels, numberOfBands, data_type, options=["BIGTIFF=YES", "INTERLEAVE=BAND"])
    raster.SetGeoTransform(tile_metadata.transform)
    raster.SetProjection(tile_metadata.projection)
    index = 1
    stats_file.close()
    for band in bands:
        stack_band = raster.GetRasterBand(index)
        stack_band.SetNoDataValue(-999)
        stack_band.WriteArray(best_data[band])
        stack_band.ComputeStatistics(True)
        index+=1
        stack_band.FlushCache()
        del stack_band
    raster.FlushCache()
    del raster
    cache.set(c_name,f_name+".tif")
    return f_name+".tif"
    
 
 
@celery.task()
def matrix_obtain_mosaic(x,y,start,end, bands, satellite,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_CInt16,months=None, normalized=False):
    """
    Obtains a dict with the query results, one matrix per band
    MATRIX OBTAIN MOSAIC
    """
    StartDate = start
    EndDate = end
    print("____________________matriz_obtain_mosaic____________________")
    
    tiles = list_tiles(x=[x], y=[y],acq_min=StartDate,acq_max=EndDate,satellites=satellite,dataset_types=[DatasetType.ARG25,DatasetType.PQ25], sort=SortType.ASC)
    tile_metadata = None
    tile_count = 0
    tile_filled = False
    total_ins = 0
    
    all_bands={}
    avgs_band={}
    st_band={}
    count_band={}
    for tile in tiles:
        if tile_filled:
           break
        if months:
            print tile.start_datetime.month
            if not tile.start_datetime.month in months:
                continue
        tile_count+=1
        dataset =  DatasetType.ARG25 in tile.datasets and tile.datasets[DatasetType.ARG25] or None
        if dataset is None:
            print "No dataset availible"
            tile_count-=1
            continue
        tile_metadata = get_dataset_metadata(dataset)
        if tile_metadata is None:
            print "NO METADATA"
            tile_count-=1
            continue
        
        pqa = DatasetType.PQ25 in tile.datasets and tile.datasets[DatasetType.PQ25] or None
        mask = None
        mask = get_mask_pqa(pqa,[PqaMask.PQ_MASK_CLEAR],mask=mask)
        band_data = get_dataset_data_masked(dataset, mask=mask,bands=bands)
        del mask
        for band in band_data:
            
          #  print "Adding "+band.name
            data = numpy.array(band_data[band]).astype(numpy.float32)
            non_data=numpy.in1d(data.ravel(),-999).reshape(data.shape)
            data[non_data]=numpy.NaN 
            if normalized:
                m=np.nanmean(data)
                st=np.nanstd(data)
                if not np.isnan(m):
                    avgs_band[band.name]=avgs_band[band.name]+m if avgs_band.has_key(band.name) else m
                    st_band[band.name]=st_band[band.name]+st if st_band.has_key(band.name) else st
                    count_band[band.name] =(count_band[band.name]+1) if count_band.has_key(band.name) else 1
                if not np.isnan(m):
                  #  print ("Media: "+str(m)+" STD: "+str(st))
                    data=np.true_divide(np.subtract(data,m),st)
            if not np.isnan(data).all():
                if all_bands.has_key(band.name):
                    all_bands[band.name]=numpy.dstack((all_bands[band.name], data))
                else:
                    all_bands[band.name]=data
    if normalized: 
        for band in bands:
            if count_band.has_key(band.name): 
                all_bands[band.name]=(all_bands[band.name]*(st_band[band.name]/count_band[band.name]))+(avgs_band[band.name]/count_band[band.name])
    return all_bands,tile_metadata 

    
@celery.task()
def obtain_median(validate_range,x,y,start,end, bands, satellite,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_Float32,months=None):
        median_bands,meta=matrix_obtain_mosaic(x,y,start,end, bands, satellite,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=data_type,months=None,normalized=True)
        print "OBTAIN MEDIAN"
        print "Terminó consulta"
        median_data=None
        for bandCONST in bands:
            #b =np.apply_along_axis(median_min,2,median_bands[band],validate_range) 
            band=bandCONST.name
            print band
            if not band in median_bands:
                continue
            print median_bands[band].shape
            if len(median_bands[band].shape)>2:
                b=np.nanmedian(median_bands[band],2)
                allNan=~np.isnan(median_bands[band])
                b[np.sum(allNan,2)<validate_range]=np.nan
                del allNan
            else:
                b=median_bands[band]
                if validate_range>1:
                    b[:]=np.nan
            if median_data is None:
                median_data=b
            else: 
                median_data=np.dstack((median_data, b))

        #print median_data.shape
        del median_bands
        return median_data,meta  

@celery.task()
def obtain_median_mosaic(validate_range,x,y,start,end, bands, satellite,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_CFloat32,months=None):
    medians,meta=obtain_median(validate_range,x,y,start,end, bands, satellite,iterations,xsize,ysize,file_format,data_type,months)
    if medians is None: 
        return "None"
    pprint(medians.shape)
    pprint(len(medians.shape))
    nf=medians.shape[0]
    nc=medians.shape[1]
    if len(medians.shape)>=3:
        nb=medians.shape[2]
    else:
        nb=1
    band_str = "+".join([band.name for band in bands])
    sat_str = "+".join([sat.name for sat in satellite])
    cache_id = [str(x),str(y),str(start),str(end),band_str,sat_str,str(xsize),str(ysize),file_format,str(iterations)]
    f_name = "_".join(cache_id)
    f_name = "res_median_"+f_name.replace(" ","_")
    c_name = f_name
    f_name = os.path.join("/tilestore/tile_cache",f_name)
    tile_metadata=meta
    numberOfBands=nb
    if numberOfBands == 0:
       return "None"

    numberOfPixelsInXDirection=nc
    print numberOfPixelsInXDirection
    numberOfPixelsInYDirection=nf
    print numberOfPixelsInYDirection
    pixels = numberOfPixelsInXDirection
    if numberOfPixelsInYDirection > numberOfPixelsInXDirection:
        pixels = numberOfPixelsInYDirection

    driver = gdal.GetDriverByName(file_format)
    if driver is None:
        print "No driver found for "+file_format
        return "None"
    raster = driver.Create(f_name+'.tif', pixels, pixels, numberOfBands, data_type, options=["BIGTIFF=YES", "INTERLEAVE=BAND"])
    raster.SetGeoTransform(tile_metadata.transform)
    raster.SetProjection(tile_metadata.projection)
    index = 1
    #medians[np.isnan(medians)]=-999
    for band in range (0,nb):
        stack_band = raster.GetRasterBand(index)
        stack_band.SetNoDataValue(-999)
        if nb==1:
            stack_band.WriteArray(medians)
        else:
            stack_band.WriteArray(medians[:,:,band])
        stack_band.ComputeStatistics(True)
        index+=1
        stack_band.FlushCache()
        del stack_band
    raster.FlushCache()
    del raster
    cache.set(c_name,f_name+".tif")
    return f_name+".tif"


def obtain_histogram_info(x,y,start,end, selectedBand, satellite):
    median_bands,meta=matrix_obtain_mosaic(x,y,start,end, [selectedBand], satellite,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_CInt16,months=None)
    median_data=None
    band=selectedBand.name
    if not(median_bands.has_key(band)):
        pprint('No data for period'+str(x)+' '+str(y)+' '+str(start)+' '+str(end))
        return None,[],0,0,0
    allNan=~np.isnan(median_bands[band])
    tileSizeArray=allNan.shape
    numberTiles=1
    if len(tileSizeArray)>=3:
        numberTiles=tileSizeArray[2]
    if numberTiles>1:
        matrixCount=np.sum(allNan,2)
    else:
        matrixCount=np.sum(allNan)
    del allNan
    histogram=np.histogram(np.ravel(matrixCount),density=False)
    bincount=np.bincount(np.ravel(matrixCount))
    min=np.min(matrixCount)
    max=np.max(matrixCount)
    return histogram,bincount,min,max,numberTiles
    
    
@celery.task()    
def obtain_forest_noforest(x, y, start_date, end_date, satellite = [Satellite.LS7], months = None, min_ok = 1, vegetation_rate = 0.5, ndvi_threshold = 0.7, slice_size = 3):

    period_ndvi,metadata = obtain_ndvi(x, y, start_date, end_date, satellite = satellite, months = months, min_ok = min_ok)
    if period_ndvi is None: 
        return "None"
    height = period_ndvi.shape[0]
    width = period_ndvi.shape[1]   
    
    nan_mask=np.isnan(period_ndvi)
    
    original_ndvi=period_ndvi.astype(float)
    original_nvdi=np.clip(original_ndvi,-1,1)
    for y1 in xrange(0, height, slice_size):
        for x1 in xrange(0, width, slice_size):
            
            x2 = x1 + slice_size
            y2 = y1 + slice_size
            
            if(x2 > width):
                x2 = width

            if(y2 > height):
                y2 = height
            
            submatrix = period_ndvi[y1:y2,x1:x2]
            ok_pixels = np.count_nonzero(~np.isnan(submatrix))
            
            submatrix[np.isnan(submatrix)]=-1

            if ok_pixels==0:
                period_ndvi[y1:y2,x1:x2] = 1    
            elif float(np.sum(submatrix>ndvi_threshold))/float(ok_pixels) >= vegetation_rate :
                period_ndvi[y1:y2,x1:x2] = 2
            else:
                period_ndvi[y1:y2,x1:x2] = 1


    period_ndvi[nan_mask] = np.nan
    composite_all=np.dstack((period_ndvi,original_ndvi))
    pprint("Max nvdi es:"+str(np.nanmax(original_ndvi)))
    pprint("Min nvdi es:"+str(np.nanmin(original_ndvi)))
    # Prepara el nombre base de los archivos de salida
    bands = [ Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED ]
    bands_str = '+'.join(each_band.name for each_band in bands)
    satellites_str = '+'.join(each_satellite.name for each_satellite in satellite)
    image_filename = ("_".join([str(x), str(y), str(start_date), str(end_date), bands_str, satellites_str])).replace(" ","_")

   # generate_rgb_image(period_ndvi, period_ndvi, period_ndvi, temp_directory, output_name = "FOREST_NOFOREST_" + image_filename, width = width, height = height, scale = 0.3)
    file=generate_geotiff_image(composite_all, width, height, "/tilestore/tile_cache/", metadata = metadata, output_name = "FOREST_NOFOREST_" + image_filename)

    return file




def obtain_ndvi(x, y, start_date, end_date, satellite = [Satellite.LS7], months = None, min_ok = 2):

    print "BEGIN NDVI PROCESS"

  
    # Lista las bandas necesarias para operar NDVI
    bands = [ Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED ]

    # Obtiene los compuestos de medianas del periodos 1
    period, metadata = obtain_median(min_ok,x, y, start_date, end_date,bands, satellite,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_Float32,months=None)
    if period is None:
        return None, metadata
    mask_nan=np.any(np.isnan(period),axis=2)

   
    # Separa los canales rojo e infrarrojo cercano
    period_red = period[:,:,0]
    period_red[mask_nan]=0
    period_nir = period[:,:,1]
    period_nir[mask_nan]=0  
    
    # Genera NDVI del periodo 1

    period_ndvi = np.true_divide( np.subtract(period_nir,period_red) , np.add(period_nir,period_red) )
    period_nvdi2=np.copy(period_ndvi)
    np.clip(period_ndvi,0,1,out=period_nvdi2)
    period_nvdi2[mask_nan]=np.nan

    return period_nvdi2, metadata





def obtain_bands_dict(x, y, start, end, bands, satellite, months=None):

    """ 
    Obtains a dict with the query results, one matrix per band
    """
    
    tiles = list_tiles(x=[x], y=[y],acq_min=start,acq_max=end,satellites=satellite,dataset_types=[DatasetType.ARG25,DatasetType.PQ25], sort=SortType.ASC)
    tile_metadata = None
    tile_count = 0
    tile_filled = False
    total_ins = 0
    all_bands={}
    
    for tile in tiles:
        if tile_filled:
           break
        if months:
            print tile.start_datetime.month
            if not tile.start_datetime.month in months:
                continue
        tile_count+=1
        dataset =  DatasetType.ARG25 in tile.datasets and tile.datasets[DatasetType.ARG25] or None
        if dataset is None:
            print "No dataset availible"
            tile_count-=1
            continue
        tile_metadata = get_dataset_metadata(dataset)
        if tile_metadata is None:
            print "NO METADATA"
            tile_count-=1
            continue
        
        pqa = DatasetType.PQ25 in tile.datasets and tile.datasets[DatasetType.PQ25] or None
        mask = None
        mask = get_mask_pqa(pqa,[PqaMask.PQ_MASK_CLEAR],mask=mask)
        band_data = get_dataset_data_masked(dataset, mask=mask,bands=bands)

        for band in band_data:
            data = np.array(band_data[band]).astype(np.float32)
            non_data=np.in1d(data.ravel(),-999).reshape(data.shape)
            data[non_data]=np.NaN 
            if all_bands.has_key(band.name):
                all_bands[band.name]=np.dstack((all_bands[band.name], data))
            else:
                all_bands[band.name]=np.array(data)
    return all_bands, tile_metadata

def ravel_compounds(compounds):

    flattened_compounds = None

    for compound in xrange(0, compounds.shape[2]):
        flattened_compound = compounds[:,:,compound].ravel()
        if flattened_compounds is None:
            flattened_compounds = flattened_compound
        else:
            flattened_compounds = np.vstack((flattened_compounds, flattened_compound))
    return flattened_compounds.T



def obtain_medians_compound(x, y, start, end, bands, satellite, months = None, validate_range = 2):
        median_bands, metadata = obtain_bands_dict(x, y, start, end, bands, satellite, months)
        print "Terminó consulta"
        if median_bands is None:
            return None, metadata
        median_data=None
        for bandCONST in bands:
            #b =np.apply_along_axis(median_min,2,median_bands[band],validate_range) 
            band=bandCONST.name
            print band
            print median_bands[band].shape
            if len(median_bands[band].shape)>2:
                b=np.nanmedian(median_bands[band],2)
                allNan=~np.isnan(median_bands[band])
                b[np.sum(allNan,2)<validate_range]=np.nan
                del allNan
            else:
                b=median_bands[band]
                if validate_range>1:
                    b[:]=np.nan
            if median_data is None:
                median_data=b
            else: 
                median_data=np.dstack((median_data, b))

        #print median_data.shape
        del median_bands
        return median_data,metadata  



@celery.task()
def obtain_convolution_nvdi(prueba,NDVI_result_final,percetage_ndvi=0.3,threshold_ndvi=0.7):
   
    print ("_______________obtain_convolution_nvdiL____________")
    
    [height,weight]=NDVI_result_final.shape
    #print ("Alto",height)
    #print ("Ancho",weight)
    test=(prueba+"entro convolucion")
    nueva_matriz=None
    
    for x1 in xrange(0,height,3):
        for y1 in xrange(0,weight,3):
            auxX=x1+3
            auxY=y1+3
            if(auxX>=height):
                auxX=height-1
            if(auxY>=weight):
                auxY=weight-1
            auxMatriz=NDVI_result_final[xrange(x1,auxX),:] [:,xrange(y1,auxY)]
            #print auxMatriz.shape
            count_pixel=auxMatriz.shape[0]*auxMatriz.shape[1]
            pixel_nan=np.count_nonzero(np.isnan(auxMatriz))
            pixel_forest=np.sum(np.where(auxMatriz>threshold_ndvi,1,0))
            if(x1==0 and y1==0):
                print("AUX_X______",auxX)
                print("AUX_Y_______",auxY)
                print("AUX_AUXM______",auxMatriz)
                print("AUX_COUPIX______",count_pixel)
                print("AUX_COU_NAN______",pixel_nan)
                print("AUX_PIX_FOR______",pixel_forest)
            
            if(count_pixel-pixel_nan>0):
                auxResult=(pixel_forest)/(count_pixel-pixel_nan)
                if(auxResult>percetage_ndvi):
                    #print ("ENTRO ERROR")
                    NDVI_result_final[x1:auxX, y1:auxY]=1
                else:
                    NDVI_result_final[x1:auxX, y1:auxY]=0
            else:
                NDVI_result_final[x1:auxX, y1:auxY]=np.nan

            if(x1==0 and y1==0):
                  print ("FINAL TEST",NDVI_result_final[xrange(x1,auxX),:] [:,xrange(y1,auxY)])

    print NDVI_result_final
          
        
    return test


def generate_geotiff_image(input_array, width, height, output_path, metadata, output_name = "oimage4", data_type = gdal.GDT_Float32 ):

    n_bands=1
    if len(input_array.shape)>=3:
        n_bands = input_array.shape[2]
    

    gtiff_driver = gdal.GetDriverByName('GTiff')
    f_name=output_path + output_name
    raster = gtiff_driver.Create( f_name+ '.tif', width, height, n_bands, eType = data_type, options=["BIGTIFF=YES", "INTERLEAVE=BAND"])

    if metadata:
        raster.SetGeoTransform(metadata.transform)

    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    raster.SetProjection(srs.ExportToWkt())

    for band in xrange(0,n_bands):
        raster_band = raster.GetRasterBand(band+1)
        raster_band.SetNoDataValue(-999)
        if n_bands==1:
            raster_band.WriteArray(input_array)
        else:
            raster_band.WriteArray(input_array[:,:,band])
        raster_band.ComputeStatistics(True)
        raster_band.FlushCache()

    raster.FlushCache()
    
    return f_name+ '.tif'

def generate_rgb_image(r_array, g_array, b_array, output_path, output_name = "oimage", width = None, height = None, scale = 1, format = "jpg"):

    input_array = np.zeros(((width*height),3))
    input_array[:,0] = r_array
    input_array[:,1] = g_array
    input_array[:,2] = b_array

    if len(input_array.shape) == 2:
        input_array = input_array.reshape((height, width, 3))

    max_value = np.nanmax(input_array)
    input_array = (input_array/max_value)*255

    output_img = Image.fromarray(np.uint8(input_array), 'RGB')

    width = int(np.ceil(output_img.size[0]*scale))
    height = int(np.ceil(output_img.size[1]*scale))
    output_img = output_img.resize((width, height))
    output_img.save(output_path + output_name + "." + format)
    
@celery.task()
def obtain_pca_png(validate_range,x,y,start1,end1,start2,end2, bands, satellite,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_CInt16,months=None):
       
        median_p1=obtain_median(validate_range,x,y,start1,end1, bands, satellite,iterations,xsize,ysize,file_format,data_type,months)
        median_p2=obtain_median(validate_range,x,y,start2,end2, bands, satellite,iterations,xsize,ysize,file_format,data_type,months)
        pickl.dump( median_p1, open( "median_p_1.p", "wb" ) )
        pickl.dump( median_p2, open( "median_p_2.p", "wb" ) )
        ##GUARDANDO DATOS MEDIANA
        component_p1=pre_process_ravel(median_p1)
        component_p2=pre_process_ravel(median_p2)

        #________________ciclo Normalizacion______________________________
        for x in xrange(0,component_p1.shape[1]):
            component_p2[:,x]=normalize(component_p1[:,x],component_p2[:,x])
        #________________ciclo mascara______________________________

        mask_comp = None
        for x in xrange(0,component_p1.shape[1]):
            if(mask_comp is None) :
                mask_comp = combine_masks(np.zeros(len(component_p1[:,x])),component_p1[:,x])
                mask_comp = combine_masks(mask_comp,component_p2[:,x])
            else:
                mask_comp = combine_masks(mask_comp,(combine_masks(component_p1[:,x],component_p2[:,x])))
        
        #________________ciclo change NAN______________________________
  
        pre_pca_bands=numpy.concatenate((component_p1,component_p2),1)
        a= pre_pca_bands.flatten()
        median_array_pre_pca=np.nanmedian(a)
        print("MEDIANA PREPCA",median_array_pre_pca)
        for x in xrange(0,pre_pca_bands.shape[1]):
            pre_pca_bands[:,x]=convert_nan_to_median(pre_pca_bands[:,x],median_array_pre_pca)

        print ("RESULTADO FINAL",pre_pca_bands.shape)
    


        print("COMPUESTO SIN NAN",pre_pca_bands)   
        print ("RESULTADO MASCARA PARA COMPARAR DATOS ",mask_comp)

        ##GUARDANDO DATOS TEST
       
        print ("GUARDE LOS DATOS")
        f_pca=PCA(pre_pca_bands)
        size_ma=f_pca.Y.T.shape
        pickl.dump( f_pca, open( "f_pca2.p", "wb" ) )
        pickl.dump( mask_comp, open( "mask_comp2.p", "wb" ) )


        presult=f_pca.Y[:,0].reshape(3705,3705)
        presult2=f_pca.Y[:,2].reshape(3705,3705)
        #normalizacion
        presult *= (255.0/presult.max())
        
        im = Image.fromarray(np.uint8(cm.gist_earth(presult)*255))
        im2 = Image.fromarray(np.uint8(cm.gist_earth(presult2)*255))
        print ("MATRIX ok2",im)
        im.save('test__TEST2.jpeg')
        im2.save('test72.png')

        
        return 0
      
@celery.task()
def obtain_median_png(validate_range,x,y,start1,end1,start2,end2, bands, satellite,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_CInt16,months=None):

        mediana= pickl.load( open( "median_p1.p", "rb" ) )      
        print("PRUEBA",prueba)
        print("PRUEBA2",prueba.shape)
        print mediana
        print mediana.shape
        #rgbArray = np.zeros((512,512,3), 'uint8')
        r=mediana[..., 0]
        g=mediana[..., 1] 
        b=mediana[..., 1] 
        print("PRUEBA",mediana)
        print("R",r)
        print("G",g)
        print("B",b)

        
        return 0



def obtain_pca_all(validate_range,x,y,start1,end1,start2,end2, bands, satellite,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_CInt16,months=None):
        print("OBTAIN PCA_ALL")
        raw_b1,meta=obtain_median(validate_range,x,y,start1,end1, bands, satellite,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_CInt16,months=None)
        median_p1=raw_b1
        nf=raw_b1.shape[0]
        nc=raw_b1.shape[1]
        nb=raw_b1.shape[2]*2

        median_p2,meta2=obtain_median(validate_range,x,y,start2,end2, bands, satellite,iterations,xsize,ysize,file_format,data_type,months)

        pickl.dump( median_p1, open( "26_median_p_1_all_f.p", "wb" ) )
        pickl.dump( median_p2, open( "26_median_p_2_all_f.p", "wb" ) )
        ##GUARDANDO DATOS MEDIANA
        component_p1=pre_process_ravel(raw_b1)
        component_p2=pre_process_ravel(median_p2)




        #________________ciclo Normalizacion______________________________
        for x in xrange(0,component_p1.shape[1]):
            component_p2[:,x]=normalize(component_p1[:,x],component_p2[:,x])
        #________________ciclo mascara______________________________

        mask_comp = None
        for x in xrange(0,component_p1.shape[1]):
            if(mask_comp is None) :
                mask_comp = component_p1[:,x]
                mask_comp = combine_masks(mask_comp,component_p2[:,x])
            else:
                mask_comp = combine_masks(mask_comp,(combine_masks(component_p1[:,x],component_p2[:,x])))
        
        #________________ciclo change NAN______________________________
  
        pre_pca_bands=numpy.concatenate((component_p1,component_p2),1)
        a= pre_pca_bands.flatten()
        median_array_pre_pca=np.nanmedian(a)
        print("MEDIANA PREPCA",median_array_pre_pca)
        for x in xrange(0,pre_pca_bands.shape[1]):
            pre_pca_bands[:,x]=convert_nan_to_median(pre_pca_bands[:,x],median_array_pre_pca)

        print ("RESULTADO FINAL",pre_pca_bands.shape)
    


        print("COMPUESTO SIN NAN",pre_pca_bands)   
        print ("RESULTADO MASCARA PARA COMPARAR DATOS ",mask_comp)

        ##GUARDANDO DATOS TEST
       
        print ("GUARDE LOS DATOS")
        f_pca=PCA(pre_pca_bands)

        size_ma=f_pca.Y.T.shape
        presult=f_pca.Y.T

        pickl.dump( f_pca, open( "26_pca_final_25.p", "wb" ) )
        pickl.dump( presult, open( "26_pca_final_trasn.p", "wb" ) )

        presult1=f_pca.Y[:,0].reshape(3705,3705)
        presult2=f_pca.Y[:,2].reshape(3705,3705)
        #normalizacion
        presult1 *= (255.0/presult1.max())
        
        im = Image.fromarray(np.uint8(cm.gist_earth(presult1)*255))
        im2 = Image.fromarray(np.uint8(cm.gist_earth(presult2)*255))
        print ("MATRIX ok2",im)
        im.save('26_presentacion.jpeg')
        im2.save('26_presentacion_norma.jpeg')


#-_-------------------_-----------------------

        km_centroids,_=kmeans(f_pca.Y, 2) #Generar los centroides
        
        print km_centroids
        """
        Guardar el archivo: 
        """
        band_str = "+".join([band.name for band in bands])
        sat_str = "+".join([sat.name for sat in satellite])
        cache_id = [str(x),str(y),str(start1),str(end1),str(start2),str(end2),band_str,sat_str,str(xsize),str(ysize),file_format,str(iterations)]
        f_name = "_".join(cache_id)
        f_name = "26_celery"+f_name.replace(" ","_")
        c_name = f_name
        driver = gdal.GetDriverByName(file_format)
        if driver is None:
            print "No driver found for "+file_format
            return "None"
        c_file=os.path.join("/tilestore/tile_cache","centroids_"+f_name+".csv")
        print c_file
        numpy.savetxt(c_file,km_centroids)
        f_name = os.path.join("/tilestore/tile_cache",f_name)
        
        raster = driver.Create(f_name+'.tif', nf, nc, nb, data_type, options=["BIGTIFF=YES", "INTERLEAVE=BAND"])
        raster.SetGeoTransform((x-0.00025, 0.00027, 0.0, y+1.0002400000000002, 0.0, -0.00027)) #Debemos obtenerlo del original, o calcularlo bien
        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS("WGS84")
        raster.SetProjection(srs.ExportToWkt())
        index = 1
        for bn in presult:
            stack_band = raster.GetRasterBand(index)
            stack_band.SetNoDataValue(-999)
            bn[numpy.isnan(bn)]=-999
            stack_band.WriteArray(bn.reshape(nf,nc))
            stack_band.ComputeStatistics(False)
            index+=1
            stack_band.FlushCache()
            del stack_band
        raster.FlushCache()
        del raster
        cache.set(c_name,f_name+".tif")
        return f_name+".tif"
        
  



#Funcion que aplica los elementos de mask2 en array1
@celery.task()
def apply_nan(array1,mask2):
    if (len(array1)==len(mask2)):
        i = 0
        while i < len(array1):
           if(np.isnan(mask2[i])):
               array1[i] = np.nan
           i+=1
        return array1
    else:
        print("ERROR DE TAMANOS DE MASCARA DIFERENTES DIFERENTES")



def generate_component_kmean(km_centroids,pca_final_with_nan):
   
    indices = [numpy.where(km_centroids<=x)[0][0] for x in pca_final_with_nan]

    print indices

    return 99

@celery.task()
def convert_nan_to_median(array1,median_array_pre_pca):
    f_result=[]

    i=0
    media=median_array_pre_pca
    #print ("media ",media)
    while i<len(array1) :
       
         
        if(np.isnan(array1[i])):
          f_result.append(media)
        else:
          f_result.append(array1[i])
        i+=1
    return f_result
     


@celery.task()
def combine_masks(mask1, mask2):
   
   if (len(mask1)==len(mask2)):
       i = 0
       while i < len(mask1):
           if(np.isnan(mask2[i])):
               mask1[i] = np.nan
           i+=1
       return mask1
   else:
        print("ERROR DE TAMANOS DE MASCARA DIFERENTES DIFERENTES")


@celery.task()
def normalize(final_composite1,final_composite2):
    desv_final_mask2=np.nanstd(final_composite2)
    mean_final_1=np.nanmean(final_composite1)
    mean_final_2=np.nanmean(final_composite2)
    temp_mask2=((final_composite2-mean_final_2)/desv_final_mask2)+mean_final_1
    return temp_mask2

   
@celery.task()
def pre_process_ravel(pre_pca):
   
    new_pca_input=None
    for d in xrange(0,pre_pca.shape[2]):
        b=pre_pca[:,:,d].ravel()
        if new_pca_input is None:
            new_pca_input=b
        else:
            new_pca_input=numpy.vstack((new_pca_input,b))
    #print ("ENVIO_VSTACK",new_pca_input.T.shape)
    return new_pca_input.T



@celery.task()
def median_min(array_bands,validate_range):
   
    count_no_nan=np.count_nonzero(np.isnan(array_bands))
    len_data=len(array_bands)
    
    if((len_data - count_no_nan)<=validate_range):
        return np.nanmedian(array_bands)
    else:
        return   np.nan  


@celery.task()
def mask_range(array_bands,validate_range):
   
    count_nan=np.count_nonzero(np.isnan(array_bands))
    len_data=len(array_bands)
    
    if((len_data - count_nan)>validate_range):
        return True
    else:
        return   False 

 
@celery.task()
def validate_mask(array_bands):
    count_nan=np.count_nonzero(np.isnan(array_bands))
    len_data=len(array_bands)     
    if count_nan!=len_data :
            return False
    else:
            return True   


@celery.task()                                                      
def obtain_mask(validate_range,x,y,start,end, bands, satellite,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_CInt16,months=None):
        mosaic_bands,meta=matrix_obtain_mosaic(x,y,start,end, bands, satellite,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_CInt16,months=None)
        mask_data=None
        for band in mosaic_bands:
            b =np.apply_along_axis(mask_range,2,mosaic_bands[band],validate_range) 
            if mask_data is None:
                mask_data=b
            else: 
                mask_data=np.dstack((mask_data, b))
        print mask_data.shape
        return mask_data   




@celery.task()
def assemble_mosaic(file_list):
    print "Assembling mosaic"
    print file_list
    
    fl = None
    try:
        if type(file_list) is list:
            
            fl = [f for f in file_list if f!="None"]
        else:
            fl = [file_list]
    except:
        fl = [file_list]
    if len(fl) <1:
        return "None"
    c_name = hashlib.sha512("_".join(fl)).hexdigest()[0:32]
    cmd = "gdalbuildvrt -hidenodata /tilestore/tile_cache/"+c_name+".vrt "+" ".join(fl)
    print cmd
    os.system(cmd)
    if not os.path.exists("/tilestore/tile_cache/"+c_name+".vrt"):
        return "None"
    res = "/tilestore/tile_cache/"+c_name+".vrt"
    ret_prod = []
    ret_prod.append(res)
    for fi in fl:
        ret_prod.append(fi)
    return ret_prod

@celery.task()
def get_bounds(input_file):
    in_file = None
    print input_file
    if isinstance(input_file,(str)):
        if input_file == "None":
            return "None"
        else:
            in_file = input_file
    else:
        in_file = input_file[0]
    ds = gdal.Open(in_file)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    gt = ds.GetGeoTransform()
    bb1 = originx = gt[0]
    bb4 = originy = gt[3]
    pixelWidth = gt[1]
    pixelHeight = gt[5]
    width = cols*pixelWidth
    height = rows*pixelHeight
    bb3 = originx+width
    bb2 = originy+height
    del ds
    return str(bb2)+","+str(bb1)+","+str(bb4)+","+str(bb3)

@celery.task()
def translate_files(file_list,file_format,output_scale,output_size,output_datatype,output_bands,additional_arguments=None):
    print file_list
    fl = None
    try:
        if type(file_list) is list:
            
            fl = [f for f in file_list if f!="None"]
        else:
            fl = [file_list]
    except:
        fl = [file_list]
    addy = ""
    b_arg= ""
    if output_bands is not None:
        b_arg = " ".join(["-b "+str(b) for b in output_bands])
    res = []
    if additional_arguments:
        addy = " "+" ".join(additional_arguments)
    for f in fl:
        print "Translating "+f
        ds = gdal.Open(f)
        rc = ds.RasterCount
        if output_bands is not None:
            if rc < len(output_bands):
                print "Less bands than requested!"
                b_arg = "-b 1"
        del ds
        out_scale = ""
        out_dt = ""
        out_size = ""
        b_l_arg = ""
        if output_scale is not None and b_arg != "-b 1":
            out_scale = " -scale "+output_scale
        if output_datatype is not None:
            out_dt = " -ot "+output_datatype
        if output_size is not None:
            out_size = " -outsize "+output_size
        if output_bands is not None and b_arg != "-b 1":
            b_l_arg = " "+b_arg
        b_tmp = ""
        if output_bands is not None:
            b_tmp = "_".join([str(b) for b in output_bands])
        c_arr = [f,str(file_format),str(output_scale),str(output_size),str(output_datatype),b_tmp,addy]
        c_name = "_".join(c_arr)
        c_name = hashlib.sha512(c_name).hexdigest()[0:32]
        tar_img = os.path.join("/tilestore/tile_cache/",c_name+FILE_EXT[file_format])
        tar_img_marked = os.path.join("/tilestore/tile_cache/",c_name+"_marked"+FILE_EXT[file_format])
        
        
        cmd = "gdal_translate -of "+file_format+out_dt+out_scale+out_size+b_l_arg+addy+" "+f+" "+tar_img
        print cmd
        os.system(cmd)
        if os.path.exists(tar_img):
            if file_format == "png" or file_format == "PNG":
                cmd = "convert -transparent \"#000000\" "+tar_img+" "+tar_img
                os.system(cmd);
                cmd = "convert "+tar_img+" -background red -alpha remove "+tar_img_marked
                os.system(cmd)
            res.append(tar_img)
            res.append(tar_img_marked)
    return res
    
@celery.task()
def apply_color_table_to_files(file_list,output_band,color_table):
    print file_list
    fl = None
    try:
        if type(file_list) is list:
            
            fl = [f for f in file_list if f!="None"]
        else:
            fl = [file_list]
    except:
        fl = [file_list]
    
    
    res = []
    
    for f in fl:
        print "Coloring "+f
        c_arr = [f,str(output_band),color_table]
        c_name = "_".join(c_arr)
        c_name = hashlib.sha512(c_name).hexdigest()[0:32]
        tar_img = os.path.join("/tilestore/tile_cache/",c_name+".tif")
        tmp_img = os.path.join("/tilestore/tile_cache/",c_name)
        cmd = "gdal_translate "+f+" "+tmp_img+"_"+str(output_band)+".tif"+" -b "+str(output_band)
        os.system(cmd)
        print "Applying color table"
        cmd = "gdaldem color-relief -of GTiff "+tmp_img+"_"+str(output_band)+".tif"+" "+color_table+" "+tar_img
        print cmd
        os.system(cmd)
        
        if os.path.exists(tar_img):
            #cmd = "convert -transparent \"#000000\" "+tar_img+" "+tar_img
            #os.system(cmd);            
            res.append(tar_img)
    return res

@celery.task()
def preview_cloudfree_mosaic(x,y,start,end, bands, satellite,iterations=0,xsize=2000,ysize=2000,file_format="GTiff",data_type=gdal.GDT_CInt16):
    def resize_array(arr,size):
       r = numpy.array(arr).astype(numpy.int16)
       i = Image.fromarray(r)
       i2 = i.resize(size,Image.NEAREST)
       r2 = numpy.array(i2)
       del i2
       del i
       del r
       return r2
    StartDate = start
    EndDate = end
    
    best_data = {}
    band_str = "+".join([band.name for band in bands])
    sat_str = "+".join([sat.name for sat in satellite])
    cache_id = ["preview",str(x),str(y),str(start),str(end),band_str,sat_str,str(xsize),str(ysize),file_format,str(iterations)]
    f_name = "_".join(cache_id)
    f_name = f_name.replace(" ","_")
    c_name = f_name
    cached_res = cache.get(c_name)
    if cached_res:
        return str(cached_res)
    f_name = os.path.join("/tilestore/tile_cache",f_name)
    tiles = list_tiles(x=[x], y=[y],acq_min=StartDate,acq_max=EndDate,satellites=satellite,dataset_types=[DatasetType.ARG25,DatasetType.PQ25], sort=SortType.ASC)
    tile_metadata = None
    tile_count = 0
    tile_filled = False
    for tile in tiles:
        if tile_filled:
           break
        print "merging on tile "+str(tile.x)+", "+str(tile.y)
        tile_count+=1
        dataset =  DatasetType.ARG25 in tile.datasets and tile.datasets[DatasetType.ARG25] or None
        if dataset is None:
            print "No dataset availible"
            tile_count-=1
            continue
        tile_metadata = get_dataset_metadata(dataset)
        if tile_metadata is None:
            print "NO METADATA"
            tile_count-=1
            continue
        pqa = DatasetType.PQ25 in tile.datasets and tile.datasets[DatasetType.PQ25] or None
        mask = None
        mask = get_mask_pqa(pqa,[PqaMask.PQ_MASK_CLEAR],mask=mask)
        band_data = get_dataset_data_masked(dataset, mask=mask,bands=bands)
        swap_arr = None
        for band in band_data:
            if not band in best_data:
                print "Adding "+band.name
                bd = resize_array(band_data[band],(2000,2000))
                print bd
                best_data[band]=bd
                del bd
            else:
                best = resize_array(best_data[band],(2000,2000))
               
                swap_arr=numpy.in1d(best.ravel(),-999).reshape(best.shape)
                b_data = numpy.array(band_data[band])
                best[swap_arr]=b_data[swap_arr]
                best_data[band]=numpy.copy(best)
                del b_data
                del best
        del swap_arr
        if iterations > 0:
            if tile_count>iterations:
                print "Exiting after "+str(iterations)+" iterations"
                break
    numberOfBands=len(bands)
    if numberOfBands == 0:
       return "None"
    if bands[0] not in best_data:
       print "No data was merged for "+str(x)+", "+str(y)
       return "None"
    numberOfPixelsInXDirection=len(best_data[bands[0]])
    numberOfPixelsInYDirection=len(best_data[bands[0]][0])   
    if tile_count <1:
        print "No tiles found for "+str(x)+", "+str(y)
        return "None"
    driver = gdal.GetDriverByName(file_format)
    if driver is None:
        print "No driver found for "+file_format
        return "None"
    print f_name+'.tif'
    raster = driver.Create(f_name+'.tif', numberOfPixelsInXDirection, numberOfPixelsInYDirection, numberOfBands, data_type, options=["BIGTIFF=YES", "INTERLEAVE=BAND"])
    gt = tile_metadata.transform
    gt2 = (gt[0],gt[1]*2.0,gt[2],gt[3],gt[4],gt[5]*2.0)
    tile_metadata.transform = gt2
    raster.SetGeoTransform(tile_metadata.transform)
    print tile_metadata.transform
    raster.SetProjection(tile_metadata.projection)
    index = 1
    for band in bands:
        stack_band = raster.GetRasterBand(index)
        stack_band.SetNoDataValue(-999)
        stack_band.WriteArray(best_data[band])
        stack_band.ComputeStatistics(True)
        index+=1
        stack_band.FlushCache()
        del stack_band
    raster.FlushCache()
    del raster
    cache.set(c_name,f_name+".tif")
    return f_name+".tif"

    

import hashlib
#TODO: Implement proper masking support
@celery.task()
def obtain_file_from_math(input_file,expressions_list,file_format="GTiff",data_type=gdal.GDT_CFloat32,input_ndv=-999,output_ndv=-999):
    """
    ex. band4,band3, (band4-band3)/(band4+band3) AKA NDVI
    """
    """
    Read in file
    """
    if input_file == "None":
        return "None"
    driver = gdal.GetDriverByName(file_format)
    ds = gdal.Open(input_file,0)
    if ds is None:
        return "None"
    arrays = []
    
    band_count = ds.RasterCount
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    exp_str = "_".join(expressions_list)
    cache_id = [os.path.splitext(os.path.basename(input_file))[0],exp_str,str(xsize),str(ysize),file_format]
    f_name = "_".join(cache_id)
    f_name = hashlib.sha512(f_name).hexdigest()[0:32]
    c_name = f_name
    cached_res = cache.get(c_name)
    if cached_res:
        return cached_res
    f_name = os.path.join("/tilestore/tile_cache",f_name)
    for i in range(band_count):
        RB = ds.GetRasterBand(i+1)
        arrays.append(RB.ReadAsArray(0,0,xsize,ysize).astype(numpy.float32))
        del RB
    
    var_identifier = "A"+''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    #test if we've used this id in this scope
    var_test = var_identifier+"_band1"
    while var_test in globals():
        var_identifier = "A"+''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        var_test = var_identifier+"_band1"
    for band_num in range(len(arrays)):
        globals()[var_identifier+'_band'+str(band_num+1)]=arrays[band_num]
    results = []
    expressions = [expression.replace("band",var_identifier+"_band") for expression in expressions_list]
    for expression in expressions:
        results.append(ne.evaluate(expression))
    
    raster = driver.Create(f_name+'.tif', xsize, ysize, len(expressions_list), data_type, options=["BIGTIFF=YES", "INTERLEAVE=BAND"])
    raster.SetGeoTransform(gt)
    raster.SetProjection(proj)
    index = 1
    for band in results:
        stack_band = raster.GetRasterBand(index)
        stack_band.SetNoDataValue(output_ndv)
        stack_band.WriteArray(band)
        stack_band.ComputeStatistics(True)
        index+=1
        stack_band.FlushCache()
        del stack_band
    raster.FlushCache()
    del raster
    del ds
    del results
    cache.set(c_name,f_name+".tif")
    return f_name+".tif"
    

@celery.task()
def shrink_raster_file(input_file,size=(2000,2000)):
    if len(size)!=2:
        return "None"
    if input_file=="None":
        return "None"
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    if size[0] ==0 or size[1]==0:
        return "None"
    gdal.AllRegister()
    c_arr = [file_name,str(size)]
    c_name = "_".join(c_arr)
    c_name = c_name.replace(" ","_")
    c_name = c_name.replace(",","")
    c_name = c_name.replace("(","")
    c_name = c_name.replace(")","")
    f_name = c_name+".tif"
    f_name = os.path.join("/tilestore/tile_cache",f_name)
    ds = gdal.Open(input_file,0)
    band_count = ds.RasterCount
    if band_count == 0:
        return "None"
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ndv = ds.GetRasterBand(1).GetNoDataValue()
    dt = ds.GetRasterBand(1).DataType
    bands = []
    for i in range(band_count):
        RB = ds.GetRasterBand(i+1)
        r = numpy.array(RB.ReadAsArray(0,0,xsize,ysize)).astype(numpy.float32)
        print r
        i = Image.fromarray(r)
        i2 = i.resize(size,Image.NEAREST)
        bands.append(numpy.array(i2))
        del i2
        del i
        del r
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(f_name, size[0], size[1], band_count, dt, options=["BIGTIFF=YES", "INTERLEAVE=BAND"])
    raster.SetGeoTransform(gt)
    raster.SetProjection(proj)
    index = 1
    for band in bands:
        stack_band = raster.GetRasterBand(index)
        stack_band.SetNoDataValue(ndv)
        stack_band.WriteArray(band)
        stack_band.ComputeStatistics(True)
        index+=1
        stack_band.FlushCache()
        del stack_band
    raster.FlushCache()
    del raster
    return f_name

@celery.task()
def merge_files_on_value(input_files_list,merge_value=-999, input_ndv=-999,output_ndv=-999):
    input_files = input_files_list
    input_files = [fl for fl in input_files if fl != "None"]
    if len(input_files)<2:
        if len(input_files)==1:
            return input_files[0]
        else:
            return "None"
    
    file_name_list = [os.path.splitext(os.path.basename(in_file))[0] for in_file in input_files]
    file_names_str = "_".join(file_name_list)
    c_name_arr = [file_names_str,str(merge_value),str(input_ndv),str(output_ndv)]
    c_name= "_".join(c_name_arr)
    f_name = c_name+".tif"
    f_name = os.path.join("/tilestore/tile_cache",f_name)
    gdal.AllRegister()
    arrays = []
    ds = None
    ndv_array = None
    swap_array = None
    xsize = 0
    ysize = 0
    gt = None
    proj = None
    band_count = 0
    ds = gdal.Open(file_path,0)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    band_count = ds.RasterCount
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    """
    Load the first file
    """
    for i in range(band_count):
        RB = ds.GetRasterBand(i+1)
        arrays.append(RB.ReadAsArray(0,0,xsize,ysize))
        del RB
        ds = None
    for file_path in input_files[1:]:
        ds = gdal.Open(file_path,0)
        if ds.RasterCount == band_count:
            for i in range(band_count):
                RB = ds.GetRasterBand(i+1)
                RA = RB.ReadAsArray(0,0,xsize,ysize)
                ndv_array = numpy.in1d(arrays[0].ravel(),ndv).reshape(arrays[0].shape)
                swap_array = numpy.in1d(arrays[0].ravel(),merge_value).reshape(arrays[0].shape)
                arrays[i][swap_array]=RA[swap_array]
                arrays[i][ndv_array]=output_ndv
                del RB
                del RA
                ndv_array = None
                swap_array = None
        ds = None
        
    """
    Write the merged file
    """
    raster = driver.Create(f_name+'.tif', xsize, ysize, band_count, gdal.GDT_CFloat32, options=["BIGTIFF=YES", "INTERLEAVE=BAND"])
    raster.SetGeoTransform(gt)
    raster.SetProjection(proj)
    index = 1
    for band in arrays:
        stack_band = raster.GetRasterBand(index)
        stack_band.SetNoDataValue(output_ndv)
        stack_band.WriteArray(band)
        stack_band.ComputeStatistics(True)
        index+=1
        stack_band.FlushCache()
        del stack_band
    raster.FlushCache()
    del raster
    return f_name
        
        
        


@celery.task()
def merge_2files_on_value(input_file1, input_file2, merge_value=-999, input_ndv=-999,output_ndv=-999):
    driver = gdal.GetDriverByName(file_format)
    ds1 = gdal.Open(input_file1,0)
    if ds1 is None:
        return "None"
    ds2 = gdal.Open(input_file2,0)
    if ds2 is None:
        return "None"
    arrays1 = []
    arrays2 = []
    band_count = ds1.RasterCount
    xsize = ds1.RasterXSize
    ysize = ds1.RasterYSize
    gt = ds1.GetGeoTransform()
    proj = ds1.GetProjection()
    for i in range(band_count):
        RB = ds1.GetRasterBand(i+1)
        arrays1.append(RB.ReadAsArray(0,0,xsize,ysize))
        del RB
    for i in range(band_count):
        RB = ds2.GetRasterBand(i+1)
        arrays2.append(RB.ReadAsArray(0,0,xsize,ysize))
        del RB
    for i in arrays1:
        ndv_array = numpy.in1d(arrays1[0].ravel(),ndv).reshape(arrays1[0].shape)
        swap_array = numpy.in1d(arrays1[0].ravel(),merge_value).reshape(arrays1[0].shape)
        arrays1[i][swap_array]=arrays2[i][swap_array]
        arrays1[i][ndv_array]=output_ndv
        del ndv_array
        del swap_array
    del arrays2
    cache_id = [os.path.splitext(os.path.basename(input_file1))[0],os.path.splitext(os.path.basename(input_file2))[0],str(merge_value),str(input_ndv),str(output_ndv)]
    f_name = "_".join(cache_id)
    f_name = hashlib.sha512(f_name).hexdigest()[0:32]
    f_name = os.path.join("/tilestore/tile_cache",f_name)
    raster = driver.Create(f_name+'.tif', xsize, ysize, band_count, data_type, options=["BIGTIFF=YES", "INTERLEAVE=BAND"])
    raster.SetGeoTransform(gt)
    raster.SetProjection(proj)
    index = 1
    for band in arrays1:
        stack_band = raster.GetRasterBand(index)
        stack_band.SetNoDataValue(output_ndv)
        stack_band.WriteArray(band)
        stack_band.ComputeStatistics(True)
        index+=1
        stack_band.FlushCache()
        del stack_band
    raster.FlushCache()
    del raster
    del ds1
    del ds2
    return f_name+".tif"

@celery.task()
def obtain_pca_test(validate_range,x,y,start1,end1,start2,end2, bands, satellite,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_CFloat32,months=None):
        print("OBTAIN PCA_ALL")
        medians,meta=obtain_median(validate_range,x,y,start1,end1, bands, satellite,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_CFloat32,months=None)
        median_p2,meta2=obtain_median(validate_range,x,y,start2,end2, bands, satellite,iterations,xsize,ysize,file_format,data_type,months)
        if medians is None or median_p2 is None:
            return "None"
        nf=medians.shape[0]
        nc=medians.shape[1]
        nb=medians.shape[2]*2
        mask_nan=np.any(np.isnan(np.concatenate((medians, median_p2),axis=2)),axis=2)
        ##GUARDANDO DATOS MEDIANA_APLANANDO
        component_p1=pre_process_ravel(medians)
        component_p2=pre_process_ravel(median_p2)

        #________________ciclo Normalizacion______________________________
        for xat in xrange(0,component_p1.shape[1]):
            component_p2[:,xat]=normalize(component_p1[:,xat],component_p2[:,xat])
  
        pre_pca_bands=numpy.concatenate((component_p1,component_p2),1)
        
        for xat in xrange(0,pre_pca_bands.shape[1]):
            a=pre_pca_bands[:,xat]
            a[np.isnan(a)]=np.nanmedian(a)
            pre_pca_bands[:,xat]=a
        f_pca=PCA(pre_pca_bands)
        del medians
        del median_p2
        presult=f_pca.Y.T

#-_-------------------_-----------------------

        """
        Guardar el archivo: 
        """
        band_str = "+".join([band.name for band in bands])
        sat_str = "+".join([sat.name for sat in satellite])
        cache_id = [str(x),str(y),str(start1),str(end1),str(start2),str(end2),band_str,sat_str,str(xsize),str(ysize),file_format,str(iterations)]
        f_name = "_".join(cache_id)
        f_name = "pca_"+f_name.replace(" ","_")
        c_name = f_name
        driver = gdal.GetDriverByName(file_format)
        if driver is None:
            print "No driver found for "+file_format
            return "None"
        
        f_name = os.path.join("/tilestore/tile_cache/",f_name)
        t=max(nf,nc)
        raster = driver.Create(f_name+'.tif', t, t, nb, data_type, options=["BIGTIFF=YES", "INTERLEAVE=BAND"])
        #raster.SetGeoTransform((x-0.00025, 0.00027, 0.0, y+1.0002400000000002, 0.0, -0.00027)) #Debemos obtenerlo del original, o calcularlo bien
        srs = osr.SpatialReference()

        raster.SetGeoTransform(meta.transform)
        #raster.SetProjection(tile_metadata.projection)
        srs.SetWellKnownGeogCS("WGS84")
        raster.SetProjection(srs.ExportToWkt())
        index = 1
        for bn in presult:
            stack_band = raster.GetRasterBand(index)
            stack_band.SetNoDataValue(-999)
            bn=bn.reshape(nf,nc)
            bn[mask_nan]=np.nan
            stack_band.WriteArray(bn)
            stack_band.ComputeStatistics(True)
            index+=1
            stack_band.FlushCache()
            del stack_band
        raster.FlushCache()
        del presult
        del f_pca
        cache.set(c_name,f_name+".tif")
        return f_name+".tif"

@celery.task()
def obtain_pca_2002_2014L8(x,y):
        validate_range=1
        st = datetime.strptime('2002-01-01','%Y-%m-%d')
        en = datetime.strptime('2002-12-31','%Y-%m-%d')
        st2 = datetime.strptime('2014-01-01','%Y-%m-%d')
        en2 = datetime.strptime('2014-12-31','%Y-%m-%d')
        file_format="GTiff"
        data_type=gdal.GDT_CFloat32
        iterations=0
        bands1=[ Ls57Arg25Bands.RED, Ls57Arg25Bands.NEAR_INFRARED, Ls57Arg25Bands.SHORT_WAVE_INFRARED_1,Ls57Arg25Bands.SHORT_WAVE_INFRARED_2]
        satellite1=[Satellite.LS7]
        medians,meta=obtain_median(validate_range,x,y,st,en, bands1, satellite1,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_CFloat32,months=None)
        print "consulta 1"
        nf=medians.shape[0]
        nc=medians.shape[1]
        nb=medians.shape[2]*2
        
        bands2=[Ls8Arg25Bands.RED, Ls8Arg25Bands.NEAR_INFRARED, Ls8Arg25Bands.SHORT_WAVE_INFRARED_1, Ls8Arg25Bands.SHORT_WAVE_INFRARED_2]
        satellite2=[Satellite.LS8]
        median_p2,meta2=obtain_median(validate_range,x,y,st2,en2, bands2, satellite2,iterations=0,xsize=4000,ysize=4000,file_format="GTiff",data_type=gdal.GDT_CFloat32,months=None)
        print "consulta 2"
        
        mask_nan=np.any(np.isnan(np.concatenate((medians, median_p2),axis=2)),axis=2)
        ##GUARDANDO DATOS MEDIANA_APLANANDO
        component_p1=pre_process_ravel(medians)
        component_p2=pre_process_ravel(median_p2)

        #________________ciclo Normalizacion______________________________
        for xat in xrange(0,component_p1.shape[1]):
            component_p2[:,xat]=normalize(component_p1[:,xat],component_p2[:,xat])
  
        pre_pca_bands=numpy.concatenate((component_p1,component_p2),1)
        
        for xat in xrange(0,pre_pca_bands.shape[1]):
            a=pre_pca_bands[:,xat]
            a[np.isnan(a)]=np.nanmedian(a)
            pre_pca_bands[:,xat]=a
        f_pca=PCA(pre_pca_bands)
        del medians
        del median_p2
        presult=f_pca.Y.T

#-_-------------------_-----------------------

        """
        Guardar el archivo: 
        """
        band_str = "+".join([band.name for band in bands1])
        sat_str = "+".join([sat.name for sat in satellite1])
        cache_id = [str(x),str(y),str(st),str(en),str(st2),str(en2),band_str,sat_str,file_format,str(iterations)]
        f_name = "_".join(cache_id)
        f_name = "pca_"+f_name.replace(" ","_")
        c_name = f_name
        driver = gdal.GetDriverByName(file_format)
        if driver is None:
            print "No driver found for "+file_format
            return "None"
        
        f_name = os.path.join("/tilestore/tile_cache/",f_name)
        t=max(nf,nc)
        raster = driver.Create(f_name+'.tif', t, t, nb, data_type, options=["BIGTIFF=YES", "INTERLEAVE=BAND"])
        
        srs = osr.SpatialReference()

        raster.SetGeoTransform(meta.transform)
        #raster.SetProjection(tile_metadata.projection)
        srs.SetWellKnownGeogCS("WGS84")
        raster.SetProjection(srs.ExportToWkt())
        index = 1
        for bn in presult:
            stack_band = raster.GetRasterBand(index)
            stack_band.SetNoDataValue(-999)
            bn=bn.reshape(nf,nc)
            bn[mask_nan]=np.nan
            stack_band.WriteArray(bn)
            stack_band.ComputeStatistics(True)
            index+=1
            stack_band.FlushCache()
            del stack_band
        raster.FlushCache()
        del presult
        del f_pca
        cache.set(c_name,f_name+".tif")
        return f_name+".tif"           