"""GeoTiff相关工具"""

from osgeo import gdal
import os


# 读取tiff文件
def ReadGeoTIFF(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据

    im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    im_proj = dataset.GetProjection()  # 获取投影信息
    return im_data, im_geotrans, im_proj


# 写入tiff文件
def CreateGeoTiff(outRaster, image, geo_transform, projection, dtype='float', compress=False):
    no_bands = 0
    rows = 0
    cols = 0

    driver = gdal.GetDriverByName('GTiff')
    if len(image.shape) == 2:
        no_bands = 1
        rows, cols = image.shape
    elif len(image.shape) == 3:
        no_bands, rows, cols = image.shape

    if dtype == 'int16' or dtype == 'int':
        dtype = gdal.GDT_Int16
    elif dtype == 'int32':
        dtype = gdal.GDT_Int32
    else:
        dtype = gdal.GDT_Float32
    if compress:
        DataSet = driver.Create(outRaster, cols, rows, no_bands, dtype,
                                options=["TILED=YES", "COMPRESS={0}".format("LZW")])
    else:
        DataSet = driver.Create(outRaster, cols, rows, no_bands, dtype)

    DataSet.SetGeoTransform(geo_transform)
    DataSet.SetProjection(projection)

    if no_bands == 1:
        DataSet.GetRasterBand(1).WriteArray(image)  # 写入数组数据
    else:
        for i in range(no_bands):
            DataSet.GetRasterBand(i + 1).WriteArray(image[i])
    del DataSet


# 压缩tiff文件
def CompressGeoTiff(path, method="LZW"):
    """使用gdal进行文件压缩，
          LZW方法属于无损压缩"""
    dataset = gdal.Open(path)
    driver = gdal.GetDriverByName('GTiff')
    target_path = path.replace('.tif', '_temp.tif')

    driver.CreateCopy(target_path, dataset, strict=1, options=["TILED=YES", "COMPRESS={0}".format(method)])
    os.remove(path)
    os.rename(target_path, path)
    del dataset
