import xarray as xr
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import shapely.geometry as sgeom
import cartopy.feature as cfeature
from copy import copy

# Define functions for plotting

def find_side(ls, side):
    """
    Given a shapely LineString which is assumed to be rectangular, return the
    line corresponding to a given side of the rectangle.
    
    """
    minx, miny, maxx, maxy = ls.bounds
    points = {'left': [(minx, miny), (minx, maxy)],
              'right': [(maxx, miny), (maxx, maxy)],
              'bottom': [(minx, miny), (maxx, miny)],
              'top': [(minx, maxy), (maxx, maxy)],}
    return sgeom.LineString(points[side])


def lambert_xticks(ax, ticks):
    """
    Draw ticks on the bottom x-axis of a Lambert Conformal projection.
    
    """
    te = lambda xy: xy[0]
    lc = lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2], b[3], n))).T
    xticks, xticklabels = _lambert_ticks(ax, ticks, 'bottom', lc, te)
    ax.xaxis.tick_bottom()
    ax.set_xticks(xticks)
    ax.set_xticklabels([ax.xaxis.get_major_formatter()(xtick) for xtick in xticklabels])
    

def lambert_yticks(ax, ticks):
    """
    Draw ticks on the left y-axis of a Lambert Conformal projection.
    
    """
    te = lambda xy: xy[1]
    lc = lambda t, n, b: np.vstack((np.linspace(b[0], b[1], n), np.zeros(n) + t)).T
    yticks, yticklabels = _lambert_ticks(ax, ticks, 'left', lc, te)
    ax.yaxis.tick_left()
    ax.set_yticks(yticks)
    ax.set_yticklabels([ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels])

def _lambert_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    """
    Get the tick locations and labels for an axis of a Lambert Conformal projection.
    
    """
    outline_patch = sgeom.LineString(ax.outline_patch.get_path().vertices.tolist())
    axis = find_side(outline_patch, tick_location)
    n_steps = 30
    extent = ax.get_extent(ccrs.PlateCarree())
    _ticks = []
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        proj_xyz = ax.projection.transform_points(ccrs.Geodetic(), xy[:, 0], xy[:, 1])
        xyt = proj_xyz[..., :2]
        ls = sgeom.LineString(xyt.tolist())
        locs = axis.intersection(ls)
        if not locs:
            tick = [None]
        else:
            tick = tick_extractor(locs.xy)
        _ticks.append(tick[0])
    # Remove ticks that aren't visible:    
    ticklabels = copy(ticks)
    while True:
        try:
            index = _ticks.index(None)
        except ValueError:
            break
        _ticks.pop(index)
        ticklabels.pop(index)
    return _ticks, ticklabels

def plot_250hPa_winds(lon, lat, u, v, wspd, mode):
    """
    Plot filled contours overlayed with vectors

    Input
    -------
    lon = lon values extracted from xarray dataset (1-D)

    lat = lat values extracted from xarray dataset (1-D)

    u = U-wind at 250 hPa, shape = lon X lat

    v = V-wind at 250 hPa, shape = lon X lat
    
    wspd = Wind speed at 250 hPa, shape = lon X lat
    
    mode = 'A' for anomaly data, 'LM' for long term means, and 'EM' for extreme precipitation days
    
    Output
    --------
    matplotlib figure with filled contours of wind speed overlayed with wind vectors
    
    """
    # change data and lon to cyclic coordinates
    u, lon_new = add_cyclic_point(u.values, coord = lon.values)
    v, lon_new = add_cyclic_point(v.values, coord = lon.values)
    wspd, lon = add_cyclic_point(wspd.values, coord = lon.values)
    
    # Create a figure
    fig = plt.figure(figsize = (10, 5))
    
    # Set the GeoAxes to the PlateCarree projection
    ax = plt.axes(projection = ccrs.PlateCarree())
    
    # Add coastlines
    ax.coastlines('50m', linewidth = 0.8)
    
    # Assign data for filled contour    
    data = wspd
    
    if mode == 'EM' or mode == 'LM':
        
        # Plot filled contours
        plt.contourf(lon, lat, data, 20, transform = ccrs.PlateCarree(), 
                     cmap = get_cmap("viridis"))
        
        # Add a color bar
        cbar = plt.colorbar(ax = ax, shrink = .75)
        cbar.ax.set_ylabel('m/s', fontsize = 18)
        
        # Plot the vectors and reference vector
        rd = 5 #regrid_delta
        quiver = plt.quiver(lon[::rd], lat[::rd], u[::rd, ::rd], v[::rd, ::rd], 
                            transform = ccrs.PlateCarree(), headwidth = 5., headlength = 5.)
        ax.quiverkey(quiver, X = 0.9, Y = 1.03, U = 20., label = '20 m/s',
                     coordinates='axes', labelpos='E')
    
    elif mode == 'A':
        
        # Plot filled contours
        maxval, minval = np.abs(np.amax(data)), np.abs(np.amin(data))
        normmax = np.amax([maxval, minval])
        norm = mpl.colors.Normalize(vmin = -normmax, vmax = normmax)
        plt.contourf(lon, lat, data, 20, transform = ccrs.PlateCarree(), 
                     norm = norm, cmap = get_cmap("RdBu_r"))
        
        # Add a color bar
        cbar = plt.colorbar(ax = ax, shrink = .75)
        cbar.ax.set_ylabel('m/s', fontsize = 18)
        
        # Plot the vectors and reference vector
        rd = 5 #regrid_delta
        quiver = plt.quiver(lon[::rd], lat[::rd], u[::rd, ::rd], v[::rd, ::rd], 
                            transform = ccrs.PlateCarree(), headwidth = 5., headlength = 5.)
        ax.quiverkey(quiver, X = 0.9, Y = 1.03, U = 3., label = '3 m/s', 
                     coordinates = 'axes', labelpos = 'E')
    
    # *must* call draw in order to get the axis boundary used to add ticks:
    fig.canvas.draw()

    # Add the tick marks
    xticks = np.arange(0., 360., 30.)
    yticks = np.arange(-90., 100., 15.)
    
    # Label the end-points of the gridlines using the custom tick makers:
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    lambert_xticks(ax, xticks)
    lambert_yticks(ax, yticks)
    
    # Set title and figure name
    if mode == 'LM':
        plt.title('250 hPa Winds'+'\n'+'long term mean', fontsize=18)
        pname = 'p250_longterm.png'
    elif mode == 'EM':
        plt.title('250 hPa Winds'+'\n'+'extreme precipitation days', fontsize=18)
        pname = 'p250_extreme.png'
    elif mode == 'A':
        plt.title('250 hPa Winds'+'\n'+'anomaly fields', fontsize=18)
        pname = 'p250_anom.png'
        
    ax.set_global(); ax.gridlines();
    plt.tight_layout()
    #plot_dir = '/mnt/a/u/sciteam/chug/Laplata_tracers/plots/dipole_assessment/'
    #pname = plot_dir + name + '.png'
    plt.savefig(pname, bbox_inches = 'tight')
    plt.show()

def plot_500hPa_winds_geopot(lon, lat, u, v, z, mode):
    
    """
    Plot filled contours overlayed with vectors

    Input
    -------
    lon = lon values extracted from xarray dataset (1-D)

    lat = lat values extracted from xarray dataset (1-D)

    u = U-wind at 500 hPa, shape = lon X lat

    v = V-wind at 500 hPa, shape = lon X lat
    
    z = Geopotential height at 500 hPa, shape = lon X lat
    
    mode = 'A' for anomaly data, 'LM' for long term means, and 'EM' for extreme precipitation days
    
    Output
    --------
    matplotlib figure with filled contours of geopotential height overlayed with wind vectors
    
    """
    # change data and lon to cyclic coordinates
    u, lon_new = add_cyclic_point(u.values, coord = lon.values)
    v, lon_new = add_cyclic_point(v.values, coord = lon.values)
    z, lon = add_cyclic_point(z.values, coord = lon.values)
    
    # Create a figure
    fig = plt.figure(figsize = (10, 5))
    
    # Set the GeoAxes to the PlateCarree projection
    ax = plt.axes(projection = ccrs.PlateCarree())
    
    # Add coastlines
    ax.coastlines('50m', linewidth=0.8)
    
    # Assign data for filled contour    
    data = z
    
    if mode == 'EM' or mode == 'LM':
        # Plot filled contours
        plt.contourf(lon, lat, data, 20, transform = ccrs.PlateCarree(), 
                     cmap = get_cmap("viridis"))
        
        # Add a color bar
        cbar = plt.colorbar(ax = ax, shrink = .75)
        cbar.ax.set_ylabel('m', fontsize = 18)
        
        # Plot the vectors and reference vector
        rd = 5 #regrid_delta
        quiver = plt.quiver(lon[::rd], lat[::rd], u[::rd, ::rd], v[::rd, ::rd], 
                            transform = ccrs.PlateCarree(), headwidth = 5., headlength = 5.)
        ax.quiverkey(quiver, X = 0.9, Y = 1.03, U = 10., label = '10 m/s', 
                     coordinates = 'axes', labelpos = 'E')
    
    elif mode == 'A':
        # Plot filled contours
        maxval, minval = np.abs(np.amax(data)), np.abs(np.amin(data))
        normmax = np.amax([maxval, minval])
        norm = mpl.colors.Normalize(vmin = -normmax, vmax = normmax)
        plt.contourf(lon, lat, data, 20, transform = ccrs.PlateCarree(), 
                     norm = norm, cmap = get_cmap("RdBu_r"))
        
        # Add a color bar
        cbar = plt.colorbar(ax = ax, shrink = .75)
        cbar.ax.set_ylabel('m', fontsize = 18)
        
        # Plot the vectors and reference vector
        rd = 5 #regrid_delta
        quiver = plt.quiver(lon[::rd], lat[::rd], u[::rd, ::rd], v[::rd, ::rd], 
                            transform = ccrs.PlateCarree(), headwidth = 5., headlength = 5.)
        ax.quiverkey(quiver, X = 0.9, Y = 1.03, U = 3., label = '3 m/s', 
                     coordinates = 'axes', labelpos = 'E')
    
    # *must* call draw in order to get the axis boundary used to add ticks:
    fig.canvas.draw()

    # Add the tick marks
    xticks = np.arange(0., 360., 30.)
    yticks = np.arange(-90., 100., 15.)
    
    # Label the end-points of the gridlines using the custom tick makers:
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    lambert_xticks(ax, xticks)
    lambert_yticks(ax, yticks)
    
    #Set title and figure name
    if mode == 'LM':
        plt.title('500 hPa Winds, GPH'+'\n'+'long term mean', fontsize=18)
        pname = 'p500_longterm.png'
    elif mode == 'EM':
        plt.title('500 hPa Winds, GPH'+'\n'+'extreme precipitation days', fontsize=18)
        pname = 'p500_extreme.png'
    elif mode == 'A':
        plt.title('500 hPa Winds, GPH'+'\n'+'anomaly fields', fontsize=18)
        pname = 'p500_anom.png'
        
    ax.set_global(); ax.gridlines();
    plt.tight_layout()
    #plot_dir = '/mnt/a/u/sciteam/chug/Laplata_tracers/plots/dipole_assessment/'
    #pname = plot_dir + name + '.png'
    plt.savefig(pname, bbox_inches = 'tight')
    plt.show()

def plot_850hPa(lon, lat, u, v, t, q, mode):
    """
    Plot filled contours overlayed with contours and vectors

    Input
    -------
    lon = lon values extracted from xarray dataset (1-D)

    lat = lat values extracted from xarray dataset (1-D)

    u = U-wind at 850 hPa, shape = lon X lat

    v = V-wind at 850 hPa, shape = lon X lat
    
    t = Temperature at 850 hPa, shape = lon X lat
    
    q = Specific humidity at 850 hPa, shape = lon X lat
    
    mode = 'A' for anomaly data, 'LM' for long term means, and 'EM' for extreme precipitation days
    
    Output
    --------
    matplotlib figure with filled contours of temperature overlayed with contours of spec humidity and wind vectors
    
    """
    # change data and lon to cyclic coordinates
    u, lon_new = add_cyclic_point(u.values, coord = lon.values)
    v, lon_new = add_cyclic_point(v.values, coord = lon.values)
    q, lon_new = add_cyclic_point(q.values, coord = lon.values)
    t, lon = add_cyclic_point(t.values, coord = lon.values)
    
    # Create a figure
    fig = plt.figure(figsize = (10, 5))
    
    # Set the GeoAxes to the PlateCarree projection
    ax = plt.axes(projection = ccrs.PlateCarree())
    
    # Add coastlines
    ax.coastlines('50m', linewidth = 0.8)
    
    # Assign data for filled contour    
    data = t
    
    if mode == 'EM' or mode == 'LM':
        # Plot filled contours
        plt.contourf(lon, lat, data, 20, transform = ccrs.PlateCarree(), 
                     cmap = get_cmap("viridis"))
        
        # Add a color bar
        cbar = plt.colorbar(ax = ax, shrink = .75)
        cbar.ax.set_ylabel('$^{o}C$', fontsize = 18)
        
        # Plot contours
        plt.contour(lon, lat, q, transform = ccrs.PlateCarree(), colors = 'w')
        
        # Plot the vectors and reference vector
        rd = 5 #regrid_delta
        quiver = plt.quiver(lon[::rd], lat[::rd], u[::rd, ::rd], v[::rd, ::rd], 
                            transform = ccrs.PlateCarree(), headwidth = 5., headlength = 5.)
        ax.quiverkey(quiver, X = 0.9, Y = 1.03, U = 8., label = '8 m/s', 
                     coordinates = 'axes', labelpos = 'E')
    
    elif mode == 'A':
        # Plot filled contours
        maxval, minval = np.abs(np.amax(data)), np.abs(np.amin(data))
        normmax = np.amax([maxval, minval])
        norm = mpl.colors.Normalize(vmin = -normmax, vmax = normmax)
        plt.contourf(lon, lat, data, 20, transform = ccrs.PlateCarree(), 
                     norm = norm, cmap = get_cmap("RdBu_r"))
        
        # Add a color bar
        cbar = plt.colorbar(ax = ax, shrink = .75)
        cbar.ax.set_ylabel('$^{o}C$', fontsize = 18)
        
        # Plot contours
        plt.contour(lon, lat, q, transform = ccrs.PlateCarree())
        
        # Plot the vectors and reference vector
        rd = 5 #regrid_delta
        quiver = plt.quiver(lon[::rd], lat[::rd], u[::rd, ::rd], v[::rd, ::rd], 
                            transform = ccrs.PlateCarree(), headwidth = 5., headlength = 5.)
        ax.quiverkey(quiver, X = 0.9, Y = 1.03, U = 3., label = '3 m/s', 
                     coordinates = 'axes', labelpos = 'E')
    
    # *must* call draw in order to get the axis boundary used to add ticks:
    fig.canvas.draw()

    # Add the tick marks
    xticks = np.arange(0., 360., 30.)
    yticks = np.arange(-90., 100., 15.)
    
    # Label the end-points of the gridlines using the custom tick makers:
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    lambert_xticks(ax, xticks)
    lambert_yticks(ax, yticks)
    
    # Set title and figure name
    if mode == 'LM':
        plt.title('850 hPa Winds, Temp, Humidity'+'\n'+'long term mean', fontsize = 18)
        pname = 'p850_longterm.png'
    elif mode == 'EM':
        plt.title('850 hPa Winds, Temp, Humidity'+'\n'+'extreme precipitation days', fontsize = 18)
        pname = 'p850_extreme.png'
    elif mode == 'A':
        plt.title('850 hPa Winds, Temp, Humidity'+'\n'+'anomaly fields', fontsize = 18)
        pname = 'p850_anom.png'
        
    ax.set_global(); ax.gridlines();
    plt.tight_layout()
    #plot_dir = '/mnt/a/u/sciteam/chug/Laplata_tracers/plots/dipole_assessment/'
    #pname = plot_dir + name + '.png'
    plt.savefig(pname, bbox_inches = 'tight')
    plt.show()

def plot_sfc_winds_skt(lonu, latu, u, v, lont, latt, t, mode):
    """
    Plot filled contours overlayed with contours and vectors

    Input
    -------
    lonu = lon values extracted from wind dataset (1-D)

    latu = lat values extracted from wind dataset (1-D)

    u = U-wind at surface, shape = lonu X latu

    v = V-wind at surface, shape = lonu X latu
    
    lont = lon values extracted from skin temperature dataset (1-D)

    latt = lat values extracted from skin temperature dataset (1-D)
    
    t = Skin temperature, shape = lont X latt
    
    mode = 'A' for anomaly data, 'LM' for long term means, and 'EM' for extreme precipitation days
    
    Output
    --------
    matplotlib figure with filled contours of skin temperature overlayed with wind vectors
    
    """
    # change data and lon to cyclic coordinates
    u, lonu_new = add_cyclic_point(u.values, coord = lonu.values)
    v, lonu = add_cyclic_point(v.values, coord = lonu.values)
    t, lont = add_cyclic_point(t.values, coord = lont.values)
    
    # Create a figure
    fig = plt.figure(figsize=(10, 5))
    
    # Set the GeoAxes to the PlateCarree projection
    ax = plt.axes(projection = ccrs.PlateCarree())
    
    # Add coastlines
    ax.coastlines('50m', linewidth = 0.8)
    
    # Assign data for filled contour    
    data = t
    
    if mode == 'EM' or mode == 'LM':
        # Plot filled contours
        plt.contourf(lont, latt, data, 20, transform = ccrs.PlateCarree(), 
                     cmap = get_cmap("viridis"))
        # Add a color bar
        cbar = plt.colorbar(ax = ax, shrink = .75)
        cbar.ax.set_ylabel('$^{o}C$', fontsize = 18)
        
        # Plot the vectors and reference vector
        rd = 5 #regrid_delta
        quiver = plt.quiver(lonu[::rd], latu[::rd], u[::rd, ::rd], v[::rd, ::rd], 
                            transform = ccrs.PlateCarree(), headwidth = 5., headlength = 5.)
        ax.quiverkey(quiver, X = 0.9, Y = 1.03, U = 5., label = '5 m/s', 
                     coordinates = 'axes', labelpos = 'E')
    
    elif mode == 'A':
        # Plot filled contours
        maxval, minval = np.abs(np.amax(data)), np.abs(np.amin(data))
        normmax = np.amax([maxval, minval])
        norm = mpl.colors.Normalize(vmin = -normmax, vmax = normmax)
        plt.contourf(lont, latt, data, 20, transform = ccrs.PlateCarree(), 
                     norm = norm, cmap = get_cmap("RdBu_r"))
        
        # Add a color bar
        cbar = plt.colorbar(ax = ax, shrink = .75)
        cbar.ax.set_ylabel('$^{o}C$', fontsize = 18)

        # Plot the vectors and reference vector
        rd = 5 #regrid_delta
        quiver = plt.quiver(lonu[::rd], latu[::rd], u[::rd, ::rd], v[::rd, ::rd], 
                            transform = ccrs.PlateCarree(), headwidth = 5., headlength = 5.)
        ax.quiverkey(quiver, X = 0.9, Y = 1.03, U = 3., label = '3 m/s', 
                     coordinates = 'axes', labelpos = 'E')
    
    # *must* call draw in order to get the axis boundary used to add ticks:
    fig.canvas.draw()

    # Add the tick marks
    xticks = np.arange(0., 360., 30.)
    yticks = np.arange(-80., 80., 20.)
    
    # Label the end-points of the gridlines using the custom tick makers:
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    lambert_xticks(ax, xticks)
    lambert_yticks(ax, yticks)
    
    # Set title and figure name
    if mode == 'LM':
        plt.title('Surface Winds, Skin temp'+'\n'+'long term mean', fontsize = 18)
        pname = 'sfc_longterm.png'
    elif mode == 'EM':
        plt.title('Surface Winds, Skin temp'+'\n'+'extreme precipitation days', fontsize = 18)
        pname = 'sfc_extreme.png'
    elif mode == 'A':
        plt.title('Surface Winds, Skin temp'+'\n'+'anomaly fields', fontsize = 18)
        pname = 'sfc_anom.png'
        
    ax.set_global(); ax.gridlines();
    plt.tight_layout()
    #plot_dir = '/mnt/a/u/sciteam/chug/Laplata_tracers/plots/dipole_assessment/'
    #pname = plot_dir + name + '.png'
    plt.savefig(pname, bbox_inches = 'tight')
    plt.show()

def plot_TCWV(lon, lat, q, mode):
    """
    Plot filled contours of total column water vapor

    Input
    -------
    lon = lon values extracted from xarray dataset (1-D)

    lat = lat values extracted from xarray dataset (1-D)

    q = Total column water vapor, shape = lon X lat

    mode = 'A' for anomaly data, 'LM' for long term means, and 'EM' for extreme precipitation days
    
    Output
    --------
    matplotlib figure with filled contours of total column water vapor
    
    """
    # change data and lon to cyclic coordinates
    q, lon = add_cyclic_point(q.values, coord = lon.values)
    # Create a figure
    fig = plt.figure(figsize=(10, 5))
    
    # Set the GeoAxes to the PlateCarree projection
    ax = plt.axes(projection = ccrs.PlateCarree())
    
    # Add coastlines
    ax.coastlines('50m', linewidth = 0.8)
    
    # Assign data for filled contour    
    data = q
    
    if mode == 'EM' or mode == 'LM':
        data[data > 80.] = 80.
        # Plot filled contours
        plt.contourf(lon, lat, data, 20, transform = ccrs.PlateCarree(), 
                     cmap = get_cmap("viridis"))
        # Add a color bar
        cbar = plt.colorbar(ax = ax, shrink = .75)
        cbar.ax.set_ylabel('$mm$', fontsize = 18)
    
    elif mode == 'A':
        maxval, minval = np.abs(np.amax(data)), np.abs(np.amin(data))
        normmax = np.amax([maxval, minval])
        norm = mpl.colors.Normalize(vmin = -normmax, vmax = normmax)
        plt.contourf(lon, lat, data, 20, transform = ccrs.PlateCarree(), 
                     norm = norm, cmap = get_cmap("RdBu_r"))
        
        # Add a color bar
        cbar = plt.colorbar(ax = ax, shrink = .75)
        cbar.ax.set_ylabel('$mm$', fontsize = 18)
    
    # *must* call draw in order to get the axis boundary used to add ticks:
    fig.canvas.draw()

    # Add the tick marks
    xticks = np.arange(0., 360., 30.)
    yticks = np.arange(-80., 80., 20.)
    
    # Label the end-points of the gridlines using the custom tick makers:
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    lambert_xticks(ax, xticks)
    lambert_yticks(ax, yticks)
    
    # Set title and figure name
    if mode == 'LM':
        plt.title('Total column water vapor'+'\n'+'long term mean', fontsize = 18)
        pname = 'tcwv_longterm.png'
    elif mode == 'EM':
        plt.title('Total column water vapor'+'\n'+'extreme precipitation days',fontsize = 18)
        pname = 'tcwv_extreme.png'
    elif mode == 'A':
        plt.title('Total column water vapor'+'\n'+'anomaly field',fontsize = 18)
        pname = 'tcwv_anom.png'
        
    ax.set_global(); ax.gridlines();
    plt.tight_layout()
    #plot_dir = '/mnt/a/u/sciteam/chug/Laplata_tracers/plots/dipole_assessment/'
    #pname = plot_dir + name + '.png'
    plt.savefig(pname, bbox_inches = 'tight')
    plt.show()

###############################
# Open datasets and plot data #

# Set path to netcdf files
path = 'atms597_proj3/data/'

# First let's plot the anomalies 

# 250 hPa anomalies
xrdata = xr.open_dataset(path+'pressure_anomaly.nc')
lat = xrdata['lat']
lon = xrdata['lon']
u = xrdata['u_wind_250']
v = xrdata['v_wind_250']
xrdata = xr.open_dataset('atms597_proj3/data/pressure_anomaly_new.nc')
wspd = xrdata['wind_spd_250']
plot_250hPa_winds(lon, lat, u, v, wspd, 'A')

# 500 hPa anomalies
u = xrdata['u_wind_500']
v = xrdata['v_wind_500']
z = xrdata['height_500']
plot_500hPa_winds_geopot(lon, lat, u, v, z, 'A')

# 850 hPa anomalies
u = xrdata['u_wind_850']
v = xrdata['v_wind_850']
t = xrdata['temp_850']
q = xrdata['q_850']
plot_850hPa(lon, lat, u, v, t, q, 'A')

# Next we move to surface anomalies
xrdata = xr.open_dataset(path+'surface_anomaly.nc')
latu = xrdata['lat']
lonu = xrdata['lon']
u = xrdata['sfc_u_wind_surface']
v = xrdata['sfc_v_wind_surface']
xrdata = xr.open_dataset(path+'surface_gauss_anomaly.nc')
t = xrdata['skin_temp_surface']-273 #convert to Celcius
latt = xrdata['lat']
lont = xrdata['lon']
plot_sfc_winds_skt(lonu, latu, u, v, lont, latt, t, 'A')

# TCWV anomalies
xrdata = xr.open_dataset(path+'total_column_anomaly.nc')
lat = xrdata['lat']
lon = xrdata['lon']
q = xrdata['total_column_q']
plot_TCWV(lon, lat, q, 'A')

# Next we plot the long term means

# 250 hPa long term means
xrdata = xr.open_dataset(path+'pressure_long_term_mean.nc')
lat = xrdata['lat']
lon = xrdata['lon']
u = xrdata['u_wind_250']
v = xrdata['v_wind_250']
wspd = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
plot_250hPa_winds(lon, lat, u, v, wspd, 'LM')

# 500 hPa long term means
u = xrdata['u_wind_500']
v = xrdata['v_wind_500']
z = xrdata['height_500']
plot_500hPa_winds_geopot(lon, lat, u, v, z, 'LM')

# 850 hPa long term means
u = xrdata['u_wind_850']
v = xrdata['v_wind_850']
t = xrdata['temp_850']
q = xrdata['q_850']
plot_850hPa(lon, lat, u, v, t, q, 'LM')

# surface long term means
xrdata = xr.open_dataset(path+'surface_long_term_mean.nc')
latu = xrdata['lat']
lonu = xrdata['lon']
u = xrdata['sfc_u_wind_surface']
v = xrdata['sfc_v_wind_surface']
xrdata = xr.open_dataset(path+'surface_gauss_long_term_mean.nc')
t = xrdata['skin_temp_surface']
latt = xrdata['lat']
lont = xrdata['lon']
plot_sfc_winds_skt(lonu, latu, u, v, lont, latt, t, 'LM')

# TCWV long term means
xrdata = xr.open_dataset(path+'total_column_long_term_mean.nc')
lat = xrdata['lat']
lon = xrdata['lon']
q = xrdata['total_column_q']
plot_TCWV(lon, lat, q, 'LM')

# Finally we plot the mean of extreme precipitation days

# 250 hPa extreme means
xrdata = xr.open_dataset(path+'pressure_extreme_precip_mean.nc')
lat = xrdata['lat']
lon = xrdata['lon']
u = xrdata['u_wind_250']
v = xrdata['v_wind_250']
wspd = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
plot_250hPa_winds(lon, lat, u, v, wspd, 'EM')

# 500 hPa extreme means
u = xrdata['u_wind_500']
v = xrdata['v_wind_500']
z = xrdata['height_500']
plot_500hPa_winds_geopot(lon, lat, u, v, z, 'EM')

# 850 hPa extreme means
u = xrdata['u_wind_850']
v = xrdata['v_wind_850']
t = xrdata['temp_850']
q = xrdata['q_850']
plot_850hPa(lon, lat, u, v, t, q, 'EM')

# surface extreme means
xrdata = xr.open_dataset(path+'surface_extreme_precip_mean.nc')
latu = xrdata['lat']
lonu = xrdata['lon']
u = xrdata['sfc_u_wind_surface']
v = xrdata['sfc_v_wind_surface']
xrdata = xr.open_dataset(path+'surface_gauss_extreme_precip_mean.nc')
t = xrdata['skin_temp_surface']-273
latt = xrdata['lat']
lont = xrdata['lon']
plot_sfc_winds_skt(lonu, latu, u, v, lont, latt, t, 'EM')

# TCWV extreme means
xrdata = xr.open_dataset(path+'total_column_extreme_precip_mean.nc')
lat = xrdata['lat']
lon = xrdata['lon']
q = xrdata['total_column_q']
plot_TCWV(lon, lat, q, 'EM')
    