###########################################################################
# GAUSSIAN PLUME MODEL FOR TEACHING PURPOSES                              #
# PAUL CONNOLLY (UNIVERSITY OF MANCHESTER, 2017)                          #
# THIS CODE IS PROVIDED `AS IS' WITH NO GUARANTEE OF ACCURACY             #
# IT IS USED TO DEMONSTRATE THE EFFECTS OF ATMOSPHERIC STABILITY,         #
# WINDSPEED AND DIRECTION AND MULTIPLE STACKS ON THE DISPERSION OF        #
# POLLUTANTS FROM POINT SOURCES                                           #
###########################################################################

import numpy as np
import sys
from scipy.special import erfcinv as erfcinv
import tqdm as tqdm
import time
import yaml 
import pandas as pd
import calendar
import xarray as xr

from gauss_func import gauss_func

import matplotlib.pyplot as plt
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=False)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

###########################################################################
# Do not change these variables                                           #
###########################################################################
# SECTION 0: Definitions (normally don't modify this section)
# view
PLAN_VIEW=1;
HEIGHT_SLICE=2;
SURFACE_TIME=3;
NO_PLOT=4;

# wind field
CONSTANT_WIND=1;
FLUCTUATING_WIND=2;
PREVAILING_WIND=3;

# number of stacks
ONE_STACK=1;
TWO_STACKS=2;
THREE_STACKS=3;

# stability of the atmosphere
CONSTANT_STABILITY=1;
ANNUAL_CYCLE=2;
stability_str=['Very unstable','Moderately unstable','Slightly unstable', \
    'Neutral','Moderately stable','Very stable'];
# Aerosol properties
HUMIDIFY=2;
DRY_AEROSOL=1;

dxy=100;          # resolution of the model in both x and y directions
dz=10;
x=np.mgrid[-2500:2500+dxy:dxy]; # solve on a 5 km domain
y=x              # x-grid is same as y-grid
###########################################################################
fin = sys.argv[1]
with open(fin, 'r') as stream:
    config_var = yaml.load(stream, Loader = yaml.FullLoader)
print(config_var)

# SECTION 1: Configuration
# Variables can be changed by the user+++++++++++++++++++++++++++++++++++++
postproc = config_var['type_postproc']

dry_size=np.float(config_var['diameter']);
humidify=config_var['humid'];
nu=np.float(config_var['nu']);
rho_s=np.float(config_var['rho']);
Ms=np.float(config_var['M']);
Mw=18e-3;
mass=np.pi/6.*rho_s*dry_size**3.;
moles=mass/Ms;

output=config_var['output'];
stack_x=np.float(config_var['x'])
stack_y=np.float(config_var['y'])

H=np.float(config_var['hauteur']); # stack height, m
# emissions
profilQ = config_var['proftps']
if profilQ == 'constant':
  daysemis = 1
  Qread=np.float(config_var['emiss']); # mass emitted per unit time
else:
  hourly = pd.read_csv(config_var['proftps']+'hourly.txt', sep=' ',header=None)
  daily = pd.read_csv(config_var['proftps']+'daily.txt', sep=' ',header=None)
  monthly = pd.read_csv(config_var['proftps']+'monthly.txt', sep=' ',header=None)
  daysemis = 365
  yeartype = 2013
  Q = np.zeros(daysemis*24)
  hh = 0
  for m in range(12):
    for d in range(calendar.mdays[m+1]):
      wd = calendar.weekday(yeartype,m+1,d+1)
      for h in range(24):
        coeff = monthly[1][m]* daily[1][wd] * hourly[1][h]
        Q[hh] = coeff * np.float(config_var['emiss'])
        hh = hh + 1
 
# SECTION 2: Act on the configuration information

# Set the wind based on input flags++++++++++++++++++++++++++++++++++++++++
if config_var['type_vent'] == 'constant':
   wind = CONSTANT_WIND
   days= max(daysemis, 1)
   nb_timesteps = days * 24 
   if profilQ == 'constant':
     Q = Qread*np.ones((nb_timesteps,1));
   stab1=config_var['stabilite']; # set from 1-6
   stability_used = CONSTANT_STABILITY;
   RH1=config_var['rh']
   year = 2013 # cas type
   list_hours = pd.date_range(start='01/01/'+str(year),periods = nb_timesteps ,freq='H')
else:
   wind = FLUCTUATING_WIND
   stability_used =  ANNUAL_CYCLE
   # read file
   ts = pd.read_csv(config_var['type_vent'], sep = ' ', header=None, names=['Date', 'temp', 'speed', 'dir', 'grad', 'humidity'])
   ts['date']=pd.to_datetime(ts['Date'],format='%Y%m%d%H')
   ts.set_index('date', inplace=True)
   year = ts.index[0].year
   days = 365
   if calendar.isleap(year):
      days = days +1 
   nb_timesteps = days * 24 
   list_hours = pd.date_range(start='01/01/'+str(year),periods = nb_timesteps ,freq='H')
   ts_full = ts.reindex(list_hours).fillna(method = 'ffill')
   list_days = pd.date_range(start='01/01/'+str(year),periods = days ,freq='D')
   if profilQ == 'constant':
     Q = Qread*np.ones((nb_timesteps,1));

#--------------------------------------------------------------------------
times=np.mgrid[1:(days)*24+1:1]/24.;   
if wind == CONSTANT_WIND:
   wind_dir=np.float(config_var['direction'])*np.ones((nb_timesteps,1));
   wind_dir_str='Constant wind';
   wind_speed=np.float(config_var['vitesse'])*np.ones((nb_timesteps,1)); # m/s
elif wind == FLUCTUATING_WIND:
   wind_dir = ts_full['dir'].values
   wind_dir_str='Yearly time series of wind';
   wind_speed=ts_full['speed'].values
elif wind == PREVAILING_WIND:
   wind_dir=-np.sqrt(2.)*erfcinv(2.*np.random.rand(nb_timesteps,1))*40.; #norminv(rand(days.*24,1),0,40);
   # note at this point you can add on the prevailing wind direction, i.e.
   # wind_dir=wind_dir+200;
   wind_dir[np.where(wind_dir>=360.)]= \
        np.mod(wind_dir[np.where(wind_dir>=360)],360);
   wind_dir_str='Prevailing wind';
   wind_speed=np.float(config_var['vitesse'])*np.ones((nb_timesteps,1)); # m/s
else:
   sys.exit()

# Decide which stability profile to use
if stability_used == CONSTANT_STABILITY:   
   stability=stab1*np.ones((nb_timesteps,1));
   stability_str=stability_str[stab1-1];
   rh = RH1*np.ones((nb_timesteps,1))
elif stability_used == ANNUAL_CYCLE:
   # implementing method temperature gradient
   ts_full['stability'] = 4
   ts_full['stability'][ts_full['grad'] < -1.9e-2] = 6
   ts_full['stability'][(ts_full['grad'] >= -1.9e-2) & (ts_full['grad'] < -1.7e-2) ] = 5
   ts_full['stability'][(ts_full['grad'] >= -1.7e-2) & (ts_full['grad'] < -1.5e-2) ] = 4
   ts_full['stability'][(ts_full['grad'] >= -1.5e-2) & (ts_full['grad'] < -0.55e-2) ] = 3
   ts_full['stability'][(ts_full['grad'] >= -0.55e-2) & (ts_full['grad'] < 1.5e-2) ] = 2
   ts_full['stability'][ts_full['grad'] >= 1.5e-2] = 1
   stability= ts_full['stability'].values
   rh = ts_full['humidity'].values
   rh = rh /100.
  # implementing method ecart-type de direction?
   stability_str='Yearly time series of stability';
else:
   sys.exit()


# decide what kind of run to do, plan view or y-z slice, or time series
if output == PLAN_VIEW or output == SURFACE_TIME or output == NO_PLOT:
   C1=np.zeros((len(x),len(y),nb_timesteps)); # array to store data, initialised to be zero
   [x,y]=np.meshgrid(x,y); # x and y defined at all positions on the grid
   z=np.zeros(np.shape(x)) + int(config_var['zlev']);    # z = 0 at ground level.

elif output == HEIGHT_SLICE:
   z=np.mgrid[0:500+dz:dz];       # z-grid
   dirslice = config_var['dirslice']
   if dirslice == 'X':
     C1=np.zeros((len(y),len(z),nb_timesteps)); # array to store data, initialised to be zero
     [y,z]=np.meshgrid(y,z); # y and z defined at all positions on the grid
     x_slice=config_var['posslice']; # position (1-50) to take the slice in the x-direction
     x=x[x_slice]*np.ones(np.shape(y));    # x is defined to be x at x_slice       
   elif dirslice == 'Y':
     C1=np.zeros((len(x),len(z),nb_timesteps)); # array to store data, initialised to be zero
     [x,z] = np.meshgrid(x,z); # x and z defined at all positions on the grid
     y_slice=config_var['posslice']; # position (1-50) to take the slice in the y-direction
     y=y[y_slice]*np.ones(np.shape(x));    # y is defined to be y at y_slice    
   else:
      print('vertical slices along X or Y directions only')
      sys.exit()  
else:
   sys.exit()

#--------------------------------------------------------------------------

# SECTION 3: Main loop
# For all times...
for i in tqdm.tqdm(range(0,nb_timesteps)):
    C=gauss_func(Q[i],wind_speed[i],wind_dir[i],x,y,z,
        stack_x,stack_y,H,stability[i]);
    C1[:,:,i]=C1[:,:,i]+C;
    # decide whether to humidify the aerosol and hence increase the mass
    if humidify == DRY_AEROSOL and i ==0:
        print('do not humidify')
    if humidify == HUMIDIFY:
        if i == 0: 
            print('humidify')
        nw=rh[i]*nu*moles/(1.-rh[i])
        mass2=nw*Mw+moles*Ms
        C1[:,:,i]=C1[:,:,i]*mass2/mass

# SECTION 4: Post process / output
 

# output the plots
if output == PLAN_VIEW:
   plt.figure()
   plt.ion()
   if postproc == 'mean':
      plt.pcolor(x,y,np.mean(C1,axis=2), cmap='jet')
   elif postproc == 'min':
      plt.pcolor(x,y,np.min(C1,axis=2), cmap='jet')
   elif postproc == 'max':
      plt.pcolor(x,y,np.max(C1,axis=2), cmap='jet')
   elif postproc == 'freq':
      limit = np.float(config_var['threshold'])
      plt.pcolor(x,y,np.count_nonzero(C1 > limit, axis=2), cmap='jet')
   else:
      print('post-processing type unknown')
      sys.exit()
   plt.clim((np.float(config_var['min']), np.float(config_var['max'])));
   plt.title(stability_str + '\n' + wind_dir_str);
   plt.xlabel('x (metres)');
   plt.ylabel('y (metres)');
   cb1=plt.colorbar();
   cb1.set_label('$\mu$g$\cdot$m$^{-3}$');

   plt.savefig('carte_'+config_var['etiq']+'.png')
   plt.show()


elif output == HEIGHT_SLICE:
   plt.figure();
   plt.ion()
   if dirslice == 'X':
     plt.xlabel('y (metres)');
     if postproc == 'mean':
        plt.pcolor(y,z,np.mean(C1,axis=2), cmap='jet')
     elif postproc == 'min':
        plt.pcolor(y,z,np.min(C1,axis=2), cmap='jet')
     elif postproc == 'max':
        plt.pcolor(y,z,np.max(C1,axis=2), cmap='jet')
     elif postproc == 'freq':
        limit = np.float(config_var['threshold'])
        plt.pcolor(y,z,np.count_nonzero(C1 > limit, axis=2), cmap='jet')
     else:
        print('post-processing type unknown')
        sys.exit()
   if dirslice == 'Y':
     plt.xlabel('x (metres)');
     if postproc == 'mean':
        plt.pcolor(x,z,np.mean(C1,axis=2), cmap='jet')      
     elif postproc == 'min':
        plt.pcolor(x,z,np.min(C1,axis=2), cmap='jet')     
     elif postproc == 'max':
        plt.pcolor(x,z,np.max(C1,axis=2), cmap='jet')     
     elif postproc == 'freq':
       limit = np.float(config_var['threshold'])
       plt.pcolor(x,z,np.count_nonzero(C1 > limit,axis=2), cmap='jet')
     else:
        print('post-processing type unknown')
        sys.exit()
   plt.clim((np.float(config_var['min']), np.float(config_var['max'])));
   plt.ylabel('z (metres)');
   plt.title(stability_str + '\n' + wind_dir_str);
   cb1=plt.colorbar();
   cb1.set_label('$\mu$g$\cdot$m$^{-3}$');

   plt.savefig('coupe_'+config_var['etiq']+'.png')
   plt.show()


elif output == SURFACE_TIME:  
   plt.figure()
   plt.ion()
   x_slice = config_var['xpos'] # position (1-50) to plot concentrations vs time 
   y_slice = config_var['ypos'] # position (1-50) to plot concentrations vs time 
   cts = xr.Dataset( coords={'lon': (['x', 'y'], x),'lat': (['x', 'y'], y), \
                      'time': list_hours})   
   cts['concs'] = (['x','y','time'], C1)
   freq = config_var['freqtps']
   if freq == 'mois':
     toplot = cts.groupby('time.month').mean()
     toplot.isel(x = x_slice, y=y_slice)['concs'].plot.line(x='month', marker = '.') 
     plt.xlabel('time (months)');
   elif freq == 'jour':
     toplot = cts.resample(time='1D').mean() 
     toplot.isel(x = x_slice, y=y_slice)['concs'].plot.line(x='time', linestyle = None) 
     plt.xlabel('time (days)')
     #plt.xticks([list_days[k] for k in range(len(list_days)) if (k%15==0)])
   plt.ylabel('Mass loading (microg/m3)')
   plt.title(stability_str +'\n' + wind_dir_str)

   plt.savefig('serie_temporelle_'+config_var['etiq']+'.png')
   plt.show()

   
elif output == NO_PLOT:
   print('don''t plot');
else:
   sys.exit()
   

maxconc = np.max(np.mean(C1,axis=2)) + 20

print('La plus haute concentration atteinte sur le domaine est de ', maxconc, 'microg/m^3 = ', (maxconc/40)*100,'% du maximum autorise')


if maxconc>40:
    print('POLLUTION NON-REGLEMENTAIRE')
else:
    print('Seuil non-franchi')