import pandas as pd
import numpy as np

constants = {
    'SCALING' : ['density', 'spectrum'][1], # For PSDs
    'NPERSEG' : 96,
    'FPS' : 30,
    'F_ECOG': 500, # Downsampled from 1K
    'MAX_TF_FREQ': 150,    
}

constants['DATA_DIR'] = '/path/to/data/'
# constants['DATA_DIR'] = '/data2/users/satsingh/ST2020_20200602/'
constants['PSD_LABEL'] = 'PSD [V**2/Hz]' if constants['SCALING'] is 'density' else 'PSD [V**2]'


constants['PATIENT_IDS_PAPER'] = [
  'a0f66459', 
  'c95c1e82', 
  'cb46fd46', 

  'fcb01f7a',
  'ffb52f92', 
  
  'b4ac1726', 
  'f3b79359', 
  'ec761078', 
  
  'f0bbc9a9',
  'abdb496b',
  'ec168864', 
  'b45e3f7b', 
]

constants['HEMISPHERES'] = {
  'a0f66459':'L', 
  'c95c1e82':'R', 
  'cb46fd46':'L', 

  'ffb52f92':'R', 
  'fcb01f7a':'R',
  'b4ac1726':'L', 
  'f3b79359':'R', 
  'abdb496b':'L',
    
  'ec168864':'L', 
  'b45e3f7b':'L', 
  'ec761078':'R', 
  'f0bbc9a9':'L',
}


constants['ECOG_DAYS_PAPER'] = { 
  'a0f66459': [3, 4, 5],
  'c95c1e82': [3, 4, 5],
  'cb46fd46': [2, 3, 4], # Day 5 is anomalous (dip in events/video issues?)

  'fcb01f7a': [7, 8, 9], # Note: (day-2) Hospital data starts late
  # 'ffb52f92': [5, 6, 7], # Testing day has v. little data after filters
  # 'ffb52f92': [4,5,6,7], # No events on Day 6 (ECoG issues) 
  'ffb52f92': [2, 3, 4], # Usually avoid days 1/2 

  'b4ac1726': [3, 4, 5],
  'f3b79359': [3, 4, 5],
  'ec761078': [3, 4, 5],
  'f0bbc9a9': [3, 4, 5], # Too much sleep 
  'abdb496b': [3, 4, 5],
  'ec168864': [3, 4, 5],
  'b45e3f7b': [3, 4, 5],
}


