import numpy as np
import pandas as pd
import os, pdb
from nilearn import plotting as ni_plt
import matplotlib.pyplot as plt

#Subfunctions to put elsewhere

def _setup_subplot_view(locs,sides_2_display,figsize):
    """
    Decide whether to plot L or R hemisphere based on x coordinates
    """
    if sides_2_display=='auto':
        average_xpos_sign = np.mean(np.asarray(locs['x']))
        if average_xpos_sign>0:
            sides_2_display='yrz'
        else:
            sides_2_display='ylz'
    
    #Create figure and axes
    if sides_2_display=='ortho':
        N = 1
    else:
        N = len(sides_2_display)
        
    if sides_2_display=='yrz' or sides_2_display=='ylz':
        gridspec.GridSpec(0,3)
        fig,axes=plt.subplots(1,N, figsize=figsize)
    else:
        fig,axes=plt.subplots(1,N, figsize=figsize)
    return N,axes,sides_2_display

def _plot_electrodes(locs,node_size,colors,axes,sides_2_display,N,node_edge_colors,alpha,edge_linewidths):
    """
    Handles plotting
    """
    if N==1:
        ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                               node_kwargs={'alpha': alpha, 'edgecolors': node_edge_colors,'linewidths':edge_linewidths},
                               node_size=node_size, node_color=colors,axes=axes,display_mode=sides_2_display)
    elif sides_2_display=='yrz' or sides_2_display=='ylz':
        colspans=[5,6,5] #different sized subplot to make saggital view similar size to other two slices
        current_col=0
        total_colspans=int(np.sum(np.asarray(colspans)))
        for ind,colspan in enumerate(colspans):
            axes[ind]=plt.subplot2grid((1,total_colspans), (0,current_col), colspan=colspan, rowspan=1)
            ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                               node_kwargs={'alpha': alpha, 'edgecolors': node_edge_colors,'linewidths':edge_linewidths},
                               node_size=node_size, node_color=colors,axes=axes[ind],display_mode=sides_2_display[ind])
            current_col+=colspan
    else:
        for i in range(N):
            ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                                   node_kwargs={'alpha': alpha, 'edgecolors': node_edge_colors,'linewidths':edge_linewidths},
                                   node_size=node_size, node_color=colors,axes=axes[i],display_mode=sides_2_display[i])
            
def plot_ecog_electrodes_mni_in_order(elec_locs_fnames,chan_labels='all',num_grid_chans=64,colors_in=None,node_size=50,
                                      figsize=(16,6),sides_2_display='auto',node_edge_colors=None,
                                      alpha=0.5,edge_linewidths=3,ax_in=None,rem_zero_chans=False,
                                      allLH=False,zero_rem_thresh=.99,elec_col_suppl_in=None,
                                      sort_vals_in=None,sort_abs=False,rem_zero_chans_show=False,rem_show_col=[0,0,0]):
    """
    Plots ECoG electrodes from MNI coordinate file in order based on a value (only for specified labels)
        
    NOTE: If running in Jupyter, use '%matplotlib inline' instead of '%matplotlib notebook'
    """ 
    for i,fID in enumerate(elec_locs_fnames):
        #Load channel locations
        chan_info = pd.read_csv(fID)
        chan_info = chan_info.transpose()
        if (colors_in is not None) and isinstance(colors_in, list):
            colors = colors_in.copy() #one subject
        elif (colors_in is not None) and isinstance(colors_in, np.ndarray):
            colors = colors_in[i] #multiple subjects
        else:
            colors = None
        
        if (elec_col_suppl_in is not None) and isinstance(elec_col_suppl_in, list):
            elec_col_suppl = elec_col_suppl_in.copy() #one subject
        elif (elec_col_suppl_in is not None) and isinstance(elec_col_suppl_in, np.ndarray):
            elec_col_suppl = elec_col_suppl_in[i].copy() #multiple subjects
        else:
            elec_col_suppl = None

        if (sort_vals_in is not None) and isinstance(sort_vals_in, list):
            sort_vals = sort_vals_in.copy() #one subject
        elif (sort_vals_in is not None) and isinstance(sort_vals_in, np.ndarray):
            sort_vals = sort_vals_in[i] #multiple subjects
        else:
            sort_vals = None
            
        #Create dataframe for electrode locations
        if chan_labels== 'all':
            locs = chan_info.loc[['X','Y','Z'],:].astype('float64').transpose()
        elif chan_labels== 'allgood':
            locs = chan_info.loc[['X','Y','Z','goodChanInds'],:].astype('float64').transpose()
        else:
            locs = chan_info.loc[['X','Y','Z'],chan_labels].astype('float64').transpose()
        if (colors is not None):
            if (locs.shape[0]>len(colors)) & isinstance(colors, list):
                locs = locs.iloc[:len(colors),:]
        locs.rename(columns={'X':'x','Y':'y','Z':'z'}, inplace=True)
        chan_loc_x = chan_info.loc['X',:].astype('float64').values
        
        #Remove NaN electrode locations (no location info)
        nan_drop_inds = np.nonzero(np.isnan(chan_loc_x))[0]
        locs.dropna(axis=0,inplace=True) #remove NaN locations
        if (colors is not None) & isinstance(colors, list):
            colors_new,sort_vals_new,loc_inds_2_drop = [],[],[]
            for s,val in enumerate(colors):
                if not (s in nan_drop_inds):
                    colors_new.append(val)
                    if (sort_vals is not None):
                        sort_vals_new.append(sort_vals[s])
                else:
                    loc_inds_2_drop.append(s)
            colors = colors_new.copy()
            sort_vals = sort_vals_new.copy()

            if elec_col_suppl is not None:
                loc_inds_2_drop.reverse() #go from high to low values
                for val in loc_inds_2_drop:
                    del elec_col_suppl[val]

        if chan_labels=='allgood':
            goodChanInds = chan_info.loc['goodChanInds',:].astype('float64').transpose()
            inds2drop = np.nonzero(locs['goodChanInds']==0)[0]
            locs.drop(columns=['goodChanInds'],inplace=True)
            locs.drop(locs.index[inds2drop],inplace=True)

            if colors is not None:
                colors_new,sort_vals_new,loc_inds_2_drop = [],[],[]
                for s,val in enumerate(colors):
                    if not (s in inds2drop):
    #                     np.all(s!=inds2drop):
                        colors_new.append(val)
                        if (len(sort_vals)>0):
                            sort_vals_new.append(sort_vals[s])
                    else:
                        loc_inds_2_drop.append(s)
                colors = colors_new.copy()
                sort_vals = sort_vals_new.copy()

                if elec_col_suppl is not None:
                    loc_inds_2_drop.reverse() #go from high to low values
                    for val in loc_inds_2_drop:
                        del elec_col_suppl[val]
        
        if rem_zero_chans:
            #Remove channels with zero values (white colors)
            colors_new,sort_vals_new,loc_inds_2_drop = [],[],[]
            for s,val in enumerate(colors):
                if np.mean(val)<zero_rem_thresh:
                    colors_new.append(val)
                    if (len(sort_vals)>0):
                        sort_vals_new.append(sort_vals[s])
                else:
                    loc_inds_2_drop.append(s)
            colors = colors_new.copy()
            sort_vals = sort_vals_new.copy()
            locs.drop(locs.index[loc_inds_2_drop],inplace=True)

            if elec_col_suppl is not None:
                loc_inds_2_drop.reverse() #go from high to low values
                for val in loc_inds_2_drop:
                    del elec_col_suppl[val]
        elif rem_zero_chans_show:
            #Channels with zero values are white
            for s,val in enumerate(colors):
                if np.mean(val)>=zero_rem_thresh:
                    elec_col_suppl[s] = rem_show_col.copy()

        #Decide whether to plot L or R hemisphere based on x coordinates
        if len(sides_2_display)>1:
            N,axes,sides_2_display = _setup_subplot_view(locs,sides_2_display,figsize)
        else:
            N = 1
            axes = ax_in
            if allLH:
                #Automatically flip electrodes to LH
                average_xpos_sign = np.mean(np.asarray(locs['x']))
                if average_xpos_sign>0:
                    locs['x'] = -locs['x']
                sides_2_display ='l'

        if elec_col_suppl is not None:
            colors = elec_col_suppl.copy()
        
        if i == 0:
            locs2 = locs.copy()
            colors2 = colors.copy()
            sort_vals2 = sort_vals.copy()
        else:
            locs2 = pd.concat([locs2,locs],axis=0,ignore_index=True)
            colors2.extend(colors)
            sort_vals2.extend(sort_vals)
    
    #Re-order by magnitude
    if (len(colors2)>0) and (len(sort_vals2)>0):
        if sort_abs:
            #Use absolute value
            sort_vals2 = [abs(val) for val in sort_vals2]
        sort_inds = np.argsort(np.asarray(sort_vals2))
        colors2_np = np.asarray(colors2)
        colors_out = colors2_np[sort_inds,:].tolist()
        locs_out = locs2.iloc[sort_inds,:]
    else:
        colors_out = colors2.copy()
        locs_out = locs2.copy()
    
    #Plot the result
    _plot_electrodes(locs_out,node_size,colors_out,axes,sides_2_display,N,node_edge_colors,alpha,edge_linewidths)