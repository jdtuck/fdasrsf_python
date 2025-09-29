import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap


def getPPDinfo(t, Fa, lam, th):
    n_lams = lam.shape[0]

    # Compute the mean of each function in FN_temp over its rows
    FNm = np.zeros(t.shape[0], len(Fa))
    FNm[:, 0] = Fa[0].mean(axis=1)
    
    # Find indices of local maxima in the first function's mean
    idxMaxFirst = find_peaks(FNm[:, 0])

    # Initalize labels and Locations for the first function
    Labels = []
    Locs = []
    Labels.append(np.arange(0, idxMaxFirst.shape[0]))
    Locs.append(idxMaxFirst)

    # Initialize the maximum label number
    labelMax = Labels[0].max()

    # Process each function to assign labels and locate peaks
    for i in range(n_lams - 1):
        currentLabel = Labels[i]
        Labelst, labelMax = peak_successor(Fa[i], Fa[i+1], currentLabel, labelMax, 1)
        Labels.append(Labelst)

        # Find peak locations in the next function's mean
        FnmNextMean = Fa[i+1].mean(axis=1)
        idxMaxNext = find_peaks(FnmNextMean)
        Locs.append(idxMaxNext)

        # Update the mean function
        FNm[:, i+1] = FnmNextMean
    
    IndicatorMatrix, Heights, Heights2 = PreprocessingForPPD(t, lam, Labels, Locs, labelMax, FNm, th)

    if np.all(np.isnan(Heights2)):
        NameError("All peaks are ignored. A smaller threshold is required.")

    return (IndicatorMatrix, Heights, Locs, Labels, FNm)


def peak_successor(f1, f2, labels1, labelMax, smooth_parameter):
    # Combine f1 and f2 into a 3D array and compute the mean across
    # the second dimension
    F = np.concatenate((f1[:,:,np.newaxis], f2[:,:,np.newaxis]), axis=2)
    fm = F.mean(axis=1)

    # compute peak ranges and labes for fm[:,0]
    ranges, idx_max1 = computePeakRanges(fm[:, 0])

    # compute indices of local maxima in fm[:,1]
    idx_max2 = find_peaks(fm[:,1])

    if idx_max1.size==0:
        labels2 = labelMax + np.arange(1, idx_max2.shape[0]+1)
    else:
        # assign labels to peaks in fm[:,1] based on matching ranges in fm[:,0]
        labels2 = assignLabelsToPeaks(idx_max2, ranges, idx_max1, labels1)

        # ensure no overlapping labels in lables2
        labels2 = resolveOverlappingLabels(labels2, idx_max1, idx_max2, labels1)

        # assign new labels to unmatched peaks in fm[:,1]
        unmatched = labels2 == 0
        labels2[unmatched] = labelMax + np.arange(1,np.sum(unmatched)+1)
        labelMax = labelMax + sum(unmatched)
    
    return(labels2, labelMax)


def computePeakRanges(data):
    # computes peak ranges defined by adjacent minima in the data
    idx_max = find_peaks(data)
    if idx_max.size == 0:
        idx_max = []
        ranges = []
    
    idx_min = find_peaks(-1*data)
    idx_min = np.unique(np.concatenate((0, data.shape[0], idx_min)))
    ranges = np.zeros((idx_max.shape[0], 2))
    for i in range(idx_max.shape[0]):
        ranges[i,0] = np.max(idx_min[idx_min<idx_max[i]])
        ranges[i,1] = np.min(idx_min[idx_min>idx_max[i]])
    # remove degenerate ranges
    idx = ranges[:,0] == ranges[:,1]
    ranges = np.delete(ranges, idx, 0)
    return(ranges, idx_max)


def assignLabelsToPeaks(idx_max2, ranges, idx_max1, labels1):
    # assigns labels to peaks in idx_max2 based on matching ranges in idx_max1
    labels = np.zeros(idx_max2.shape[0])
    for i in range(idx_max2.shape):
        # find the range in fm[:,0] that contains the current peak in fm[:,1]
        in_range = np.logical_and(idx_max2[i] >= ranges[:,0], idx_max2[i] <= ranges[:,1])
        matching_ranges = np.argwhere(in_range)

        if matching_ranges.size != 0:
            # choose the closest peak if multiple ranges match
            if matching_ranges.size > 1:
                tmp = np.abs(idx_max1[matching_ranges] - idx_max2[i])
                closest_idx = tmp.argmin()
            else:
                matching_range = matching_ranges.copy()
            
            labels[i] = labels1[matching_range]

    return labels


def resolveOverlappingLabels(labels, idx_max1, idx_max2, labels1):
    # ensures no overlapping labels in the assigned labels
    unique_labels = np.unique(labels[labels > 0])
    for label in unique_labels:
        duplicates = np.argwhere(labels==label)
        if duplicates.size > 1:
            # keep the closest peak and reset others
            distances = np.abs(idx_max1[labels1==label] - idx_max2[duplicates])
            min_idx = distances.arg_min()
            duplicates = np.delete(duplicates, min_idx, 0)
            labels[duplicates] = 0
    return labels


def PreprocessingForPPD(t, lam, Labels, Locs, labelMax, FNm, th):
    K = lam.shape[0]
    # initialize output matrices
    IndicatorMatrix = np.full((K, labelMax), np.nan) 
    curvatures = np.zeros((K, labelMax))
    Heights = np.full((K, labelMax), np.nan)

    # assume t is uniformly spaced; compute time step
    dx = t[1] - t[0]

    # process each function
    for i in range(K):
        # extract the function values for the current parameter lambda
        fnm = FNm[:, i]

        # compute negative curvature (sec)
        negCurvature = -1*np.gradient(np.gradient(fnm,dx), dx)

        # Ensure non-negative curvature values
        negCurvature[negCurvature < 0] = 0

        # normalize negative curvature to [0,1] if possible 
        maxNegCurvature = negCurvature.max()
        if maxNegCurvature > 0:
            negCurvature = negCurvature / negCurvature
        
        # retrieve peak locations and labels for the current function
        locsCurrent = Locs[i]
        labelsCurrent = Labels[i]

        # select negative curvature values at specified peak locations
        negCurvSelected = negCurvature[locsCurrent]

        # update curvatures and heights matrices at the appropriate labels
        curvatures[i, labelsCurrent] = negCurvSelected
        Heights[i, labelsCurrent] = fnm[locsCurrent]

        # apply threshold to select significant peaks based on curvature
        significantLabels = labelsCurrent[negCurvSelected >= th]

        # update the indicator matrix for signficiant peaks
        IndicatorMatrix[i, significantLabels] = 1
    
    # compute heighs2 by multiply heighs with the indicator matrix 
    Heights2 = IndicatorMatrix * Heights

    return(IndicatorMatrix, Heights, Heights2)


def getPersistentPeaks(IndicatorMatrix):

    if IndicatorMatrix.size == 0:
        Clt2 = []
        return Clt2
    
    # count the number of ones (occurences) for each peak (ignore nans)
    occurrenceCounts = np.nansum(IndicatorMatrix, 1)

    data = np.append(occurrenceCounts, 0)
    if np.all(data == 0) or np.unique(occurrenceCounts).size == 1:
        Clt2 = []
        return Clt2
    
    # compute pairwise distances between observations
    pairwiseDistances = pdist(data, metric='euclidean')
    Y = linkage(pairwiseDistances, 'ward')

    # Cluster the data into the specified number of clusters
    clusterAssignments = fcluster(Y, 2, 'maxclust')
    referenceCluster = clusterAssignments[-1]
    clusterAssignments = clusterAssignments[:-1]

    # identify indices wehre cluster assignmetns differ from the reference
    Clt2 = np.argwhere(clusterAssignments != referenceCluster)

    return Clt2


def drawPPDBarChart(IndicatorMatrix, Heights, lam, idx_opt):
    lam_diff = lam[1] - lam[0]
    len_lam, labelMax = IndicatorMatrix.shape

    fig, ax = plt.subplots()

    for i in range(len_lam):
        label_all_peaks = np.argwhere(np.logical_not(np.isnan(Heights[i,:])))
        label_persistent_peaks = np.argwhere(np.logical_not(np.isnan(IndicatorMatrix[i,:])))

        if label_all_peaks.sum() == 0:
            continue

        for j in range(label_all_peaks.shape[0]):
            x = lam[i]
            y = label_all_peaks[j]

            if y in label_persistent_peaks:
                rect = Rectangle((x, y), lam_diff, 1, facecolor='black', edgecolor='none')
            else:
                rect = Rectangle((x, y), lam_diff, 0.5, 1, facecolor='grey', edgecolor='none')
            
            ax.add_patch(rect)
    
    for j in range(labelMax):
        plt.hlines(y=j+0.5, xmin=plt.xlim()[0], xmax=plt.xlim()[1])
    
    plt.axvline(x=lam[idx_opt], color="red", linestyles='dashed', linewdith=2)

    plt.xlim((lam[0], lam[-1]))
    plt.ylim((0.5, labelMax+0.5))

    plt.yticks(np.arange(1,labelMax+1))

    plt.xlabel('$\\lambda$')
    plt.ylabel('Peak Index')


def drawPPDSurface(t,lam,FNm,Heights,Locs,IndicatorMatrix,Labels,idx_opt):
    parula_colors = [
        (0.24, 0.15, 0.66),  # Dark blue
        (0.26, 0.44, 0.82),  # Medium blue
        (0.28, 0.70, 0.81),  # Cyan
        (0.60, 0.80, 0.44),  # Greenish-yellow
        (0.99, 0.81, 0.19)   # Yellow
    ]

    # Create the custom colormap
    parula_cmap = LinearSegmentedColormap.from_list('parula_custom', parula_colors)

    
    n_lams = lam.shape[0]
    labelMax = IndicatorMatrix.shape[1]

    LocationMatrix_full = np.full((n_lams, labelMax), np.nan)
    for i in range(n_lams):
        LocationMatrix_full[i, Labels[i]] = Locs[i]
    
    LocationMatrix_sig = LocationMatrix_full * IndicatorMatrix
    HeighMatrix_full = Heights.copy()
    HeightMatrix_sig = HeighMatrix_full * IndicatorMatrix

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(t, lam)
    ax.plot_surface(X, Y, FNm.T, cmap=parula_cmap, linewidth=0.5)

    plt.xlim(t[0], t[-1])

    ax.plot(t, lam[idx_opt]*np.ones(t.shape[0]), FNm[:, idx_opt], linestyle='dashed', linewidth=2, color='magenta')
    # loop through each label
    for j in range(labelMax):

        # find non-nan indices for full location matrix and plot
        idx_full = np.argwhere(np.logical_not(np.isnan(LocationMatrix_full[:, j])))
        ax.plot(t[LocationMatrix_full[idx_full, j]], lam[idx_full], HeighMatrix_full[idx_full, j], marker='o', color='black', linewidth=1.5, markersize=5, mfc='black')

        # find non-nan indices for significant location matrix and plot
        idx_sig = np.argwhere(np.logical_not(np.isnan(LocationMatrix_sig[:, j])))
        ax.plot(t[LocationMatrix_sig[:, j]], lam[idx_sig], HeightMatrix_sig[idx_sig, j], linestyle='dashed', linewidth=2, color='magenta', markersize=6)
    
    plt.xlabel('$t$')
    plt.ylabel('$\\lambda$')
    plt.zlabel('$\\hat{g}_\\lambda(t)$')

    plt.grid(True)
