# Code for plotting bands and fatbands from Quantum Espresso

# Written by Soham S. Ghosh, 2019.


import numpy as np
np.set_printoptions(threshold=np.nan)
import sys, os, re, shutil
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



# This function checks if a variable is int
def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

# This function extracts thhe Fermi energy from the scf output file
def Ef(scf=None, key='Fermi'):
    Fermi_E = 'None'
    if scffile:
        f = open(scf, 'r')
        for line in f:
            if key in line:
                Fermi_E = float(line.strip().split()[4])
        f.close()
    return Fermi_E


# This function gets the high symmetry points from the output of bands.x
def hsym_pts(bandsout=None):
    hspts = np.zeros(0)
    if bandsout:
        f = open(bandsout,'r')
        for i in f:
            if "high-symmetry" in i:
                hspts = np.append(hspts,float(i.split()[-1]))
        f.close()
    return hspts

# This function gets the high symmetry labels from pw.x (with 'calculation=bands/nscf')
def hsym_labels(nscf=None):
    hslabels = []
    if nscf:
        f = open(nscf, 'r')
        flag = False  # used to know if the K-POINTS namelist has been reached
        for line in f:
            if 'K_POINTS' in line.upper():
                flag = True
            # check if it is a bandstructure calculation
            if "calculation" in line.lower():
                if ('nscf' not in line.lower()) and ('bands' not in line.lower()):
                    print ("This does not seem to be a bandstructure calculation...Aborting")
                    sys.exit(0)
            if flag:
                lst = line.strip().split()
                if len(lst) ==5:
                    hslabels.append(lst[-1].strip('!'))
        f.close()
    return hslabels


# This function plots the normal bands
def band_plot(gnufile,plotfile="fatbands.png",Ef=0.0,hspts='None',hslabels='None',Ebound='default',projection='None',lw=2.0,fontsize=18,savefig_dir='default',mul=100):

    # We set a list of colors
    lst_colors = ['red','green','yellow','darkgreen','orange', 'aquamarine']
    len_colors = len(lst_colors)
    # Hold the chunks for each band separately in a list (Eband) and append those lists to a big list (Ek)
    Ek, Eband = [], []
    f = open(gnufile,'r')
    for line in f:
        if line.strip() or line in ['\n', '\r\n']:
            Eband.append(line.strip().split())
        else:
            Ek.append(Eband)
            Eband = []
    Ek.append(Eband)
    f.close()
    # I don't like this part because I don;t like loading a potntially large file into memory.
    # But I need this to know the default x and y boundaries
    array= np.loadtxt(gnufile)
    k_array = array[:,0]
    E_array = array[:,1] - Ef
    if (Ebound is 'default'):
        Ebound=[np.min(E_array),np.max(E_array)]
    # Set x and y-axes bounds
    axes = plt.gca()
    axes.set_xlim(np.min(k_array), np.max(k_array))
    axes.set_ylim(Ebound)

    # plot normal band structure
    for band_index, band in enumerate(Ek):  # get the list for each band (band = [[k1, E(k1)], [k2, E(k2)], ...[Kn, E(kn)]])
        band = np.array(band, dtype='float64')  # change it from string
        # plot fatbands
        if projection !='None':
            lw = float(lw)/5
            for group_index, group in enumerate(projection):
                color = lst_colors[group_index%len_colors]
                # The following matches the weights to the band index and picks out the weights for
                # the band being plotted
                weights = group[:, 2][(group[:, 1].astype(int)) == (band_index+1)]
                #norm_weights = mul*np.array([float(i)/np.sum(weights) for i in weights])
                weights = mul*weights
                plt.scatter(band[:,0], band[:,1] - Ef, s=weights, c=color, alpha=0.2)
        plt.plot(band[:,0], band[:,1] - Ef,'k', lw=lw)  # plot each band
    # plot high symmetry lines
    if hspts.any():
        for xc in hspts:
            plt.axvline(lw=1,x=xc)
    # Show high-symmetry points
    # First check if you have high symmetry labels. If you do, make sure they match up with high symmetry points
    if hspts.any() and any(hslabels):
        if (len(hslabels) != len(hspts)):
            print ("unequal number of Symm points and Symm labels...Aborting")
            sys,exit(0)
        plt.xticks(hspts, hslabels)
    # plot Fermi level
    plt.axhline(linewidth=1, color='k')
    hfont = {'fontname':'sans-serif'}
    plt.ylabel('Energy (eV)', **hfont)
    if re.search(r'png', plotfile[-3:]):
        plt.savefig(plotfile, dpi=300)
    elif re.search(r'eps', plotfile[-3:]):
        plt.savefig(plotfile)
    else:
        print ("plotting file type unknown...Aborting")
        sys.exit(0)



def proj(projout, projdat, plotfile, mul, input_states='None', bandfun='band_plot'):

    # This function plots the normal bands along with orbital projected weights

    # Here we read the output of projwfc.x to link the state number to atom label and (l,m) values
    # By reading the data below "Atomic states used for projection".
    #This gives a bitmore flexibility than to hardcode which line number in the projwfc.x dat file to start
    # looking at. However it is vulnerable to modifications to the pattern of the out file.
    if input_states=='None':
        print ("I need at least one orbital to project...Abort")
        sys.exit(0)
    # split the group of states by comma into a list of strings
    lst_input_states = [s.strip() for s in input_states.split(",")]
    # input_states_obj is a list of lists, each inner level list made up of 1 or more states
    # Items in the inner list are states (either 'state_num' or ('atom','l','m') tuple
    input_states_obj = []
    # If the ...atom l m... style is used in input
    if re.search(r'[A-Za-z]', input_states):
        for states in lst_input_states:
            # for each group, findall creates a list of tuples, each tuple being of the form (atom l m)
            # each such group is then sppended to make a list of lists
            input_states_obj.append(re.findall(r'(\w+\d*)\s+(\d)\s+(\d)', states))
    else:       # If the state number style is used
        for states in lst_input_states:
            input_states_obj.append(re.findall(r'\d+', states))
    print (input_states_obj)
    num_of_groups = len(input_states_obj)

    # Here we read the projwfc.x output file to know which state number corresponds to which atom, l,m
    f = open(projout, 'r')
    dict_proj_index = {}
    for line in f:
        # we look for a line like the following
        # state #   3: atom   1 (S  ), wfc  2 (l=1 m= 2)
        if 'state' in line and 'atom' in line:
            state_num = line.strip().split()[2].strip(':')
            atom = re.sub(r'[(),]', '', line.strip().split()[5])
            orbital_obj = re.search(r'l\s*=\s*(\d)\s*m\s*\=\s*(\d)', line)
            if orbital_obj:
                l_val = orbital_obj.group(1)
                m_val = orbital_obj.group(2)
            else:
                print ("there is something wrong with the structure of projwfc.x output file...Abort")
                sys.exit(0)
            # Collect the information in a {state_num: [group_index, (atom, l, m)] dict}
            for group_index, states in enumerate(input_states_obj):
                if state_num in states:
                    dict_proj_index[state_num] = [group_index, (atom, l_val, m_val)]  # Store in a tuple
                elif (atom, l_val, m_val) in states:
                    dict_proj_index[state_num] = [group_index, (atom, l_val, m_val)]  # Store in a tuple
    print ("The projection will include following states:")
    for state_num in dict_proj_index.keys():
        print (state_num, dict_proj_index[state_num])
    print ("")
    print ("Summed in the following groups")
    for state_group in input_states_obj:
        print (state_group)
    print ("########################################")
    f.close()

    # Here we read the projection data file generated by projwfc.x
    f = open(projdat, 'r')
    list_proj = [[] for x in range(num_of_groups)]
    array_proj_sum_states = [[] for x in range(num_of_groups)]
    list_proj_single_state = []
    # The way I have structured it, its stricly to find the group index of the single state list.
    # Because when the time comes (in the if block following immediately) to append the
    # single state list to the full state list, the group which the single state list
    # should belong to has already been updated to the group which the next matching
    # single state list will belong to.
    # For this reason, I implement a list (lst_group_index) like a queue.
    # The new group index is appended to it at the right end. The old group index,
    # which is the index the single state list should be appended to, is poped out from the left.
    lst_group_index = []
    state_num = 'None'
    for line in f:
        lst = line.strip().split()
        # fish out lines with 4 or more inputs and whose 1st value is an integer
        # This is the heading of projection for a single orbital as a function of k and band.
        if len(lst) > 3 and isint(lst[0]):
            if lst[0] in dict_proj_index:
                key = lst[0]
                atom,l_val,m_val=dict_proj_index[key][-1][0],dict_proj_index[key][-1][1],dict_proj_index[key][-1][2]
            else:
                key,atom,l_val,m_val = 'None', 'None', 'None', 'None'
                # The following combined with the if len(lst) > 3 and isint(lst[0]) gives a strict criterion
            if key in dict_proj_index and atom in lst and l_val == lst[-2] and m_val ==lst[-1]:
                lst_group_index.append(dict_proj_index[key][0])
                state_num = int(key)
                # Here we append the list for a single state to its group list
                # The group list is one of the lists (at position = group_index) of the list list_proj
                if list_proj_single_state:
                    group_index = lst_group_index.pop(0)
                    list_proj[group_index].append(list_proj_single_state)
                    list_proj_single_state = []
            else:
                state_num = 'None'
        if len(lst) == 3 and state_num != 'None':
            list_proj_single_state.append([state_num, int(lst[0]), int(lst[1]), float(lst[2])])

    # We have one final list of single state weights that must be added to the full list
    # Some consistency checks
    if len(lst_group_index) != 1:
        print ("At the end of reading the projwfc.x data file, there should have been one group index left")
        print ("But there are  "+len(lst_group_index)+ "  left and the group index queue is"+ lst_group_index)
        print ("Aborting...")
        sys.exit(0)
    else:
        group_index = lst_group_index[0]
        list_proj[group_index].append(list_proj_single_state)
    some_list = []
    array_proj = np.array(list_proj)
    for group_index, group in enumerate(array_proj):
        for state in group:
            some_list.append(state[:,-1].tolist())
        some_array = (np.array(some_list)).sum(axis=0)
        if array_proj_sum_states[group_index]:
            array_proj_sum_states[group_index][:,-1] = some_array
        else:
            array_proj_sum_states[group_index] = group[0][:,[1,2,3]]

    # Important sorting mechanism coming up, I can't come up with this on my own If I lose this
    for group_index, group in enumerate(array_proj_sum_states):
        idx=np.lexsort((group[:,0],group[:,1]))
        array_proj_sum_states[group_index] = group[idx]

    print ("size of weight array--->", len(array_proj_sum_states))
    #for group_index, group in enumerate(array_proj_sum_states):
        #print ("group_index---->", group_index)
        #print ("group_size----->", len(group))
        #for item in group:
            #list_item = item.tolist()
            #print (int(list_item[0]), int(list_item[1]), float(list_item[2]))
    band_plot(gnufile, plotfile,Ef, hspts, hslabels, 'default', array_proj_sum_states, mul=mul)

##########################################################################################################################
#
#               User modified area begins below
#
##########################################################################################################################


# The following files must exist in the same directory
scffile = 'H3S.bands.out'             # Output from the scf pw.x calculation - to get Ef
nscffile = 'H3S.bands.in'             # input from the nscf bands calculation of pw.x - to get high-symmetry labels
bandsoutfile = 'H3S.plotbands.out'    # output of bands.x - to get the high-symmetry points
gnufile = 'H3S.plotbands.dat.dat'     # data file putput of bands.x
projout = "H3S.projwfc.out"           # Output file of projwfc.x
projdat = "H3S-proj.dat.projwfc_up"   # Output data file from projwfc.x
plotfile = "H3S.python.fatbands.png"  # name of the file you are plotting along with the extention

mul = 100   # parameter that controls the thickness of the fatbands

# Call the function to find Fermi energy
Ef = Ef(scffile)
# Call the fucntion to find high-symmetry points
hspts = hsym_pts(bandsoutfile)
# Call the function to find high-symmetry labels
hslabels=hsym_labels(nscffile)

# Call this if you want only the normal bandplot with no projections
#band_plot(gnufile, plotfile=plotfile, Ef, hspts,hslabels)

# Call this if you want normal bands and projections. This internally calls the function to plot normal bands.
# Give comma separated groups of states. Each group of states can be a single state, or in itself be a group of states.
# If they are a group of states their contribution will be summed.

# States can be given as numbers.
# these numbers correspond to the number of the states as given in projwfc.x output.
# mul makes the fatbands thicker or thinner
#Two examples are given below
proj(projout, projdat, plotfile, mul, "17 33 49, 2 3 4")  # Projections  from tates 17, 33 and 49 are summed and shown with a single color on the plot.
# In a second group, contributions from states 2, 3 and 4 are summed and shown in another color.
#proj(projout, projdat, "2 3 4")

# Alternatively, instead of state numbers you can use the format  <atom l m>. Two examples are given below.
# If there are multiple states between two commas, they will be summed.
#proj(projout, projdat, "H 0 1, Ca 2 1 Ca 2 2 Ca 2 3 Ca 2 4 Ca 2 5, Ca 1 1 Ca 1 2 Ca 1 3, Ca 0 1 ")
# The following xample shows projected bands for one single state - Hydrogen l=0, m=1
#proj(projout, projdat, "H 0 1")
