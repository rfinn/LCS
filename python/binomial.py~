#!/usr/bin/env python

import numpy as N

###########################################################################
#for two given numbers of events calculate the binomial upper and
#lower limits for the 1-sided confidence interval of their ratio
#(n1/(n1 + n2).  Formulas are (18), (21), (26) from Gehrels, 1986, ApJ, 303,
#336. 0.8413, 0.9772, and 0.9987 correspond to the usual 1-sided
#intervals for 1, 2, and 3 sigma gaussian probabilities
def Binomial_lim(n1,n2):

    #n2=ntot-n1
#  my n1 = shift
#  my n2 = shift
#  my conflim_d = shift         
    conflim_d=0.84
#  my (imatch, iconf)
    
#  my (diff_d, conf_up, conf_dn, w_d, h_d, lambda_d, epsilon_d)

#  my (p1u_d, p1l_d, p2u_d)

  #allow only these limits
    conf_d = N.array([0.8413, 0.90, 0.95, 0.975, 0.9772, 0.990, 0.995, 0.9987, 0.999, 0.9995],'f')

  #'S' parameter for these limits
    S_d = N.array([1.0, 1.282, 1.645, 1.960, 2.00, 2.326, 2.576, 3.000, 3.090, 3.291],'f')
    
    ntot = n1 + n2
    
  #in this version just use the value of @conf_d that's closest to the
  #input confidence limit
  #find closest value

    diff_d = 1.e2
    for iconf in range(len(conf_d)):
	delta = abs(conf_d[iconf]-conflim_d)
	if (delta < diff_d):
	    diff_d = delta
	    imatch = iconf



  #upper limits
  #special cases
    if(n2 == 1):
	p1u_d = conflim_d**(1.0/ntot)
    elif(n2 == 0):#adding this as a place holder so the code doesn't crash.  Need to implement a proper handling of n2 = 0
	#p1u_d = 1-conflim_d**(1.0/ntot)+1
	p1u_d = 1.#added this based on Vandana's code
    elif (n1 == 0):
	#print "n1 = 0"
	p1u_d = 1 - (1 - conflim_d)**(1.0/n2)
    else:
    
    #general case
	lambda_d = (S_d[imatch]**2 - 3.0) / 6.0
	
	h_d = 2.0 * (1.0 / (2.0 * n2 - 1.0) + 1.0 / (2.0 * n1 + 1))**(-1)
	
	w_d = S_d[imatch]*((h_d + lambda_d)**0.5)/h_d + (1.0 /(2.0 * n2 - 1.0) - 1.0/(2.0*n1 + 1))*(lambda_d + 5.0/6.0 - 2.0/(3.0 * h_d))
	
	epsilon_d = 0.64 * (1 - S_d[imatch]) * N.exp(-n2)

	p1u_d = ( (n1 + 1.0) * N.exp(2.0 * w_d) + epsilon_d * n2)/( (n1 + 1.0) * N.exp(2.0 * w_d) + n2)

  #lower limits
  #recalculate but switch n2 and n1
  #special cases
    if (n1 == 0):
	p2u_d = 1.0
    else:
        if(n1 == 1):
	    p2u_d = conflim_d**(1.0/ntot)
	elif (n2 == 0):
	    p2u_d = 1 - (1 - conflim_d)**(1.0/n1)
	else:
    
    #general case
	    lambda_d = (S_d[imatch]**2 - 3.0) / 6.0

	    h_d = 2.0*(1.0/(2.0*n1 - 1.0) + 1.0/(2.0 * n2 + 1))**(-1)
	    #print "h_d = ",h_d,lambda_d
	    w_d = S_d[imatch]*((h_d + lambda_d)**0.5)/h_d + (1.0/(2.0*n1 - 1.0) - 1.0/(2.0 * n2 + 1))*(lambda_d + 5.0/6.0 - 2.0 / (3.0 * h_d))

	    epsilon_d = 0.64 * (1 - S_d[imatch]) * N.exp(-n1)

	    p2u_d = ( (n2 + 1.0) * N.exp(2.0 * w_d) + epsilon_d * n1)/( (n2 + 1.0) * N.exp(2.0 * w_d) + n1)

    p1l_d = 1 - p2u_d
  
    conf_dn = p1l_d
    conf_up = p1u_d
  
    return conf_dn, conf_up
