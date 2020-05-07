# //------------------------------------------------------------------------------ //
# //   Nonlinear curve fitting class                                               //
# //------------------------------------------------------------------------------ //
#    Version 1.2.1
#
# [Requirements]
#   - Python 3.x
#   - numpy
#   - scipy
#   - cdata.py (included in this git repository)
#
# [How to fit a data]
#   1. Import data as cdata.CSpectrum
#        e.g. data=cdata.CSpectrum("foo.dat").read()
#
#   2. Create and prepare fitting object
#        e.g. lf=lfitting.lfitting()               #-> creates fitting object
#             lf.lf.set_function("Lorentzian")     #-> selects model function
#             lf.set_data(data)                    #-> selects data
#             lf.set_param([1000,1300,10,0])       #-> sets initial guess parameters
#   3. Fit
#        e.g. fit_param=lf.fit()
#
#   4. Check results
#        fit() will automatically print fit results as well as variance (standard deviation)
#        of the fitted value. The definition of variance is compatible with that of Origin
#        or Matlab.
#        See https://jp.mathworks.com/help/stats/nlinfit.html
#        and https://jp.mathworks.com/help/releases/R2008a/toolbox/optim/index.html?/help/releases/R2008a/toolbox/optim/ug/f19096.html      
#        for details
#
# [Advanced use: convoluted fitting models]
#   Fitting functions ending with "Conv" convolutes instrument response function (IRF)
#   to the fitting model. This is useful for fitting time-resolved PL decay curves.
#   The instrument response curve should be prepared in a dat or ascii format (converted
#   from SPCM), and can be loaded using "cdata"
#      irf=cdata.CDecayCurve("foo.asc").read(mode="asc").generate_irf()
#   or
#      irf=cdata.CDecayCurve("foo.dat").read(mode="dat").generate_irf()
#   This will automatically subtract background and normalize the intensities.
#   Then, IRF should be loaded to this fitting module by setting the global variable
#      lfitting.irf=irf
#   Everything else is the same as the normal fitting, but the initial guess should be
#   good enough to obtain reliable results. Always check the resulting fitting curves.
#   When fitting to bi-exponential decay, the smaller decay component might not fit well
#   with standard settings. In such cases,
#    1) First try setting loss function to 'soft_l1'
#         e.g. lf.loss='soft_l1' #default: 'linear'
#       This will reduce the impact on the fitting residuals by the main decay component.
#    2) If 1) does not work, try fit_scalar() function
#         e.g. lf.fit_scalar() # instead of fit()
#       This will try to minimize weighted chi-squared
#         Sum [  (yi-fi)^2/yi/(n-p)  ],
#       where n is the number of data point, p is the number of parameters,
#       yi is the experimental data, and fi is the fitted data. See software manual for
#       SPCImage for this definition (page 10).
#
# [Advanced use: fix parameters]
#   You can set some of the parameters as 'fixed' parameters. More precisely, the minimization
#   is performed along the parameter axes where the 'mask' is set to True (default: all are True).
#   Other parameters are kept constant to the initial guess value.
#     e.g. lf.set_mask([True, False, True])
#          #-> this will fix 2nd parameter and try to minimize the fitting residual by adjusting
#          #   1st and 3rd parameters only.
#
# [Advanced use: adding fit functions]
#   Fit function should be defined in this class definition file. If you need frequent change in
#   in the model functions, consider directly calling scipy.optimize.least_squares() function.
#   Frequently used fitting functions such as Lorentzian, Gaussian, sine wave, ... should be defined
#   here. The fit function should take two arguments: x and p.
#     - x is a scalar or a one-dimensional array containing the x axis of measured data.
#     - p is a one-dimensional array of fitting parameters.
#   Using x and p, the fit function should return a scalar or an array corresponding to the Y value
#   of the fitted curve. A reference to the model functions should be added to the global variable
#   'fitmodel_funcs' with the dictionary key (string) being the search key during set_function() call.
#   

import math
import numpy as np
from scipy.optimize import least_squares, minimize, dual_annealing, brute, differential_evolution, basinhopping, shgo
import analysispy.data


# ---
# Fit function definition
#   func(x, p)
#     - x is a np.array
#     - p is a tuple
# ---
# p[0]: peak area
# p[1]: center
# p[2]: fwhm
# p[3]: offset
def Lorentz(x,p):
    return p[3]+2*p[0]/math.pi*p[2]*np.reciprocal(4*np.power(x-p[1],2)+p[2]**2)

   
def Gauss(x,p):
    a=p[0]
    x0=p[1]
    w=p[2]
    y0=p[3]
    return y0+a*np.exp(-(x-x0)**2/2/w**2)


# p[0:3] Lorentz
# p[4] coeff
def LorentzPlusLinear(x,p):
    return Lorentz(x,p[0:4])+x*p[4]
    
# p[0]: offset
# p[1]: peak area for peak#0
# p[2]: peak center for peak#0
# p[3]: peak fwhm for peak#0
# p[4]: peak area for peak#1
# ...
# len(p) should be 3n+1
def NLorentz(x,p):
    y=p[0]
    # p is a tuple
    i=1
    while i<len(p):
        y=y+Lorentz(x,np.append(p[i:i+3],0))
        i=i+3
    return y


#Exponential Decay With Offset
# p[0]: offset y0
# p[1]: time offset t0
# p[2]: decay time tau
# p[3]: amplitude at t=t0 a
# y=y0+a*exp(-(t-t0)/tau) (if t>=t0), y0 (if t<t0)

def MonoExpDecay(x,p):
    if isinstance(x,float) or isinstance(x,int):
        if x < p[1]:
            return p[0]
        else:
            return p[0]+p[3]/p[2]*math.exp(-(x-p[1])/p[2])
    else:
        y=np.ones(len(x))*p[0]
        mask=np.where(x>p[1])
        ymask=np.zeros(len(x))
        ymask[mask]=1
        y=y+p[3]/p[2]*ymask*np.exp(-(x-p[1])/p[2])
        return y

#Exponential Decay With Offset
# p[0]: offset y0
# p[1]: time offset t1
# p[2]: decay time tau1
# p[3]: amplitude at t=t0 a1
# p[4]: t2
# p[5]: tau2
# p[6]: a2

def BiExpDecay(x,p):
    if isinstance(x,float) or isinstance(x,int):
        y=p[0]
        if x >= p[1]:
            y=y+p[0]+p[3]/p[2]*math.exp(-(x-p[1])/p[2])
        if x >= p[4]:
            y=y+p[0]+p[6]/p[5]*math.exp(-(x-p[4])/p[5])
    else:
        y=np.ones(len(x))*p[0]
        mask=np.where(x>p[1])
        ymask=np.zeros(len(x))
        ymask[mask]=1
        y=y+p[3]/p[2]*ymask*np.exp(-(x-p[1])/p[2])
        mask=np.where(x>p[4])
        ymask=np.zeros(len(x))
        ymask[mask]=1
        y=y+p[6]/p[2]*ymask*np.exp(-(x-p[4])/p[5])
        return y

def MonoExpDecayConv(x,p):
    y=MonoExpDecay(x,p)-p[0]
    if irf is None:
        return y+p[0]
    else:
        #do not do convolutions for the background signal!
        return np.convolve(y,irf.y,mode='same')+p[0]
    
def BiExpDecayConv(x,p):
    y=BiExpDecay(x,p)-p[0]
    if irf is None:
        return y+p[0]
    else:
        #do not do convolutions for the background signal!
        return np.convolve(y,irf.y,mode='same')+p[0]

def BiExpDecayFixOffsetConv(x,p):
    pp=np.zeros(7)
    pp[0]=p[0]
    pp[1]=p[1]
    pp[2]=p[2] #tau
    pp[3]=p[3] #amp
    pp[4]=p[1] #t0
    pp[5]=p[4] #tau2
    pp[6]=p[5] #amp2
    y=BiExpDecay(x,pp)-pp[0]
    if irf is None:
        return y+pp[0]
    else:
        #do not do convolutions for the background signal!
        return np.convolve(y,irf.y,mode='same')+pp[0]

irf=None
log_residual_linear_factor=0

fitmodel_funcs={
    'Lorentz':Lorentz,
    'Gauss':Gauss,
    'NLorentz': NLorentz, 
    'LorentzPlusLinear': LorentzPlusLinear, 
    'MonoExpDecay': MonoExpDecay,
    'BiExpDecayConv': BiExpDecayConv, 
    'MonoExpDecayConv': MonoExpDecayConv,
    'BiExpDecayFixOffsetConv':BiExpDecayFixOffsetConv
}


# returns reference to the function
def get_function(funcname):
    if funcname in fitmodel_funcs:
        return fitmodel_funcs[funcname]
    return None
# returns complete array of parameters
def unmask(var_param, fixed_param, fixed_param_i):
    complete_param = np.copy(var_param) #not necessary?
    for i in range(0,len(fixed_param)):
        complete_param=np.insert(complete_param,fixed_param_i[i],fixed_param[i])
    return complete_param


    # calculates the residual vector
#   - p: free parameter
#   - args[0]: func
#   - args[1]: data
#   - args[2]: fixed paramater tuple
def residual(p,*args):
    x=args[1].x
    y_data=args[1].y
    pp=unmask(p,args[2][0],args[2][1])
    y_func=args[0](x,pp)
    return y_data-y_func

def residual_chisquared(p,*args):
    x=args[1].x
    y_data=args[1].y
    pp=unmask(p,args[2][0],args[2][1])
    y_func=args[0](x,pp)
    return np.sum(np.power(y_data-y_func,2)/y_data/(len(y_data)-len(p)))

def rms_residual(p,*args):
    x=args[1].x
    y_data=args[1].y
    pp=unmask(p,args[2][0],args[2][1])
    y_func=args[0](x,pp)
    return np.sqrt(np.sum(np.power(y_data-y_func,2))/len(x))


#returns a scalar (sum[y-f(p)]^2)
def residual_log(p, *args):
    x=args[1].x
    y_data=args[1].y
    offset=0
    #if amin(ydata)==0:
    #    offset=0.001
    pp=unmask(p,args[2][0], args[2][1])
    y_func=args[0](x,pp)
    return np.sum(np.power(np.log(y_data+offset)-np.log(y_func+offset),2))+log_residual_linear_factor*math.log(np.sum(np.power(y_data-y_func,2)))

class fit:
    def __init__(self):
        self.torelance = 1e-9
        self.method = 'trf'
        self.func=None
        self.param=[]
        self.fit_param=[]
        self.mask=[]
        self.data=analysispy.data.CSpectrum("")
        self.result=None
        self.tol=1e-9
        self.bounds=None
        self.cov=None
        self.chi_squared=[-1, -1]
        self.quiet=False
        self.calculate_covariance=True
        self.loss="linear" #one of: linear, soft-l1, huber, cauchy, arctan

    def set_function(self,funcname):
        self.func=get_function(funcname)

    def set_data(self,spectrum):
        self.data=spectrum

    def set_param(self,param0):
        self.param=np.array(param0)
        self.fit_param=np.array(param0)
        if len(self.mask)==0:
            mask=[]
            for i in range(0,len(param0)):
                mask.append(True)
            self.set_mask(mask)
        if self.bounds is None:
            lb=[-math.inf]*len(self.param)
            ub=[math.inf]*len(self.param)
            self.bounds=(lb,ub)

    def set_bounds(self,i,lower=None,upper=None):
        #print("Note: setting method to TRF to handle constrained least squares")
        self.method="trf"
        if lower is not None:
            self.bounds[0][i]=lower
        if upper is not None:
            self.bounds[1][i]=upper

    def set_mask(self,mask):
        self.mask=mask

    def apply_mask(self):
        fixed_param = []
        variable_param = []
        fixed_param_index = []
        lb=[]
        ub=[]
        for i in range(0,len(self.param)):
            if self.mask[i]:
                #variable
                variable_param.append(self.param[i])
                lb.append(self.bounds[0][i])
                ub.append(self.bounds[1][i])
            else:
                #fixed
                fixed_param.append(self.param[i])
                fixed_param_index.append(i)
        return variable_param,(fixed_param,fixed_param_index),(lb,ub)

    def fit(self, retry=False):
        verbose_flag=1
        if self.quiet:
            verbose_flag=0
        if retry:
            self.param=self.fit_param
        #check
        if self.func == None:
            raise ValueError("function name is not defined")
        if len(self.param)==0:
            raise ValueError("empty initial guess")
        if self.mask.count(True)==0:
            raise ValueError("all parameters are fixed or mask is not initialized")
        vp,fp,_bounds=self.apply_mask()
        #perform fitting
        #  least_squares(func(p, *args),p0,*args)
        try:
            result=least_squares(residual,vp,
                                method=self.method,
                                ftol=self.tol,
                                xtol=self.tol,
                                gtol=self.tol,
                                args=(self.func,self.data,fp),
                                bounds=_bounds,
                                verbose=verbose_flag,
                                jac='3-point',
                                loss=self.loss)
        except Exception as e:
            print(e)
            self.result=None
            return None
        self.result=result
        self.fit_param=unmask(result.x,fp[0],fp[1])
        chi0,chi1=self.calculate_chi_squared()
        if verbose_flag==1:
            print("Chi-squared (before): {}".format(chi0))
            print("Chi-squared (after): {}".format(chi1))
        #calculate covariance
        #see https://jp.mathworks.com/help/stats/nlinfit.html
        #and https://jp.mathworks.com/help/releases/R2008a/toolbox/optim/index.html?/help/releases/R2008a/toolbox/optim/ug/f19096.html
        self.variance=np.zeros(len(self.param))
        try:
            if self.calculate_covariance:
                jac=self.result.jac
                q,r=np.linalg.qr(jac)
                rinv=np.linalg.inv(r)
                rms=np.sum(np.power(self.data.y-self.func(self.data.x,self.fit_param),2)/(len(self.data.y)-len(self.fit_param)))
                covmat=np.matmul(rinv,np.transpose(rinv))*rms
                variance=np.sqrt(np.diag(covmat))
                j=0
                for i in range(0,len(self.param)):
                    if self.mask[i]:
                    #variable
                        self.variance[i]=variance[j]
                        j=j+1
                    #keep zero for fixed params
                j=0
        except Exception as e:
            print(e)
        if verbose_flag==1:
            for i in range(0,len(self.param)):
                if self.variance[i]!=0:
                    print("#Fit result {}: value {}, error {}".format(i,self.fit_param[i],self.variance[i]))
                    j=j+1
                else:
                    print("#Fit result {}: value {}, fixed".format(i,self.fit_param[i]))
        return self.fit_param
    
    def fit_scalar(self):
        #check
        if self.func == None:
            raise ValueError("function name is not defined")
        if len(self.param)==0:
            raise ValueError("empty initial guess")
        if self.mask.count(True)==0:
            raise ValueError("all parameters are fixed or mask is not initialized")
        vp,fp,_bounds=self.apply_mask()
        #perform fitting
        #  least_squares(func(p, *args),p0,*args)
        try:
            verbose_flag=1
            if self.quiet:
                verbose_flag=0
            result=minimize(residual_chisquared,vp,args=(self.func,self.data,fp),
                            method='Nelder-Mead',options={'maxiter':1e4, 'adaptive':True, 'fatol':1e-15, 'xatol':1e-15})
        except Exception as e:
            print(e)
            self.result=None
            return None
        self.result=result
        self.fit_param=unmask(result.x,fp[0],fp[1])
        chi0,chi1=self.calculate_chi_squared()
        if verbose_flag==1:
            print("Chi-squared (before): {}".format(chi0))
            print("Chi-squared (after): {}".format(chi1))
        return self.fit_param

    def global_optim(self):
        #check
        if self.func == None:
            raise ValueError("function name is not defined")
        if len(self.param)==0:
            raise ValueError("empty initial guess")
        if self.mask.count(True)==0:
            raise ValueError("all parameters are fixed or mask is not initialized")
        vp,fp,_bounds=self.apply_mask()
        #perform fitting
        #  least_squares(func(p, *args),p0,*args)
        try:
            result=basinhopping(rms_residual,vp,
                                T=1,
                                niter=200,
                                minimizer_kwargs={"args":(self.func,self.data,fp),"method":"trust-constr"},
                                disp=True
                                )
        except Exception as e:
            print(e)
            self.result=None
            print("Fit did not succeed")
            return None
        self.result=result
        self.fit_param=unmask(result.x,fp[0],fp[1])
        chi0,chi1=self.calculate_chi_squared()
        print("Chi-squared (before): {}".format(chi0))
        print("Chi-squared (after): {}".format(chi1))
        return self.fit_param

    def get_mse(self):
        n=len(self.data.x)
        p=len(self.param)
        f=self.func
        x=self.data.x
        y=self.data.y
        r=f(x,self.fit_param)-y
        return np.sum(np.power(r,2)/(n-p))

    def calculate_chi_squared(self):
        n=len(self.data.x)
        p=len(self.param)
        f=self.func
        x=self.data.x
        y=self.data.y
        #before optimization
        chi_before = np.sum(np.power(y-f(x,self.param),2)/y/(n-p))
        #after optimization
        chi_after = np.sum(np.power(y-f(x,self.fit_param),2)/y/(n-p))
        self.chi_squared=[chi_before, chi_after]
        return chi_before, chi_after


    #decay curve fit.
    def fit_log(self):
        #check
        if self.func == None:
            raise ValueError("function name is not defined")
        if len(self.param)==0:
            raise ValueError("empty initial guess")
        if self.mask.count(True)==0:
            raise ValueError("all parameters are fixed or mask is not initialized")
        vp,fp,_bounds=self.apply_mask()
        
        bnds=[]
        for vpi in vp:
            bnds.append((vpi*0.85,vpi*1.15))
        #perform fitting
        #  least_squares(func(p, *args),p0,*args)
        result=minimize(residual_log,vp,args=(self.func,self.data,fp),method='Nelder-Mead',options={'maxiter':1e3, 'adaptive':True, 'fatol':1e-15, 'xatol':1e-9})
        #result=dual_annealing(residual_log,bnds,args=(self.func,self.data,fp),maxiter=10000)
        print(result['message'])
        #result=minimize(residual,vp,method=self.method,ftol=self.tol,args=(self.func,self.data,fp),bounds=_bounds)
        #except:
        #    self.result=None
        #    return None
        self.result=result
        return unmask(result.x,fp[0],fp[1])
