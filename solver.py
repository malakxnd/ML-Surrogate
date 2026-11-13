import mpmath as mp # fun fact: mp stands for multi-precision

mp.mp.dps=120 # safe margin for calculations, esp the det
I=mp.mpc(0,1) # 0+1i

def MaterialConstants(**mc):
    m = dict(C11=7.41e10, C13=3.93e10, C33=8.36e10, C44=1.32e10, rho=5504.0, T0=298.0,
             e15=-0.138, e31=-0.160, e33=0.347, beta1=0.621e6, beta3=0.551e6, p3=-2.94e-6,
             k1=9.0, k3=9.0, eps11=8.26e-11, eps33=9.03e-11, ce=260.0,
             b1=2.14e-2, b3=19.2e-8, b_diff=9e5, b3s=4.7e-2,
             al1=1.05e-8, al3=0.58e-8, astar=3.7,
             mu0=4*mp.pi*1e-7, eps0=8.854e-12,
             tau_th=0.3, tau_q=4, tau_st=0.1,
             omega=2.3, a=6, F1=1.0, F2=1.0, x=0.1, t=0.9, H0=0.0)

    # if we ever wish to change the default values
    m.update(mc)

    m = {
        k:mp.mpf(v) if not isinstance(v, mp.mpf)
        else v
        for k,v in m.items()
    }

    m['c']=m['omega']/m['a']

    return m

# ----------------------------------------------------------------
# calulate non-diminsional coeffs, i named them g's
# ----------------------------------------------------------------
def g_coeff(m):
    C11=m['C11'];C13=m['C13'];C33=m['C33'];C44=m['C44'];rho=m['rho'];T0=m['T0']
    e15=m['e15'];e31=m['e31'];e33=m['e33'];be1=m['beta1'];be3=m['beta3'];p3=m['p3']
    k1=m['k1'];k3=m['k3'];ep11=m['eps11'];ep33=m['eps33'];ce=m['ce']
    b1=m['b1'];b3=m['b3'];b=m['b_diff'];b3s=m['b3s'];a1=m['al1'];a3=m['al3'];As=m['astar']
    mu0=m['mu0'];eps0=m['eps0'];H0=m['H0']

    g = {}

    g[1]=C44/C11; g[2]=(C13+C44)/C11; g[3]=(e15+e31)/e33
    g[4]= (C11/(T0*be1))
    g[5]=H0**2*mu0/C11; g[6]=(H0**2*eps0*mu0**2+rho)/rho
    g[7]=C33/C11; g[8]=e15/e33; g[9]=be3/be1; g[10]=C11*b3/(T0*b1*be1)
    g[11]=(e15+e31)*be1/(C11*p3); g[12]=e15*be1/(C11*p3); g[13]=e33*be1/(C11*p3)
    g[14]=ep11*be1/(e33*p3); g[15]=ep33*be1/(e33*p3)
    g[16]=C11*b3s/(T0*b1*p3)
    g[17]=a1*b*ce*rho/k1; g[18]=a3*b*ce*rho/k1
    g[19]=T0*a1*b1**2*be1*ce*rho/(C11**2*k1); g[20]=T0*a1*b1*b3*be1*ce*rho/(C11**2*k1)
    g[21]=T0*a3*b1**2*be1*ce*rho/(C11**2*k1); g[22]=T0*a3*b1*b3*be1*ce*rho/(C11**2*k1)
    g[23]=T0*a1*b1*b3s*be1*ce*rho/(C11*e33*k1); g[24]=T0*a3*b1*b3s*be1*ce*rho/(C11*e33*k1)
    g[25]=T0*a1*As*b1*ce*rho/(C11*k1); g[26]=T0*a3*As*b1*ce*rho/(C11*k1)
    g[27]=k3/k1; g[28]=T0*be1**2/(C11*ce*rho); g[29]=T0*be1*be3/(C11*ce*rho)
    g[30]=T0*be1*p3/(ce*e33*rho); g[31]=C11*As/(b1*ce*rho)

    return g

# ----------------------------------------------------------------
# build matrix coeffs, i named it as Mn(D)
# we can change the tau's but it must be that τ_q > τ_θ
# ----------------------------------------------------------------
def Mn_D(D, g, m, tau_th=None, tau_q=None, tau_st=None):
    a=m['a'];c=m['c']
    tth=m['tau_th'] if tau_th is None else tau_th
    tq=m['tau_q'] if tau_q is None else tau_q
    tst=m['tau_st'] if tau_st is None else tau_st
    ia=I*a
    M=mp.zeros(5,5)
    M[0,0]=(1+g[5])*ia**2+g[1]*D**2+g[6]*a**2*c**2
    M[0,1]=(g[2]+g[5])*ia*D 
    M[0,2]=g[3]*ia*D 
    M[0,3]=-ia 
    M[0,4]=-g[4]*ia

    M[1,0]=(g[2]+g[5])*ia*D
    M[1,1]=g[1]*ia**2+(g[7]+g[5])*D**2+g[6]*a**2*c**2
    M[1,2]=g[8]*ia**2+D**2
    M[1,3]=-g[9]*D 
    M[1,4]=-g[10]*D

    M[2,0]=g[11]*ia*D
    M[2,1]=g[12]*ia**2+g[13]*D**2
    M[2,2]=-g[14]*ia**2-g[15]*D**2
    M[2,3]=D; M[2,4]=g[16]*D

    opd=1-tst*ia*c
    M[3,0]=-g[19]*ia**3-g[21]*ia*D**2
    M[3,1]=-g[20]*D*ia**2-g[22]*D**3
    M[3,2]=g[23]*D*ia**2+g[24]*D**3
    M[3,3]=g[25]*ia**2+g[26]*D**2
    M[3,4]=g[17]*ia**2+g[18]*D**2+opd*ia*c

    opq=1-tq*ia*c; opt=1-tth*ia*c
    M[4,0]=-opq*g[28]*(-ia*c)*ia
    M[4,1]=-opq*g[29]*(-ia*c)*D
    M[4,2]=-opq*(-g[30])*(-ia*c)*D
    M[4,3]=opt*(ia**2+g[27]*D**2)-opq*(-ia*c)
    M[4,4]=-opq*g[31]*(-ia*c)

    return M

# ----------------------------------------------------------------------
# building eq18-eq23 from the paper, i used d's for renaming the coeffs
# ----------------------------------------------------------------------
def dcoef(m):
    C11=m['C11'];C13=m['C13'];C33=m['C33'];C44=m['C44'];rho=m['rho'];T0=m['T0']
    e15=m['e15'];e31=m['e31'];e33=m['e33'];be1=m['beta1'];be3=m['beta3'];p3=m['p3']
    ep11=m['eps11'];ep33=m['eps33']
    b1=m['b1'];b3=m['b3'];b=m['b_diff'];b3s=m['b3s'];As=m['astar']

    d = {}

    d[1]=C13/C11; d[2]=e31/e33
    d[3]=C11/(T0*be1); d[4]=C13/C11
    d[5]=C33/C11; d[6]=be3/be1
    d[7]=C11*b3/(T0*b1*be1); d[8]=C44/C11
    d[9]=e15/e33; d[10]=T0*be1*e15/(C11*e33)
    d[11]=T0*be1*ep11/e33**2; d[12]=T0*be1*e31/(C11*e33)
    d[13]=T0*be1/C11; d[14]=T0*be1*ep33/e33**2
    d[15]=T0*p3/e33; d[16]=C11*b3s/(b1*e33)
    d[17]=b*rho/b1; d[18]=T0*b1*be1*rho/C11**2
    d[19]=T0*b3*be1*rho/C11**2; d[20]=T0*b3s*be1*rho/(C11*e33)
    d[21]=T0*As*rho/C11

    return d

def phys_nd(v, D, d, m):
    # nondimensional stresses etc. for one mode. v=(U,W,Phi,T,C), D=d/dz.
    a=m['a']
    U,W,P,T,C=v
    Ux=I*a*U; Wz=D*W; Pz=D*P; Uz=D*U; Wx=I*a*W; Px=I*a*P
    sxx=Ux+d[1]*Wz+d[2]*Pz-T-d[3]*C
    szz=d[4]*Ux+d[5]*Wz+Pz-d[6]*T-d[7]*C
    sxz=d[8]*(Uz+Wx)+d[9]*Px
    Dx=d[10]*(Uz+Wx)-d[11]*Px
    Dz=d[12]*Ux+d[13]*Wz-d[14]*Pz+d[15]*T+d[16]*C
    mu=d[17]*C-d[18]*Ux-d[19]*Wz+d[20]*Pz+d[21]*T
    return [sxx,szz,sxz,Dx,Dz,mu]

# ----------------------------------------------------------------------
# build the polynomial from the det 
# have you noticed the a "wave number" in my paper? it appears almost 
# in all the equations, that's why we choose it as a scale let rho=D^2
# ----------------------------------------------------------------------
def get_poly(M_fun, scale):
    input=[scale*(k+1) for k in range(6)]
    output=[mp.det(M_fun(mp.sqrt(pt))) for pt in input]

    A =mp.matrix(6,6)
    for i, p in enumerate(input):
        for j in range(6):
            A[i,j]=p**j

    poly_coefs = mp.lu_solve(A, mp.matrix(output))
    return[poly_coefs[i] for i in range(5,-1,-1)] # highest first

def pick_roots(poly):
    r = mp.polyroots(poly, maxsteps=2000,extraprec=400)
    out = []
    for D in r:
        sol = mp.sqrt(D)
        # mp.sqrt(r2) gives one square root; if its real part is negative, 
        # -s flips it so we always store the root with a positive real part
        # since we don't know which sign mp.sqrt() will hand us in the form we want
        out.append(sol if mp.re(sol)>0 else -sol)
    return out

#---------------------------------------------------
# Build the matrix for the selected roots
#---------------------------------------------------
def null_vec(M_fun, r):
    M=M_fun(-r)
    '''I'll take 4 of the 5 equations, solve for the 4 unknowns by setting U=1, 
    then check whether my answer also satisfies the 5th equation'''
    best=None

    for drop in range(5):
        rows=[i for i in range(5) if i!=drop]
        A=mp.matrix(4,4); rhs_U=mp.matrix(4,1)

        for index,row in enumerate(rows):
            rhs_U[index]=-M[row,0]
            for j in range(4): A[index,j]=M[row,j+1]

        try:
            sol=mp.lu_solve(A,rhs_U)
        except ZeroDivisionError:
            continue

        v=[mp.mpc(1)]+[sol[i] for i in range(4)] # [1, w, phi, T, C]
        error=max(abs(sum(M[i,j]*v[j] for j in range(5))) for i in range(5)) # M*v=0
        if best is None or error<best[0]: best=(error,v)
    return best[1], best[0] # vector, error

def solve(m, **opt_taus):
    g=g_coeff(m)
    d=dcoef(m)
    Mf=lambda D: Mn_D(D,g,m,**opt_taus)

    poly=get_poly(Mf, m['a']**2)
    roots=pick_roots(poly)
    modes=[]
    phys_eqs=[]

    for r in roots:
        v,res=null_vec(Mf,r)
        modes.append(v)
        phys_eqs.append(phys_nd(v,-r,d,m))

    BC=mp.matrix(5,5)
    for j in range(5):
        BC[0,j]=phys_eqs[j][1]
        BC[1,j]=phys_eqs[j][2]
        BC[2,j]=modes[j][3]
        BC[3,j]=-roots[j]*modes[j][2]
        BC[4,j]=-roots[j]*modes[j][4]

    rhs=mp.matrix([-m['F1'],0,m['F2'],0,0])
    Mc=mp.lu_solve(BC,rhs)
    return dict(g=g,d=d,roots=roots,modes=modes,phys_eqs=phys_eqs,Mc=Mc,BC=BC,rhs=rhs,poly=poly)

def mapTo_ct(sol, m, z, x=None, t=None):
    x=m['x'] if x is None else x
    t=m['t'] if t is None else t

    expo=mp.exp(I*m['a']*(x-m['c']*t))
    out={}

    names=['u','w','phi','T','C']
    pn=['sxx','szz','sxz','Dx','Dz','mu']

    for k,nm in enumerate(names):
        out[nm]=mp.re(sum(sol['Mc'][n]*sol['modes'][n][k]*mp.exp(-sol['roots'][n]*z) for n in range(5))*expo)
    for k,nm in enumerate(pn):
        out[nm]=mp.re(sum(sol['Mc'][n]*sol['phys_eqs'][n][k]*mp.exp(-sol['roots'][n]*z) for n in range(5))*expo)
    return out


    # if we want to do amplitude analysis, we can calculate the absolute value of the complex solution, which is the envelope of the wavee
def mapTo_envelope(sol, z):
    out={}

    names=['u','w','phi','T','C']
    pn=['sxx','szz','sxz','Dx','Dz','mu']

    for k,nm in enumerate(names):
        out[nm]=abs(sum(sol['Mc'][n]*sol['modes'][n][k]*mp.exp(-sol['roots'][n]*z) for n in range(5)))
    for k,nm in enumerate(pn):
        out[nm]=abs(sum(sol['Mc'][n]*sol['phys_eqs'][n][k]*mp.exp(-sol['roots'][n]*z) for n in range(5)))
    return out 
















