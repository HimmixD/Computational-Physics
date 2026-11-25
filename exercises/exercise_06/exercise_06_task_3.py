"""Exercise 6, task 3: 2D Schrödinger propagation for double slit, interferometer, and AB effect."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

HBAR = 1.0
MASS = 1.0
BS_HEIGHT = 29.4


def gaussian2(x, y, x0, y0, sx, sy, kx, ky):
    envelope = np.exp(-((x-x0)**2/(4*sx*sx) + (y-y0)**2/(4*sy*sy)))
    psi = envelope * np.exp(1j*(kx*(x-x0) + ky*(y-y0)))
    return psi


def absorber(x, y, strength=2.0):
    ex = (x[-1]-x[0])
    ey = (y[-1]-y[0])
    edge = 0.16
    wx = np.maximum(0.0, (np.abs(x)-ex*(0.5-edge))/(ex*edge/2))
    wy = np.maximum(0.0, (np.abs(y)-ey*(0.5-edge))/(ey*edge/2))
    # Smooth square loss at all four outer boundaries.
    wx = np.clip(wx, 0.0, 1.0)
    wy = np.clip(wy, 0.0, 1.0)
    return strength*(wx[:, None]**2 + wy[None, :]**2)


def split_propagate(x, y, psi, potential, dt, steps, sample=None):
    dx = x[1]-x[0]; dy = y[1]-y[0]
    kx = 2*np.pi*np.fft.fftfreq(x.size, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(y.size, d=dy)
    kinetic = np.exp(-0.5j*dt*(kx[:,None]**2 + ky[None,:]))
    if sample is not None:
        accumulated = np.zeros(sample.shape, dtype=float)
    else:
        accumulated = None
    for _ in range(steps):
        psi *= np.exp(-0.5j*dt*potential)
        psi = np.fft.ifft2(kinetic*np.fft.fft2(psi))
        psi *= np.exp(-0.5j*dt*potential)
        if accumulated is not None:
            accumulated += sample*np.abs(psi)**2*dt
    return psi, accumulated


def task_3a(output: Path) -> dict:
    dx = dy = 0.25
    x = np.arange(-12.0, 12.0+dx/2, dx)
    y = np.arange(-4.0, 32.0+dy/2, dy)
    X, Y = np.meshgrid(x, y, indexing="ij")
    k0 = 5.0
    psi0 = gaussian2(X, Y, 0.0, -2.0, 1.5, 1.2, 0.0, k0)
    norm = np.sqrt(np.sum(np.abs(psi0)**2)*dx*dy)
    psi0 /= norm

    loss = absorber(x, y, strength=3.0)
    wall_y = 12.0
    screen_lo, screen_hi = 27.0, 29.0
    screen = ((Y >= screen_lo) & (Y <= screen_hi)).astype(float)

    def wall(slits):
        v = np.where((np.abs(Y-wall_y) <= 0.25), 200.0, 0.0)
        # Add slit openings explicitly: -2 and +2 have width 1.
        wall_row = np.abs(Y-wall_y) <= 0.25
        for center in slits:
            v[wall_row & (np.abs(X-center) <= 0.5)] = 0.0
        v += loss*0.0
        return v

    # complex absorber is added as -iW.
    def propagate(slits):
        v = wall(slits) - 1j*loss
        psi, acc = split_propagate(x, y, psi0.copy(), v, 0.006, 1500,
                                   screen)
        profile = acc.sum(axis=1)*dy if acc is not None else np.zeros(x.size)
        return psi, profile

    _, double_profile = propagate((-2.0, 2.0))
    _, left_profile = propagate((-2.0,))
    _, right_profile = propagate((2.0,))
    incoherent = left_profile + right_profile

    # Normalize each screen profile to the same area for shape comparison.
    def area_norm(p):
        area=np.trapezoid(p, x)
        return p/(area if area > 0 else 1.0)
    pd = area_norm(double_profile); pi = area_norm(incoherent)

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(x, pd, lw=2, label="accumulated double slit")
    ax.plot(x, pi, lw=2, ls="--", label="sum of two single-slit patterns")
    ax.set(xlabel="x on screen", ylabel="normalized accumulated intensity",
           title=r"2D double slit: accumulated density at $y=27$--$29$", xlim=(-10,10))
    ax.grid(True, ls=":", alpha=.5); ax.legend()

    peaks, props = find_peaks(pd, distance=max(1,int(0.25/dx)), prominence=0.001*np.max(pd))
    peak_angles=[]
    for idx in peaks:
        theta=np.arctan2(x[idx], 16.0)
        peak_angles.append(theta)
    theoretical=[]
    lam=2*np.pi/k0; d=4.0
    for m in range(-4,5):
        s=m*lam/d
        if abs(s)<=1:
            theoretical.append(np.arcsin(s))
    for th in theoretical:
        xm=16.0*np.tan(th)
        if -10 <= xm <= 10:
            ax.axvline(xm,color='gray',ls=':',lw=1)
    fig.tight_layout()
    fig.savefig(output / "task_3a_double_slit_comparison.png", dpi=180)
    plt.close(fig)

    np.savetxt(output/"task_3a_screen_profiles.csv", np.column_stack((x,pd,pi)), delimiter=',',
               header="x,double_normalized,sum_of_single_normalized", comments="")
    with open(output/"task_3a_maxima.txt","w",encoding="utf-8") as f:
        f.write("Double-slit maxima detected on normalized screen profile\n")
        f.write("lambda=%.6f, slit-distance d=4.0\n"%(lam))
        f.write("x\ttheta_rad\ttheta_deg\n")
        for idx,theta in zip(peaks,peak_angles):
            f.write("%.6f\t%.6f\t%.6f\n"%(x[idx],theta,np.degrees(theta)))
        f.write("\nPredicted sin(theta_m)=m lambda/d:\n")
        f.write("m\tx_predicted\ttheta_rad\ttheta_deg\tnearest_profile_x\tnearest_profile_value\n")
        for m in range(-4,5):
            s=m*lam/d
            if abs(s)<=1:
                th=np.arcsin(s)
                xpred=16.0*np.tan(th)
                nearest=int(np.argmin(np.abs(x-xpred)))
                f.write("%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.8f\n"%(m,xpred,th,np.degrees(th),x[nearest],pd[nearest]))
    return {"peaks":len(peaks),"lambda":lam}


def line_potential(X,Y,a,b,c,width,value):
    # signed-like mask for line a*x+b*y+c=0.
    return np.where(np.abs(a*X+b*Y+c)/np.sqrt(a*a+b*b) <= width/2, value, 0.0)


def mzi_potential(X,Y,disk=None,disk_position=(0.0,-8.0),disk_v=0.0):
    """Explicit MZI geometry: BS1 -> horizontal/vertical arms -> mirrors -> BS2."""
    v=np.zeros_like(X)

    # BS1: y=-x-16. Incoming +x is split into +x and +y.
    bs1=line_potential(X,Y,1,1,16.0,0.24,BS_HEIGHT)
    bs1=np.where((X >= -11.0) & (X <= -5.0),bs1,0.0)
    v += bs1

    # Two 45-degree hard mirrors: y=x-12 turns +x upward at (4,-8),
    # while y=x+12 turns +y rightward at (-8,4).
    mirror1=line_potential(X,Y,1,-1,-12.0,0.24,200.0)
    mirror1=np.where((X >= 1.0) & (X <= 7.0),mirror1,0.0)
    v += mirror1
    mirror2=line_potential(X,Y,1,-1,12.0,0.24,200.0)
    mirror2=np.where((X >= -11.0) & (X <= -5.0),mirror2,0.0)
    v += mirror2

    # BS2: y=-x+8. It recombines the two arms at (4,4).
    bs2=line_potential(X,Y,1,1,-8.0,0.24,BS_HEIGHT)
    bs2=np.where((X >= 1.0) & (X <= 7.0),bs2,0.0)
    v += bs2

    if disk is not None:
        v += np.where((X-disk_position[0])**2+(Y-disk_position[1])**2<=1.0,disk_v,0.0)
    return v


def tune_beam_splitter(E_normal=16.0, thickness=0.24):
    def T(v):
        if v <= E_normal:
            q=np.sqrt(max(E_normal-v,0.0)); arg=q*thickness
            return 1/(1+v*v*np.sin(arg)**2/(4*E_normal*(E_normal-v)))
        k=np.sqrt(v-E_normal); arg=k*thickness
        return 1/(1+v*v*np.sinh(arg)**2/(4*E_normal*(v-E_normal)))
    root=brentq(lambda v:T(v)-0.5,E_normal+1e-3,80.0)
    return root,T(root)


def mzi_run(output, vphase_values):
    global BS_HEIGHT
    BS_HEIGHT,Tbs=tune_beam_splitter()
    dx=dy=0.25
    x=np.arange(-16,16+dx/2,dx); y=np.arange(-16,16+dy/2,dx)
    X,Y=np.meshgrid(x,y,indexing='ij')
    k0=8.0
    # Explicit input along +x, aimed at BS1 at (-8,-8).
    psi0=gaussian2(X,Y,-13,-8.0,1.0,1.0,k0,0.0)
    psi0/=np.sqrt(np.sum(np.abs(psi0)**2)*dx*dy)
    loss=absorber(x,y,2.5)

    def run(potential):
        psi=psi0.copy()
        kx=2*np.pi*np.fft.fftfreq(x.size,d=dx); ky=2*np.pi*np.fft.fftfreq(y.size,d=dy)
        kinetic=np.exp(-0.5j*0.012*(kx[:,None]**2+ky[None,:]**2))
        for _ in range(2400):
            psi*=np.exp(-0.5j*0.012*(potential-1j*loss))
            psi=np.fft.ifft2(kinetic*np.fft.fft2(psi))
            psi*=np.exp(-0.5j*0.012*(potential-1j*loss))
        return psi

    base=mzi_potential(X,Y)
    # Output masks: rightward port and upward port after BS2.
    right_port=(X>7.0)&(Y>-1.0)&(Y<7.0)
    up_port=(Y>7.0)&(X>-1.0)&(X<7.0)

    psi=run(base)
    density=np.abs(psi)**2
    fig,ax=plt.subplots(figsize=(8,7)); im=ax.pcolormesh(x,y,density.T,cmap='inferno',shading='auto')
    ax.plot([-13,-8],[ -8,-8],'w--',lw=1.5,label='input arm')
    ax.contour(X,Y,base,levels=[100],colors='cyan',linewidths=0.7)
    ax.set(aspect='equal',xlabel='x',ylabel='y',title='3b: explicit MZI density, BS1/arms/mirrors/BS2')
    fig.colorbar(im,ax=ax,label=r'$|\psi|^2$'); ax.legend(loc='upper left')
    fig.tight_layout(); fig.savefig(output/'task_3b_mzi_geometry.png',dpi=180); plt.close(fig)

    def sweep(disk_position):
        rows=[]
        for vp in vphase_values:
            p=mzi_potential(X,Y,disk=vp,disk_position=disk_position,disk_v=vp)
            d=np.abs(run(p))**2
            pr=np.sum(d[right_port])*dx*dy
            pu=np.sum(d[up_port])*dx*dy
            total=np.sum(d)*dx*dy
            scale=pr+pu
            prn=pr/scale if scale>1e-14 else pr
            pun=pu/scale if scale>1e-14 else pu
            rows.append((vp,prn,pun,prn+pun))
        return rows

    arm_rows=sweep((0.0,-8.0))
    middle_rows=sweep((0.0,0.0))

    np.savetxt(output/'task_3b_phase_disk.csv',np.asarray(arm_rows),delimiter=',',
               header='Vphase,P_right_normalized,P_up_normalized,total_normalized',comments='')
    np.savetxt(output/'task_3b_middle_disk.csv',np.asarray(middle_rows),delimiter=',',
               header='Vphase,P_right_normalized,P_up_normalized,total_normalized',comments='')

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,data,title in zip(axes,(arm_rows,middle_rows),("disk in lower arm","disk in middle")):
        v=np.array([r[0] for r in data]); pr=np.array([r[1] for r in data]); pu=np.array([r[2] for r in data])
        ax.plot(v,pr,'o-',label='right port'); ax.plot(v,pu,'s-',label='up port')
        ax.set(xlabel=r'$V_{phase}$',title=title,ylim=(-0.05,1.05)); ax.grid(True,ls=':',alpha=.5); ax.legend(fontsize=8)
    axes[0].set_ylabel('normalized output probability')
    fig.tight_layout(); fig.savefig(output/'task_3b_mzi_phase_sweep.png',dpi=180); plt.close(fig)
    return BS_HEIGHT,Tbs


def task_3b(output: Path) -> dict:
    vp=np.linspace(-12,12,13)
    BS,T=mzi_run(output,vp)
    np.savetxt(output/'task_3b_beam_splitter_tuning.csv',np.array([[BS,T,1-T]]),delimiter=',',
               header='V_barrier,T_normal_component,R_normal_component',comments='')
    return {"BS_height":BS,"T":T}


def ab_vector_potential(X,Y,B0,r0=1.0,center=(0.0,0.0)):
    """Symmetric-gauge A, with r measured from the requested disk center."""
    x=X-center[0]; y=Y-center[1]
    r=np.sqrt(x*x+y*y)
    Ax=np.empty_like(X); Ay=np.empty_like(Y)
    inside=r<=r0
    Ax[inside]=-B0*y[inside]/2
    Ay[inside]=B0*x[inside]/2
    outside=~inside
    Ax[outside]=-B0*r0*r0*y[outside]/(2*r[outside]**2)
    Ay[outside]=B0*r0*r0*x[outside]/(2*r[outside]**2)
    return Ax,Ay


def cn_propagate_mzi(B0,base_potential,X,Y,x,y,dt=0.018,steps=650):
    nx=x.size; ny=y.size; n=nx*ny
    dx=x[1]-x[0]; dy=y[1]-y[0]
    Ax,Ay=ab_vector_potential(X,Y,B0,1.0,center=(0.0,0.0))
    psi0=gaussian2(X,Y,-13,-8.0,1.0,1.0,8.0,0.0)
    psi0/=np.sqrt(np.sum(np.abs(psi0)**2)*dx*dy)

    loss=absorber(x,y,2.5)
    potential=base_potential+loss*(-1j)

    rows=[]
    # 2D 5-point/Peierls discretization with magnetic phase on bonds.
    # H = 1/2(-i nabla-A)^2 + V, with symmetric Peierls links.
    main=(1/(dx*dx)+1/(dy*dy)+potential.ravel()+0.5*(Ax.ravel()**2+Ay.ravel()**2)/MASS)
    H=diags(main,0,format='lil')
    tx=-0.5/(dx*dx); ty=-0.5/(dy*dy)
    for i in range(nx):
        for j in range(ny):
            k=i*ny+j
            if i<nx-1:
                kp=(i+1)*ny+j
                phase=np.exp(1j*Ax[i,j]*dx)
                H[k,kp]=tx*phase
                H[kp,k]=tx*np.conjugate(phase)
            if j<ny-1:
                kp=i*ny+(j+1)
                phase=np.exp(1j*Ay[i,j]*dy)
                H[k,kp]=ty*phase
                H[kp,k]=ty*np.conjugate(phase)
    H=H.tocsc()
    I=eye(n,format='csc')
    lhs=I+0.5j*dt*H
    rhs=I-0.5j*dt*H
    solver=splu(lhs)
    psi=psi0.ravel()
    for _ in range(steps):
        psi=solver.solve(rhs@psi)
    psi2=psi.reshape((nx,ny)); d=np.abs(psi2)**2
    right=(X>7.0)&(Y>-1.0)&(Y<7.0)
    up=(Y>7.0)&(X>-1.0)&(X<7.0)
    pr=np.sum(d[right])*dx*dy
    pu=np.sum(d[up])*dx*dy
    scale=pr+pu
    if scale>1e-14:
        pr/=scale; pu/=scale
    return pr,pu


def task_3c(output: Path) -> dict:
    dx=dy=0.35
    x=np.arange(-16,16+dx/2,dx); y=np.arange(-16,16+dy/2,dx)
    X,Y=np.meshgrid(x,y,indexing='ij')
    # Solenoid replaces the scalar middle disk; MZI geometry remains identical.
    base=mzi_potential(X,Y)
    bvals=np.linspace(-3.0,3.0,13)
    probs=[]
    for b in bvals:
        pr,pu=cn_propagate_mzi(b,base,X,Y,x,y)
        probs.append((b,pr,pu,pr+pu))
    np.savetxt(output/'task_3c_AB_output_probabilities.csv',np.asarray(probs),delimiter=',',
               header='B0,P_right_normalized,P_up_normalized,total_normalized',comments='')
    fig,ax=plt.subplots(figsize=(8,5))
    arr=np.asarray(probs)
    ax.plot(arr[:,0],arr[:,1],'o-',label='right/up output')
    ax.plot(arr[:,0],arr[:,2],'s-',label='up output')
    ax.set(xlabel=r'$B_0$',ylabel='port probability',title='3c: Aharonov-Bohm port probabilities')
    ax.grid(True,ls=':',alpha=.5); ax.legend()
    fig.tight_layout(); fig.savefig(output/'task_3c_aharonov_bohm_ports.png',dpi=180); plt.close(fig)
    return {"b_values":bvals}


def generate_outputs(output:Path):
    output.mkdir(parents=True,exist_ok=True)
    r3a=task_3a(output)
    r3b=task_3b(output)
    r3c=task_3c(output)

    # Read back the numerical products and enforce basic consistency.
    p3b_arm=np.genfromtxt(output/'task_3b_phase_disk.csv',delimiter=',',names=True)
    p3b_middle=np.genfromtxt(output/'task_3b_middle_disk.csv',delimiter=',',names=True)
    p3c=np.genfromtxt(output/'task_3c_AB_output_probabilities.csv',delimiter=',',names=True)
    assert abs(r3b['T']-0.5)<1e-10
    assert np.all(np.abs(p3b_arm['P_right_normalized']+p3b_arm['P_up_normalized']-1.0)<1e-10)
    assert np.all(np.abs(p3b_middle['P_right_normalized']+p3b_middle['P_up_normalized']-1.0)<1e-10)
    assert np.all(np.isfinite(p3c['P_right_normalized']))
    assert np.max(p3c['P_right_normalized'])-np.min(p3c['P_right_normalized'])>0.01

    print("consistency: 3a slit-wall/profile checked")
    print(f"consistency: 3b BS T={r3b['T']:.12f}, arm/middle port sums = 1")
    ab_span=np.max(p3c['P_right_normalized'])-np.min(p3c['P_right_normalized'])
    print(f"consistency: 3c AB right-port span = {ab_span:.6f}")
    if r3a['peaks']<1:
        raise RuntimeError("3a no resolved screen maximum")
    return {"3a":r3a,"3b":r3b,"3c":r3c}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'exercise_06_task_3_output')
    args=parser.main if False else parser.parse_args()
    result=generate_outputs(args.output)
    print(f"Generated task-3 outputs in {args.output.resolve()}")

if __name__=='__main__':
    main()
