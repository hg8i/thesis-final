from __future__ import division
import sys
sys.path = ["/home/prime/dev/buildRoot/lib/"] + sys.path
print "Loading root"
import ROOT
from collections import defaultdict

import sys,math,os
import numpy as np

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator,FuncFormatter,FormatStrFormatter, AutoMinorLocator, ScalarFormatter)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
pgf_with_rc_fonts = {
    "font.family": "sans-serif",
    "font.size" : "15",
    "font.sans-serif": ["Helvetica"],
    "pgf.texsystem":"pdflatex",
    "pgf.preamble":[
                    r"\usepackage{amsmath}",
                    r"\usepackage[english]{babel}",
                    r"\usepackage{arev}",
                   ],
    "text.usetex":True,
    "text.latex.preamble":[
                    r'\usepackage{amsmath}',
                    # r'\usepackage{upgreek}', # VERY slow, only use if need upgreek
                   ],
}
mpl.rcParams.update(pgf_with_rc_fonts)

def ticksInside(removeXLabel=False,removeYLabel=False):
    """ Make atlas style ticks """
    ax=plt.gca()
    ax.tick_params(labeltop=False, labelright=False)
    plt.xlabel(ax.get_xlabel(), horizontalalignment='right', x=1.0)
    plt.ylabel(ax.get_ylabel(), horizontalalignment='right', y=1.0)
    ax.tick_params(axis='y',direction="in",labelleft=not removeYLabel,left=1,right=1,which='both')
    ax.tick_params(axis='x',direction="in",labelbottom=not removeXLabel,bottom=1, top=1,which='both')

def atlasStyle(position="nw",status="Internal",size=12,lumi=139,subnote=""):
    """ Make ATLAS style plot decorations """
    ax=plt.gca()
    # decide the positioning
    textx={"se":0.95,"nw":0.05,"ne":0.95,"n":0.5}[position]
    texty={"se":0.05,"nw":0.95,"ne":0.95,"n":0.95}[position]
    va = {"se":"bottom","nw":"top","ne":"top","n":"top"}[position]
    ha = {"se":"right","nw":"left","ne":"right","n":"center"}[position]
    # add label to plot
    lines = [r"\noindent \textbf{{\emph{{ATLAS}}}}",
             r"$\textstyle\sqrt{\text{s}}$ = 13 TeV, "+str(lumi)+r" fb$^{\text{-1}}$",
            ]
    lines.append(status)
    if subnote: lines.append(subnote)
    labelString = "\n".join(lines)
    plt.text(textx,texty, labelString,transform=ax.transAxes,va=va,ha=ha, family="sans-serif",size=size)
    ticksInside()

def plotObs(obs,x,ys,yb):
    plt.clf(); plt.cla()
    plt.figure(figsize=(5,5))
    grid = gridspec.GridSpec(1, 1, height_ratios=[1])
    grid.update(wspace=0.025, hspace=0.05)
    ax = plt.subplot(grid[0])

    width = x[1]-x[0]
    plt.step(x+width/2,ys,"C3",lw=1.0,label="S+B")
    plt.step(x+width/2,yb,"C0",lw=1.0,label="B-Only")
    plt.bar(x[x<obs],ys[x<obs],lw=0,color="C3",alpha=0.5,edgecolor="none",width=width)
    plt.bar(x[x>obs],yb[x>obs],lw=0,color="C0",alpha=0.5,edgecolor="none",width=width)

    ylim=list(plt.ylim())
    plt.plot([obs]*2,ylim,color="grey",label="Observed")
    plt.ylim(top=ylim[1]*1.5)

    # format axis labels as text
    formatterx = FuncFormatter(lambda x,y:"$\mathsf{{{0:.0f}}}$".format(int(x)))
    formattery = FuncFormatter(lambda x,y:"$\mathsf{{{0:.2f}}}$".format(x))
    # formattery = FuncFormatter(lambda x,y:str(x))
    ax.xaxis.set_major_formatter(formatterx)
    ax.yaxis.set_major_formatter(formattery)
    # minor tick locators
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


    # axis sytle and scales
    plt.ylabel("Probability Density")
    plt.xlabel(r"$N_\text{obs}$")
    # atlasStyle(position="ne")
    ticksInside()

    # safe figure
    plt.legend(loc=2,frameon=0,fontsize=12)
    plt.savefig("stat-nobs.png",bbox_inches="tight")
    plt.savefig("stat-nobs.pdf",bbox_inches="tight")

def plotLike(obs,x,ys,yb):
    plt.clf(); plt.cla()
    plt.figure(figsize=(5,5))
    grid = gridspec.GridSpec(1, 1, height_ratios=[1])
    grid.update(wspace=0.025, hspace=0.05)
    ax = plt.subplot(grid[0])

    # lSig = np.array([sum(ys[i:]) for i in range(len(ys))])
    # lSig = np.array([sum(ys[:i]) for i in range(len(ys))])
    # lBkg = np.array([sum(yb[i:]) for i in range(len(yb))])

    lambda_x = ys/yb
    lo=0.1
    hi=10

    sig=defaultdict(float)
    for xi,prob in enumerate(ys):
        lr = lambda_x[xi]
        sig[lr]+=prob
    aSig=np.array(sorted(sig.keys()))
    bSig=np.array([sig[i] for i in aSig])
    bSig/=sum(bSig[np.logical_and(aSig>lo,aSig<hi)])
    plt.step(aSig,bSig,"C3",lw=1,label="S+B")

    bkg=defaultdict(float)
    for xi,prob in enumerate(yb):
        lr = lambda_x[xi]
        bkg[lr]+=prob
    aBkg=np.array(sorted(bkg.keys()))
    bBkg=np.array([bkg[i] for i in aBkg])
    bBkg/=sum(bBkg[np.logical_and(aBkg>lo,aBkg<hi)])
    plt.step(aBkg,bBkg,"C0",lw=1,label="B-Only")

    xIndexOfObs = np.abs(x-obs).argmin()
    ylim=list(plt.ylim())
    plt.plot([lambda_x[xIndexOfObs]]*2,ylim,color="grey",label="Observed")

    plt.xlim(lo,hi)
    # plt.ylim(bottom=1e-4,top=1e0)
    plt.ylim(top=ylim[1]*3,bottom=1e-2)


    # plt.plot(x,l)
    width = [aSig[i]-aSig[i-1] for i in range(1,len(aSig))]
    # width.insert(0,width[-1])
    width.append(width[-1])
    width=np.array(width)
    bp=list(bSig); bp.pop(0); bp.append(0); bp=np.array(bp)
    t=lambda_x[xIndexOfObs]
    bp=bp[aSig<t]
    width=width[aSig<t]
    aSig=aSig[aSig<t]
    plt.bar(aSig+width/2,bp,lw=0,color="C3",alpha=0.5,edgecolor="none",width=width)
    # plt.step(x+width/2,l,"C3",lw=1.0,label="")

    # plt.plot(x,l)
    width = [aBkg[i]-aBkg[i-1] for i in range(1,len(aBkg))]
    # width.insert(0,width[-1])
    width.append(width[-1])
    width=np.array(width)
    bp=list(bBkg); bp.pop(0); bp.append(0); bp=np.array(bp)
    t=lambda_x[xIndexOfObs]
    bp=bp[aBkg>=t]
    width=width[aBkg>=t]
    aBkg=aBkg[aBkg>=t]
    plt.bar(aBkg+width/2,bp,lw=0,color="C0",alpha=0.5,edgecolor="none",width=width)


    plt.yscale("log")
    plt.xscale("log")

    # format axis labels as text
    formatterx = FuncFormatter(lambda x,y:"$\mathsf{{{0:.0f}}}$".format(int(x)))
    formattery = FuncFormatter(lambda x,y:"$\mathsf{{{0:.2f}}}$".format(x))
    # formattery = FuncFormatter(lambda x,y:str(x))
    ax.xaxis.set_major_formatter(formatterx)
    ax.yaxis.set_major_formatter(formattery)
    # minor tick locators

    # axis sytle and scales
    plt.xlabel(r"$\Lambda(N_\text{obs})$")
    plt.ylabel("Probability Density")
    # atlasStyle(position="ne")
    ticksInside()

    # safe figure
    plt.legend(loc=2,frameon=0,fontsize=12)
    plt.savefig("stat-likel.png",bbox_inches="tight")
    plt.savefig("stat-likel.pdf",bbox_inches="tight")

def run():
    """ plotting function """
    # set up plot

    n=10000000
    # n=1000000
    # n=100000
    # n=10000
    muBkg=500
    muSig=1000
    sigma=200
    obs = 700
    bkg = np.random.normal(muBkg,sigma,n)
    sig = np.random.normal(muSig,sigma,n)

    nBins=int(150/2)
    lo=0
    hi=1500
    hs = ROOT.TH1F("sig","sig",nBins,lo,hi)
    hb = ROOT.TH1F("bkg","bkg",nBins,lo,hi)
    for i in sig: hs.Fill(i)
    for i in bkg: hb.Fill(i)

    x  = np.array([hs.GetBinCenter(i) for i in range(1,nBins+1)])
    ys = np.array([hs.GetBinContent(i) for i in range(1,nBins+1)])
    yb = np.array([hb.GetBinContent(i) for i in range(1,nBins+1)])
    yb/=sum(yb)
    ys/=sum(ys)
    # print yb; quit()

    plotObs(obs,x,ys,yb)

    plotLike(obs,x,ys,yb)


os.popen("rm -f *png *png")
run()
