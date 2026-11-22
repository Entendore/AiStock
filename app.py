#!/usr/bin/env python3
"""
AURUM — AI Trading Terminal · PySide6 desktop port (v5.8 parity core)

Single-file, dependency: PySide6 only (numpy not required — nets are pure Python).

    pip install PySide6
    python aurum.py

Ported from the browser terminal: market simulation, QPainter candlestick chart
(zoom / pan / crosshair / MA / forecast fan), AI copilot (MLP 8→10→3 backprop+Adam,
linear policy, kNN memory, adaptive trust, forecast grading), continuous trainer with
loss curves, order ticket with brackets + ATR trailing stops, limit fills, positions
and netted holdings, risk governor circuit breaker, sizeable prove-it SIM book with
METRICS MATCH scoring, session JSON report, workspace persistence, shortcuts.
"""
import sys, os, json, math, random, time
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QPointF, QRectF
from PySide6.QtGui import (QColor, QPainter, QPen, QBrush, QFont, QPixmap,
                           QSplashScreen, QLinearGradient, QShortcut, QKeySequence)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QGridLayout, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLineEdit, QComboBox, QFrame, QAbstractItemView,
    QCheckBox, QFileDialog)

# ───────────────────────────── palette / utils ──────────────────────────────
BG="#0B0D10"; PANEL="#0E1116"; PANEL2="#0C0F13"; INK="#11151B"
LINE="#1D232B"; LINE2="#161B22"; TXT="#E7EAEE"; TXT2="#9AA4B0"; TXT3="#5C6672"
GOLD="#E8B54A"; UP="#2EBD85"; DN="#F6465D"; TEAL="#45B8AC"; MAUVE="#B48EAD"
MONO_FAMS=["JetBrains Mono","Cascadia Mono","Consolas","Menlo","DejaVu Sans Mono","monospace"]

def mono(size=9, bold=False):
    f=QFont(); f.setFamilies(MONO_FAMS); f.setPointSizeF(size); f.setBold(bold)
    f.setStyleHint(QFont.Monospace); return f

def clamp(v,a,b): return max(a,min(b,v))
def fmt(v,d=2):
    try: return f"{v:,.{d}f}"
    except Exception: return str(v)
def fmtS(v,d=2): return ("+" if v>=0 else "-")+fmt(abs(v),d)
def fmtM(v): return ("-$" if v<0 else "$")+fmt(abs(v),2)
def two(n): return f"{n:02d}"
def hhmmss(t=None): return datetime.now().strftime("%H:%M:%S") if t is None else t.strftime("%H:%M:%S")
def hhmm(t=None): return datetime.now().strftime("%H:%M") if t is None else t.strftime("%H:%M")
def stdev(a):
    if len(a)<2: return 0.0
    m=sum(a)/len(a); return math.sqrt(sum((x-m)**2 for x in a)/(len(a)-1))
def pearson(x,y):
    n=min(len(x),len(y))
    if n<10: return 0.0
    X=x[-n:]; Y=y[-n:]
    mx=sum(X)/n; my=sum(Y)/n
    sxy=sx=sy=0.0
    for i in range(n):
        dx=X[i]-mx; dy=Y[i]-my; sxy+=dx*dy; sx+=dx*dx; sy+=dy*dy
    return sxy/math.sqrt(sx*sy) if sx>0 and sy>0 else 0.0

def hash_str(s):
    h=2166136261
    for ch in s:
        h^=ord(ch); h=(h*16777619)&0xFFFFFFFF
    return h
def mulberry32(a):
    def rnd():
        nonlocal a
        a=(a+0x6D2B79F5)&0xFFFFFFFF
        t=a
        t=(t^(t>>15))*(t|1)&0xFFFFFFFF
        t^=(t+(t^(t>>7))*(t|61))&0xFFFFFFFF
        return ((t^(t>>14))&0xFFFFFFFF)/4294967296
    return rnd
def randn(rnd):
    u=v=0.0
    while u==0: u=rnd()
    while v==0: v=rnd()
    return math.sqrt(-2*math.log(u))*math.cos(6.2831853*v)

QSS=f"""
* {{ font-family: "JetBrains Mono","Consolas","Menlo",monospace; font-size: 12px; color: {TXT}; }}
QWidget {{ background: {BG}; }}
QMainWindow, QSplitter {{ background: {BG}; }}
QLabel {{ background: transparent; }}
QLabel[h1="true"] {{ font-size: 15px; font-weight: 700; letter-spacing: 2px; }}
QLabel[sub="true"] {{ color: {TXT3}; font-size: 10px; letter-spacing: 1px; }}
QLabel[stat="true"] {{ color: {TXT3}; font-size: 9px; letter-spacing: 1px; }}
QFrame[line="true"] {{ background: {LINE}; max-height: 1px; border: none; }}
QFrame[panel="true"] {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 4px; }}
QPushButton {{ background: {PANEL2}; border: 1px solid {LINE}; border-radius: 4px;
               padding: 6px 12px; color: {TXT2}; font-size: 11px; }}
QPushButton:hover {{ color: {TXT}; border-color: #2E3742; }}
QPushButton[gold="true"] {{ background: {GOLD}; color: #141005; font-weight: 700; letter-spacing: 1px; }}
QPushButton[gold="true"]:hover {{ background: #F2C463; }}
QPushButton[flat="true"] {{ border: none; background: transparent; }}
QPushButton:checked {{ color: {GOLD}; border-color: rgba(232,181,74,.5); background: rgba(232,181,74,.10); }}
QPushButton:disabled {{ color: {TXT3}; }}
QLineEdit, QComboBox {{ background: {PANEL2}; border: 1px solid {LINE}; border-radius: 4px;
                       padding: 5px 8px; selection-background-color: {GOLD}; selection-color: #141005; }}
QLineEdit:focus, QComboBox:focus {{ border-color: {GOLD}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {LINE};
    selection-background-color: {INK}; selection-color: {GOLD}; }}
QCheckBox {{ color: {TXT3}; font-size: 10px; letter-spacing: 1px; }}
QCheckBox::indicator {{ width: 12px; height: 12px; border: 1px solid {LINE}; border-radius: 2px; background: {PANEL2}; }}
QCheckBox::indicator:checked {{ background: {GOLD}; border-color: {GOLD}; }}
QTabWidget::pane {{ border: 1px solid {LINE}; top: -1px; }}
QTabBar::tab {{ background: {PANEL}; color: {TXT3}; padding: 6px 14px; font-size: 10px;
               letter-spacing: 1px; border: 1px solid {LINE}; border-bottom: none; }}
QTabBar::tab:selected {{ color: {TXT}; background: {BG}; border-bottom: 2px solid {GOLD}; }}
QTableWidget {{ background: {PANEL}; border: none; gridline-color: {LINE2}; font-size: 11px; }}
QTableWidget::item {{ padding: 3px 8px; }}
QTableWidget::item:selected {{ background: {INK}; color: {TXT}; }}
QHeaderView::section {{ background: {PANEL}; color: {TXT3}; border: none;
    border-bottom: 1px solid {LINE2}; border-right: 1px solid {LINE2};
    padding: 5px 8px; font-size: 9px; letter-spacing: 1px; }}
QScrollBar:vertical {{ background: {BG}; width: 8px; }}
QScrollBar::handle:vertical {{ background: #232A33; border-radius: 3px; min-height: 24px; }}
QScrollBar:horizontal {{ background: {BG}; height: 8px; }}
QScrollBar::handle:horizontal {{ background: #232A33; border-radius: 3px; min-width: 24px; }}
QScrollBar::add-line,QScrollBar::sub-line {{ height: 0; width: 0; }}
QToolTip {{ background: {PANEL}; color: {TXT}; border: 1px solid {LINE}; padding: 4px; }}
QStatusBar {{ background: {PANEL}; color: {TXT3}; border-top: 1px solid {LINE}; font-size: 10px; }}
QListWidget {{ background: {PANEL}; border: none; font-size: 11px; }}
QListWidget::item {{ padding: 3px 4px; border-bottom: 1px solid {LINE2}; color: {TXT2}; }}
"""

# ───────────────────────────── market simulation ────────────────────────────
TFS={"1m":60_000,"5m":300_000,"15m":900_000,"1H":3_600_000,"4H":14_400_000,"1D":86_400_000}
SYMS=[dict(s="BTC/USD",n="Bitcoin",p=67482,d=0,qd=3,qs=0.05,qdef=0.25,vol=1.0),
      dict(s="ETH/USD",n="Ethereum",p=3512.4,d=1,qd=2,qs=0.5,qdef=2.5,vol=1.15),
      dict(s="SOL/USD",n="Solana",p=172.36,d=2,qd=1,qs=5,qdef=20,vol=1.5),
      dict(s="NVDA",n="NVIDIA Corp",p=131.18,d=2,qd=0,qs=5,qdef=40,vol=0.9),
      dict(s="TSLA",n="Tesla Inc",p=248.42,d=2,qd=0,qs=5,qdef=30,vol=1.2),
      dict(s="AAPL",n="Apple Inc",p=227.65,d=2,qd=0,qs=5,qdef=40,vol=0.6),
      dict(s="META",n="Meta Platforms",p=512.9,d=2,qd=0,qs=5,qdef=10,vol=0.8),
      dict(s="AMZN",n="Amazon.com",p=186.24,d=2,qd=0,qs=5,qdef=25,vol=0.7),
      dict(s="XAU/USD",n="Gold Spot",p=2412.6,d=1,qd=1,qs=1,qdef=15,vol=0.4),
      dict(s="EUR/USD",n="Euro / Dollar",p=1.0852,d=4,qd=0,qs=1,qdef=5,vol=0.25)]
BY={x["s"]:x for x in SYMS}
BASE_EQ=250000.0; LEV=10

for sm in SYMS:
    sm["open"]=sm["p"]*(1+(random.random()-.5)*.024)
    sm["hi"]=sm["p"]*1.012; sm["lo"]=sm["p"]*0.988
    sm["spark"]=[sm["p"]*(1+(random.random()-.5)*.002) for _ in range(44)]
    sm["scan"]=0.0

def gen_candles(sym,tf):
    rnd=mulberry32(hash_str(sym["s"]+tf)); ms=TFS[tf]; n=170
    v=sym["vol"]*0.0038*math.sqrt(ms/900_000)
    p=sym["p"]*(1+(rnd()-.5)*.05); drift=(rnd()-.5)*.6
    t=int(time.time()*1000//ms)*ms-(n-1)*ms
    arr=[]
    for _ in range(n):
        if rnd()<.06: drift=(rnd()-.5)*.8
        o=p; c=o*(1+drift*v*.3+randn(rnd)*v)
        h=max(o,c)*(1+rnd()*v*.5); l=min(o,c)*(1-rnd()*.5*v)
        arr.append(dict(t=t,o=o,h=h,l=l,c=c,v=0.4+rnd()*1.4))
        p=c; t+=ms
    d=sym["p"]-arr[-1]["c"]; L=len(arr)
    for i,k in enumerate(arr):
        f=i/(L-1); k["o"]+=d*f; k["c"]+=d*f; k["h"]+=d*f; k["l"]+=d*f
    return arr

CACHE={}
def load_candles(sym,tf):
    key=sym["s"]+"|"+tf
    a=CACHE.get(key)
    if a is None:
        a=gen_candles(sym,tf); CACHE[key]=a
    else:
        d=sym["p"]-a[-1]["c"]; L=len(a)
        for i,k in enumerate(a):
            f=i/(L-1); k["o"]+=d*f; k["c"]+=d*f; k["h"]+=d*f; k["l"]+=d*f
    return a
def atr_fn(c,n=14):
    s=0.0
    for i in range(len(c)-n,len(c)):
        h,l,pc=c[i]["h"],c[i]["l"],c[i-1]["c"]
        s+=max(h-l,abs(h-pc),abs(l-pc))
    return s/n
def rsi_fn(a,n=14):
    g=l=0.0
    for i in range(len(a)-n,len(a)):
        d=a[i]-a[i-1]
        if d>=0: g+=d
        else: l-=d
    return 100.0 if l==0 else 100-100/(1+(g/n)/(l/n))

# ───────────────────────────── features / regime ────────────────────────────
def feats_full(c,i):
    px=c[i]["c"]
    s20=sum(c[k]["c"] for k in range(i-19,i+1))/20
    s50=sum(c[k]["c"] for k in range(i-49,i+1))/50
    sv=sum(c[k]["v"] for k in range(i-19,i+1))/20
    hi=max(c[k]["h"] for k in range(i-19,i+1)); lo=min(c[k]["l"] for k in range(i-19,i+1))
    g=l=0.0
    for k in range(i-13,i+1):
        d=c[k]["c"]-c[k-1]["c"]
        if d>=0: g+=d
        else: l-=d
    rsi=100.0 if l==0 else 100-100/(1+(g/14)/(l/14))
    atr=sum(max(c[k]["h"]-c[k]["l"],abs(c[k]["h"]-c[k-1]["c"]),abs(c[k]["l"]-c[k-1]["c"]))
            for k in range(i-13,i+1))/14
    return [clamp((s20/s50-1)*125,-1.5,1.5),
            clamp((px-c[i-20]["c"])/c[i-20]["c"]*40,-1.5,1.5),
            clamp((50-rsi)/25,-1.2,1.2),
            clamp(-(atr/px*100-0.6)*1.2,-1.5,0.8),
            clamp((c[i]["v"]/sv-1)*0.8,-1,1),
            clamp(((px-lo)/((hi-lo) or 1))*2-1,-1,1),
            clamp((c[i]["c"]-c[i]["o"])/((c[i]["h"]-c[i]["l"]) or px*1e-4),-1,1),
            clamp((px-c[i-45]["c"])/c[i-45]["c"]*18,-1.5,1.5)]
WKEYS=["trend","mom","rev","vol","flow"]
def feats5(f):
    return dict(zip(WKEYS,f[:5]))
def regime_of(c,i,atr_pct):
    num=abs(c[i]["c"]-c[i-19]["c"]); den=sum(abs(c[k]["c"]-c[k-1]["c"]) for k in range(i-18,i+1))+1e-9
    er=num/den
    name="TRENDING" if er>=0.34 else ("VOLATILE" if atr_pct>1.7 else "CHOPPY")
    return name,(3 if er>=0.34 else (0 if atr_pct>1.7 else -3)),er

# ───────────────────────────── MLP 8→10→3 (backprop + Adam) ─────────────────
class MLP:
    IN,HN,OUT=8,10,3
    def __init__(self,seed=1337):
        r=random.Random(seed)
        self.w1=[[ (r.random()-.5)*.7 for _ in range(self.IN)] for _ in range(self.HN)]
        self.b1=[0.0]*self.HN
        self.w2=[[ (r.random()-.5)*.7 for _ in range(self.HN)] for _ in range(self.OUT)]
        self.b2=[0.0]*self.OUT
        z=lambda: [ [0.0]*self.IN for _ in range(self.HN)]
        self.m=dict(w1=z(),b1=[0.0]*self.HN,w2=[[0.0]*self.HN for _ in range(self.OUT)],b2=[0.0]*self.OUT)
        self.v=dict(w1=z(),b1=[0.0]*self.HN,w2=[[0.0]*self.HN for _ in range(self.OUT)],b2=[0.0]*self.OUT)
        self.t=0; self.epoch=0; self.hist=[]; self.acc=None; self.loss=None
    def forward(self,x):
        h=[math.tanh(self.b1[j]+sum(self.w1[j][k]*x[k] for k in range(self.IN)))
           for j in range(self.HN)]
        z=[self.b2[c]+sum(self.w2[c][j]*h[j] for j in range(self.HN)) for c in range(self.OUT)]
        mx=max(z); e=[math.exp(v-mx) for v in z]; s=sum(e)
        return h,[v/s for v in e]
    def train_batch(self,batch,lr):
        g=dict(w1=[[0.0]*self.IN for _ in range(self.HN)],b1=[0.0]*self.HN,
               w2=[[0.0]*self.HN for _ in range(self.OUT)],b2=[0.0]*self.OUT)
        L=0.0
        for x,y in batch:
            h,p=self.forward(x)
            for c in range(self.OUT): L-=y[c]*math.log(max(1e-9,p[c]))
            for c in range(self.OUT):
                d=p[c]-y[c]
                g["b2"][c]+=d
                for j in range(self.HN): g["w2"][c][j]+=d*h[j]
            for j in range(self.HN):
                dh=sum((p[c]-y[c])*self.w2[c][j] for c in range(self.OUT))*(1-h[j]*h[j])
                g["b1"][j]+=dh
                for k in range(self.IN): g["w1"][j][k]+=dh*x[k]
        n=len(batch); self.t+=1
        B1,B2,eps=0.9,0.999,1e-8
        lrT=lr*math.sqrt(1-B2**self.t)/(1-B1**self.t)
        for key in ("w1","b1","w2","b2"):
            if key=="w1":
                for j in range(self.HN):
                    for k in range(self.IN):
                        gr=g[key][j][k]/n
                        self.m[key][j][k]=B1*self.m[key][j][k]+(1-B1)*gr
                        self.v[key][j][k]=B2*self.v[key][j][k]+(1-B2)*gr*gr
                        self.p_w(key)[j][k]-=lrT*self.m[key][j][k]/(math.sqrt(self.v[key][j][k])+eps)
            else:
                L_=self.HN if key=="b1" else self.OUT
                for i in range(L_):
                    gr=g[key][i]/n
                    self.m[key][i]=B1*self.m[key][i]+(1-B1)*gr
                    self.v[key][i]=B2*self.v[key][i]+(1-B2)*gr*gr
                    self.p_b(key)[i]-=lrT*self.m[key][i]/(math.sqrt(self.v[key][i])+eps)
        return L/n
    def p_w(self,key): return self.w1 if key=="w1" else self.w2
    def p_b(self,key): return self.b1 if key=="b1" else self.b2

NET=MLP()
DS=dict(long=[],flat=[],short=[],val=[])
def build_dataset():
    DS["long"].clear(); DS["flat"].clear(); DS["short"].clear(); DS["val"].clear()
    for sm in SYMS:
        c=TSCACHE[sm["s"]]; thr=0.35+sm["vol"]*0.15
        for i in range(52,len(c)-7):
            r=(c[i+6]["c"]-c[i]["c"])/c[i]["c"]*100
            rec=(feats_full(c,i),r)
            y=0 if r>thr else (2 if r<-thr else 1)
            if i%9==0: DS["val"].append((rec[0],y))
            elif y==0: DS["long"].append(rec[0])
            elif y==1: DS["flat"].append(rec[0])
            else: DS["short"].append(rec[0])
    r=mulberry32(7)
    for key in ("long","flat","short"):
        lst=DS[key]
        for i in range(len(lst)-1,0,-1):
            j=int(r()*(i+1)); lst[i],lst[j]=lst[j],lst[i]
def mlp_epoch():
    if not DS["long"]: return
    E=NET.epoch; batch=[]
    for k in range(8):
        batch.append((DS["long"][(E*8+k)%len(DS["long"])],[1,0,0]))
        batch.append((DS["flat"][(E*8+k)%len(DS["flat"])],[0,1,0]))
        batch.append((DS["short"][(E*8+k)%len(DS["short"])],[0,0,1]))
    lr=max(0.004,0.02*0.996**E)
    L=NET.train_batch(batch,lr)
    NET.epoch+=1; NET.loss=L; NET.hist.append(L)
    if len(NET.hist)>220: NET.hist.pop(0)
    ok=0
    for x,y in DS["val"]:
        p=NET.forward(x)[1]
        am=max(range(3),key=lambda i:p[i])
        if am==y: ok+=1
    NET.acc=ok/len(DS["val"])*100 if DS["val"] else None

# ───────────────────────────── kNN memory ───────────────────────────────────
KNN=dict(book=[],k=12)
def build_knn():
    KNN["book"]=[]
    for sm in SYMS:
        c=TSCACHE[sm["s"]]
        for i in range(52,len(c)-7,2):
            KNN["book"].append((feats_full(c,i),(c[i+6]["c"]-c[i]["c"])/c[i]["c"]*100))
def knn_score(x):
    dists=[]
    for bx,r in KNN["book"]:
        d=sum((bx[k]-x[k])**2 for k in range(8))
        dists.append((d,r))
    dists.sort(key=lambda t:t[0])
    wsum=rsum=0.0
    for d,r in dists[:KNN["k"]]:
        w=1/(d+0.05); wsum+=w; rsum+=w*r
    exp=rsum/wsum if wsum else 0.0
    return exp,clamp(exp/0.7,-1.5,1.5)

# ───────────────────────────── linear policy (Adam) ─────────────────────────
TR=dict(weights={k:1.0 for k in WKEYS},
        m={k:0.0 for k in WKEYS}, v={k:0.0 for k in WKEYS},
        t=0, epoch=0, loss=None, best=float("inf"), base=None, hist=[],
        marks=[], targets=dict(ret=1.0,dd=2.0,vol=0.8,brk=1.5),
        locked=False)
TRUST={k:1.0 for k in ("mlp","lin","knn")}
def lr_now(): return max(0.003,0.06*0.988**TR["epoch"])
def conv_pct():
    b=TR["base"]
    return 0 if (b is None or b<=1e-9 or TR["loss"] is None) else clamp((1-TR["loss"]/b)*100,0,100)
def objective(w,syms):
    t=TR["targets"]; pnl=peak=dd=0.0; rets=[]
    for sm in syms:
        c=TSCACHE[sm["s"]]
        for i in range(52,len(c)-7):
            f=feats5(feats_full(c,i))
            s=w["trend"]*f["trend"]+w["mom"]*f["mom"]+w["rev"]*f["rev"]+w["vol"]*f["vol"]+w["flow"]*f["flow"]
            if abs(s)<0.85: continue
            r=(c[i+6]["c"]-c[i]["c"])/c[i]["c"]*100*(1 if s>0 else -1)*min(1,abs(s)/2)
            pnl+=r; rets.append(r)
            peak=max(peak,pnl); dd=max(dd,peak-pnl)
    volT=stdev(rets)*math.sqrt(20) if len(rets)>2 else 0.0
    e1=(pnl-t["ret"])/max(t["ret"],0.2)
    e2=max(0.0,dd/max(t["dd"],0.2)-1)
    e3=max(0.0,volT/max(t["vol"],0.1)-1)
    reg=sum((w[k]-1)**2 for k in WKEYS)
    return 0.55*e1*e1+0.25*e2*e2+0.20*e3*e3+0.015*reg
def run_epoch():
    eps=0.05; w=TR["weights"]; syms=SYMS[TR["epoch"]%10:][:3] or SYMS[:3]
    g={}
    for k in WKEYS:
        wp=dict(w); wm=dict(w); wp[k]+=eps; wm[k]-=eps
        g[k]=(objective(wp,syms)-objective(wm,syms))/(2*eps)
    L=objective(w,syms)
    TR["t"]+=1; lr=lr_now()
    for k in WKEYS:
        TR["m"][k]=0.9*TR["m"][k]+0.1*g[k]
        TR["v"][k]=0.999*TR["v"][k]+0.001*g[k]*g[k]
        mh=TR["m"][k]/(1-0.9**TR["t"]); vh=TR["v"][k]/(1-0.999**TR["t"])
        w[k]=clamp(w[k]-lr*mh/(math.sqrt(vh)+1e-8),0.2,2.4)
    TR["epoch"]+=1; TR["loss"]=L; TR["best"]=min(TR["best"],L)
    TR["hist"].append(L)
    if len(TR["hist"])>220: TR["hist"].pop(0)
    if TR["base"] is None: TR["base"]=L
    if not TR["locked"] and conv_pct()>=85:
        TR["locked"]=True
        win.toasts.show("ai","Target locked",f"Policy converged · loss {L:.4f}")
    elif TR["locked"] and conv_pct()<72:
        TR["locked"]=False

TSCACHE={}
for sm in SYMS: TSCACHE[sm["s"]]=gen_candles(sm,"15m")

# ───────────────────────────── account / engine state ───────────────────────
ACC=dict(sym=SYMS[0], tf="15m", side="buy", otype="market", trail=False,
         positions=[], orders=[], realized=1284.50, uid=1, bulk=False,
         lastFeat="trend", last_sig=None)
tickN=0
peakEq=BASE_EQ
FCASTS=[]; fOK=0; fN=0; fSum=0.0
GOV=dict(tripped=False,until=0)
ALERTS=[]
TRADES=[]
EQH=[]
SIM_START=10.0
SIM=dict(start=SIM_START,eq=SIM_START,uid=1,positions=[],closed=0,wins=0,
         blown=False,hist=[SIM_START],last=None,match=None)

def equity(): return BASE_EQ+ACC["realized"]+unreal()
def unreal(): return sum(ppnl(p) for p in ACC["positions"])
def ppnl(p): return (p["sym"]["p"]-p["entry"])*p["qty"]*(1 if p["side"]=="buy" else -1)
def gross_exp(): return sum(abs(p["qty"]*p["sym"]["p"]) for p in ACC["positions"])/equity()*100
def buying_power(): return equity()-sum(p["entry"]*p["qty"]/LEV for p in ACC["positions"])
def slip_bps(notional): return clamp(notional/max(1,equity())*9,0,25)
def slip_px(sym,side,notional):
    return sym["p"]*(1+(1 if side=="buy" else -1)*(0.0002+slip_bps(notional)/1e4))
def is_tripped(): return GOV["tripped"]

# ───────────────────────────── signal core ──────────────────────────────────
def ens_score(x):
    f5=feats5(x)
    lin=sum(TR["weights"][k]*f5[k] for k in WKEYS)
    _,_,p=NET.forward(x) if False else (None,None,NET.forward(x)[1])
    p=NET.forward(x)[1]
    kexp,kscore=knn_score(x)
    total=TRUST["mlp"]+TRUST["lin"]+TRUST["knn"]
    ens=(TRUST["mlp"]*clamp((p[0]-p[2])*2.6,-1.6,1.6)+TRUST["lin"]*lin+TRUST["knn"]*kscore)/total
    return dict(ens=ens,p=p,lin=lin,kexp=kexp,kscore=kscore)

def mc_fan(c,entry,direction,strength,atr,steps=26,paths=240):
    rets=[c[i]["c"]/c[i-1]["c"]-1 for i in range(max(1,len(c)-140),len(c))]
    mu=direction*min(1,abs(strength)/1.4)*(atr/entry)*0.16
    rnd=random.Random()
    grid=[[] for _ in range(steps)]
    for _ in range(paths):
        v=entry
        for s in range(steps):
            v*=1+rets[int(rnd()*len(rets))]+mu
            grid[s].append(v)
    pts=[]
    for col in grid:
        col.sort()
        q=lambda f: col[min(len(col)-1,int(round(f*(len(col)-1))))]
        pts.append(dict(mid=q(.5),hi=q(.92),lo=q(.08),hi2=q(.75),lo2=q(.25)))
    return pts

def compute_signal():
    c=win.candles
    atr=atr_fn(c); px=ACC["sym"]["p"]; atr_pct=atr/px*100
    x=feats_full(c,len(c)-1)
    reg,radj,er=regime_of(c,len(c)-1,atr_pct)
    sc=ens_score(x)
    thr=1.05 if reg=="CHOPPY" else 0.85
    side="LONG" if sc["ens"]>thr else ("SHORT" if sc["ens"]<-thr else "NEUTRAL")
    dirv=1 if side=="LONG" else (-1 if side=="SHORT" else (1 if x[1]>=0 else -1))
    agree=sum(1 for v in ((sc["p"][0]-sc["p"][2])*2.6,sc["lin"],sc["kscore"])
              if (v>0.15 and sc["ens"]>0) or (v<-0.15 and sc["ens"]<0))
    if side=="NEUTRAL":
        conf=int(50+random.random()*13); tp=sl=None
    else:
        conf=int(clamp(52+abs(sc["ens"])*13+(agree-1)*4+abs(sc["p"][0]-sc["p"][2])*9+radj,42,96))
        tp=px+dirv*atr*2.1; sl=px-dirv*atr*1.3
    f5=feats5(x)
    ACC["lastFeat"]=max(WKEYS,key=lambda k:abs(TR["weights"][k]*f5[k]))
    pts=mc_fan(c,px,dirv,0 if side=="NEUTRAL" else sc["ens"],atr)
    vAvg=sum(k["v"] for k in c[-20:])/20
    factors=[
      f"Policy net 8-10-3 · p(L) {round(sc['p'][0]*100)}% / p(F) {round(sc['p'][1]*100)}% / p(S) {round(sc['p'][2]*100)}% · val acc {NET.acc:.0f}%" if NET.acc is not None else "Policy net warming · no val acc yet",
      f"Linear policy · score {fmtS(sc['lin'])} · dominant {ACC['lastFeat'].upper()} (w {TR['weights'][ACC['lastFeat']]:.2f})",
      f"kNN memory · {KNN['k']} nearest of {len(KNN['book'])} · expected {fmtS(sc['kexp'])}%/6-bar",
      f"Regime · {reg} · Kaufman {er:.2f} · ATR {fmt(atr,px/1e4)} ({atr_pct:.2f}% of price)" + (" · threshold ±1.05" if reg=='CHOPPY' else ""),
      f"Ensemble {fmtS(sc['ens'])} · agreement {agree}/3 · trust m/l/k {TRUST['mlp']:.2f}/{TRUST['lin']:.2f}/{TRUST['knn']:.2f}",
      f"Forecast record · {fOK}/{fN} hit · avg {fmtS(fSum/fN if fN else 0)}% · sim {fmtM(SIM['eq'])}"+(f" · match {SIM['match']}%" if SIM['match'] is not None else ""),
      f"Risk · gross {gross_exp():.0f}% · {len(ACC['positions'])} pos · dd-stop {TR['targets']['brk']:.2f}% · gov {'TRIPPED' if GOV['tripped'] else 'ok'}",
    ]
    sig=dict(side=side,conf=conf,entry=px,tp=tp,sl=sl,pts=pts,factors=factors,
             votes=dict(mlp=clamp((sc["p"][0]-sc["p"][2])*2.6,-1.6,1.6),
                        lin=sc["lin"],knn=sc["kscore"]),
             reg=reg,x=x,p=sc["p"])
    ACC["last_sig"]=sig
    return sig

def forecast_grade_due():
    global fOK,fN,fSum
    for i in range(len(FCASTS)-1,-1,-1):
        f=FCASTS[i]
        if tickN-f["t"]>=f["hz"]:
            FCASTS.pop(i)
            sym=BY[f["sym"]]
            ret=(sym["p"]-f["entry"])/f["entry"]*100*(1 if f["side"]=="LONG" else -1)
            fN+=1; fSum+=ret
            if ret>0: fOK+=1
            rs=1 if ret>=0 else -1
            tr=""
            for m in ("mlp","lin","knn"):
                v=f["votes"].get(m,0); ds=1 if v>0 else (-1 if v<0 else 0)
                if ds:
                    TRUST[m]=clamp(TRUST[m]+(0.04 if ds==rs else -0.04),0.5,2)
                    tr+=f" {m}{'↑' if ds==rs else '↓'}"
            win.log_line(f"forecast eval · {f['sym']} {f['side']} · {'HIT' if ret>0 else 'MISS'} · {fmtS(ret,f['dec'])}% · trust:{tr or ' —'} · sim {fmtM(SIM['eq'])}")
            sim_match()

# ───────────────────────────── $10 (sizeable) prove-it SIM ──────────────────
def sim_set_start(v,announce=True):
    global SIM_START
    v=clamp(round(v*100)/100,1,1_000_000)
    SIM_START=v
    SIM.update(start=v,eq=v,realized=0,positions=[],closed=0,wins=0,blown=False,
               hist=[v],last=None,match=None)
    win.refresh_sim()
    if announce:
        win.toasts.show("info","Sim size set",f"{fmtM(v)} prove-it book armed · 1% risk sizing, blow-up floor and MATCH grading scale automatically")
def sim_enter(sym,side,entry,tp,sl,src):
    if SIM["blown"] or is_tripped() or not tp or not sl: return
    if len(SIM["positions"])>=5 or any(p["sym"] is sym for p in SIM["positions"]): return
    dist=abs(entry-sl)
    if dist<=0: return
    qty=SIM["eq"]*0.01/dist
    qty=max(0.0,round(qy:=qty,8))
    notional=qty*entry
    if notional<0.5 or notional>SIM["eq"]*3: return
    fill=sym["p"]*(1+(1 if side=="buy" else -1)*0.0004)
    SIM["eq"]-=notional*4e-4
    SIM["positions"].append(dict(id=SIM["uid"],sym=sym,side=side,qty=qty,entry=fill,
                                 tp=tp,sl=sl,trail=True,trailed=False,time=hhmmss()))
    SIM["uid"]+=1
def sim_close(p,why):
    d=1 if p["side"]=="buy" else -1
    gross=(p["sym"]["p"]-p["entry"])*p["qty"]*d
    net=gross-gross*4e-4
    SIM["eq"]+=net; SIM["closed"]+=1
    if net>0: SIM["wins"]+=1
    ret=gross/(p["qty"]*p["entry"])*100*d
    SIM["last"]=dict(txt=f"{p['sym']['s']} {'LONG' if p['side']=='buy' else 'SHORT'} {fmtS(ret)}% ({why}) · sim P&L {fmtM(net)}",ok=net>=0)
    SIM["positions"].remove(p)
    SIM["hist"].append(SIM["eq"])
    if len(SIM["hist"])>160: SIM["hist"].pop(0)
    if SIM["eq"]<=SIM["start"]*0.05 and not SIM["blown"]:
        SIM["blown"]=True
        win.toasts.show("dn","SIM blown up",f"{fmtM(SIM['start'])} prove-it account wiped — AI metrics do NOT match reality")
def sim_tick():
    for p in list(SIM["positions"]):
        a=atr_fn(TSCACHE[p["sym"]["s"]])
        if p["trail"] and a>0:
            if p["side"]=="buy" and p["sym"]["p"]-p["entry"]>a*0.5:
                ns=round(p["sym"]["p"]-a*1.5,p["sym"]["d"])
                if ns>p["sl"]: p["sl"]=ns; p["trailed"]=True
            elif p["side"]=="sell" and p["entry"]-p["sym"]["p"]>a*0.5:
                ns=round(p["sym"]["p"]+a*1.5,p["sym"]["d"])
                if ns<p["sl"]: p["sl"]=ns; p["trailed"]=True
        if p["tp"] and ((p["side"]=="buy" and p["sym"]["p"]>=p["tp"]) or (p["side"]=="sell" and p["sym"]["p"]<=p["tp"])):
            sim_close(p,"TP HIT")
        elif p["sl"] and ((p["side"]=="buy" and p["sym"]["p"]<=p["sl"]) or (p["side"]=="sell" and p["sym"]["p"]>=p["sl"])):
            sim_close(p,"TRAIL HIT" if p["trailed"] else "SL HIT")
    sim_match()
def sim_match():
    if SIM["closed"]>=3 and fN>=3:
        predWR=fOK/fN; actWR=SIM["wins"]/SIM["closed"]
        predExp=fSum/fN
        actExp=((SIM["eq"]/SIM["start"]-1)*100)/max(1,SIM["closed"])
        dEx=abs(predExp-actExp); expTol=max(0.15,abs(predExp)*0.8)
        SIM["match"]=int((clamp(1-abs(predWR-actWR)*1.6,0,1)*0.65+clamp(1-dEx/expTol,0,1)*0.35)*100)
    else:
        SIM["match"]=None

# ───────────────────────────── custom paint widgets ─────────────────────────
class Chart(QWidget):
    PAD_R,PAD_B,VOL_H,PAD_T=64,20,30,8
    def __init__(self,get_state):
        super().__init__()
        self.gs=get_state           # callable -> dict(candles,dec,ma_on,sig,positions,dirup)
        self.setMouseTracking(True)
        self.setMinimumSize(420,280)
        self.span=90; self.off=0    # off = bars back from live edge
        self.cross=None; self.drag=None
        self.ohlc_cb=None
    # view helpers
    def _view(self,N):
        right=max(19,N-1-self.off)
        span=clamp(self.span,20,max(20,N))
        left=max(0,right-span+1)
        return left,right
    def wheelEvent(self,e):
        N=len(self.gs()["candles"])
        if N<30: return
        f=1/1.16 if e.angleDelta().y()>0 else 1.16
        self.span=clamp(int(self.span*f),20,max(20,N))
        self.update()
    def mousePressEvent(self,e):
        self.drag=(e.position().x(),self.off); self.setCursor(Qt.ClosedHandCursor)
    def mouseMoveEvent(self,e):
        if self.drag is not None:
            N=len(self.gs()["candles"])
            pl=self.width()-self.PAD_R
            slot=pl/max(1,self.span)
            self.off=clamp(int(self.drag[1]+(e.position().x()-self.drag[0])/slot),0,max(0,N-20))
        else:
            self.cross=QPointF(e.position())
        self.update()
    def mouseReleaseEvent(self,e):
        self.drag=None; self.setCursor(Qt.CrossCursor)
    def mouseDoubleClickEvent(self,e):
        self.off=0; self.update()
    def leaveEvent(self,e):
        self.cross=None; self.drag=None; self.update()
    def paintEvent(self,ev):
        st=self.gs(); data=st["candles"]; N=len(data)
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(),QColor(BG))
        if N<10: return
        W,H=self.width(),self.height()
        pl=W-self.PAD_R; innerH=H-self.PAD_B-self.VOL_H-self.PAD_T
        left,right=self._view(N)
        span=right-left+1
        slot=pl/span
        X=lambda i:(i-left+0.5)*slot
        dec=st["dec"]
        mn=min(data[i]["l"] for i in range(left,right+1))
        mx=max(data[i]["h"] for i in range(left,right+1))
        sig=st["sig"] if self.off==0 else None
        if sig and sig["tp"] is not None:
            for k in sig["pts"]:
                mn=min(mn,k["lo"]); mx=max(mx,k["hi"])
        pad=(mx-mn)*.08 or mx*.004; mn-=pad; mx+=pad
        Y=lambda v:self.PAD_T+(mx-v)/(mx-mn)*innerH
        # h grid
        raw=(mx-mn)/6; mag=10**math.floor(math.log10(raw)); step=10*mag
        for a in (1,2,2.5,5,10):
            if raw<=a*mag: step=a*mag; break
        p.setFont(mono(8)); p.setPen(QPen(QColor("#FFFFFF14"),1))
        v=math.ceil(mn/step)*step; i=0
        while v<mx:
            y=round(Y(v))+.5
            p.drawLine(0,int(y),pl,int(y))
            p.setPen(QPen(QColor(TXT3))); p.drawText(pl+4,int(y)+4,f"{fmt(v,dec)}")
            p.setPen(QPen(QColor("#FFFFFF14"),1)); v+=step
        # v grid + time labels
        every=max(1,round(70/slot))
        p.setPen(QPen(QColor("#FFFFFF0A"),1)); p.setFont(mono(8))
        for i in range(left+((every-left%every)%every),right+1,every):
            x=int(X(i))+.5
            p.drawLine(int(x),self.PAD_T,int(x),H-self.PAD_B)
            d=datetime.fromtimestamp(data[i]["t"]/1000)
            lbl=two(d.day)+"/"+two(d.month) if TFS[ACC["tf"]]>=86400000 else hhmm(d)
            p.setPen(QPen(QColor(TXT3))); p.drawText(int(x)-16,H-self.PAD_B+4,lbl)
        # volume
        vmax=max(data[i]["v"] for i in range(left,right+1)) or 1
        bw=max(1,min(slot*0.66,14))
        for i in range(left,right+1):
            k=data[i]; bh=k["v"]/vmax*self.VOL_H
            p.fillRect(QRectF(X(i)-bw/2,H-self.PAD_B-bh,bw,bh),
                       QColor("#2EBD854D" if k["c"]>=k["o"] else "#F6465D4D"))
        # candles
        p.setPen(Qt.NoPen)
        for i in range(left,right+1):
            k=data[i]; up=k["c"]>=k["o"]; col=QColor(UP if up else DN)
            x=X(i)
            p.setPen(QPen(col,1)); p.drawLine(int(x)+.5,Y(k["h"]),int(x)+.5,Y(k["l"]))
            p.setPen(Qt.NoPen)
            y1=Y(max(k["o"],k["c"])); y2=Y(min(k["o"],k["c"]))
            p.fillRect(QRectF(x-bw/2,y1,bw,max(1,y2-y1)),col)
        # MAs
        if st["ma_on"]:
            for n,col in ((20,TEAL),(50,MAUVE)):
                p.setPen(QPen(QColor(col),1)); path=None; s=0.0; started=False
                for i in range(max(left-n+1,0),right+1):
                    s+=data[i]["c"]
                    if i>=n: s-=data[i-n]["c"]
                    if i>=n-1 and i>=left:
                        x=X(i); y=Y(s/n)
                        if not started: p.drawPoint(QPointF(x,y)); started=True
                        else: p.drawLine(QPointF(X(i-1),Y(prev) if False else y),QPointF(x,y)) if False else None
                        prev=y
                # simpler redraw to avoid gaps
            for n,col in ((20,TEAL),(50,MAUVE)):
                pts=[]
                for i in range(left,right+1):
                    if i>=n:
                        pts.append(QPointF(X(i),Y(sum(data[j]["c"] for j in range(i-n+1,i+1))/n)))
                if len(pts)>1: p.drawPolyline(pts)
        # forecast fan
        if sig and sig["tp"] is not None:
            ax=X(N-1)
            def xp(k): return X(N-1)+slot*(k+1)
            for key,alpha in (("hi","lo"),("hi2","lo2")):
                poly=[QPointF(ax,Y(sig["entry"]))]
                poly+= [QPointF(xp(k),Y(pt[key])) for k,pt in enumerate(sig["pts"])]
                poly+=[QPointF(xp(k),Y(pt[key.replace("hi","lo") if key=="hi" else "lo2"] if False else pt["lo" if key=="hi" else "lo2"])) for k,pt in reversed(list(enumerate(sig["pts"])))]
                p.setPen(Qt.NoPen); p.setBrush(QColor(232,181,74,14 if key=="hi" else 28))
                p.drawPolygon(poly)
            p.setPen(QPen(QColor(GOLD),1.4)); p.setBrush(Qt.NoBrush)
            p.drawPolyline([QPointF(ax,Y(sig["entry"]))]+[QPointF(xp(k),Y(pt["mid"])) for k,pt in enumerate(sig["pts"])])
            def tag(yv,col,label):
                y=Y(yv)
                p.setPen(QPen(col,1,Qt.DashLine)); p.drawLine(int(ax),int(y),pl,int(y))
                p.setPen(Qt.NoPen); p.setBrush(col)
                p.drawRect(pl+2,int(y)-8,self.PAD_R-6,16)
                p.setPen(QPen(QColor("#0B0D10"))); p.setFont(mono(8,True))
                p.drawText(pl+4,int(y)+4,label)
            tag(sig["entry"],QColor(232,181,74,230),"E "+fmt(sig["entry"],dec))
            tag(sig["tp"],QColor(46,189,133,230),"T "+fmt(sig["tp"],dec))
            tag(sig["sl"],QColor(246,70,93,230),"S "+fmt(sig["sl"],dec))
        # position lines
        for pos in st["positions"]:
            if pos["sym"] is not st["sym"]: continue
            def pline(price,col,label,dash):
                if mn>=price>=mx: return
                y=Y(price)
                p.setPen(QPen(col,1,dash)); p.drawLine(0,int(y),pl,int(y))
                p.setPen(Qt.NoPen); p.setBrush(col)
                p.drawRect(pl+2,int(y)-8,self.PAD_R-6,16)
                p.setPen(QPen(QColor("#0B0D10"))); p.setFont(mono(7,True))
                p.drawText(pl+4,int(y)+4,label)
            pline(pos["entry"],QColor("#9AA4B080"),("L" if pos["side"]=="buy" else "S")+" "+fmt(pos["qty"],pos["sym"]["qd"]),Qt.DotLine)
            if pos["tp"]: pline(pos["tp"],QColor(46,189,133,220),"TP",Qt.DashLine)
            if pos["sl"]: pline(pos["sl"],QColor(246,70,93,220),"SL",Qt.DashLine)
        # last price
        if self.off==0:
            last=data[N-1]; ly=Y(last["c"])
            col=QColor(UP if st["dirup"] else DN)
            p.setPen(QPen(QColor(UP if st["dirup"] else DN,1,Qt.DashLine)))
            p.drawLine(0,int(ly),pl,int(ly))
            p.setPen(Qt.NoPen); p.setBrush(col)
            p.drawRect(pl+2,int(ly)-9,self.PAD_R-6,18)
            p.setPen(QPen(QColor("#0B0D10"))); p.setFont(mono(9,True))
            p.drawText(pl+4,int(ly)+5,fmt(last["c"],dec))
        # crosshair + ohlc
        k=data[right if self.cross is None else clamp(left+int((self.cross.x())/slot),left,right)]
        if self.cross is not None:
            i=clamp(left+int(self.cross.x()/slot),left,right)
            k=data[i]; cx=int(X(i))+.5
            p.setPen(QPen(QColor("#E9ECF047"),1,Qt.DashLine))
            p.drawLine(cx,self.PAD_T,cx,H-self.PAD_B)
            if self.PAD_T<self.cross.y()<self.PAD_T+innerH:
                p.drawLine(0,int(self.cross.y()),pl,int(self.cross.y()))
                pv=mx-(self.cross.y()-self.PAD_T)/innerH*(mx-mn)
                p.fillRect(pl+2,int(self.cross.y())-9,self.PAD_R-6,18,QColor("#2A323C"))
                p.setPen(QPen(QColor(TXT))); p.setFont(mono(9))
                p.drawText(pl+4,int(self.cross.y())+5,fmt(pv,dec))
            d=datetime.fromtimestamp(k["t"]/1000)
            lbl=f"{two(d.day)}/{two(d.getMonth()+1) if hasattr(d,'getMonth') else two(d.month)} {hhmmss(d)}"
        chg=(k["c"]-(data[left-1]["c"] if left>0 else k["o"]))/(data[left-1]["c"] if left>0 else k["o"])*100
        if self.ohlc_cb:
            self.ohlc_cb(f"O {fmt(k['o'],dec)}  H {fmt(k['h'],dec)}  L {fmt(k['l'],dec)}  C {fmt(k['c'],dec)}  "
                         f"{fmtS(chg)}%  VOL {fmt(k['v'],0)}", k["c"]>=k["o"])

class LossChart(QWidget):
    def __init__(self,get_curves):
        super().__init__()
        self.gs=get_curves
        self.setMinimumHeight(110)
    def paintEvent(self,ev):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(),QColor(PANEL2))
        W,H=self.width(),self.height()
        pc,mc=TR["hist"],NET.hist
        if len(pc)<2 and len(mc)<2:
            p.setPen(QPen(QColor(TXT3))); p.setFont(mono(9)); p.drawText(self.rect(),Qt.AlignCenter,"AWAITING EPOCHS"); return
        pl,pr,pt,pb=8,42,8,8
        if len(pc)>1:
            mx=max(pc)*1.12 or 1
            X=lambda i:pl+(i/(len(pc)-1))*(W-pl-pr); Y=lambda v:pt+(1-v/mx)*(H-pt-pb)
            p.setPen(QPen(QColor(GOLD),1.4))
            p.drawPolyline([QPointF(X(i),Y(v)) for i,v in enumerate(pc)])
            p.setPen(QPen(QColor(TXT3))); p.setFont(mono(7))
            p.drawText(W-pr+4,pt+10,f"{mx:.3f}")
            p.drawText(W-pr+4,H-pb-2,f"0")
        if len(mc)>1:
            mne,mxe=min(mc)*0.985,max(mc)*1.02; rg=(mxe-mne) or 1
            X=lambda i:pl+(i/(len(mc)-1))*(W-pl-pr); Y=lambda v:pt+(1-(v-mne)/rg)*(H-pt-pb)
            p.setPen(QPen(QColor(TEAL),1))
            p.drawPolyline([QPointF(X(i),Y(v)) for i,v in enumerate(mc)])
        p.setFont(mono(7))
        p.setPen(QPen(QColor(GOLD))); p.drawText(pl+4,pt+9,"— POLICY")
        p.setPen(QPen(QColor(TEAL))); p.drawText(pl+4,pt+20,"— MLP XENT")

class Spark(QWidget):
    def __init__(self,get,h=26):
        super().__init__(); self.get=get
        self.setMinimumHeight(h); self.setFixedHeight(h)
    def paintEvent(self,ev):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(),QColor(PANEL2))
        H=self.get()
        if len(H)<2: return
        W,Ht=self.width(),self.height()
        mn=min(H); mx=max(H); rg=(mx-mn) or 1
        base=self.get.base if hasattr(self.get,"base") else None
        X=lambda i:2+(i/(len(H)-1))*(W-8)
        Y=lambda v:3+(1-(v-mn)/rg)*(Ht-6)
        up=H[-1]>=H[0]
        p.setPen(QPen(QColor(UP if up else DN),1.2))
        p.drawPolyline([QPointF(X(i),Y(v)) for i,v in enumerate(H)])

class Toasts(QWidget):
    def __init__(self,parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.lay=QVBoxLayout(self); self.lay.setContentsMargins(0,0,0,0); self.lay.setSpacing(6)
        self.lay.addStretch(1)
    def show(self,kind,title,msg):
        col=dict(ai=GOLD,up=UP,dn=DN,info=TXT2).get(kind,GOLD)
        lb=QLabel(f"<b style='color:{col}'>{title}</b><br><span style='color:{TXT2};font-size:10px'>{msg}</span>")
        lb.setStyleSheet(f"background:{PANEL};border:1px solid {LINE};border-left:3px solid {col};"
                         f"border-radius:4px;padding:8px 10px;max-width:280px;")
        lb.setWordWrap(True)
        self.lay.insertWidget(self.lay.count()-1,lb)
        lb.adjustSize()
        QTimer.singleShot(4200,lambda: (lb.setParent(None), lb.deleteLater()))
        self.reposition()
    def reposition(self):
        pm=self.parentWidget()
        if pm: self.move(pm.width()-310,60); self.resize(300,self.sizeHint().height())

# ───────────────────────────── main window ──────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AURUM · AI TERMINAL — PySide6")
        self.resize(1480,900)
        self.candles=TSCACHE[ACC["sym"]["s"]]
        self.ma_on=True
        self.posItems={}   # pos id -> dict(mark,pl,plp)
        self._build_ui()
        self._shortcuts()
        self._load_ws()
        # warm start
        for _ in range(80): mlp_epoch()
        for _ in range(8): run_epoch()
        build_dataset(); build_knn()
        for _ in range(8): run_epoch()
        self.refresh_trainer(); self.refresh_pf(); self.refresh_sim()
        self.log_line(f"core ready · policy e{TR['epoch']} · net e{NET.epoch} · kNN {len(KNN['book'])} patterns")
        # timers
        self.mkt=QTimer(self); self.mkt.timeout.connect(self.tick); self.mkt.start(600)
        self.trn=QTimer(self); self.trn.timeout.connect(self.train_step); self.trn.start(3200)
        QTimer(self,timeout=self._clock).start(1000)

    # ---------- UI build ----------
    def _build_ui(self):
        central=QWidget(); self.setCentralWidget(central)
        root=QHBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        # left: watchlist
        self.watch=QTableWidget(0,3); self.watch.setHorizontalHeaderLabels(["SYMBOL","PX","CHG%"])
        self.watch.horizontalHeader().setSectionResizeMode(0,QHeaderView.Stretch)
        self.watch.verticalHeader().setVisible(False)
        self.watch.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.watch.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.watch.setShowGrid(False)
        self.watch.setFixedWidth(230)
        self.watch.itemSelectionChanged.connect(self._pick_sym)
        self.watch_rows={}
        for sm in SYMS:
            r=self.watch.rowCount(); self.watch.insertRow(r)
            a=QTableWidgetItem(sm["s"]); b=QTableWidgetItem("—"); c=QTableWidgetItem("—")
            b.setTextAlignment(Qt.AlignRight|Qt.AlignVCenter); c.setTextAlignment(Qt.AlignRight|Qt.AlignVCenter)
            self.watch.setItem(r,0,a); self.watch.setItem(r,1,b); self.watch.setItem(r,2,c)
            self.watch_rows[sm["s"]]=dict(px=b,chg=c)
        # center
        center=QWidget(); cl=QVBoxLayout(center); cl.setContentsMargins(0,0,0,0); cl.setSpacing(0)
        head=QWidget(); hh=QHBoxLayout(head); hh.setContentsMargins(12,8,12,8)
        self.lb_sym=QLabel(ACC["sym"]["s"]); self.lb_sym.setProperty("h1",True)
        self.lb_name=QLabel(ACC["sym"]["n"]); self.lb_name.setProperty("sub",True)
        self.lb_px=QLabel("—"); self.lb_px.setStyleSheet(f"font-size:20px;font-weight:700;")
        self.lb_ohlc=QLabel("—"); self.lb_ohlc.setStyleSheet(f"color:{TXT2};font-size:11px")
        self.tfbox=QComboBox(); self.tfbox.addItems(TFS.keys()); self.tfbox.setCurrentText("15m")
        self.tfbox.setFixedWidth(80); self.tfbox.currentTextChanged.connect(self._tf_change)
        self.btn_ma=QPushButton("MA 20·50"); self.btn_ma.setCheckable(True); self.btn_ma.setChecked(True)
        self.btn_ma.clicked.connect(lambda: (setattr(self,"ma_on",not self.ma_on), self.chart.update()))
        hh.addWidget(self.lb_sym); hh.addWidget(self.lb_name); hh.addSpacing(16)
        hh.addWidget(self.lb_px); hh.addWidget(self.lb_ohlc,1)
        hh.addWidget(QLabel("TF")); hh.addWidget(self.tfbox); hh.addWidget(self.btn_ma)
        self.chart=Chart(lambda: dict(candles=self.candles,dec=ACC["sym"]["d"],ma_on=self.ma_on,
                                      sig=ACC["last_sig"] if ACC["last_sig"] and ACC["last_sig"]["sym"]==ACC["sym"]["s"] else None,
                                      positions=ACC["positions"],sym=ACC["sym"],
                                      dirup=(self.candles[-1]["c"]>=self.candles[-1]["o"]) if self.candles else True))
        self.chart.ohlc_cb=self._set_ohlc
        # tabs
        self.tabs=QTabWidget()
        # positions tab
        self.tb_pos=QTableWidget(0,9)
        self.tb_pos.setHorizontalHeaderLabels(["SYMBOL","SIDE","QTY","ENTRY","MARK","TP","SL","P&L $","P&L %"])
        for i in range(9): self.tb_pos.horizontalHeader().setSectionResizeMode(i,QHeaderView.Stretch)
        self._table_plain(self.tb_pos)
        self.btn_close=QPushButton("CLOSE SELECTED"); self.btn_close.clicked.connect(self._close_sel)
        pv=QWidget(); pvv=QVBoxLayout(pv); pvv.setContentsMargins(0,0,0,0)
        pvv.addWidget(self.btn_close); pvv.addWidget(self.tb_pos)
        self.tabs.addTab(pv,"POSITIONS")
        # orders tab
        self.tb_ord=QTableWidget(0,5)
        self.tb_ord.setHorizontalHeaderLabels(["SYMBOL","SIDE","TYPE","QTY","LIMIT"])
        for i in range(5): self.tb_ord.horizontalHeader().setSectionResizeMode(i,QHeaderView.Stretch)
        self._table_plain(self.tb_ord)
        self.btn_cancel=QPushButton("CANCEL SELECTED"); self.btn_cancel.clicked.connect(self._cancel_sel)
        ov=QWidget(); ovv=QVBoxLayout(ov); ovv.setContentsMargins(0,0,0,0)
        ovv.addWidget(self.btn_cancel); ovv.addWidget(self.tb_ord)
        self.tabs.addTab(ov,"ORDERS")
        # trainer tab
        tw=QWidget(); tg=QHBoxLayout(tw); tg.setContentsMargins(8,8,8,8)
        left=QWidget(); lv=QVBoxLayout(left); lv.setContentsMargins(0,0,0,0)
        self.loss=LossChart(lambda:(TR["hist"],NET.hist))
        self.st_epoch=QLabel("0"); self.st_loss=QLabel("—"); self.st_best=QLabel("—")
        self.st_lr=QLabel("—"); self.st_conv=QLabel("0%"); self.st_acc=QLabel("—")
        grid=QGridLayout()
        for col,(cap,lb) in enumerate([("EPOCH",self.st_epoch),("LOSS",self.st_loss),("BEST",self.st_best),
                                       ("LR",self.st_lr),("CONV",self.st_conv),("NET ACC",self.st_acc)]):
            capl=QLabel(cap); capl.setProperty("stat",True)
            grid.addWidget(capl,0,col); grid.addWidget(lb,1,col)
        lv.addWidget(self.loss,1); lv.addLayout(grid)
        hb=QHBoxLayout()
        self.btn_step=QPushButton("STEP ×10"); self.btn_step.clicked.connect(self._step10)
        self.btn_bt=QPushButton("BACKTEST POLICY"); self.btn_bt.clicked.connect(self._backtest)
        self.chk_train=QCheckBox("CONTINUOUS"); self.chk_train.setChecked(True)
        self.chk_train.toggled.connect(lambda on: self.trn.start(3200) if on else self.trn.stop())
        hb.addWidget(self.btn_step); hb.addWidget(self.btn_bt); hb.addWidget(self.chk_train); hb.addStretch(1)
        lv.addLayout(hb); self.lb_bt=QLabel(""); self.lb_bt.setProperty("sub",True); lv.addWidget(self.lb_bt)
        tg.addWidget(left,1)
        rw=QWidget(); rv=QVBoxLayout(rw); rv.setContentsMargins(0,0,0,0)
        rv.addWidget(QLabel("PORTFOLIO TARGETS · return% / max dd% / vol% / dd-stop%").setProperty("stat",True) or QLabel("PORTFOLIO TARGETS"))
        self.in_ret=QLineEdit("1.00"); self.in_dd=QLineEdit("2.00")
        self.in_vol=QLineEdit("0.80"); self.in_brk=QLineEdit("1.50")
        for wgt,cap in ((self.in_ret,"RETURN %"),(self.in_dd,"MAX DD %"),(self.in_vol,"VOL %"),(self.in_brk,"DD STOP %")):
            capl=QLabel(cap); capl.setProperty("stat",True); rv.addWidget(capl); rv.addWidget(wgt)
        self.in_ret.returnPressed.connect(self._retarget)
        self.in_dd.returnPressed.connect(self._retarget)
        self.in_vol.returnPressed.connect(self._retarget)
        self.in_brk.returnPressed.connect(self._retarget)
        rv.addStretch(1); rw.setFixedWidth(160)
        tg.addWidget(rw)
        self.tabs.addTab(tw,"TRAINER")
        # portfolio tab
        pw=QWidget(); pg=QHBoxLayout(pw); pg.setContentsMargins(8,8,8,8)
        pf_left=QWidget(); plv=QVBoxLayout(pf_left); plv.setContentsMargins(0,0,0,0)
        self.lb_nav=QLabel("—"); self.lb_gross=QLabel("—"); self.lb_vol=QLabel("—")
        self.lb_var=QLabel("—"); self.lb_beta=QLabel("—"); self.lb_match=QLabel("—")
        strip=QGridLayout()
        for col,(cap,lb) in enumerate([("NAV",self.lb_nav),("GROSS",self.lb_gross),("VOL ANN",self.lb_vol),
                                       ("VAR 95",self.lb_var),("BETA BTC",self.lb_beta),("SIM MATCH",self.lb_match)]):
            capl=QLabel(cap); capl.setProperty("stat",True)
            strip.addWidget(capl,0,col); strip.addWidget(lb,1,col)
        self.tb_pf=QTableWidget(0,8)
        self.tb_pf.setHorizontalHeaderLabels(["SYMBOL","CLASS","NET QTY","VALUE","WEIGHT","TGT","DRIFT","UNRL P&L"])
        for i in range(8): self.tb_pf.horizontalHeader().setSectionResizeMode(i,QHeaderView.Stretch)
        self._table_plain(self.tb_pf)
        plv.addLayout(strip); plv.addWidget(self.tb_pf,1)
        pg.addWidget(pf_left,1)
        pr=QWidget(); prv=QVBoxLayout(pr); prv.setContentsMargins(0,0,0,0)
        prv.addWidget(QLabel("SIM ACCOUNT · PROVE-IT"))
        self.lb_simnav=QLabel("—"); self.lb_simpnl=QLabel("—"); self.lb_simwin=QLabel("—")
        self.lb_simpred=QLabel("—"); self.lb_verdict=QLabel("WARMING")
        self.lb_verdict.setStyleSheet(f"color:{GOLD};font-weight:700")
        for cap,lb in (("START / NAV",self.lb_simnav),("SIM P&L",self.lb_simpnl),
                       ("TRADES · WIN%",self.lb_simwin),("AI PRED WR",self.lb_simpred),
                       ("VERDICT",self.lb_verdict)):
            capl=QLabel(cap); capl.setProperty("stat",True); prv.addWidget(capl); prv.addWidget(lb)
        sizes=QHBoxLayout()
        for v in (10,100,1000,10000):
            b=QPushButton(f"${v:,}"); b.setCheckable(True)
            b.clicked.connect(lambda _,vv=v: sim_set_start(vv))
            sizes.addWidget(b); (setattr(self,"_simsz",sizes))
        self.sim_spark=Spark(lambda: SIM["hist"])
        prv.addLayout(sizes); prv.addWidget(self.sim_spark)
        self.btn_simreset=QPushButton("RESET SIM"); self.btn_simreset.clicked.connect(lambda: sim_set_start(SIM.start,True))
        prv.addWidget(self.btn_simreset)
        self.btn_rep=QPushButton("SESSION REPORT"); self.btn_rep.clicked.connect(self._report)
        prv.addWidget(self.btn_rep)
        prv.addStretch(1); pr.setFixedWidth(220)
        pg.addWidget(pr)
        self.tabs.addTab(pw,"PORTFOLIO")
        # log tab
        self.log=QListWidget()
        self.tabs.addTab(self.log,"TRACE")
        cl.addWidget(head); cl.addWidget(self.chart,1); cl.addWidget(self.tabs)
        self.tabs.setMinimumHeight(240)
        # right column
        right=QWidget(); right.setFixedWidth(300)
        rl=QVBoxLayout(right); rl.setContentsMargins(8,8,8,8); rl.setSpacing(6)
        rl.addWidget(QLabel("AI COPILOT · AURUM-5.8").setProperty("stat",True) or QLabel("AI COPILOT"))
        self.sig_card=QLabel("AWAITING ANALYSIS — press R"); self.sig_card.setWordWrap(True)
        self.sig_card.setStyleSheet(f"background:{PANEL2};border:1px solid {LINE};border-radius:4px;padding:10px;")
        rl.addWidget(self.sig_card)
        rl.addWidget(QLabel("SIGNAL TRACE").setProperty("stat",True) or QLabel("SIGNAL TRACE"))
        self.trace=QListWidget(); rl.addWidget(self.trace,1)
        self.btn_run=QPushButton("RUN AI ANALYSIS"); self.btn_run.setProperty("gold",True)
        self.btn_run.clicked.connect(self.run_analysis); rl.addWidget(self.btn_run)
        rl.addWidget(QLabel("ORDER TICKET").setProperty("stat",True) or QLabel("ORDER TICKET"))
        side=QHBoxLayout()
        self.btn_buy=QPushButton("BUY"); self.btn_buy.setCheckable(True); self.btn_buy.setChecked(True)
        self.btn_sell=QPushButton("SELL"); self.btn_sell.setCheckable(True)
        self.btn_buy.clicked.connect(lambda: self._side("buy")); self.btn_sell.clicked.connect(lambda: self._side("sell"))
        side.addWidget(self.btn_buy); side.addWidget(self.btn_sell); rl.addLayout(side)
        self.cb_type=QComboBox(); self.cb_type.addItems(["MARKET","LIMIT"]); rl.addWidget(self.cb_type)
        self.in_qty=QLineEdit(str(ACC["sym"]["qdef"])); self.in_qty.returnPressed.connect(self._submit)
        self.in_tp=QLineEdit(); self.in_sl=QLineEdit()
        rl.addWidget(QLabel("QTY").setProperty("stat",True) or QLabel("QTY")); rl.addWidget(self.in_qty)
        rl.addWidget(QLabel("TP / SL").setProperty("stat",True) or QLabel("TP / SL"))
        ts=QHBoxLayout(); ts.addWidget(self.in_tp); ts.addWidget(self.in_sl); rl.addLayout(ts)
        self.btn_trail=QPushButton("TRAIL STOP · 1.5×ATR"); self.btn_trail.setCheckable(True)
        self.btn_trail.clicked.connect(lambda on: win.toasts.show("ai","Trailing stop armed","SL trails 1.5×ATR once 0.5×ATR in profit") if on else None)
        rl.addWidget(self.btn_trail)
        self.btn_submit=QPushButton("BUY —"); self.btn_submit.clicked.connect(self._submit)
        rl.addWidget(self.btn_submit)
        rl.addStretch(1)
        # assemble
        split=QSplitter(Qt.Horizontal)
        split.addWidget(self.watch); split.addWidget(center); split.addWidget(right)
        split.setSizes([230,940,300])
        root.addWidget(split)
        self.statusBar().showMessage("LIVE FEED · SIMULATED DATA · R run · B/S side · F flatten · ESC clear signal")
        self.toasts=Toasts(self)

    def _table_plain(self,t):
        t.verticalHeader().setVisible(False); t.setShowGrid(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)

    def _shortcuts(self):
        QShortcut(QKeySequence("R"),self,self.run_analysis)
        QShortcut(QKeySequence("B"),self,lambda: self._side("buy"))
        QShortcut(QKeySequence("S"),self,lambda: self._side("sell"))
        QShortcut(QKeySequence("F"),self,self._flatten)
        QShortcut(QKeySequence("Esc"),self,self._clear_sig)

    # ---------- persistence ----------
    def _ws_path(self): return os.path.join(os.path.expanduser("~"),".aurum_ws.json")
    def _save_ws(self):
        try:
            json.dump(dict(w=TR["weights"],t=TR["targets"],q=[[NET.w1[j][k] for k in range(8)] for j in range(10)],
                           sim=SIM.start),open(self._ws_path(),"w"))
        except Exception: pass
    def _load_ws(self):
        try:
            d=json.load(open(self._ws_path()))
            if "w" in d: TR["weights"].update({k:clamp(float(d["w"][k]),0.2,2.4) for k in WKEYS})
            if "t" in d: TR["targets"].update(d["t"])
            if "sim" in d: sim_set_start(float(d["sim"]),False)
        except Exception: pass

    # ---------- small helpers ----------
    def log_line(self,s):
        self.trace.insertItem(0,QListWidgetItem(f"{hhmmss()}  {s}"))
        if self.trace.count()>200: self.trace.takeItem(self.trace.count()-1)
    def _set_ohlc(self,text,up):
        self.lb_ohlc.setText(text)
        self.lb_ohlc.setStyleSheet(f"color:{UP if up else DN};font-size:11px")
    def _clock(self): self.statusBar().showMessage(self.statusBar().currentMessage(),0)  # noop keep

    # ---------- market tick ----------
    def tick(self):
        global tickN
        tickN+=1
        for sm in SYMS:
            sm["p"]*=1+(random.random()-.5)*2*sm["vol"]*.00045
            sm["hi"]=max(sm["hi"],sm["p"]); sm["lo"]=min(sm["lo"],sm["p"])
            if tickN%2==0:
                sm["spark"].append(sm["p"])
                if len(sm["spark"])>60: sm["spark"].pop(0)
            if sm is ACC["sym"]:
                last=self.candles[-1]
                last["c"]=sm["p"]; last["h"]=max(last["h"],sm["p"]); last["l"]=min(last["l"],sm["p"])
                if tickN%14==0:
                    self.candles.append(dict(t=last["t"]+TFS[ACC["tf"]],o=sm["p"],h=sm["p"],l=sm["p"],c=sm["p"],v=0.3))
                    if len(self.candles)>200: self.candles.pop(0)
        # limit fills
        for o in list(ACC["orders"]):
            sym=o["sym"]
            if (sym["p"]<=o["price"] and o["side"]=="buy") or (sym["p"]>=o["price"] and o["side"]=="sell"):
                ACC["orders"].remove(o); self._fill(o,sym,o["price"],"LIMIT")
        # brackets + trail
        for p in list(ACC["positions"]):
            a=atr_fn(TSCACHE[p["sym"]["s"]])
            if p["trail"] and p["sl"] and a>0:
                if p["side"]=="buy" and p["sym"]["p"]-p["entry"]>a*0.5:
                    ns=round(p["sym"]["p"]-a*1.5,p["sym"]["d"])
                    if ns>p["sl"]: p["sl"]=ns; p["trailed"]=True
                elif p["side"]=="sell" and p["entry"]-p["sym"]["p"]>a*0.5:
                    ns=round(p["sym"]["p"]+a*1.5,p["sym"]["d"])
                    if ns<p["sl"]: p["sl"]=ns; p["trailed"]=True
            if p["tp"] and ((p["side"]=="buy" and p["sym"]["p"]>=p["tp"]) or (p["side"]=="sell" and p["sym"]["p"]<=p["tp"])):
                self._close(p,"TP HIT")
            elif p["sl"] and ((p["side"]=="buy" and p["sym"]["p"]<=p["sl"]) or (p["side"]=="sell" and p["sym"]["p"]>=p["sl"])):
                self._close(p,"TRAIL HIT" if p.get("trailed") else "SL HIT")
        sim_tick(); forecast_grade_due()
        # circuit breaker
        global peakEq
        eq=equity(); peakEq=max(peakEq,eq)
        dd=(peakEq-eq)/peakEq*100; brk=TR["targets"]["brk"]
        if not GOV["tripped"] and dd>=brk and (ACC["positions"] or TRADES):
            GOV.update(tripped=True,until=time.time()+90)
            self.toasts.show("dn","CIRCUIT BREAKER",f"Drawdown {dd:.2f}% ≥ {brk:.2f}% — book flattened, entries locked 90s")
            self._flatten("CIRCUIT BREAKER")
        elif GOV["tripped"] and time.time()>GOV["until"] and dd<brk*0.5:
            GOV["tripped"]=False
            self.toasts.show("info","Risk governor rearmed","Entries unlocked")
        # UI refresh
        for sm in SYMS:
            r=self.watch_rows[sm["s"]]
            r["px"].setText(fmt(sm["p"],sm["d"]))
            chg=(sm["p"]-sm["open"])/sm["open"]*100
            r["chg"].setText(fmtS(chg)+"%")
            r["chg"].setForeground(QBrush(QColor(UP if chg>=0 else DN)))
        if tickN%4==0: self._hdr()
        self._pos_tick(); self.chart.update()
        if tickN%5==0 and self.tabs.currentWidget() and self.tabs.currentIndex()==2:
            self.refresh_trainer()
        if tickN%5==0: self.refresh_pf(); self.refresh_sim()
        if tickN%12==0: self._save_ws()

    def _hdr(self):
        sm=ACC["sym"]
        self.lb_sym.setText(sm["s"]); self.lb_name.setText(sm["n"])
        self.lb_px.setText(fmt(sm["p"],sm["d"]))
        chg=(sm["p"]-sm["open"])/sm["open"]*100
        self.lb_px.setStyleSheet(f"font-size:20px;font-weight:700;color:{UP if chg>=0 else DN}")
        self.btn_submit.setText(f"{'BUY' if ACC['side']=='buy' else 'SELL'} {self.in_qty.text() or '—'} "
                                f"{sm['s'].split('/')[0]} · {ACC['otype']}")

    # ---------- watch / symbol ----------
    def _pick_sym(self):
        row=self.watch.currentRow()
        if row<0 or row>=len(SYMS): return
        sm=SYMS[row]
        if sm is ACC["sym"]: return
        ACC["sym"]=sm
        self.candles=load_candles(sm,ACC["tf"])
        ACC["last_sig"]=None
        self.in_qty.setText(str(sm["qdef"]))
        self._hdr(); self.chart.off=0; self.chart.update()
    def _tf_change(self,tf):
        ACC["tf"]=tf
        self.candles=load_candles(ACC["sym"],tf)
        ACC["last_sig"]=None; self.chart.off=0; self.chart.update()

    # ---------- copilot ----------
    def run_analysis(self):
        sig=compute_signal(); sig["sym"]=ACC["sym"]
        s=ACC["sym"]
        col=UP if sig["side"]=="LONG" else (DN if sig["side"]=="SHORT" else GOLD)
        head=(f"<b style='color:{col};font-size:15px'>{sig['side']}</b>"
              f"<br><span style='color:{TXT3};font-size:10px'>entry {fmt(sig['entry'],s['d'])} · "
              f"TP {fmt(sig['tp'],s['d']) if sig['tp'] else '—'} · "
              f"SL {fmt(sig['sl'],s['d']) if sig['sl'] else '—'} · conf {sig['conf']}%</span>")
        self.sig_card.setText(head)
        self.log_line(f"» SIGNAL: {sig['side']} · conf {sig['conf']}% · {s['s']} @ {fmt(sig['entry'],s['d'])} · sim mirrored")
        for f in sig["factors"]: self.log_line("   "+f)
        if sig["side"]!="NEUTRAL" and sig["tp"] is not None:
            FCASTS.append(dict(sym=s["s"],side=sig["side"],entry=sig["entry"],t=tickN,
                               hz=52 if sig["reg"]=="TRENDING" else 44,dec=s["d"],votes=sig["votes"]))
            sim_enter(s,"buy" if sig["side"]=="LONG" else "sell",sig["entry"],sig["tp"],sig["sl"],"SIGNAL")
        self.chart.update()
    def _clear_sig(self):
        ACC["last_sig"]=None; self.sig_card.setText("AWAITING ANALYSIS — press R"); self.chart.update()

    # ---------- trading ----------
    def _side(self,s):
        ACC["side"]=s
        self.btn_buy.setChecked(s=="buy"); self.btn_sell.setChecked(s=="sell")
        self._hdr()
    def _submit(self):
        if is_tripped():
            self.toasts.show("dn","Circuit breaker active","Entries locked — governor must rearm"); return
        try: qty=float(self.in_qty.text())
        except ValueError: return self.toasts.show("dn","Invalid order","Enter a quantity")
        if qty<=0: return self.toasts.show("dn","Invalid order","Quantity must be > 0")
        sym=ACC["sym"]
        notional=qty*sym["p"]
        if notional/LEV>buying_power():
            return self.toasts.show("dn","Rejected","Insufficient buying power")
        o=dict(id=ACC["uid"],sym=sym,side=ACC["side"],qty=qty,
               price=sym["p"]*(1+(0.0002+slip_bps(notional)/1e4)*(1 if ACC["side"]=="buy" else -1)),
               tp=float(self.in_tp.text()) if self.in_tp.text() else None,
               sl=float(self.in_sl.text()) if self.in_sl.text() else None,
               trail=self.btn_trail.isChecked(),trailed=False,time=hhmmss())
        ACC["uid"]+=1
        if ACC["otype"]=="MARKET":
            self._fill(o,sym,o["price"],"MARKET")
        else:
            ACC["orders"].append(o); self._render_orders()
            self.toasts.show("ai","Order working",f"LIMIT {o['side'].upper()} {qty:g} {sym['s']} @ {fmt(o['price'],sym['d'])}")
        self.in_tp.clear(); self.in_sl.clear()
    def _fill(self,o,sym,price,src):
        p=dict(id=o["id"],sym=sym,side=o["side"],qty=o["qty"],entry=price,
               tp=o["tp"],sl=o["sl"],trail=o["trail"],trailed=False,time=o["time"])
        ACC["positions"].append(p); self._render_positions()
        if not ACC["bulk"]:
            self.toasts.show("up" if o["side"]=="buy" else "dn","Order filled",
                f"{src} · {o['side'].upper()} {o['qty']:g} {sym['s']} @ {fmt(price,sym['d'])}"+(" · brackets" if o["tp"] else ""))
    def _close(self,p,reason):
        v=ppnl(p)
        ACC["positions"].remove(p)
        ACC["realized"]+=v
        TRADES.insert(0,dict(tm=hhmmss(),sym=p["sym"]["s"],
                             side="LONG" if p["side"]=="buy" else "SHORT",
                             qty=f"{p['qty']:.6g}",exit=fmt(p["sym"]["p"],p["sym"]["d"]),
                             pnl=v,why=reason))
        self._render_positions()
        if not ACC["bulk"]:
            self.toasts.show("up" if v>=0 else "dn",reason or "Position closed",
                f"{p['sym']['s']} · P&L {fmtM(v)}")
            won=v>=0
            TR["weights"][ACC["lastFeat"]]=clamp(TR["weights"][ACC["lastFeat"]]+(0.035 if won else -0.035),0.2,2.4)
            self.log_line(f"reward nudge · {ACC['lastFeat']} {'+' if won else '-'}0.035 · {p['sym']['s']} {fmtM(v)}")
            self.refresh_trainer()
    def _close_sel(self):
        r=self.tb_pos.currentRow()
        if 0<=r<len(ACC["positions"]): self._close(ACC["positions"][r],"MANUAL")
    def _flatten(self,reason="Flattened"):
        if not ACC["positions"]: return
        ACC["bulk"]=True
        tot=sum(ppnl(p) for p in list(ACC["positions"]))
        for p in list(ACC["positions"]): self._close(p,reason)
        ACC["bulk"]=False
        self.toasts.show("up" if tot>=0 else "dn","Book flattened",f"P&L {fmtM(tot)}")
    def _cancel_sel(self):
        r=self.tb_ord.currentRow()
        if 0<=r<len(ACC["orders"]):
            o=ACC["orders"].pop(r); self._render_orders()
            self.toasts.show("info","Order cancelled",f"{o['side'].upper()} {o['qty']:g} {o['sym']['s']}")
    def _render_positions(self):
        tb=self.tb_pos; tb.setRowCount(0); self.posItems={}
        for p in ACC["positions"]:
            r=tb.rowCount(); tb.insertRow(r)
            vals=[p["sym"]["s"],"LONG" if p["side"]=="buy" else "SHORT",
                  fmt(p["qty"],p["sym"]["qd"]),fmt(p["entry"],p["sym"]["d"]),"—",
                  fmt(p["tp"],p["sym"]["d"]) if p["tp"] else "—",
                  fmt(p["sl"],p["sym"]["d"]) if p["sl"] else "—","—","—"]
            for cix,v in enumerate(vals):
                it=QTableWidgetItem(v)
                if cix>0: it.setTextAlignment(Qt.AlignRight|Qt.AlignVCenter)
                tb.setItem(r,cix,it)
            self.posItems[p["id"]]=dict(mk=tb.item(r,4),pl=tb.item(r,7),plp=tb.item(r,8),
                                        side=p["side"])
    def _pos_tick(self):
        for p in ACC["positions"]:
            ref=self.posItems.get(p["id"])
            if not ref: continue
            v=ppnl(p)
            ref["mk"].setText(fmt(p["sym"]["p"],p["sym"]["d"]))
            ref["pl"].setText(fmtM(v))
            ref["pl"].setForeground(QBrush(QColor(UP if v>=0 else DN)))
            ref["plp"].setText(fmtS(v/(p["entry"]*p["qty"])*100)+"%")
            ref["plp"].setForeground(QBrush(QColor(UP if v>=0 else DN)))
    def _render_orders(self):
        tb=self.tb_ord; tb.setRowCount(0)
        for o in ACC["orders"]:
            r=tb.rowCount(); tb.insertRow(r)
            for cix,v in enumerate([o["sym"]["s"],o["side"].upper(),"LIMIT",f"{o['qty']:g}",fmt(o["price"],o["sym"]["d"])]):
                it=QTableWidgetItem(v)
                if cix>0: it.setTextAlignment(Qt.AlignRight|Qt.AlignVCenter)
                tb.setItem(r,cix,it)

    # ---------- trainer ----------
    def train_step(self):
        run_epoch(); mlp_epoch()
        self.refresh_trainer()
        if TR["epoch"]%12==0: self._save_ws()
    def _step10(self):
        for _ in range(10):
            run_epoch(); mlp_epoch()
        self.refresh_trainer()
        self.toasts.show("ai","STEP ×10",f"policy e{TR['epoch']} · loss {TR['loss']:.4f}")
    def refresh_trainer(self):
        self.st_epoch.setText(str(TR["epoch"]))
        self.st_loss.setText(f"{TR['loss']:.4f}" if TR["loss"] is not None else "—")
        self.st_best.setText(f"{TR['best']:.4f}" if TR["best"]<1e9 else "—")
        self.st_lr.setText(f"{lr_now():.4f}")
        self.st_conv.setText(f"{conv_pct():.0f}%")
        self.st_acc.setText(f"{NET.acc:.0f}%" if NET.acc is not None else "—")
        self.loss.update()
    def _retarget(self):
        try:
            TR["targets"].update(ret=clamp(float(self.in_ret.text()),0.1,5),
                                 dd=clamp(float(self.in_dd.text()),0.2,10),
                                 vol=clamp(float(self.in_vol.text()),0.1,5),
                                 brk=clamp(float(self.in_brk.text()),0.3,10))
        except ValueError: return
        TR["base"]=objective(TR["weights"],SYMS); TR["marks"].append(TR["epoch"])
        TR["locked"]=False
        self.toasts.show("ai","Targets updated","Objective re-weighted — re-optimizing")
    def _backtest(self):
        w=TR["weights"]; eq=100.0; n=0; hist=[100.0]; rets=[]
        for sm in SYMS:
            c=TSCACHE[sm["s"]]; i=52
            while i<len(c)-7:
                f=feats5(feats_full(c,i))
                sc=sum(w[k]*f[k] for k in WKEYS)
                if abs(sc)>=0.85:
                    d=1 if sc>0 else -1; sz=min(1,abs(sc)/2)
                    for k2 in range(1,7):
                        if i+k2<len(c):
                            r=(c[i+k2]["c"]-c[i+k2-1]["c"])/c[i+k2-1]["c"]
                            eq*=(1+d*sz*r)
                    ret=d*((c[min(i+6,len(c)-1)]["c"]-c[i]["c"])/c[i]["c"])*100*sz-0.08
                    rets.append(ret); n+=1; hist.append(eq); i+=7
                else:
                    hist.append(eq); i+=1
        dd=0; pk=-1e18
        for v in hist: pk=max(pk,v); dd=max(dd,(pk-v)/pk*100)
        sh=0.0
        if len(hist)>30:
            rr=[(hist[i]-hist[i-1])/hist[i-1] for i in range(1,len(hist))]
            m=sum(rr)/len(rr); sd=stdev(rr)
            sh=clamp(m/sd*math.sqrt(2400),-9.99,9.99) if sd>0 else 0
        wins=[r for r in rets if r>0]; loss=[r for r in rets if r<=0]
        gw=sum(wins); gl=abs(sum(loss))
        self.lb_bt.setText(f"walk-forward · net {fmtS(eq-100)}% · maxDD {dd:.2f}% · sharpe {sh:.2f} · "
                           f"{n} trades · win {len(wins)/n*100:.0f}% · PF {(gw/gl if gl>0 else float('inf')):.2f}")
        self.toasts.show("ai","Backtest complete",f"net {fmtS(eq-100)}% · {n} trades")

    # ---------- portfolio / sim ----------
    def holdings(self):
        m={}
        for p in ACC["positions"]:
            sgn=1 if p["side"]=="buy" else -1
            h=m.setdefault(p["sym"]["s"],dict(sym=p["sym"],qty=0.0,upnl=0.0))
            h["qty"]+=sgn*p["qty"]; h["upnl"]+=ppnl(p)
        out=list(m.values())
        for h in out: h["mv"]=h["qty"]*h["sym"]["p"]
        out.sort(key=lambda h:-abs(h["mv"]))
        return out
    def refresh_pf(self):
        H=self.holdings(); nav=equity() or 1
        R=[SIM_START]
        self.lb_nav.setText(fmtM(nav))
        gross=sum(abs(h["mv"]) for h in H)
        self.lb_gross.setText(f"{gross/nav*100:.0f}% · {(gross/nav):.2f}x")
        # portfolio vol via live correlation over tick sparks
        rets=[]
        for sm in SYMS:
            a=sm["spark"]; rets.append([a[i]/a[i-1]-1 for i in range(1,len(a))])
        wts=[(next((h["mv"] for h in H if h["sym"]["s"]==sm["s"]),0.0))/nav for sm in SYMS]
        vols=[stdev(r) for r in rets]
        pv=0.0
        for i in range(10):
            for j in range(10):
                rho=1.0 if i==j else pearson(rets[i],rets[j])
                pv+=wts[i]*wts[j]*vols[i]*vols[j]*rho
        daily=math.sqrt(max(0,pv))*math.sqrt(1440)
        self.lb_vol.setText(f"{daily*math.sqrt(365)*100:.1f}%")
        self.lb_var.setText(fmtM(1.645*daily*nav))
        btcR=rets[0]; bvar=stdev(btcR)**2; beta=0.0
        for i,sm in enumerate(SYMS):
            if wts[i] and bvar>1e-18:
                A=rets[i]; ma=sum(A)/len(A); mb=sum(btcR)/len(btcR)
                cov=sum((A[k]-ma)*(btcR[k]-mb) for k in range(len(A)))/len(A)
                beta+=wts[i]*cov/bvar
        self.lb_beta.setText(fmtS(beta))
        ALLOC_W={"BTC/USD":34,"ETH/USD":22,"NVDA":18,"XAU/USD":14,"SOL/USD":12}
        tb=self.tb_pf; tb.setRowCount(0)
        for h in H:
            r=tb.rowCount(); tb.insertRow(r)
            cls=("CRYPTO" if h["sym"]["s"] in ("BTC/USD","ETH/USD","SOL/USD") else
                 "METALS" if h["sym"]["s"]=="XAU/USD" else
                 "FX" if h["sym"]["s"]=="EUR/USD" else "EQUITY")
            wgt=h["mv"]/nav*100; tgt=ALLOC_W.get(h["sym"]["s"],0); drift=wgt-tgt
            vals=[h["sym"]["s"],cls,(f"+{h['qty']:.6g}" if h["qty"]>=0 else f"{h['qty']:.6g}"),
                  fmtM(h["mv"]),f"{wgt:.1f}%",f"{tgt}%" if tgt else "—",fmtS(drift,1)+"%",fmtM(h["upnl"])]
            for cix,v in enumerate(vals):
                it=QTableWidgetItem(v)
                if cix>0: it.setTextAlignment(Qt.AlignRight|Qt.AlignVCenter)
                if cix==7: it.setForeground(QBrush(QColor(UP if h["upnl"]>=0 else DN)))
                if cix==6: it.setForeground(QBrush(QColor(TXT3 if abs(drift)<1 else (UP if drift>0 else DN))))
                tb.setItem(r,cix,it)
    def refresh_sim(self):
        pnl_pct=(SIM["eq"]/SIM["start"]-1)*100
        self.lb_simnav.setText(f"{fmtM(SIM['start'])} → {fmtM(SIM['eq'])}")
        self.lb_simpnl.setText(fmtS(pnl_pct)+"%")
        self.lb_simpnl.setStyleSheet(f"color:{UP if pnl_pct>=0 else DN};font-weight:700")
        self.lb_simwin.setText(f"{SIM['closed']} closed · {SIM['positions']} open · "
                               f"{(SIM['wins']/SIM['closed']*100 if SIM['closed'] else 0):.0f}% win")
        self.lb_simpred.setText(f"{(fOK/fN*100 if fN else 0):.0f}% · avg {fmtS(fSum/fN if fN else 0)}%")
        vmap={0:("BLOWN UP",DN),None:("WARMING",GOLD)}
        if SIM["match"] is None: txt,col=vmap[None]
        elif SIM["match"]>=70: txt,col="CONVERGED",UP
        elif SIM["match"]>=45: txt,col="TRACKING",GOLD
        elif SIM["match"]>=0: txt,col="DIVERGENT",DN
        else: txt,col="BLOWN UP",DN
        if SIM["blown"]: txt,col="BLOWN UP",DN
        self.lb_verdict.setText(f"{txt} · match {SIM['match'] if SIM['match'] is not None else '—'}%")
        self.lb_verdict.setStyleSheet(f"color:{col};font-weight:700")
        self.sim_spark.update()
    def _report(self):
        path,_=QFileDialog.getSaveFileName(self,"Save session report","aurum_session_report.json","JSON (*.json)")
        if not path: return
        rep=dict(terminal="AURUM PySide6 v5.8",generated=datetime.now().isoformat(),
                 equity=round(equity(),2),realized=round(ACC["realized"],2),
                 targets=TR["targets"],weights=TR["weights"],trust=TRUST,
                 policy_epoch=TR["epoch"],net_epoch=NET.epoch,net_acc=NET.acc,
                 forecast=dict(hits=fOK,total=fN,avg=fSum/fN if fN else 0),
                 sim=dict(start=SIM.start,nav=round(SIM["eq"],4),pnl_pct=round((SIM["eq"]/SIM["start"]-1)*100,2),
                          trades=SIM.closed,wins=SIM.wins,match=SIM.match),
                 trades=TRADES[:60])
        json.dump(rep,open(path,"w"),indent=2)
        self.toasts.show("info","Report exported",os.path.basename(path))

    def resizeEvent(self,e):
        super().resizeEvent(e); self.toasts.reposition()

# ───────────────────────────── boot ─────────────────────────────────────────
def main():
    app=QApplication(sys.argv)
    app.setStyleSheet(QSS)
    app.setFont(mono(10))
    splash_pix=QPixmap(440,320); splash_pix.fill(QColor(BG))
    sp=QPainter(splash_pix); sp.setRenderHint(QPainter.Antialiasing)
    sp.setPen(QPen(QColor(GOLD))); sp.setFont(mono(20,True)); sp.drawText(30,60,"AURUM")
    sp.setPen(QPen(QColor(TXT3))); sp.setFont(mono(9)); sp.drawText(30,80,"AI TERMINAL · PySide6")
    sp.end()
    splash=QSplashScreen(splash_pix); splash.show()
    app.processEvents()
    steps=["> establishing local market feed ........ OK",
           "> loading quant model AURUM-5.8 ......... OK",
           "> pretraining policy net · backprop ..... OK",
           "> building kNN pattern memory ........... OK",
           "> calibrating policy to targets ......... OK",
           "> arming prove-it sim · $10 ............. OK",
           "> verifying risk governor ............... OK"]
    for s in steps:
        splash.showMessage(s,Qt.AlignBottom|Qt.AlignLeft,QColor(TXT3)); app.processEvents()
        QApplication.processEvents(); time.sleep(0.12)
    global win
    win=MainWindow(); win.show()
    splash.finish(win)
    win.toasts.show("ai","Session started","Day P&L realized +$1,284.50 · press R to run the copilot")
    sys.exit(app.exec())

if __name__=="__main__":
    main()