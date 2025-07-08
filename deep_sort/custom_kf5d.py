from __future__ import annotations
"""custom_kf5d.py
放置于 deep_sort/deep_sort/ 目录下，使其成为 deep_sort.custom_kf5d 模块。
实现稳定 5 维卡尔曼滤波器:
状态 x=[u,v,h,Z,dZ]^T  测量 z=[u,v,h]^T
"""
import numpy as np

class KF5D:
    chi2inv95 = {1:3.8415,2:5.9915,3:7.8147}

    def __init__(self, dt: float, f_px: float, cx: float, cy: float, H_real: float):
        self.dt = float(dt)
        self.f,self.cx,self.cy,self.H = f_px,cx,cy,H_real
        self.n=5; self.m=3
        self.I = np.eye(self.n)

    # ---------- 内部工具 ----------
    def _Q(self,h):
        q=(h/40)**2; return np.diag([q,q,(h/40)**2,0.25,0.25])
    def _R(self,h):
        r=(h/20)**2; return np.diag([r,r,(h/20)**2])
    def uvh_to_Z(self,h):
        return self.f*self.H/max(h,1e-6)
    def Z_to_h(self,Z):
        return self.f*self.H/max(Z,1e-6)
    def proj(self,X,Y,Z):
        u=self.f*X/Z+self.cx; v=self.f*Y/Z+self.cy; h=self.Z_to_h(Z); return u,v,h
    def backproj(self,u,v,h):
        Z=self.uvh_to_Z(h); X=(u-self.cx)*Z/self.f; Y=(v-self.cy)*Z/self.f; return X,Y,Z

    # ---------- KF 接口 ----------
    def initiate(self,meas):
        u,v,a,h = meas
        Z=self.uvh_to_Z(h); dZ=-8.0
        mean=np.array([u,v,h,Z,dZ])
        P=np.diag([(h/2)**2]*3+[1.0,4.0])
        return mean,P

    def predict(self,mean,cov):
        u,v,h,Z,dZ=mean
        Zp=max(0.1,Z+dZ*self.dt)
        X,Y,_=self.backproj(u,v,h)
        up,vp,hp=self.proj(X,Y,Zp)
        mean_pred=np.array([up,vp,hp,Zp,dZ])
        F=self.I.copy(); F[3,4]=self.dt
        cov_pred=F@cov@F.T + self._Q(hp)
        return mean_pred,cov_pred

    def project(self,mean,cov):
        H=np.zeros((self.m,self.n)); H[0,0]=H[1,1]=H[2,2]=1
        S=H@cov@H.T + self._R(mean[2])
        z_pred=mean[:3]
        return z_pred,S,H

    def update(self,mean,cov,meas):
        z_pred,S,H=self.project(mean,cov)
        K=cov@H.T@np.linalg.inv(S)
        delta=meas[[0,1,3]]-z_pred
        mean_new=mean+K@delta
        cov_new=(self.I-K@H)@cov
        return mean_new,cov_new

    def gating_distance(self,mean,cov,measurements,only_position=False):
        z_pred,S,_=self.project(mean,cov)
        diff=measurements[:,[0,1,3]]-z_pred
        invS=np.linalg.inv(S)
        d=np.einsum('ij,jk,ik->i',diff,invS,diff)
        return d 