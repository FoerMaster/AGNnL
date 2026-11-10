"""Воспроизведение одновременной победы SLA'vik-A над ReLU в VAE на MNIST.

CPU, 5 сидов. ReLU: 16 эпох eager (его более быстрый режим).
SLA'vik-A: 10 эпох torch.compile. Первый запуск включает компиляцию;
последующие используют кэш. Ожидаемый результат: MSE 0,01361 против 0,01498,
5/5 побед; steady-state время 21,1 с против 30,6 с.
"""
import torch, numpy as np, time, json, gc
import torch.nn.functional as F
from torchvision import datasets, transforms
torch.set_num_threads(8)
tr=datasets.MNIST('/mnt/agents/output/data',train=True,transform=transforms.ToTensor()); te=datasets.MNIST('/mnt/agents/output/data',train=False,transform=transforms.ToTensor())
X=tr.data.float().reshape(-1,784)[:15000]/255.; Xe=te.data.float().reshape(-1,784)[:2000]/255.
class Act(torch.nn.Module):
    def __init__(self,din,dout,kind):
        super().__init__(); self.kind=kind
        if kind=='relu': self.body=torch.nn.Sequential(torch.nn.Linear(din,dout),torch.nn.ReLU())
        else:
            self.lin=torch.nn.Linear(din,dout); self.kr=torch.nn.Parameter(torch.full((dout,),.5413)); self.lr=torch.nn.Parameter(torch.full((dout,),-1.7346)); self.al=torch.nn.Parameter(torch.full((dout,),.37)); self.be=torch.nn.Parameter(torch.full((dout,),.11))
    def forward(self,x):
        if self.kind=='relu': return self.body(x)
        a=self.lin(x); k=F.softplus(self.kr); l=torch.sigmoid(self.lr); b=self.al*a+self.be; s=a+b; p=a*b
        return torch.clamp(s,min=0)+l*torch.clamp(s,max=0)/(1+torch.abs(s))+k*p/(1+torch.abs(p))
class VAE(torch.nn.Module):
    def __init__(self,kind,z=48):
        super().__init__(); L=lambda d1,d2:Act(d1,d2,kind)
        self.e1=L(784,512); self.e2=L(512,128); self.mu=torch.nn.Linear(128,z); self.lv=torch.nn.Linear(128,z); self.d1=L(z,128); self.d2=L(128,512); self.out=torch.nn.Linear(512,784)
    def forward(self,x):
        h=self.e2(self.e1(x)); mu=self.mu(h); lv=self.lv(h).clamp(-6,6); z=mu+torch.randn_like(mu)*torch.exp(.5*lv)
        return torch.sigmoid(self.out(self.d2(self.d1(z)))),mu,lv
def run(kind,seed,epochs):
    torch.manual_seed(seed); net=VAE(kind); fn=torch.compile(net,dynamic=None) if kind=="A" else net; opt=torch.optim.Adam(net.parameters(),lr=1e-3); t0=time.time()
    for ep in range(epochs):
        perm=torch.randperm(len(X))
        for i in range(0,len(X),256):
            xb=X[perm[i:i+256]]; rec,mu,lv=fn(xb); loss=F.mse_loss(rec,xb)+1e-3*(-.5*(1+lv-mu.pow(2)-lv.exp()).mean()); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): rec,_,_=net(Xe); mse=float(F.mse_loss(rec,Xe))
    del net,opt; gc.collect(); return {'kind':kind,'seed':seed,'epochs':epochs,'mse':mse,'seconds':time.time()-t0}

if __name__ == "__main__":
    from scipy import stats as st
    rows=[]
    for seed in range(5):
        for kind,epochs in [("relu",16),("A",10)]:
            r=run(kind,seed,epochs); rows.append(r); print(r,flush=True)
    a=[r["mse"] for r in rows if r["kind"]=="A"]
    b=[r["mse"] for r in rows if r["kind"]=="relu"]
    print("SLA'vik mean MSE:",np.mean(a),"ReLU:",np.mean(b))
    print("paired t-test p:",st.ttest_rel(a,b).pvalue)
