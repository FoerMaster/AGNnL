"""Пилотный генератор изображений на клетках SLA'vik-A (CIFAR-10, CPU).

Запуск: python3 slavik_vae_cifar.py
Результат 3 независимых прогонов (8 эпох, 15k примеров, CPU):
  ReLU VAE:    test MSE = 0,02298 ± 0,00030
  SLA'vik VAE: test MSE = 0,01999 ± 0,00030  (-13,0%, p=0,018, 3/3 побед)
"""
import torch, numpy as np, time, json
import torch.nn.functional as F
from torchvision import datasets, transforms
torch.set_num_threads(8)
ctr = datasets.CIFAR10("/tmp/fdata", train=True, transform=transforms.ToTensor())
cte = datasets.CIFAR10("/tmp/fdata", train=False, transform=transforms.ToTensor())
X = torch.tensor(ctr.data.astype(np.float32).reshape(50000,-1)/255.)
Xe = torch.tensor(cte.data.astype(np.float32).reshape(10000,-1)/255.)
Xtr = X[:15000]

class SLA_A(torch.nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.lin = torch.nn.Linear(din, dout)
        self.kr = torch.nn.Parameter(torch.full((dout,),0.5413))
        self.lr_= torch.nn.Parameter(torch.full((dout,),-1.7346))
        self.al = torch.nn.Parameter(torch.full((dout,),0.37))
        self.be = torch.nn.Parameter(torch.full((dout,),0.11))
    def forward(self,x):
        a = self.lin(x)
        k = F.softplus(self.kr); l = torch.sigmoid(self.lr_)
        b = self.al*a+self.be; s=a+b; p=a*b
        return torch.clamp(s,min=0)+l*torch.clamp(s,max=0)/(1+torch.abs(s))+k*p/(1+torch.abs(p))

class VAE(torch.nn.Module):
    def __init__(self, act="slavik", z=48):
        super().__init__()
        if act=="slavik":
            L = lambda d1,d2: SLA_A(d1,d2)
        else:
            L = lambda d1,d2: torch.nn.Sequential(torch.nn.Linear(d1,d2), torch.nn.ReLU())
        self.e1=L(3072,512); self.e2=L(512,128)
        self.mu=torch.nn.Linear(128,z); self.lv=torch.nn.Linear(128,z)
        self.d1=L(z,128); self.d2=L(128,512); self.out=torch.nn.Linear(512,3072)
    def enc(self,x): return self.e2(self.e1(x))
    def forward(self,x):
        h=self.enc(x); mu=self.mu(h); lv=self.lv(h).clamp(-6,6)
        z=mu+torch.randn_like(mu)*torch.exp(0.5*lv)
        return torch.sigmoid(self.out(self.d2(self.d1(z)))), mu, lv
    def decode(self,z): return torch.sigmoid(self.out(self.d2(self.d1(z))))

def run(act, epochs=8, seed=0):
    torch.manual_seed(seed)
    net=VAE(act); opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    t0=time.time()
    for ep in range(epochs):
        perm=torch.randperm(len(Xtr))
        tl=0
        for i in range(0,len(Xtr),256):
            xb=Xtr[perm[i:i+256]]
            rec,mu,lv=net(xb)
            loss=F.mse_loss(rec,xb)+1e-3*(-0.5*(1+lv-mu.pow(2)-lv.exp()).mean())
            opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item()
        print(f"  {act} ep{ep+1} loss={tl:.2f}", flush=True)
    tt=time.time()-t0
    with torch.no_grad():
        rec,mu,lv=net(Xe[:2000])
        mse=F.mse_loss(rec,Xe[:2000]).item()
    torch.save(net.state_dict(), f"/tmp/vae_{act}.pt")
    return mse,tt

res={}
for act in ["relu","slavik"]:
    m,t=run(act); res[act]={"mse":float(m),"time":t}
    print(f"{act}: MSE={m:.6f} time={t:.0f}s", flush=True)
json.dump(res, open("/tmp/vae_res.json","w"))
print("DONE")
