import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O='D:/ai_ml_project/paper/evidence/adc_bits_test'; R={b:json.load(open(f'{O}/res_b{b}.json')) for b in (1,2,3,4)}
M=list(R[1]['pool'].keys())
L=['# SUMMARY: pooled unknown-SNR Pd (paper-style; strict is identical, no dropped scenes), 450 fresh scenes per bit depth (150 at each of 0/10/20 dB, seeds 14000/14010/14020), L=3, peak-AGC ADC, ideal hardware','',
'| method | 1 bit | 2 bits | 3 bits | 4 bits |','|---|---|---|---|---|']
for m in M: L.append(f'| {m} | '+' | '.join(f'{R[b]["pool"][m]["paper"]:.3f}' for b in (1,2,3,4))+' |')
L+=['','Pd loss when the assumed noise level is wrong (ML known sigma minus variant, pooled; positive = loss):','','| variant | 1 bit | 2 bits | 3 bits | 4 bits |','|---|---|---|---|---|']
def loss(b,m): return R[b]['pool']['ML known sigma']['paper']-R[b]['pool'][m]['paper']
for m in ['ML snr+5','ML snr-5','ML nominal 10dB','ML joint sigma']: L.append(f'| {m} | '+' | '.join(f'{loss(b,m):+.3f}' for b in (1,2,3,4))+' |')
L+=['','Best method per bit depth (point estimate): '+'; '.join(f'{b} bit: '+max(M,key=lambda m:R[b]['pool'][m]['paper'])+f' {max(R[b]["pool"][m]["paper"] for m in M):.3f}' for b in (1,2,3,4)),
'','## Reading','- CNN read-out beats NOMP only at 1-2 bits (+0.151*, +0.199*), ties at 3 bits (+0.022 n.s.) and loses at 4 bits (-0.196*). It is below quantiser ML with known sigma at every bit depth (-0.095*, -0.089*, -0.154*, -0.262*).',
'- Against SNR-blind ML (fixed 10 dB) the CNN read-out is worse at 1 bit (-0.046*), equal at 2 bits (+0.020, n.s.), worse at 3-4 bits: "network needs no noise level" helps only at ~2 bits.',
'- Hybrid (ML from CNN seeds) helps at 1 bit (+0.014 to +0.019* over matching ML), is neutral at 2-3 bits, and slightly hurts at 4 bits (-0.005 to -0.009*). 3-4 bits favour the classical decoder, as expected.']
open(f'{O}/SUMMARY.md','w').write('\n'.join(L)+'\n'); print('\n'.join(L))
fig,ax=plt.subplots(1,2,figsize=(11,4.2)); bs=[1,2,3,4]
for m in M:
    ax[0].plot(bs,[R[b]['pool'][m]['paper'] for b in bs],marker='o',ms=3,lw=1.6 if m in('NOMP','CNN read-out','ML known sigma','hybrid nominal 10dB') else 0.8,label=m)
ax[0].set_xlabel('ADC bits'); ax[0].set_ylabel('pooled Pd (paper-style)'); ax[0].set_xticks(bs); ax[0].set_title('Pd vs bits (unknown SNR, pooled)'); ax[0].legend(fontsize=5.5,ncol=2)
for m in ['ML snr+5','ML snr-5','ML nominal 10dB','ML joint sigma']: ax[1].plot(bs,[loss(b,m) for b in bs],marker='o',label=m)
ax[1].axhline(0,c='k',lw=.5); ax[1].set_xlabel('ADC bits'); ax[1].set_ylabel('Pd loss vs ML known sigma'); ax[1].set_xticks(bs); ax[1].set_title('Cost of a wrong / unknown noise level'); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{O}/fig_bits.png',dpi=90)
