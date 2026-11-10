import json,glob,os
R={os.path.basename(f)[:-5]:json.load(open(f)) for f in glob.glob('bank_json/*.json')}
D=['NOMP - U-Net','hybrid - NOMP','NOMP+selfcal - hybrid','NOMP+selfcal - NOMP']
print(len(R),'banks')
for k in ['pd_paper','pd_strict']:
    for d in D:
        pos=[n for n,r in R.items() if r['diffs'][d][k][1]>0]; neg=[n for n,r in R.items() if r['diffs'][d][k][2]<0]
        print(k,d,'interval>0:',len(pos),'interval<0:',len(neg),'contains 0:',len(R)-len(pos)-len(neg))
        if d!='NOMP - U-Net': print('   <0:',neg); print('   >0:',pos)
print({d:[round(x,4) for x in R['official']['diffs'][d]['pd_strict']] for d in D})
print({d:[round(x,4) for x in R['official']['diffs'][d]['rmse']] for d in D})
print({m:round(R['official']['methods'][m]['mean']['rmse'],4) for m in R['official']['methods']})
