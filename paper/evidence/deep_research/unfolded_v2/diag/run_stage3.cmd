@echo off
cd /d D:\ai_ml_project\paper\evidence\deep_research\unfolded_v2\diag
python -u gen_cands.py nosep > gen_cands_nosep.log 2>&1
python -u gen_cands.py sep10 > gen_cands_sep10.log 2>&1
python -u configs.py official > configs_official.log 2>&1
python -u configs.py val > configs_val.log 2>&1
python -u analyze.py nosep sep10 > analyze.log 2>&1
echo STAGE3_DONE > stage3.done
