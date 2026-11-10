cd D:/ai_ml_project/paper/evidence/new_results
mkdir -p logs
A="official,sep_1deg_snr15,sep_2deg_snr15,sep_3deg_snr15,sep_5deg_snr15,sep_10deg_snr15,sep_20deg_snr15,sep_30deg_snr15,tail_snr-25,tail_snr-20,tail_snr30,tail_snr35"
B="T2_d1,T2_d2,L1_snr0,L1_snr15,L2_snr0,L2_snr15,L4_snr0,L4_snr15"
C="T2_d5,L5_snr0,L5_snr15,L6_snr0,L6_snr15,phase_d0_snr0,phase_d0_snr15,phase_d1_snr0,phase_d1_snr15,phase_d2_snr0,phase_d2_snr15,phase_d5_snr0,phase_d5_snr15"
D="L7_snr0,L7_snr15,L8_snr0,L8_snr15,L10_snr0,L10_snr15,gain_g0.5_snr0,gain_g0.5_snr15,gain_g1_snr0,gain_g1_snr15,gain_g2_snr0,gain_g2_snr15,gain_g4_snr0,gain_g4_snr15,oodphase_d10_snr0,oodphase_d10_snr15,oodphase_d15_snr0,oodphase_d15_snr15"
for X in A B C D; do eval "L=\$$X"; python e13_decode.py "$L" > logs/decode_$X.log 2>&1 & done
wait
