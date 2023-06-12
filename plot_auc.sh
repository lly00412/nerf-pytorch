Logdir=auc
names=(room horns trex)
for NAME in ${names[@]}
do
  echo ${NAME}
  python plot_auc.py --expname llff_${NAME} --savedir ${Logdir}
  python plot_auc.py --expname llff_${NAME}_few --savedir ${Logdir}
  done

python plot_auc.py --expname llff_horns_10_views --savedir auc --est h2s jacobs mcs