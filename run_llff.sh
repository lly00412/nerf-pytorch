Logdir=auc
names=(fern flower fortress horns leaves orchids room trex)
for NAME in ${names[@]}
do
  echo ${NAME}
  python run_nerf.py --config configs/${NAME}.txt --expname llff_${NAME}_10_views --fewshot 10
  python run_nerf.py --config configs/${NAME}.txt --expname llff_${NAME}_10_views --fewshot 10 --eval_only --save_video --mc_dropout --n_passes 30
  python plot_auc.py --expname llff_${NAME}_10_views --savedir ${Logdir} --est jacobs mcs --metric psnr ssim lpips
done