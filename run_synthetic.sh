python run_nerf.py --config configs/room.txt --expname debug
python run_nerf.py --config configs/horns.txt --eval_only
python run_nerf.py --config configs/trex.txt --eval_only
python run_nerf.py --config configs/room.txt --expname llff_room_few --fewshot 5 --eval_only
python run_nerf.py --config configs/horns.txt --expname llff_horns_few --fewshot 5 --eval_only
python run_nerf.py --config configs/trex.txt --expname llff_trex_few --fewshot 5 --eval_only
python run_nerf.py --config configs/horns.txt --eval_only --expname llff_horns_10_views --mc_dropout --n_passes 30