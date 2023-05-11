#python run_nerf.py --config configs/lego.txt
python run_nerf.py --config configs/horns.txt
python run_nerf.py --config configs/trex.txt
python run_nerf.py --config configs/room.txt --expname llff_room_few --fewshot 5
python run_nerf.py --config configs/horns.txt --expname llff_horns_few --fewshot 5
python run_nerf.py --config configs/trex.txt --expname llff_trex_few --fewshot 5