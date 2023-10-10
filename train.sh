python train_supernet.py -c ./NSGA/NSGAD4K7E6W8.yaml -gpu 0 --data_path dataset.pkl
python train_supernet.py -c ./NSGA/NSGAD4K57E6W8.yaml -gpu 0 --data_path dataset.pkl
python train_supernet.py -c ./NSGA/NSGAD4K357E6W8.yaml -gpu 0 --ptr nsga_output/NSGA_D4K57E6W8_1x --data_path dataset.pkl
python train_supernet.py -c ./NSGA/NSGAD4K357E46W8.yaml -gpu 0 --ptr nsga_output/NSGA_D4K357E6W8_1x --data_path dataset.pkl
python train_supernet.py -c ./NSGA/NSGAD4K357E346W8.yaml -gpu 0 --ptr nsga_output/NSGA_D4K357E46W8_1x --data_path dataset.pkl
python train_supernet.py -c ./NSGA/NSGAD34K357E346W8.yaml -gpu 0 --ptr nsga_output/NSGA_D4K357E346W8_1x --data_path dataset.pkl
python train_supernet.py -c ./NSGA/NSGAD234K357E346W8.yaml -gpu 0 --ptr nsga_output/NSGA_D34K357E346W8_1x --data_path dataset.pkl
python train_supernet.py -c ./NSGA/NSGAD1234K357E346W8.yaml -gpu 0 --ptr nsga_output/NSGA_D234K357E346W8_1x --data_path dataset.pkl
