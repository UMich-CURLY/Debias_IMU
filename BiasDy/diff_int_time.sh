#! /bin/bash


source venv/bin/activate
echo "Ablation study for different loss window sizes"

python3 BiasDy/mainEuroc.py --outputdir=results/Euroc_4 --loss_window=4 
python3 BiasDy/mainEuroc.py --outputdir=results/Euroc_8 --loss_window=8 
python3 BiasDy/mainEuroc.py --outputdir=results/Euroc_12 --loss_window=12
python3 BiasDy/mainEuroc.py --outputdir=results/Euroc_16 --loss_window=16
python3 BiasDy/mainEuroc.py --outputdir=results/Euroc_32 --loss_window=32
python3 BiasDy/mainEuroc.py --outputdir=results/Euroc_48 --loss_window=48
python3 BiasDy/mainEuroc.py --outputdir=results/Euroc_64 --loss_window=64
python3 BiasDy/mainEuroc.py --outputdir=results/Euroc_96 --loss_window=96 


python3 BiasDy/ablation.py --outputdir=results/Euroc_4
python3 BiasDy/ablation.py --outputdir=results/Euroc_8
python3 BiasDy/ablation.py --outputdir=results/Euroc_12
python3 BiasDy/ablation.py --outputdir=results/Euroc_16
python3 BiasDy/ablation.py --outputdir=results/Euroc_32
python3 BiasDy/ablation.py --outputdir=results/Euroc_48
python3 BiasDy/ablation.py --outputdir=results/Euroc_64
python3 BiasDy/ablation.py --outputdir=results/Euroc_96
python3 BiasDy/ablation.py --plot_only

