#!/bin/bash   
echo "Job Started @ `date`"

echo "Running Jobs"
cd $HOME

module load python3/3.9.2
module load cuda/11.7.0


echo "Load Python Environment named Image"
source /g/data/nk53/rm8989/software/image2/bin/activate

cd /scratch/nk53/rm8989/gene_prediction/code/hist2RNA/
python training_main.py
