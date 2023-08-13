#!/usr/bin/env bash
echo "Running Jobs"
cd $HOME


echo "Transferring Data to Temporary Directory"
cp /g/data/nk53/rm8989/data/raw_wsi_tcga_images.tar.gz $PBS_JOBFS
cd $PBS_JOBFS
echo "Extracting Data in Temporary Directory"
tar -xvf raw_wsi_tcga_images.tar.gz
