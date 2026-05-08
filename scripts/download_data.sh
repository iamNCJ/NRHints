#!/bin/bash

# Modify this to your data directory
DATA_DIR=./training_data

HF_BASE=https://huggingface.co/datasets/NRHints/NRHints/resolve/main/Data


# Real data
mkdir -p $DATA_DIR/Real

wget $HF_BASE/Real/Cat.zip -O $DATA_DIR/Real/Cat.zip -q --show-progress
unzip $DATA_DIR/Real/Cat.zip -d $DATA_DIR/Real

wget $HF_BASE/Real/FurScene.zip -O $DATA_DIR/Real/FurScene.zip -q --show-progress
unzip $DATA_DIR/Real/FurScene.zip -d $DATA_DIR/Real

wget $HF_BASE/Real/Pixiu.zip -O $DATA_DIR/Real/Pixiu.zip -q --show-progress
unzip $DATA_DIR/Real/Pixiu.zip -d $DATA_DIR/Real

wget $HF_BASE/Real/Fish.zip -O $DATA_DIR/Real/Fish.zip -q --show-progress
unzip $DATA_DIR/Real/Fish.zip -d $DATA_DIR/Real

wget $HF_BASE/Real/CatSmall.zip -O $DATA_DIR/Real/CatSmall.zip -q --show-progress
unzip $DATA_DIR/Real/CatSmall.zip -d $DATA_DIR/Real

wget $HF_BASE/Real/CupFabric.zip -O $DATA_DIR/Real/CupFabric.zip -q --show-progress
unzip $DATA_DIR/Real/CupFabric.zip -d $DATA_DIR/Real

wget $HF_BASE/Real/Pikachu.zip -O $DATA_DIR/Real/Pikachu.zip -q --show-progress
unzip $DATA_DIR/Real/Pikachu.zip -d $DATA_DIR/Real


# Synthetic data
mkdir -p $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Cup_Plane_Diffuse_PL_500.zip -O $DATA_DIR/Synthetic/Cup_Plane_Diffuse_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Cup_Plane_Diffuse_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Cup_Plane_Metal_PL_500.zip -O $DATA_DIR/Synthetic/Cup_Plane_Metal_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Cup_Plane_Metal_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Cup_Plane_Metal_Rough_PL_500.zip -O $DATA_DIR/Synthetic/Cup_Plane_Metal_Rough_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Cup_Plane_Metal_Rough_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Cup_Plane_Metal_VeryRough_PL_500.zip -O $DATA_DIR/Synthetic/Cup_Plane_Metal_VeryRough_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Cup_Plane_Metal_VeryRough_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Cup_Plane_Metal_Aniso_PL_500.zip -O $DATA_DIR/Synthetic/Cup_Plane_Metal_Aniso_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Cup_Plane_Metal_Aniso_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Cup_Plane_NonMetal_PL_500.zip -O $DATA_DIR/Synthetic/Cup_Plane_NonMetal_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Cup_Plane_NonMetal_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Cup_Plane_NonMetal_Rough_PL_500.zip -O $DATA_DIR/Synthetic/Cup_Plane_NonMetal_Rough_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Cup_Plane_NonMetal_Rough_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Cup_Plane_NonMetal_VeryRough_PL_500.zip -O $DATA_DIR/Synthetic/Cup_Plane_NonMetal_VeryRough_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Cup_Plane_NonMetal_VeryRough_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Cup_Plane_ShortFur_PL_500.zip -O $DATA_DIR/Synthetic/Cup_Plane_ShortFur_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Cup_Plane_ShortFur_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Cup_Plane_LongFur_PL_500.zip -O $DATA_DIR/Synthetic/Cup_Plane_LongFur_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Cup_Plane_LongFur_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Cup_Plane_SSS_PL_500.zip -O $DATA_DIR/Synthetic/Cup_Plane_SSS_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Cup_Plane_SSS_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/FurBall_PL_500.zip -O $DATA_DIR/Synthetic/FurBall_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/FurBall_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Basket_PL_500.zip -O $DATA_DIR/Synthetic/Basket_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Basket_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Complex_Ball_PL_500.zip -O $DATA_DIR/Synthetic/Complex_Ball_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Complex_Ball_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Drums_PL_500.zip -O $DATA_DIR/Synthetic/Drums_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Drums_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Hotdog_PL_500.zip -O $DATA_DIR/Synthetic/Hotdog_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Hotdog_PL_500.zip -d $DATA_DIR/Synthetic

wget $HF_BASE/Synthetic/Lego_PL_500.zip -O $DATA_DIR/Synthetic/Lego_PL_500.zip -q --show-progress
unzip $DATA_DIR/Synthetic/Lego_PL_500.zip -d $DATA_DIR/Synthetic
