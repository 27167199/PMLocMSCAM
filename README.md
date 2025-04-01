PMLocMSCAM: predicting miRNA subcellular localizations by miRNA similarities and cross-attention mechanism

Requirements

python == 3.7.16
Tensorflow == 2.11.0
scikit-learn ==1.0.2
node2vec ==0.4.3
networkx == 2.6.3
pandas==1.5.2
pytorch==1.12.0
scipy==1.10.0


File
dataset
Basic Information
miRNA_ID_1041.txt: Contains unique identifiers of 1041 miRNAs used in the PMiSLocMF model.
miRNA_have_loc_information_index.txt: Indices of miRNAs with known subcellular localization information.

Association Data
miRNA_disease.csv: Associations between 1041 miRNAs and 640 diseases.
miRNA_drug.csv: Associations between 1041 miRNAs and 130 drugs .
miRNA_mRNA_matrix.txt: Interaction matrix between 1041 miRNAs and 2836 mRNAs .

Similarity Data
miRNA_seq_sim.csv: Sequence similarity matrix for 1041 miRNAs .
miRNA_func_sim.csv: Functional similarity matrix for 1041 miRNAs .

Subcellular Localization Data
miRNA_localization.csv: Subcellular localization annotations for 1041 miRNAs .
mRNA_localization.txt: Subcellular localization annotations for 2836 mRNAs .

Feature
Raw Features (Node2Vec Extracted)
miRNA_seq_feature_64.csv: 64-dimensional miRNA sequence features extracted from the miRNA sequence similarity network using Node2Vec.
miRNA_disease_feature_128.csv: 128-dimensional raw miRNA-disease features extracted from the miRNA-disease association network via Node2Vec.
miRNA_drug_feature_128.csv: 128-dimensional raw miRNA-drug features extracted from the miRNA-drug association network via Node2Vec.
miRNA_mRNA_network_feature_128.csv: 128-dimensional raw miRNA-mRNA features extracted from the miRNA-mRNA association network via Node2Vec.

Co-Localization Features
miRNA_mRNA_co-localization_feature.csv: Co-localization features between miRNAs and mRNAs, derived from their subcellular localization overlaps .

High-Level Features (HGCN-Optimized)
HGCN_feature_disease_0.8_128_0.01.csv: Refined 128-dimensional miRNA-disease features enhanced by HGCN.
HGCN_feature_drug_0.8_128_0.01.csv: Refined 128-dimensional miRNA-drug features  enhanced by HGCN.
HGCN_feature_mRNA_0.8_128_0.01.csv: Refined 128-dimensional miRNA-mRNA features enhanced by HGCN.

Code
First
python“main_diease.py”,“main_drug.py”and“main_mRNA.py”
Then
python“main.py”

Contact
If you have any questions or comments, please feel free to email Cheng Yan(yancheng01@hnucm.edu.cn).